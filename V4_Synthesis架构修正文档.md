# V4 架构修正：Synthesis as Auxiliary Query

## 关键发现

### 发现 1: 已生成数据的边结构与 config 不一致

**output/ 中数据的 edge_index** (由早期 config 生成):
```
(0,2)(0,3)(1,2)(1,3)(2,4)(2,5)(3,4)(3,5)  # 时序正向
(0,1)(1,0)(2,3)(3,2)(4,5)(5,4)             # 对抗
(4,6)(5,6)                                  # 仅 Closing→Synthesis
```
共 16 条边。**Synthesis 节点只接收 2 条入边**（从 Closing），严重欠连通。

### 发现 2: 论文中 Synthesis 根本不是图节点

重读论文 **Algorithm 1**:
```
15: for each debate interaction (node) 𝑛_i in 𝐷 do    # D 是辩论日志，6 条发言
16:    Obtain node 𝑛_i embedding ℎ_i
18: Build Debate graph 𝐺=(𝑉,𝐸) with nodes 𝑉={ℎ_i}    # 图仅来自辩论交互
...
23: return ˆ𝑦, explanatory reasons R={𝑆,𝐷}            # S 只是解释，不参与预测
```

论文 §3.4.1 原话："This summary 𝑆 **bridges** the debate discourse and the subsequent analytical processes" —— Synthesis 是**桥接**，不是图节点。

---

## V4 修正方案

### 1. config.py：边结构只保留辩论节点间的边

```python
EDGE_LIST = [
    # 时序正向 (Stage1→Stage2, Stage2→Stage3): 8 条
    (0,2),(0,3),(1,2),(1,3),(2,4),(2,5),(3,4),(3,5),
    # 对抗边 (双向): 6 条
    (0,1),(1,0),(2,3),(3,2),(4,5),(5,4),
    # 时序反向 (让 Opening/Questioner 也能感知后续发言): 8 条
    (2,0),(3,0),(2,1),(3,1),(4,2),(5,2),(4,3),(5,3),
]
# 共 22 条边，只涉及节点 0-5
```

**反向时序边的意义**：
- 原始论文只有正向时序，导致 Opening 节点(0,1) 在 2 层 GAT 后仍看不到 Closing 信息
- 反向时序让 Opening 能"回顾"后续发言，更好地理解全局辩论

### 2. dataset.py：自动分离辩论节点 + Synthesis

```python
def __getitem__(self, idx):
    record = load_json()
    nodes = record['nodes']

    # 分离
    debate_nodes = [n for n in nodes if n['role_id'] != 6]  # 6 个辩论节点
    synth_text = [n['text'] for n in nodes if n['role_id'] == 6][0]

    # 按 role_id 排序
    debate_nodes.sort(key=lambda n: n['role_id'])

    # 使用 config.EDGE_LIST（不使用 JSON 里的 edge_index）
    edge_index = self.debate_edge_index.clone()

    # Tokenize 辩论节点 + 新闻 + Synthesis (独立三路)
    ...
    return Data(
        node_input_ids=...,           # [6, L]
        role_ids=...,                  # [6]
        edge_index=...,                # [2, 22]
        news_input_ids=...,            # [1, L]
        synth_input_ids=...,           # [1, L]  ← 新字段
        synth_attention_mask=...,      # [1, L]  ← 新字段
        y=label,
        num_nodes=6,
    )
```

**优势**：
- 无需重新生成数据（output/ 里的 JSON 不动）
- Dataset 自动适配新旧数据格式

### 3. networks.py：V4 架构

```
┌──────────────────────────────────────────────────────────┐
│                  TEDClassifier V4                         │
│                                                            │
│  [6 辩论节点] → BERT+Role → GAT(2层) → node_dense         │
│                                            │               │
│                         ┌──────────────────┴─────┐         │
│                         ▼                        ▼         │
│                   ┌───────────┐         ┌──────────────┐   │
│  [News] → BERT → │ News Query├─────────▶│ Debate Nodes │   │
│                  │ MHA       │  ──c_news┤ (K, V)       │   │
│                  └───────────┘         └──────────────┘   │
│                                                            │
│                  ┌──────────────┐        ┌──────────────┐  │
│  [Synth]→BERT → │Synth Query   ├───────▶│ Debate Nodes │  │
│                 │MHA           │ ─c_synth┤(K, V)        │  │
│                 └──────────────┘        └──────────────┘  │
│                                                            │
│  Pool(Pro) vs Pool(Opp) → divergence                      │
│  Pool(All) → g_global                                     │
│                                                            │
│  [g_global; c_news; c_synth; divergence] → MLP → 2 logits │
└──────────────────────────────────────────────────────────┘
```

**核心创新**：

| 组件 | 设计意图 | 对应论文 |
|------|---------|---------|
| News Cross-Attention | News 关注最相关的辩论论据 | Eq.10 (修复退化) |
| **Synth Cross-Attention** | Synthesis 作为"桥梁"再次审视辩论 | §3.4.1 "bridges" |
| Pro/Opp 双分支池化 | 显式建模对抗结构 | "two opposing teams" |
| Divergence 信号 | 量化辩论分歧强度 | "truth through debate" |

### 4. 超参数回归稳定版

V3 实验（Focal Loss + 大容量 256）反而下降，回退到：
```python
GAT_HIDDEN_DIM = 128    # 回退
PROJ_DIM = 128          # 回退
GAT_DROPOUT = 0.2
CLASSIFIER_DROPOUT = 0.2
LEARNING_RATE = 1e-4    # 稳定值
USE_FOCAL_LOSS = False  # 回退到 CE + class_weight
```

---

## V3 失败原因分析（从 train.log 看）

V3 训练动态：
```
Epoch 1: Train Loss=0.1733, Val macF1=0.3775  ← 初始化混乱
Epoch 5: Train Loss=0.0904, Val macF1=0.7598  ← 开始收敛
Epoch 11: Train Loss=0.0076, Val macF1=0.7620  ← 最佳但已过拟合
Epoch 19: 触发 Early Stopping
```

**失败原因**：
1. **Focal Loss + 大容量同时启用**：损失极小（0.007），梯度信号弱
2. **一次改太多**：容量 ↑、损失函数变、架构变，无法定位收益来源
3. **图结构仍含 Synthesis**（未修复数据一致性问题）

---

## 预期 V4 表现

| 指标 | V3 失败 | V4 预期 | 论文 |
|------|---------|--------|------|
| Acc | 0.8426 | 0.88-0.89 | 0.892 |
| macF1 | 0.7467 | 0.79-0.80 | 0.803 |
| F1(Fake) | 0.5909 | 0.65-0.68 | 0.674 |

**V4 的关键优势**：
1. **架构符合论文原意**：图只含辩论节点，Synthesis 作为桥梁
2. **修复 MHA 退化**：真正的节点级注意力
3. **双 Query 设计**：News + Synthesis 两路交互
4. **保留立场分歧**：Pro/Opp 显式建模
5. **稳定超参数**：回归论文原始值，避免噪声

---

## 运行命令

```bash
# 不需要重新生成数据！现有的 output/*.json 兼容
python main_train.py --dataset en --epochs 30 --batch_size 4 --freeze_layers 6
```

## 消融实验建议

如果 V4 效果仍不理想，按以下顺序调整：

1. **关闭 Synthesis Query**：测试其是否真的有帮助
2. **关闭 divergence 信号**：测试立场分歧的贡献
3. **恢复反向时序边移除**：仅保留论文原始 14 条边
4. **尝试 BERT_FREEZE_LAYERS=0 / 8**：调整微调深度
5. **启用 Focal Loss + gamma=1.0**：更温和的聚焦

---

## 代码改动清单

| 文件 | 改动 |
|------|------|
| config.py | EDGE_LIST 只保留节点 0-5（22 条边）；回归稳定超参 |
| insight_flow/dataset.py | 分离辩论节点/Synthesis；使用 config.EDGE_LIST |
| insight_flow/networks.py | V4 架构：双 Query Cross-Attention + Pro/Opp 池化 |
| insight_flow/train.py | forward 传递 synth_input_ids/attention_mask |
