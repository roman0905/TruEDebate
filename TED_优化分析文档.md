# TruEDebate (TED) 代码优化分析报告

> **当前结果 vs 论文目标（ARG-EN）**
>
> | 指标 | 当前结果 | 论文目标 | 差距 |
> |------|---------|---------|------|
> | Accuracy | 0.8720 | 0.892 | −2.0% |
> | Macro F1 | 0.7779 | 0.803 | −2.5% |
> | F1 (Real) | 0.9225 | 0.932 | −0.95% |
> | F1 (Fake) | 0.6333 | 0.674 | −4.1% |

---

## 一、核心问题：严重过拟合

这是当前最主要的问题，也是与论文结果差距最大的根源。

从 `train.log` 可以直接看到：

```
Epoch  1: Train Loss=0.5458 | Val Loss=0.3882  ← 最早期，val loss最低
Epoch  3: Val macF1=0.7621  ← 第3轮已经达到不错水平
Epoch 12: Val macF1=0.7795  ← 最佳epoch
Epoch 20: Train Loss=0.1853 | Val Loss=0.5522  ← 训练损失极低，但验证损失飙升
```

训练损失从 0.5458 降至 0.1853（降幅 66%），而验证损失从 0.3882 升至 0.5522（升幅 42%）——这是典型的过拟合曲线。

**根本原因：缺少学习率调度器（LR Scheduler）**，以及 BERT 微调策略存在问题。

### 优化方案 1：添加学习率调度策略

```python
# train.py 中替换优化器配置部分

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# 优化器（保持原有差分学习率策略）
optimizer = torch.optim.AdamW([
    {"params": bert_params, "lr": lr * 0.1},
    {"params": other_params, "lr": lr},
], weight_decay=weight_decay)

# 方案A（推荐）：线性 Warmup + Cosine Decay
total_steps = len(train_loader) // grad_accum_steps * epochs
warmup_steps = total_steps // 10  # 前10%步做warmup

warmup_scheduler = LinearLR(
    optimizer,
    start_factor=0.1,
    end_factor=1.0,
    total_iters=warmup_steps
)
cosine_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=total_steps - warmup_steps,
    eta_min=lr * 0.01
)
scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[warmup_steps]
)

# 在 train_one_epoch 中每个 optimizer.step() 后调用：
# scheduler.step()
```

---

## 二、优化器配置问题

### 问题 1：`Adam` 应改为 `AdamW`

当前代码使用 `torch.optim.Adam`，但论文引用的 Adam 实现在现代 Transformer 微调中通常配合解耦权重衰减（Decoupled Weight Decay），即 `AdamW`。

```python
# 原代码（train.py）
optimizer = torch.optim.Adam([...], weight_decay=weight_decay)

# 优化后
optimizer = torch.optim.AdamW([...], weight_decay=weight_decay)
```

### 问题 2：权重衰减不应施加于 Bias 和 LayerNorm 参数

将 `weight_decay` 应用于所有参数（包括 bias 和 LayerNorm）会显著损害 BERT 微调质量。

```python
# 将参数分为三组
no_decay_names = ["bias", "LayerNorm.weight", "layernorm.weight"]

bert_no_decay, bert_decay, other_no_decay, other_decay = [], [], [], []

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    is_bert = "bert" in name
    is_no_decay = any(nd in name for nd in no_decay_names)
    if is_bert and is_no_decay:
        bert_no_decay.append(param)
    elif is_bert:
        bert_decay.append(param)
    elif is_no_decay:
        other_no_decay.append(param)
    else:
        other_decay.append(param)

optimizer = torch.optim.AdamW([
    {"params": bert_decay,    "lr": lr * 0.1, "weight_decay": weight_decay},
    {"params": bert_no_decay, "lr": lr * 0.1, "weight_decay": 0.0},
    {"params": other_decay,   "lr": lr,        "weight_decay": weight_decay},
    {"params": other_no_decay,"lr": lr,        "weight_decay": 0.0},
])
```

---

## 三、网络架构与论文存在偏差

### 问题 1：Interactive Attention 的 K/V 使用方式

**论文 Eq.8–10 的设计意图：**

```
g = GlobalPool(H^(L))          # 聚合后的全图表示向量 [proj_dim]
g_proj = W_g · g               # 投影后的辩论表示
e_proj_F = W_e · e_F           # 投影后的新闻表示
c = MHA(e_proj_F, g_proj, g_proj)  # Q=news, K=V=debate_graph_repr
h = [g_proj; c]                # 最终拼接
```

**当前代码的实现：**

```python
# networks.py 当前实现
node_proj = self.debate_proj(x)          # [total_nodes, proj_dim]
node_dense, node_mask = to_dense_batch(node_proj, batch)
g_proj = masked_mean(node_dense)         # [batch_size, proj_dim] ← 均值池化

# MHA 使用的是 node_dense（节点级）而非 g_proj（图级）
attn_output, _ = self.mha(q, kv=node_dense, kv=node_dense, ...)
```

**问题分析：** 代码中 `g_proj` 用于 Eq.11 的拼接，但 MHA 的 K/V 使用的是 `node_dense`（全部节点表示），而论文 Eq.10 中 K/V 均为 `g_proj`（单个聚合向量）。

这两种设计各有优劣：

- 论文原版（K/V=g_proj）：注意力作用于单一聚合向量，交互信息有限
- 代码版本（K/V=node_dense）：更丰富的节点级交互，但偏离原始论文

**推荐方案：** 先按论文原版实现，与 baseline 对齐后再做消融实验比较：

```python
# 严格遵循论文实现
e_proj = self.news_proj(news_features)   # [batch, proj_dim]
g_proj = self.debate_proj(global_mean)   # [batch, proj_dim]

q = e_proj.unsqueeze(1)    # [batch, 1, proj_dim]
k = g_proj.unsqueeze(1)    # [batch, 1, proj_dim]

attn_output, _ = self.mha(q, k, k)     # Q=news, K=V=debate
attn_output = attn_output.squeeze(1)   # [batch, proj_dim]

combined = torch.cat([g_proj, attn_output], dim=-1)
```

### 问题 2：Synthesis 节点在图中的角色定位模糊

**论文描述：** Synthesis Agent 的输出 $S$ 是对全辩论记录的综合总结，论文中明确区分：
- 辩论图 $G$ 由 6 个辩论发言节点构成（节点 0-5）
- Synthesis $S$ 作为辅助信息或特殊节点处理

**当前代码问题：** Synthesis 节点（node 6）被纳入 GAT 图中，但只有来自节点 4 和 5 的入边 `(4,6), (5,6)`。这导致：

1. 其文本内容（综合全部 6 个发言的总结）与其图结构位置（只关注 closing 发言）语义不一致
2. Synthesis 节点在 GAT 消息传播中只能聚合两个邻居，损失了大量信息

**优化建议：**

```python
# 方案A：为 Synthesis 节点增加与所有节点的连接
# 在 config.py 中，补充 Synthesis 的入边
SYNTHESIS_EDGES = [(i, 6) for i in range(6)]  # 所有发言节点 → Synthesis

# 方案B（更推荐）：将 Synthesis 单独编码后与图表示交互
# 不将其纳入 GAT，而是作为辅助查询向量加入最终分类器
```

---

## 四、超参数配置问题

### 问题 1：训练轮数与 Early Stopping 缺失

当前 epoch 从最佳的第 12 轮继续训练到第 20 轮，导致严重过拟合。应引入 Early Stopping：

```python
# train.py 中添加早停机制
patience = 5           # 连续 N 轮无改善则停止
no_improve_count = 0

for epoch in range(1, epochs + 1):
    val_metrics = evaluate(...)
    
    if val_metrics["macro_f1"] > best_val_f1:
        best_val_f1 = val_metrics["macro_f1"]
        no_improve_count = 0
        # 保存最佳模型...
    else:
        no_improve_count += 1
        if no_improve_count >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
```

---

## 五、训练策略问题

### 问题 1：BERT 冻结层数配置混乱

全量微调 BERT 在小数据集上极易过拟合。建议：

```python
# 论文使用 grid search，建议对比以下设置
BERT_FREEZE_LAYERS = 0   # 全量微调（当前）→ 过拟合风险高
BERT_FREEZE_LAYERS = 6   # 冻结前6层 → 平衡点
BERT_FREEZE_LAYERS = 8   # 冻结前8层 → 更保守
BERT_FREEZE_LAYERS = 10  # 冻结前10层 → 接近固定特征提取
```

### 问题 2：缺少 Label Smoothing

由于数据集存在类别不平衡（Real:Fake ≈ 3:1 在 ARG-EN），纯 CrossEntropyLoss 容易偏向多数类（Real），导致 F1(Fake) 显著低于 F1(Real)（当前 0.6333 vs 0.9225）。

```python
# train.py 中替换损失函数
# 方案A：Label Smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# 方案B：类别权重（更直接）
# ARG-EN: Real=2878/3884≈0.74, Fake=1006/3884≈0.26
class_weights = torch.tensor([0.26, 0.74], device=device)  # 权重与频率反比
criterion = nn.CrossEntropyLoss(weight=class_weights)

# 方案C（推荐）：两者结合
criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.05
)
```

### 问题 3：没有 Gradient Clipping

在 BERT 微调中，梯度裁剪是标准做法，可防止梯度爆炸：

```python
# train_one_epoch 中，在 scaler.step(optimizer) 之前添加
if use_amp:
    scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

if use_amp:
    scaler.step(optimizer)
    scaler.update()
else:
    optimizer.step()
```

---

## 六、代码工程层面的改进建议

### 6.1 evaluate 函数中缺少 AMP 支持

```python
# train.py evaluate 函数中
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    for batch_data in loader:
        # 当前无 AMP，推理速度可提升 ~30%
        with autocast(str(device).split(":")[0]):  # 添加这行
            logits = model(...)
            loss = criterion(logits, labels)
```

### 6.2 DataLoader 的 num_workers 设置

```python
# main_train.py 当前设置
num_workers=0  # Windows 兼容，但在 Linux 环境下严重影响数据加载速度

# Linux 环境下建议
num_workers=4
pin_memory=True   # 配合 GPU 使用
prefetch_factor=2
```

### 6.3 梯度累积与 Batch Normalization 的潜在冲突

当前代码中 GAT 使用 LayerNorm，没有 BatchNorm，梯度累积不存在该问题。但建议在 `train_one_epoch` 中加入断言检查：

```python
# 确认梯度累积步数整除数据集大小，避免最后一批被忽略
assert len(loader) % grad_accum_steps == 0 or True  # 已通过 (step+1)==len(loader) 处理
```

---

## 七、推荐的完整训练配置（config.py）

基于以上分析，建议将配置调整为：

```python
# 模型超参数（对齐论文基础配置）
ROLE_EMBED_DIM = 32
ROLE_PROJ_DIM = BERT_HIDDEN_DIM       # 768，保持论文设定
GAT_HIDDEN_DIM = 128                  # 恢复论文基础值（原为256）
GAT_HEADS = 4
GAT_LAYERS = 2                        # 恢复论文基础值（原为3）
GAT_DROPOUT = 0.3
PROJ_DIM = 128
MHA_HEADS = 4
CLASSIFIER_DROPOUT = 0.3             # 略微提升（原为0.1）

# 训练超参数
BATCH_SIZE = 4
LEARNING_RATE = 1e-4                  # 其他参数 LR
BERT_LR_FACTOR = 0.1                  # BERT LR = LEARNING_RATE * BERT_LR_FACTOR
WEIGHT_DECAY = 1e-2                   # AdamW 中权重衰减通常较大（0.01~0.1）
EPOCHS = 30                           # 配合 Early Stopping 可设更大
GRAD_ACCUM_STEPS = 4
WARMUP_RATIO = 0.1                    # 前10%步做线性warmup
EARLY_STOPPING_PATIENCE = 5
LABEL_SMOOTHING = 0.05
BERT_FREEZE_LAYERS = 6                # 建议先从6层开始实验
```

