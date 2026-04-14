---

# 论文复现指南：TruEDebate (TED) 框架

## 1. 实验环境与依赖项 (Dependencies)
请在复现时配置以下 Python 库：
*   **大语言模型及多智能体**：`openai` (调用 GPT-4o-mini), `mesa` (定义多智能体场景与交互)
*   **深度学习框架**：`torch` (PyTorch 核心框架)
*   **图神经网络**：`torch_geometric` (PyG，用于实现 GAT)
*   **预训练语言模型**：`transformers` (HuggingFace，用于加载 BERT)
*   **数据处理与评估**：`scikit-learn`, `pandas`, `numpy`

## 2. 项目核心模块拆解
根据论文，项目整体划分为两大核心模块：**DebateFlow Agents**（基于 LLM 和 Mesa 的辩论生成）与 **InsightFlow Agents**（包含用于总结的 Synthesis Agent 和用于分类的深度图神经网络 Analysis Agent）。

### 模块 A：DebateFlow Agents（基于 Mesa 的多智能体辩论模拟）
**目标**：输入一篇新闻 $F$，生成完整的辩论记录（Debate Log）$D$。
1.  **阵营划分**：
    *   **Proponents (正方)**：支持新闻为真（True）。
    *   **Opponents (反方)**：认为新闻为假（Fake）。
2.  **角色设定 (Roles)**：
    *   每个阵营内部划分子角色，如：开篇立论者 (Opening Speaker)、质询者 (Questioner)、结案陈词者 (Closing Speaker)。
3.  **辩论流程 (Algorithm 1 逻辑)**：使用 Mesa 调度器（Scheduler）按顺序执行。
    *   **Stage 1: Opening Statement** $\phi_1$
        *   正反方开篇立论者各自基于新闻 $F$ 和自身阵营立场生成立论 $d_i^{(1)} = f_{LLM}(F, \text{Role}_i, \text{Stance}_i)$。
    *   **Stage 2: Cross-examination and Rebuttal** $\phi_2$
        *   质询者基于前一阶段的发言进行反驳 $d_i^{(2)} = f_{LLM}(d^{(1)}, \text{Stance}_i)$。
    *   **Stage 3: Closing Statement** $\phi_3$
        *   结案陈词者根据前面所有阶段的记录进行总结 $d_i^{(3)} = f_{LLM}(d^{(1)} \cup d^{(2)}, \text{Stance}_k)$。
4.  **模块输出**：将所有智能体的发言按顺序组装为列表 $D =[d_1, d_2, ..., d_n]$，并记录每个发言对应的角色标识符（Role ID）。

### 模块 B：InsightFlow Agents - Synthesis Agent（综合总结）
**目标**：将冗长的辩论记录 $D$ 转化为结构化的总结报告 $S$。
*   **实现方式**：一次简单的 LLM (GPT-4o-mini) 调用。
*   **Prompt 模板**（严格按照论文 3.4.1 节实现）：
    > "It should contain a detailed explanation of your assessment of the debate. Focus on evaluating the authenticity of the news involved in the topic by checking the following:
    > 1. Whether the news contains specific details and verifiable information.
    > 2. Whether the news cites reliable sources or news organizations.
    > 3. The tone and style of the news, with real news generally being more objective and neutral.
    > 4. Any use of emotional language, which might be a characteristic of fake news.
    > 5. Whether the information in the news can be confirmed through other reliable channels."

### 模块 C：InsightFlow Agents - Analysis Agent（PyTorch 深度学习分类器）
**目标**：结合原新闻 $F$ 和辩论图 $G$，预测新闻的真假概率 $\hat{y}$。
请指导 Claude 严格按照以下网络结构编写 `nn.Module`：

**1. Role-aware Encoder (角色感知编码器)**
*   **Text Encoder**：对中文数据集使用 `hfl/chinese-bert-wwm-ext`，英文使用 `bert-base-uncased`。获取输入文本的 `[CLS]` 向量，维度 $d_h$，记为 $\mathbf{h}_i^{enc}$。
*   **Role Embedding**：使用 `nn.Embedding` 初始化角色嵌入 $\mathbf{e}_i$，并经过 `nn.Linear` 映射矩阵 $\mathbf{W}_{role}$ 对齐维度，记为 $\mathbf{r}_i^{proj}$。
*   **Node Representation**：将文本特征与角色特征拼接 $\mathbf{h}_i^{node} =[\mathbf{h}_i^{enc} ; \mathbf{r}_i^{proj}]$。

**2. Debate Graph (辩论图构建与 GAT)**
*   **构图规则**：将辩论记录中的每一个发言视为图的一个节点 $V = \{\mathbf{h}_i^{node}\}$。根据辩论顺序和角色互动建立有向边 $E$（如时序边和反驳边）。
*   **GAT 传播**：将构建的图 $(V, E)$ 输入 `torch_geometric.nn.GATConv` 层（可堆叠多层）。输出更新后的节点特征 $\mathbf{H}^{(L)}$。
*   **Global Pooling**：对 GAT 输出进行全局平均池化 (`global_mean_pool`)，得到图级别的辩论表征 $\mathbf{g} = \text{GlobalPool}(\mathbf{H}^{(L)})$。

**3. Debate-News Interactive Attention (交互注意力机制)**
*   **新闻编码**：同样使用 BERT 编码原始新闻文本 $F$，得到 `[CLS]` 特征 $\mathbf{e}_F$。
*   **线性投影**：将 $\mathbf{g}$ 和 $\mathbf{e}_F$ 映射到相同维度：$\mathbf{g}^{proj} = \mathbf{W}_g \mathbf{g}$，$\mathbf{e}_F^{proj} = \mathbf{W}_e \mathbf{e}_F$。
*   **Multi-Head Attention (MHA)**：捕获新闻内容与辩论特征的交互。
    使用 PyTorch 的 `nn.MultiheadAttention`，其中：
    *   Query = $\mathbf{e}_F^{proj}$ (新闻内容)
    *   Key = $\mathbf{g}^{proj}$ (辩论特征)
    *   Value = $\mathbf{g}^{proj}$ (辩论特征)
    得到交互表征 $\mathbf{c} = \text{MHA}(\mathbf{e}_F^{proj}, \mathbf{g}^{proj}, \mathbf{g}^{proj})$。

**4. 分类器与训练 (Agent Training)**
*   **特征拼接**：$\mathbf{h} = [\mathbf{g}^{proj}; \mathbf{c}]$
*   **预测层**：通过一个全连接层（Linear），输出 2 维对数几率（Logits）。
*   **损失函数**：使用交叉熵损失函数 `nn.CrossEntropyLoss()` ($\mathcal{L} = - \sum y_k \log \hat{y}_k$)。
*   **优化器**：Adam 优化器 (`torch.optim.Adam`)。

---

## 3. 项目代码目录结构建议 (供 Claude Code 参考)

```text
TED_FakeNews/
│
├── data/                  # 存放 ARG-EN 和 ARG-CN 数据集 (jsonl 格式)
│
├── debate_flow/           # 模块 A 和 B (基于 LLM 和 Mesa)
│   ├── agents.py          # Mesa Agent 定义 (Proponent, Opponent)
│   ├── model.py           # Mesa Model 定义 (Scheduler 控制辩论流程)
│   └── prompts.py         # 各个阶段的 Prompt 模板与 Synthesis Agent 模板
│
├── insight_flow/          # 模块 C (基于 PyTorch 和 PyG)
│   ├── dataset.py         # PyG Dataset 类构建图数据 (节点特征、边索引)
│   ├── networks.py        # Analysis Agent 网络架构 (Role-aware, GAT, MHA)
│   └── train.py           # 训练循环、验证逻辑与评估指标计算 (macF1, Acc)
│
├── main_generate.py       # 第一阶段：读取原始新闻，调用 LLM 生成所有 Debate Logs 和 Summary
└── main_train.py          # 第二阶段：读取生成的数据，训练深度图神经网络进行预测
```

