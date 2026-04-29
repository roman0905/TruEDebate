------

## 一、推荐创新方案：EviTED——证据增强的多智能体辩论检测框架

我建议你把创新点设计成：

> **EviTED: Evidence-grounded TruEDebate with Dynamic Argument-Evidence Graph for Fake News Detection**

中文可以叫：

> **证据约束的动态论证图多智能体辩论虚假新闻检测框架**

它不是简单堆模块，而是围绕一个明确问题展开：

> 原 TED 解决了“是否能辩论”，但没有解决“辩论是否基于可靠证据”。
> EviTED 解决“如何让辩论过程、证据链条、最终分类三者一致”。

------

## 二、具体框架设计

### 1. Claim Decomposition Agent：新闻子声明拆解

输入新闻后，先不用直接辩论，而是让 LLM 抽取结构化子声明：

```text
News F → {c1, c2, ..., cn}
```

每个子声明包含：

```text
主体 / 事件 / 时间 / 地点 / 数量 / 来源 / 关键实体 / 可验证点
```

例如娱乐新闻、政治新闻、公共卫生新闻的可验证点不同，所以这里可以借鉴 D2D 的 domain-specific profile 思路，让系统先判断领域，再选择不同核查维度。D2D 在 EMNLP 2025 中使用领域画像和五阶段辩论来提升多智能体检测的透明性和逻辑一致性。([ACL Anthology](https://aclanthology.org/2025.emnlp-main.764/))

------

### 2. Multi-source Evidence Retrieval Agent：多源证据检索

对每个子声明生成多种检索 query：

```text
q_i^support, q_i^refute, q_i^neutral
```

检索来源可以包括：

```text
新闻搜索结果
事实核查网站
百科/知识库
权威媒体
社交平台评论或转发上下文
历史相似新闻
```

这一步可以借鉴 IMRRF 的思想：不仅检索证据，还要做**多源检索 + 冗余过滤**，因为 NAACL 2025 的 IMRRF 明确指出，LLM 假新闻检测中常见问题是证据不足和冗余证据干扰。([ACL Anthology](https://aclanthology.org/2025.naacl-long.461/))

输出证据卡片：

```text
Evidence Card e_k = {
  source,
  publish_time,
  credibility_score,
  stance: support / refute / neutral,
  related_claim,
  evidence_text,
  retrieval_score,
  redundancy_score
}
```

------

### 3. Evidence-grounded DebateFlow：证据约束辩论流程

原 TED 的 DebateFlow 是：

```text
Opening → Cross-examination/Rebuttal → Closing
```

建议改成五阶段：

```text
Stage 1: Claim Framing
Stage 2: Evidence Presentation
Stage 3: Cross-examination
Stage 4: Free Debate / Counter-evidence Attack
Stage 5: Closing & Verdict
```

并且强制每个智能体的发言格式为：

```text
Argument:
Evidence IDs:
Claim IDs:
Reasoning:
Confidence:
Possible Weakness:
```

这样正方不能只说“我认为是真的”，反方也不能只说“这像假的”，而是必须引用证据卡片。

这一步同时借鉴 FactAgent 的“人类事实核查流程”思想。FactAgent 把真假新闻检测拆成多个子任务，并结合 LLM 内部知识和外部工具进行核查，强调逐步透明解释。([arXiv](https://arxiv.org/pdf/2405.01593))

------

### 4. Dynamic Argument-Evidence Graph：动态论证—证据图

这是最关键的模型创新。

原 TED 的 debate graph 主要以辩论交互为节点，边表示顺序或引用关系。你可以升级为异构动态图：

```text
节点类型：
News Node
Claim Node
Evidence Node
Argument Node
Agent Role Node
Source Node

边类型：
claim-of
supports
refutes
cites
contradicts
same-event
same-source
temporal-consistent
source-reliable
agent-argues
```

原 TED 是：

```text
Debate Log → Role-aware Encoder → Debate Graph → GAT → Interactive Attention
```

EviTED 改成：

```text
News + Claims + Evidence + Debate Log
→ Heterogeneous Argument-Evidence Graph
→ Relational GAT / HGT
→ Evidence-aware Debate Representation
→ Reliability-gated News-Debate Attention
→ Prediction
```

这比原 TED 更有创新性，因为图不再只是“辩论顺序图”，而是“事实核查图”。WWW 2024 的 Defense among Competing Wisdom 已经证明，将竞争性观点拆成不同证据并建模 defense/inference 关系，有助于解释性假新闻检测。([arXiv](https://arxiv.org/abs/2405.03371))

------

### 5. Reliability-aware Gating：证据可信度门控

借鉴 ARG 的核心思想：LLM 不是最终裁判，而是 advisor；SLM 或图模型要选择性吸收 LLM rationale。AAAI 2024 的 ARG 发现，GPT-3.5 能提供多视角理由，但直接判断仍可能不如微调 BERT，因此更适合把 LLM 理由作为小模型的指导信号。([AAAI出版物](https://ojs.aaai.org/index.php/AAAI/article/view/30214))

可以设计一个门控：

```text
z_i = sigmoid(W [argument_i ; evidence_score_i ; source_score_i ; contradiction_score_i])
```

最终表示：

```text
h_final = z_i * h_debate + (1 - z_i) * h_news
```

含义是：

> 如果某条辩论论据有强证据支撑，则增加其权重；
> 如果论据没有证据、来源弱、存在冲突，则降低其影响。

这样可以避免 LLM 辩论“说得好听但不可靠”。

------

### 6. LLM-SLM Collaborative Evolution：大模型—小模型协同进化

为了降低成本，可以借鉴 AAAI 2025 MRCD 的多轮协作思想。MRCD 用两阶段检索选择最新示例和知识，并让 LLM 与 SLM 多轮学习，提升突发新闻场景下的检测表现。([AAAI出版物](https://ojs.aaai.org/index.php/AAAI/article/view/32109))

在 EviTED 中可以这样做：

```text
Round 1: LLM agents 生成证据约束辩论
Round 2: SLM/Graph model 学习辩论图和证据图
Round 3: SLM 找出低置信样本
Round 4: LLM 对低置信样本重新检索和辩论
Round 5: 蒸馏成轻量版本 EviTED-D
```

这样论文可以同时讲：

```text
Full EviTED：高性能、高解释性
EviTED-D：低成本、部署友好
```

------

