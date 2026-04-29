# EviTED: Evidence-Verified TruEDebate 方法论与创新方案

本文档基于当前 TruEDebate 代码、三篇本地论文 PDF，以及近期假新闻检测相关研究整理，目标是形成一个可以继续实现、消融、写论文的创新框架。

本地参考文件：

- `The Truth Becomes Clearer Through Debate! Multi-Agent.pdf`
- `3701716.3715517.pdf`
- `Multi-Sourced, Multi-Agent Evidence Retrieval for Fact-Checking.pdf`
- FNDCD 参考代码：`/media/dell/DATA/下载/pengjun/code/FNDCD`

联网参考：

- IMRRF: Integrating Multi-Source Retrieval and Redundancy Filtering for LLM-based Fake News Detection, NAACL 2025: https://aclanthology.org/2025.naacl-long.461/
- D2D: Debate-to-Detect, EMNLP 2025: https://aclanthology.org/2025.emnlp-main.764/
- VeraCT Scan: Retrieval-Augmented Fake News Detection with Justifiable Reasoning, ACL 2024 Demo: https://aclanthology.org/2024.acl-demos.25/
- Defense Among Competing Wisdom, WWW 2024: https://doi.org/10.1145/3589334.3645471
- FactAgent / LLM agentic fact-checking, ECAI 2024: https://doi.org/10.3233/FAIA240787
- TripleFact, ACL 2025: https://aclanthology.org/2025.acl-long.431/
- SSA, EMNLP 2025: https://aclanthology.org/2025.emnlp-main.744/

## 1. 当前基线定位

原 TED 论文的核心是：

```text
News
-> Multi-Agent DebateFlow
   Opening -> Cross-examination -> Closing
-> Synthesis Agent
-> Analysis Agent
   Role-aware BERT Encoder
   Debate Graph + GAT
   News-Debate Multi-head Interactive Attention
-> Fake/Real Prediction
```

当前项目已经不再是纯 TED，而是一个 R-TED / EviTED-Core 雏形：

- `debate_flow/model.py` 已加入 Claim Decomposition。
- `debate_flow/prompts.py` 已要求结构化输出 `Referenced Claims / Referenced Rationales / Evidence IDs / Reasoning / Weakness / Confidence`。
- `main_generate.py` 已支持 `--evidence_file`。
- `insight_flow/dataset.py` 已构造 claim / rationale / argument / synthesis / evidence 异构图。
- `insight_flow/networks.py` 已从原论文 GAT 改为 RGCN，并加入 `alpha_evidence` 等多路门控。

目前最有效的代码主干是：

```text
精简异构图
+ 独立 news encoder
+ 独立 source/time meta branch
+ RGCN
+ validation threshold calibration
```

该路线曾达到约 `Macro F1 = 0.7839`。后续 `compact argument + node_confidence` 退化到 `0.7696`，已撤回。

## 2. 三篇论文的可迁移点

### 2.1 TED 原论文

可保留：

- 多智能体正反辩论结构。
- Synthesis Agent 提供全局解释。
- 本地小模型负责最终分类，避免 LLM 直接裁判。
- 新闻原文与辩论图分开编码再融合。

当前需要补强：

- 原论文的 `MHA(news, debate graph)` 在当前网络里被多路 gate 替代。后续可以恢复为 evidence-aware MHA。
- 原论文只有 debate graph，当前已经扩展为 argument-evidence graph，需要明确作为创新而非复现。

### 2.2 FNDCD: Unseen Fake News Detection Through Causal Debiasing

FNDCD 解决的是 unseen-domain fake news detection。核心思想不是 BiGCN 本身，而是：

```text
样本是否可靠 / 是否受环境偏置影响
= 分类置信度 + 图结构可解释性
```

可迁移到 EviTED 的部分：

- 样本级因果去偏权重。
- 结构似然估计器 `p(A|X,e)`。
- 用隐变量 `e` 区分环境独立样本和环境偏置样本。
- 测试时不增加推理成本。

在 EviTED 中，`A` 不再是社交传播图，而是：

```text
Claim-Evidence-Argument-Rationale-Synthesis 关系图
```

因此应迁移为：

```text
relation-aware structure likelihood:
p(edge_type | h_src, h_dst, r)
```

而不是直接搬 BiGCN。

### 2.3 Multi-Sourced Evidence Retrieval for Fact-Checking

该论文强调：

- KG 优先，Web 补全。
- 动态检索策略，而不是固定 query。
- Web evidence 需要过滤、去重、结构化。
- 冗余证据会干扰 LLM reasoning。
- 证据置信度不能简单当作真实概率。

当前代码的 Tavily 检索只是第一步。要形成论文级创新，需要补：

- source credibility scoring
- redundancy filtering
- evidence sentence filtering
- dynamic retrieval trace
- retrieved_at / source / url 缓存

## 3. 推荐最终创新框架

建议命名：

```text
CRED-EviTED:
Causally-Reliable Evidence-Verified TruEDebate
```

中文：

```text
因果可靠证据增强的多智能体辩论假新闻检测框架
```

核心问题定义：

> 原 TED 能生成多智能体辩论，但辩论内容未必基于可靠外部证据；外部证据又可能存在噪声、冗余、来源偏置和时间泄漏。CRED-EviTED 目标是在“新闻、声明、证据、辩论、图模型”之间建立可追踪、可去偏、可校准的证据链。

整体流程：

```text
News F
-> Claim Decomposition
-> Evidence Retrieval and Filtering
   KG / Web / fact-check / authoritative news
-> Evidence-grounded DebateFlow
-> Dynamic Argument-Evidence Graph
-> Reliability-aware News-Debate-Evidence Attention
-> Causal Debias Training
-> Prediction + Explanation
```

## 4. 方法模块设计

### 4.1 Claim Decomposition

当前已实现基础版。建议输出从纯文本 claim 升级为结构化 claim card：

```json
{
  "id": "c1",
  "content": "...",
  "entities": ["..."],
  "event": "...",
  "time": "...",
  "location": "...",
  "verifiable_points": ["..."],
  "domain": "entertainment/politics/health/..."
}
```

这样能支持 D2D 类 domain profile，也能服务后续检索 query 生成。

### 4.2 Evidence Retrieval Agent

当前 Tavily 脚本可保留，但正式方案应从“固定 support/refute query”升级为动态检索。

推荐证据卡：

```json
{
  "id": "e1",
  "related_claims": ["c1"],
  "source": "...",
  "url": "...",
  "retrieved_at": "...",
  "publish_time": "...",
  "source_type": "kg/news/factcheck/wiki/web/social",
  "query_intent": "support/refute/background",
  "stance": "support/refute/neutral/unknown",
  "evidence_text": "...",
  "retrieval_score": 0.81,
  "credibility_score": 0.72,
  "redundancy_score": 0.13,
  "consistency_score": 0.68
}
```

注意：`query_intent` 不能直接等于 `stance`。当前代码已把 `stance` 默认设为 `neutral`，这是正确的。

### 4.3 Redundancy Filtering

这是 IMRRF 和多源检索论文都强调的关键点。建议先做工程可落地版本：

- URL 去重。
- 同域名上限，例如每条 claim 每个 domain 最多 1 条。
- 文本 Jaccard / cosine 相似度去重。
- Wikipedia / IMDb 等高频来源不允许垄断 evidence cards。
- `redundancy_score` 写入 evidence card，训练图中作为边权或 gate 输入。

### 4.4 Evidence-grounded DebateFlow

当前仍是三阶段 TED：

```text
Opening -> Cross-exam -> Closing -> Synthesis
```

建议升级为轻量四阶段，而不是一次性做复杂五阶段：

```text
Stage 1: Claim Framing
Stage 2: Evidence-grounded Opening
Stage 3: Cross-examination with Evidence Attack
Stage 4: Closing and Evidence Verdict
```

每个 agent 必须输出：

```text
Argument
Claim IDs
Evidence IDs
Rationale IDs
Reasoning
Evidence Conflict
Weakness
Confidence
```

这样比只让 agent “引用 evidence”更强，因为它显式建模证据冲突。

### 4.5 Dynamic Argument-Evidence Graph

图节点：

```text
News
Claim
Evidence
Rationale
Argument
Synthesis
Source
Time
```

训练时可以继续保持当前“精简图”原则：

```text
图内: claim / evidence / rationale / argument / synthesis
图外: news / source / time
```

边类型：

```text
claim-supported-by-evidence
claim-refuted-by-evidence
argument-cites-evidence
argument-cites-claim
argument-cites-rationale
synthesis-uses-evidence
argument-attacks-argument
```

后续创新点不是简单加边，而是给边加入可靠性：

```text
r_ij = f(retrieval_score, credibility_score, redundancy_score, consistency_score)
```

可以作为 RGAT/HGT 的 attention bias。

### 4.6 Reliability-aware News-Debate-Evidence Attention

当前代码是多路 sigmoid gate：

```text
alpha_news, alpha_debate, alpha_td, alpha_cs, alpha_evidence, alpha_graph
```

建议改成更贴近原 TED 且更有创新性的结构：

```text
Q = h_news
K,V = [h_argument, h_claim, h_rationale, h_evidence, h_synthesis]
attention_bias = reliability(edge/source/evidence)
```

即：

```text
h_attn = MHA(Q_news, K_graph, V_graph, bias=R_evidence)
h_final = Gate([h_news, h_attn, h_graph, h_meta])
```

这比单纯 gate 更容易写成“继承 TED 的 interactive attention，并引入 evidence reliability bias”。

### 4.7 Causal Debias Training

从 FNDCD 迁移一个轻量版：

```text
L = w_i * L_cls + lambda_struct * L_struct + lambda_kl * L_kl
```

其中：

```text
w_i = posterior reliability of sample i
```

由两部分决定：

```text
classification confidence
structure likelihood of argument-evidence graph
```

结构估计器：

```text
s_ijk = MLP([h_i, h_j, rel_emb_k])
p(edge_type=k | i,j)
```

用途：

- 下调证据稀疏、结构异常、来源偏置强的样本。
- 输出 `e_score` 作为低可信样本诊断。
- 对低可信样本触发后续重检索或重辩论。

## 5. 当前代码下一步建议

### Step 1: 不要用当前 partial evidence 做正式指标判断

当前 `evidence/en_train_evidence.jsonl` 覆盖约 648 条，标签极不均衡，且 val/test 没有 evidence。它不适合作为判断 evidence 是否有效的实验。

应该先做均衡 probe：

```text
train: real 200 + fake 200
val:   real 100 + fake 100
test:  real 100 + fake 100
```

然后只对这些样本生成 evidence，做 controlled probe。

### Step 2: 加 evidence balanced sampler 脚本

脚本功能：

- 从 `data/en/{split}.json` 中按 label 抽样 id。
- Tavily 检索时只处理这些 id。
- 生成 `evidence_probe/en_train_probe_evidence.jsonl` 等文件。

这样可以低成本判断 evidence 是否对 `F1_fake` 有帮助。

### Step 3: 做 evidence quality filtering

当前 evidence 有明显问题：

- support/refute query 可能返回同一个网页。
- evidence_text 过长，包含页面噪声。
- 同源重复较多。

优先实现：

- 截断到 1-3 句。
- URL/domain 去重。
- 过滤过短、明显无关、导航页式文本。
- 保存 `retrieved_at`。

### Step 4: 恢复 TED-style MHA 并加入 evidence bias

在 `insight_flow/networks.py` 中新增：

```text
EvidenceAwareInteractiveAttention
```

消融比较：

```text
RGCN + gate
RGCN + original MHA
RGCN + evidence-aware MHA
```

### Step 5: 加 FNDCD-lite debias loss

先不做完整因果建模，先实现低风险版本：

- model 返回 `node_proj`、`graph_repr`。
- 训练时加 relation-aware edge reconstruction loss。
- 用结构 loss 只做辅助正则，不先重加权主 loss。

如果验证有效，再打开样本权重 `w_i`。

## 6. 实验设计

主表建议：

```text
E0 TED-clean baseline
E1 R-TED: + claim/rationale graph
E2 EviTED-Core: + evidence cards + evidence graph
E3 EviTED-R: + redundancy filtering
E4 EviTED-A: + evidence-aware MHA
E5 CRED-EviTED: + causal debias structure loss
```

核心指标：

- Accuracy
- Macro F1
- F1_real
- F1_fake
- Evidence coverage
- Average evidence cards per sample
- Evidence redundancy rate
- Source diversity

消融：

```text
without claim nodes
without rationale nodes
without evidence nodes
without synthesis node
without source/time meta
without redundancy filtering
without causal debias loss
gate vs MHA vs evidence-aware MHA
```

可靠性实验：

- 按 source domain 分组。
- 按 evidence coverage 分组。
- 按 time/source_id 做偏置分析。
- 低 `e_score` 样本错误率是否更高。

## 7. 学术边界与风险控制

必须避免：

- 用 test 结果挑 seed 或挑证据参数。
- 把 Tavily query intent 当成真实 evidence stance。
- 用新闻发布后的网页证据造成时间泄漏。
- 只在 train 加 evidence，不在 val/test 加 evidence，却声称 evidence 有效。
- evidence 覆盖样本标签严重不均衡。
- 不做消融就把多个模块一起上线。

建议写法：

> We do not treat LLMs or retrieval results as final judges. Instead, retrieved evidence is converted into auditable evidence cards, filtered for redundancy, injected into debate generation, and encoded in a dynamic argument-evidence graph. A compact SLM classifier learns to selectively absorb evidence through reliability-aware interaction and causal structure regularization.

## 8. 最终可发表贡献点

可以组织成 4 个贡献：

1. **Evidence-grounded DebateFlow**
   将 TED 的自由辩论升级为证据约束辩论，要求 agent 显式引用 claim/evidence/rationale。

2. **Dynamic Argument-Evidence Graph**
   构建 claim、evidence、rationale、argument、synthesis 的动态异构图，而不是只建模辩论顺序。

3. **Reliability-aware Evidence Interaction**
   在原 TED news-debate interactive attention 基础上加入 evidence credibility、retrieval relevance、redundancy penalty 和 source/time metadata。

4. **Causal Structure Debiasing**
   迁移 FNDCD 思想，将环境偏置从社交传播图扩展到论证-证据图，通过结构似然辅助训练提升跨事件泛化和 fake 类稳定性。

推荐论文标题：

```text
CRED-EviTED: Causally Reliable Evidence-grounded Multi-Agent Debate for Fake News Detection
```

中文标题：

```text
CRED-EviTED：因果可靠证据增强的多智能体辩论假新闻检测框架
```

## 9. 最短落地路线

不建议一次性完成所有模块。最短、最稳路线：

```text
1. Balanced evidence probe
2. Evidence filtering and redundancy score
3. Evidence-aware MHA
4. Relation-aware structure regularization
5. Full evidence generation
6. Main experiments + ablations
```

当前最该做的是第 1 步：先用均衡小样本验证 evidence 是否值得继续购买额度生成全量数据。
