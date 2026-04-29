# No-API TruEDebate 创新路线评估

当前瓶颈是外部检索 API 成本过高。因此，后续创新不应依赖全量 Tavily / Web evidence，而应充分利用现有数据集中已经具备的弱证据与结构化生成结果。

## 1. 当前可用信号

现有 `data/en/{train,val,test}.json` 每条样本都有：

- `content`
- `label`
- `time`
- `source_id`
- `td_rationale`
- `td_pred`
- `td_acc`
- `cs_rationale`
- `cs_pred`
- `cs_acc`

现有 `output/en_{split}` 每条样本还有：

- `claims`
- `rationale_cards`
- 6 个结构化 debate nodes
- `referenced_claims`
- `referenced_rationales`
- `reasoning`
- `weakness`
- `confidence`
- `synthesis_structured`

这意味着项目已经具备一种“内部证据”：

```text
td_rationale: textual-detail evidence
cs_rationale: commonsense/verifiability evidence
claims: decomposed factual units
debate references: argument-to-claim/rationale links
synthesis: debate-level summary
```

虽然它不是外部检索证据，但它可以作为可解释弱证据链，用于方法创新。

## 2. 关键统计结论

基于当前数据：

```text
train: td_acc 0.7497, cs_acc 0.8061, td/cs agree 0.7989
val:   td_acc 0.7700, cs_acc 0.8234, td/cs agree 0.8148
test:  td_acc 0.7782, cs_acc 0.8315, td/cs agree 0.8235
```

`td_pred/cs_pred` 是强弱教师信号，但明显偏 real：

```text
test label_fake = 0.1860
test td_fake    = 0.0994
test cs_fake    = 0.1383
```

所以不能直接做强蒸馏，否则会压低 fake 类召回。更合理的是：

```text
只在 td/cs 一致且模型不确定时使用弱教师；
或者把 teacher disagreement 当作风险信号，而不是标签。
```

此外，`source_id` 在当前 split 中几乎是一文一源：

```text
train sources = 3884, single-source items = 3884
val sources   = 1274, single-source items = 1274
test sources  = 1258, single-source items = 1258
```

因此不建议把 `source_id` 当作可泛化来源 embedding 的主要创新。它更适合作为环境偏置标识或时间/source meta 诊断。

## 3. 推荐无 API 创新框架

建议把外部 evidence 版 EviTED 暂时降级为：

```text
IRED-TED:
Internal Rationale-Evidence Debiased TruEDebate
```

中文：

```text
内部理由证据去偏的多智能体辩论假新闻检测框架
```

核心思想：

> 不依赖外部检索，而是把数据集已有的 td/cs rationales、LLM 结构化辩论、claim-rationale 引用和辩论冲突建模为内部证据图；通过弱教师校准和因果结构正则，降低 LLM rationale 的偏置，提高 fake 类识别稳定性。

整体流程：

```text
News
-> Claim Decomposition
-> Internal Evidence Cards
   td_rationale / cs_rationale
-> Evidence-constrained Debate
-> Internal Rationale-Evidence Graph
-> Debate Conflict and Teacher Disagreement Modeling
-> Causal Debias / Structure Regularization
-> Prediction
```

## 4. 可发表创新点

### 4.1 Internal Evidence Cards

把已有 `td_rationale` 和 `cs_rationale` 正式定义为两类内部证据：

```text
TD Evidence: textual detail evidence
CS Evidence: commonsense and verifiability evidence
```

每张 evidence card 包含：

```json
{
  "id": "td_1",
  "type": "textual_detail",
  "content": "...",
  "related_claims": ["c1", "c2"],
  "teacher_pred": 0,
  "teacher_reliability": "latent"
}
```

这可以保留你当前代码已有的 `rationale_cards`，不用重新生成数据。

### 4.2 Teacher Disagreement as Uncertainty

不要直接蒸馏 `td_pred/cs_pred`。改为建模四种状态：

```text
td=real, cs=real: likely real but may be biased
td=fake, cs=fake: strong suspicious signal
td!=cs: internal evidence conflict
td/cs unavailable: unknown
```

创新点：

```text
Teacher prediction is not used as a label;
it is used as an uncertainty and evidence-conflict signal.
```

可以加入特征：

```text
teacher_agreement
teacher_fake_count
teacher_entropy
teacher_conflict
```

这些特征不直接泄漏标签，因为测试集本身也有 `td_pred/cs_pred` 字段。

### 4.3 Debate Conflict Graph

当前图已经有 pro/opp argument。建议显式增强：

```text
proponent argument pool
opponent argument pool
conflict vector = |h_pro - h_opp|
teacher conflict = td/cs disagreement
synthesis conflict = conflict_points
```

最终分类不只看 graph mean，而看：

```text
h_final = [h_news, h_graph, h_pro, h_opp, |h_pro-h_opp|, h_teacher_conflict]
```

这比外部 evidence 便宜，而且更贴合 TED 原论文“truth becomes clearer through debate”。

### 4.4 Rationale Reliability Gate Without Labels

之前用 `td_acc/cs_acc` 监督 gate 会有标签泄漏风险，也会过拟合。现在改为无监督可靠性：

```text
alpha_td = f(td_rationale, claim overlap, debate citation count, teacher agreement)
alpha_cs = f(cs_rationale, claim overlap, debate citation count, teacher agreement)
```

可以用规则弱监督或正则：

```text
if td/cs agree and both cited frequently: gate higher
if td/cs disagree: gate more conservative
if rationale is never cited: gate lower
```

这属于“内部证据选择”，不需要 API。

### 4.5 FNDCD-lite Causal Structure Regularization

迁移 FNDCD 思想，但不依赖外部传播图。

在当前 claim-rationale-argument 图上加入结构重建任务：

```text
p(edge_type | h_src, h_dst, r)
```

辅助损失：

```text
L = L_cls + lambda_struct * L_edge_recon
```

第一阶段不要做样本重加权，避免不稳定。只把结构估计作为正则，让图表示更尊重 claim/rationale/argument 引用关系。

如果有效，再加入样本可靠性权重：

```text
w_i = sigmoid(-L_edge_i + confidence_margin_i)
L = w_i * L_cls + lambda_struct * L_edge
```

### 4.6 Synthesis Tendency Repair

当前 `synthesis_structured.final_debate_tendency` 大量是 `uncertain`：

```text
train uncertain = 3558 / 3884
val   uncertain = 1180 / 1274
test  uncertain = 1152 / 1258
```

这说明 synthesis tendency 基本不可用。但 `conflict_points` 和 explanation 仍可用。

建议：

- 不再强依赖 `final_debate_tendency`。
- 把 synthesis 当作 conflict summary，而不是 verdict。
- 在模型中单独编码 `conflict_points + explanation`。
- 生成侧后续如果重跑，可以要求 synthesis 输出 `pro_score`、`opp_score`、`conflict_level`，而不是强迫 real/fake/uncertain。

## 5. 低成本实验路线

### E0 当前稳定主干

```text
精简 R-TED 图 + news/meta branch + threshold calibration
```

已知约 `0.7839`。

### E1 + Teacher Conflict Features

不重生成数据。

增加：

```text
td_pred
cs_pred
teacher_agreement
teacher_fake_count
teacher_disagreement
```

只作为特征，不做蒸馏损失。

### E2 + Internal Evidence Reliability Gate

不重生成数据。

基于：

```text
rationale citation count
claim overlap
td/cs agreement
pro/opp citation balance
```

生成 `alpha_td/alpha_cs` 的输入特征。

### E3 + Debate Conflict Modeling

不重生成数据。

强化：

```text
h_pro
h_opp
|h_pro-h_opp|
h_pro * h_opp
synthesis conflict text
```

### E4 + FNDCD-lite Structure Regularization

不重生成数据。

对当前图边做 relation-aware edge reconstruction：

```text
positive edges = existing claim/rationale/argument edges
negative edges = same graph random node pairs
```

### E5 Optional: Balanced Evidence Probe

仅当后续有少量 API 额度时再做，不作为当前主线。

## 6. 当前最建议优先做的代码修改

优先级从高到低：

1. **Teacher Conflict Feature Branch**
   成本最低，不重生成，直接利用 `td_pred/cs_pred`。

2. **Synthesis Conflict Text Branch**
   不用 `final_debate_tendency`，改用 `conflict_points + explanation`。

3. **Debate Conflict Fusion**
   在 `networks.py` 中显式加入 `h_pro * h_opp` 与更强的 divergence fusion。

4. **Relation-aware Structure Loss**
   从 FNDCD-lite 开始，只做辅助正则，不做样本重加权。

5. **Evidence API 路线暂停**
   保留代码入口，但不作为当前实验主线。

## 7. 论文写法调整

不要把当前工作写成“外部检索证据增强”，因为你没有预算生成完整外部证据。

更稳妥的论文叙事是：

```text
Existing LLM debate frameworks rely on free-form arguments and may absorb biased or overconfident rationales.
We propose an internally grounded rationale-evidence debate framework that converts pre-existing textual-detail and commonsense rationales into structured evidence cards, models pro/con argument conflicts, and regularizes the argument graph with causal structure constraints.
```

贡献点可以写为：

1. **Internal Rationale-Evidence Graph**
   把 td/cs rationales、claims、arguments、synthesis 统一成内部证据图。

2. **Teacher Conflict-Aware Fusion**
   不蒸馏弱教师标签，而是把弱教师一致性/冲突作为不确定性信号。

3. **Debate Conflict Representation**
   显式建模 pro/opp 表示差异、交互和冲突摘要。

4. **Causal Structure Regularization**
   借鉴 FNDCD，在内部论证图上做 relation-aware structure regularization，提高泛化。

## 8. 结论

在无 API 预算前提下，最合理的创新路线不是继续做 EviTED 外部检索，而是转为：

```text
IRED-TED = Internal Rationale-Evidence Debiased TruEDebate
```

这条路线完全基于现有数据和已生成 output，不需要重生成外部 evidence，且仍然能和 TED、FNDCD、多源证据论文形成清晰的方法关联。
