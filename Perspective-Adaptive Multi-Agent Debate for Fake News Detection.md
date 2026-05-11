 ## 创新点一：观点维度自适应的多智能体辩论框架

  暂定名称：Perspective-Adaptive Multi-Agent Debate for Fake News Detection

  论文的 TED 使用固定的 Proponents 和 Opponents 两队辩论。这个设计清晰，但问题是：虚假新闻并不总是只有“真/假”两个立场，它可能涉及情绪操控、因果误导、实体错配、时间错位、夸大表述、讽刺误判、
  政治立场操控等不同维度。固定正反辩手容易忽略这些细粒度角度。

  你的创新可以是：把“正反辩论”改造成“多视角专家辩论”。

  具体做法：

  - 保留 TED 的 DebateFlow 和 InsightFlow 主结构。
  - 在辩论前增加一个 Perspective Planner，由 LLM 或小模型分析新闻样本，判断当前样本最可能涉及哪些风险维度。
  - 设计多个 perspective agents，例如：
      - Factual Consistency Agent：检查文本内部事实一致性。
      - Causal Reasoning Agent：检查因果跳跃、错误归因。
      - Temporal Reasoning Agent：检查时间线、旧闻新炒、事件顺序。
      - Emotional Manipulation Agent：检查煽动性、恐慌性、情绪攻击。
      - Intent Agent：判断是否存在政治操控、商业诱导、社会心理操纵等意图。
      - Linguistic Style Agent：检查标题党、夸张表达、模糊来源等语言特征。
  - 每个 agent 不是简单支持真/假，而是从自己的视角给出：
      - 可疑点
      - 支持真实的理由
      - 支持虚假的理由
      - 置信度
  - Coordinator 或 Judge 再聚合不同视角的辩论结果。

  这个创新点可以吸收其他论文的思想，但不走检索路线：

  - 借鉴 PAMAS 的 perspective aggregation。
  - 借鉴 IConMoE 的 misinformation intent 建模。
  - 借鉴 TED 的辩论结构。
  - 但你的重点是：让辩论角色从固定正反方变成样本自适应的多视角专家群体。

  实验设计可以这样做：

  - Baseline：原始 TED。
  - Variant 1：固定多视角 agents，但不做自适应选择。
  - Variant 2：自适应选择 top-k perspective agents。
  - Variant 3：自适应选择 + 视角置信度加权聚合。
  - 指标：Accuracy、Macro-F1、AUC、解释质量、人评或 GPT 评估解释相关性。
  - 消融：
      - 去掉 Emotional Agent。
      - 去掉 Intent Agent。
      - 去掉 Perspective Planner。
      - 固定所有 agents vs 动态选择 agents。
  - 额外分析：
      - 不同类型虚假新闻分别由哪些 agent 贡献最大。
      - 多视角是否能减少 TED 中“强行正反辩论”的幻觉问题。

  这个创新点适合作为你论文的第一个主要贡献，因为它直接改了多智能体结构，代码上也容易从 TED 扩展。

  ———

  ## 创新点二：带自反思与角色反转一致性的辩论裁判机制

  暂定名称：Self-Reflective and Role-Reversal Debate Judging

  第 32 篇 TED 的另一个弱点是：辩论 agents 会被预设立场影响。支持方会努力说真，反对方会努力说假，即使证据不足也可能强行生成论点。最后 Judge 如果只看 debate log，可能被更流畅、更强势的一方误导。

  你的第二个创新点可以聚焦在 Judge 上：让 Judge 不只是“总结辩论并分类”，而是显式检查辩论过程是否可靠。

  核心设计包括两个模块。

  ### 1. 自反思裁判

  Judge 在输出最终标签前，先做一轮自检：

  - 哪些论点是事实性论点？
  - 哪些只是主观推测？
  - 哪些论点存在互相矛盾？
  - 哪些论点没有被回应？
  - 哪些论点可能是 hallucination？
  - 当前辩论是否足以支持最终结论？

  然后 Judge 输出：

  - final label
  - confidence score
  - key accepted arguments
  - rejected arguments
  - uncertainty source

  这可以借鉴 CLUE 的“不确定性来源解释”和 ART 的“支持/攻击论证树”，但仍然不需要外部检索。

  ### 2. 角色反转一致性检查

  再增加一个轻量的 role-reversal debate：

  - 原本支持真实的一方，被要求反过来论证“虚假”。
  - 原本支持虚假的一方，被要求反过来论证“真实”。
  - Judge 比较两轮辩论是否稳定。

  如果一个样本在角色反转后结论剧烈变化，说明模型可能不是基于事实判断，而是被角色 prompt 牵引。此时可以降低置信度，或者触发更严格的裁判规则。

  最终可以设计一个分数：

  或者更简单一些：

  - 原始 TED Judge 输出标签。
  - Self-Reflective Judge 输出标签和置信度。
  - Role-Reversal Judge 输出一致性分数。
  - 最终分类器融合这三个结果。

  这个创新点的价值在于，它解决 TED 的核心局限：多智能体辩论可能生成看似合理但立场驱动的论证。

  实验设计：

  - Baseline：原始 TED。
  - Variant 1：TED + Self-Reflective Judge。
  - Variant 2：TED + Role-Reversal Consistency。
  - Variant 3：TED + 两者结合。
  - 指标：
      - Accuracy / Macro-F1
      - calibration，如 ECE 或 confidence-accuracy correlation
      - explanation faithfulness
      - 对抗鲁棒性，例如情绪改写、标题改写、立场诱导 prompt
  - 消融：
      - 去掉 contradiction check。
      - 去掉 hallucination risk check。
      - 去掉 role-reversal penalty。
      - 只用一次辩论 vs 双向角色反转。

  这个创新点适合作为第二个主要贡献，因为它不要求你换掉 TED 主体框架，而是增强其推理可靠性和裁判可信度。
