# 论文方法、贡献与局限性汇总

生成日期：2026-05-08

说明：本文件覆盖当前目录下 32 个 PDF。局限性优先整理论文明确说明的内容；若论文没有明确说明，则基于方法、实验设置和模型框架补充“推断”局限性。`2511.20233v3.pdf` 与 `Self-Refining Explainable Fact-Checking via.pdf` 为同一篇 REFLEX 论文的重复文件，仍按 PDF 文件逐项列出。

## 1. Persuasion at Play: Understanding Misinformation Dynamics in Demographic-Aware Human-LLM Interactions

**文件名**：`2026.eacl-long.234.pdf`

**方法**：提出 PANDORA 框架，研究错误信息场景下人类与 LLM 的双向说服动态。框架包含三部分：LLM 生成支持/反驳性说服文本并测试人类判断；用人类立场文本测试带有人口统计 persona 的 LLM；构建两智能体同质/异质人口群组，在多轮交互中观察错误信息接受、纠正和回声室行为。

**贡献**：首次系统比较 human-to-LLM 与 LLM-to-human 说服在错误信息中的影响；引入人口统计感知的多智能体 LLM 仿真；发现不同人口 persona 的 LLM 在错误信息判断正确率上有明显差异，且同质群组更容易出现回声室，异质互动可缓解极化。

**局限性**：论文明确指出，LLM persona 对真实人口群体是粗粒度模拟，不能等同于真实人类行为；研究主要覆盖美国参与者，跨文化泛化有限；用 correctness rate 衡量说服，未覆盖信心、推理策略等信念形成过程；persona 可能强化性别、城乡、年龄刻板印象；LLM 生成说服文本存在被恶意用于操纵脆弱群体的风险。

## 2. MEVER: Multi-Modal and Explainable Claim Verification with Graph-based Evidence Retrieval

**文件名**：`2026.eacl-long.242.pdf`

**方法**：提出 MEVER，同时做多模态证据检索、声明验证和解释生成。检索阶段构建文本-图像两层多模态图，并设计 image-to-text / text-to-image 推理；验证阶段使用 token-level 与 evidence-level 融合；解释阶段采用多模态 Fusion-in-Decoder，并加入一致性正则。论文还构建 AI 领域图表声明验证数据集 AIChartClaim。

**贡献**：把证据检索、多模态验证和解释生成统一到一个模型中；面向科学 AI 图表声明构建包含图像、文本说明和解释的数据集；实验显示图结构检索与多层融合提升验证和解释质量。

**局限性**：论文明确指出，模型假设文本和图像证据同时可用；若数据只有文本，需要额外跨模态搜索图像，本文未研究。AIChartClaim 数据集规模受限于可读图表、可核查声明和领域专家标注成本；目前只覆盖 AI 领域，跨生物医学等科学领域泛化未验证；数据集只含 SUPPORT/REFUTE，没有 NEI 标签。

## 3. KG-CRAFT: Knowledge Graph-based Contrastive Reasoning with LLMs for Enhancing Automated Fact-checking

**文件名**：`2026.eacl-long.302.pdf`

**方法**：提出 KG-CRAFT。先从声明和相关报道中用 LLM 抽取实体、类别和关系，构建知识图谱；再基于 claim triples 与报告 triples 生成对比问题，并用 MMR 选择相关且多样的问题；随后让 LLM 根据报道回答这些问题并汇总成 evidence-based summary；最后只用声明和汇总证据进行真实性分类。

**贡献**：把知识图谱结构用于生成上下文相关的对比问题，增强 LLM 事实核查推理；在 LIAR-RAW 和 RAWFC 上取得优于已有方法的结果；显示少量对比问题也能带来竞争性性能，并能缩小小模型与大模型差距。

**局限性**：论文明确指出，没有对知识图谱构建、问题生成等中间组件做充分定性验证，而这些步骤对最终性能敏感；中间任务依赖固定 LLM，其他模型或微调模型可能改变结果；只评估两个英文数据集；流程多处依赖昂贵 LLM；与 prior work 比较时不同 LLM 家族带来公平性和可复现性威胁；跨语言下实体链接、关系抽取和问题生成可能退化。

## 4. User-Centric Evidence Ranking for Attribution and Fact Verification

**文件名**：`2026.eacl-long.340.pdf`

**方法**：将证据选择重构为 Evidence Ranking：排序所有候选证据句，使用户尽早读到足够验证/反驳声明的最小证据前缀。提出 MSR/IMSR 概念，并改造 MRR、Success Rate、NDCG 评估顺序效率。基于 FEVER、HoVer、WICE 构建统一 benchmark，比较 embedding、NLI、ReasonRank、LLM 的 one-shot 与 incremental ranking。

**贡献**：从用户阅读成本角度定义证据排序任务，而不是只选固定证据集合；提出适配“最早达到充分性”的评估指标；实验证明 LLM/LRM 方法表现最好，incremental ranking 更能捕捉互补证据；用户研究显示排序相比证据选择减少阅读量并提升验证表现。

**局限性**：论文明确指出，数据集没有证据句内部自然顺序，无法评估 coherence-based ordering；数据多为短而聚焦的声明，长声明和复杂声明尚未验证；用户研究规模小、样例经过筛选，参与者群体不够多样；排序错误可能让用户误判声明与证据之间的强弱关系。

## 5. ART: Adaptive Reasoning Trees for Explainable Claim Verification

**文件名**：`2026.findings-eacl.28.pdf`

**方法**：提出 ART，把声明验证建模为支持/攻击论证树。Supporter 和 Attacker LLM 递归生成支持与反驳论点；LLM-as-a-Judge 对支持/反驳论点做 pairwise tournament；用 Bradley-Terry 模型校准论点强度，再自底向上聚合得到根声明真假概率，并输出可追踪推理路径。

**贡献**：引入树状、可争辩的声明验证框架；首次在该任务中用支持/攻击论点锦标赛和 Bradley-Terry 校准统一相对强度；在多个 claim verification 数据集上优于 Direct Prompting、CoT 和 ArgLLM，并提供比普通 CoT 更结构化的解释轨迹。

**局限性**：论文明确指出，ART 相比单次调用的 Direct/CoT 需要多次 LLM 调用，包括论点生成、强度评分和 pairwise 比较，时间和成本更高；复杂、模糊、需要大量世界知识或领域知识的声明可能导致树内错误传播；LLM judge 对 prompt 敏感，可能偏向某类论点，影响公平判断和整体可信度。

## 6. Distill and Align Decomposition for Enhanced Claim Verification

**文件名**：`2026.findings-eacl.309.pdf`

**方法**：提出面向复杂声明验证的 decomposition 训练框架。先把分解重构为单次调用内的 sequential reasoning：检测可验证句、去语境化、识别关系、抽取最小子声明；再用大 teacher model 蒸馏高质量样例对 8B student 做 SFT；最后用 GRPO 和多目标 reward 微调，reward 包含格式约束、下游 verifier 对齐信号、LLM-as-a-Judge checklist 分解质量。

**贡献**：联合优化“分解质量”和“与 verifier 的对齐”，而不是分开训练；小型 8B decomposer 在多个评估设置中提升下游 verification macro-F1；人评确认生成子声明在可验证性、去语境化、限定词保留、引用明确性等方面质量较高。

**局限性**：论文明确指出，只在固定 verifier 上测试，泛化到其他 verifier 和知识源仍待验证；SFT 依赖单一 teacher 的合成样例，可能引入偏差并限制分解多样性；LLM-as-a-Judge reward 可能无法捕捉细粒度分解质量且会引入 judge 偏差；主要在英文、Wikipedia/Google Search 场景测试；未覆盖 multi-hop claim verification benchmark；句子级切分可能漏掉跨句或文档级事实关系。

## 7. Explaining Sources of Uncertainty in Automated Fact-Checking

**文件名**：`2505.17855v2.pdf`

**方法**：提出 CLUE，用于解释事实核查模型不确定性的来源。先用答案 logits 的预测熵计算 uncertainty score；再从 claim 与多条 evidence 之间抽取 span-level 交互，通过注意力头、二部图和 Louvain 社区发现定位重要 span pair，并用关系标注器标为 agree/disagree/unrelated；最后通过 prompt 或 attention steering 生成自然语言解释，说明哪些证据冲突/一致导致模型不确定。

**贡献**：从“解释 verdict”转向“解释 uncertainty source”，把不确定性明确绑定到 claim-evidence 和 evidence-evidence 的冲突/一致；在多个模型和 HealthVer、DRUID 数据集上，解释更贴合模型不确定性和预测标签；人评认为 CLUE 解释更有帮助、信息量更高、冗余更少、逻辑一致性更好。

**局限性**：论文明确指出，实验受算力限制主要使用中等规模模型，70B 级模型可能进一步提升覆盖率和降低冗余；主实验关系标注使用 GPT-4o，完全开源替代仍有小幅性能差距；多阶段流程会传播上游错误，如 span 片段不完整、关系误标；主要测试一条 claim 配两三条 evidence 的 HealthVer/DRUID 场景，更复杂多声明/多证据互动未实证；没有比较在所有场景下自然语言不确定性解释是否优于数值置信度或 verbal hedges；只解释由证据冲突导致的不确定性，未覆盖证据不足、模型知识缺口、上下文记忆冲突等来源。

## 8. Debating Truth: Debate-driven Claim Verification with Multiple Large Language Model Agents

**文件名**：`2507.19090v4.pdf`

**方法**：提出 DebateCV，用两个辩手 LLM 分别站在支持/反对立场，基于同一证据集进行多轮攻防；Moderator 每轮总结、判断是否收敛，并给出最终四分类判定与理由。进一步提出 Debate-SFT：用 AVeriTeC 非辩论数据合成辩论记录，错误样本由 Corrector 按人工标签生成纠正理由，构造 SynDeC 来微调 Moderator。

**贡献**：首次将结构化多智能体辩论用于 claim verification；针对零样本 Moderator 偏向中立判定的问题提出 Debate-SFT；在 golden/retrieved evidence 条件下较 HerO 等 SOTA 提升准确率，并在人评中提升证据使用、uncertainty 表达和推理路径质量。

**局限性**：论文未设独立 Limitations。论文明确/未来工作指出，多智能体推理成本高于单智能体方法；未来需提升推理成本效率，并增强在极端噪声或被污染证据条件下的性能。推断：SynDeC 依赖合成辩论和 LLM Corrector，训练信号可能继承生成模型偏差；方法效果也受检索证据质量和 Moderator 微调数据覆盖度影响。

## 9. Veri-R1: Toward Precise and Faithful Claim Verification via Online Reinforcement Learning

**文件名**：`2510.01932v2.pdf`

**方法**：提出 Veri-R1 在线强化学习框架，让 LLM 在 ONCV 场景中执行 `plan -> search -> information -> think -> answer` 的多轮检索与推理。训练使用 GRPO，奖励由格式奖励、证据奖励、标签奖励组成，并加入 validity weight，要求 SUPPORT/REFUTE 的标签奖励与命中足够 gold evidence 绑定，减少“猜对标签但证据不充分”。

**贡献**：把 claim verification 从给定证据的 offline 设置推进到更接近真实场景的 online 检索-推理联合训练；设计面向标签、证据、格式的任务奖励；在 FEVEROUS、EX-FEVER、FEVER、HOVER、SciFACT 上提升 joint accuracy、label accuracy 与 evidence score，并分析 reward component 与 logit-confidence 的作用。

**局限性**：论文明确指出，训练和评估都在固定本地语料库与固定 retriever 上完成；真实 fact-checking 面对更大、动态更新的语料，信息持续涌现。作者认为需要接入真实规模检索器，才能更好反映实际条件并提升可靠性与鲁棒性。

## 10. ZoFia: Zero-Shot Fake News Detection with Entity-Guided Retrieval and Multi-LLM Interaction

**文件名**：`2511.01188v3.pdf`

**方法**：提出两阶段零样本框架 ZoFia。第一阶段用 Hierarchical Salience 结合全局/局部语义为新闻实体打分，再用 SC-MMR 提取核心实体，并驱动 Wikipedia + Open Web 双源检索。第二阶段用多智能体并行拆分内容推理与事实核查，通过对抗式 debate 打破单 LLM 的 early stance locking 和 confirmation bias。

**贡献**：将实体引导检索与多 LLM 互动结合到零样本假新闻检测；提出面向新闻实体的层级显著性指标和 SC-MMR；在 PolitiFact、GossipCop 上超过零样本基线，并超过多数 few-shot 方法。

**局限性**：论文明确指出，缺少高质量、持续更新的公开数据集，导致无法评估最新新闻；外部检索模块较轻量，未使用 reranker 或更复杂 RAG；当前主要处理文本模态，尚未覆盖图文等多模态 misinformation，作者建议引入 VLM 视觉专家并研究跨模态 debate 融合。

## 11. Large Language Models Require Curated Context for Reliable Political Fact-Checking, Even with Reasoning and Web Search

**文件名**：`2511.18749v1.pdf`

**方法**：构建 PolitiFact 事实核查数据库：抓取 2007 至 2024-10 的 PolitiFact claims、verdicts 和文章全文，用 GPT-3.5 生成证据导向摘要，人工抽样检查摘要 faithful 性，并用 Chroma + MiniLM embedding 建立 curated RAG。评估 15 个 OpenAI/Google/Meta/DeepSeek 模型，在 baseline、reasoning、web search、curated RAG 不同条件下预测六档 Truth-O-Meter 标签。

**贡献**：系统比较推理模型、联网搜索模型和 curated context 对政治事实核查的影响；发现标准 LLM 表现差，reasoning 收益很小，web search 仅中等提升；而 PolitiFact 摘要构成的 curated RAG 平均 macro-F1 提升 233%，说明瓶颈主要是“是否拿到正确上下文”，而非单纯推理能力。

**局限性**：论文明确指出，只评估 PolitiFact，可能不能泛化到其他 fact-checker、平台或 claim 类型；未解决 breaking news problem，多数 claim 早于评估时间；模型依赖 vendor API，供应商更新会影响可复现性；只报告系统性能，未评估对用户信念、信任或分享行为的下游影响。

## 12. REFLEX: Self-Refining Explainable Fact-Checking via Verdict-Anchored Style Control

**文件名**：`2511.20233v3.pdf`

**方法**：提出 REFLEX，用内部激活控制而非外部检索来提升可解释 fact-checking。流程包括：将 fact-checking 改写为单轮 QA 式对话训练，联合生成 verdict 与 explanation；比较 backbone 与 fine-tuned model 在训练集上的自分歧样本，区分 reasoning gain 与 knowledge loss；从分歧样本中构造并分解 steering vectors 为 Inference Vectors 和 Knowledge Vectors，放大有利推理风格、抑制知识冲突方向，再用 token hidden-state 与向量相似度精炼 explanation。

**贡献**：提出 verdict-anchored style control，用自分歧信号解耦 fact 与 explanation style；在少量 self-refined 样本下提升 LLaMA/Qwen/Mistral 系列模型的 verdict accuracy 与 explanation faithfulness；减少 faithfulness hallucination，并展示跨模型、跨数据集迁移能力。

**局限性**：论文明确指出，实验规模受磁盘配额与实验成本限制，只覆盖 LLaMA-2、Qwen-3、Mistral-v0.1，且主要是 7B/8B 规模；在 LIAR-RAW 上采用三分类标签，无法覆盖 “pants-on-fire” 等更细粒度标签；内部知识可能过时，而现有 time-shifted 数据集如 VitaminC 缺少 explanation，难以直接验证其 explainable paradigm 对新事实的适应。

## 13. Robust Fake News Detection using Large Language Models under Adversarial Sentiment Attacks

**文件名**：`2601.15277v1.pdf`

**方法**：提出 AdSent。先用 LLM 作为 counterfeiter，把新闻改写为 positive/negative/neutral 三种情感版本，并要求事实不变；用原文与情感改写版本评估检测器鲁棒性和预测翻转。随后采用 sentiment-agnostic 训练：将训练样本中性化，用 LLaMA 类模型基于 `fake/real` token logits 做二分类，使模型更依赖事实内容而非情感线索；推理时也先中性化再检测。

**贡献**：系统揭示情感操控是 fake news detector 的 adversarial vulnerability；发现 neutral-toned 内容尤其容易被判为 real，非中性内容更容易被判为 fake；AdSent 在 PolitiFact、GossipCop 及跨数据/跨攻击场景中提升准确率和鲁棒性，并优于 SheepDog 等风格鲁棒基线。

**局限性**：论文未设独立 Limitations。论文明确/未来工作指出，未来需扩展到多模态 misinformation，研究图像中的情绪表达及其与文本情感的交互。推断：方法依赖 LLM 中性化保持事实不变，若改写引入遗漏或细微事实漂移，会污染训练/评估；主要聚焦情感攻击，对实体替换、时序篡改、图文不一致等其他攻击类型覆盖有限。

## 14. ExDR: Explanation-driven Dynamic Retrieval Enhancement for Multimodal Fake News Detection

**文件名**：`2601.15820v1.pdf`

**方法**：提出 ExDR，面向多模态假新闻检测的 explanation-driven dynamic RAG。先让 LVLM 输出检测标签和解释；retrieval triggering 模块用解释中的三类置信度信号判断是否检索：label-level uncertainty、token-level support、sentence-level confidence，并用两阶段 hybrid search 确定阈值；evidence retrieval 模块从解释中抽取关键实体，构建 entity-enriched multimodal hybrid index，并基于细粒度 deception labels 做正负对比证据检索。

**贡献**：首次将 dynamic RAG 系统化用于多模态假新闻检测，避免对每个样本盲目检索；提出解释驱动的触发与实体增强检索机制；设计 Retrieval Identification Rate 与 Retrieval Efficiency 两个指标；在 AMG 和 MR2 上提升 retrieval triggering、retrieval quality 和最终检测效果，并展示跨域泛化。

**局限性**：论文未设独立 Limitations。推断：检索语料主要由 AMG 训练集构建，跨域 MR2 仍复用 AMG 语料和微调模型，真实开放世界覆盖度有限；依赖模型生成 explanation 来触发检索和抽实体，若初始解释错误或遗漏关键实体，后续检索会受影响；使用 AMG 的细粒度 deception labels 支持对比检索，迁移到无此类标注的数据集需要额外标注或替代机制。

## 15. R^3: Replay, Reflection, and Ranking Rewards for LLM Reinforcement Learning

**文件名**：`2601.19620v2.pdf`

**方法**：面向 GRPO 在困难推理任务中“组内 reward 全相同”导致 advantage collapse 的问题，提出 R^3 强化学习框架。核心组件包括 Cross-Context Replay，为当前全对或全错的 rollout group 检索同一 query 的历史相反 reward 样本，重建组内 reward 方差；In-Context Self-Reflection，对历史平均 reward 低于阈值的 hard query，将历史错误轨迹作为反思上下文加入 prompt；Structural Entropy Ranking Reward，对截断或失败样本，用 token-level peak entropy 与 global entropy 的偏序排名分配相对 reward。

**贡献**：将历史轨迹缓冲区用于同一 query 的跨上下文 replay，使原本会被丢弃或梯度无效的 homogeneous groups 仍能产生训练信号；用 ISR 显式利用失败样本，让模型在困难题上基于历史错误进行自我修正；用 SERR 为无法验证最终答案的截断轨迹提供无监督、相对密集的过程信号。

**局限性**：推断：方法依赖同一 query 的历史轨迹缓冲区；对一次性、无重复采样或无法按 UID 聚合历史尝试的任务，CCR/ISR 的作用会受限。SERR 假设“高 peak entropy + 低 global entropy”对应更有价值的失败推理轨迹，但这是一种代理信号，未必在数学以外的开放式任务中稳定成立。实验集中在数学推理和 DeepSeek-R1-Distill-Qwen 系列，尚未证明对代码、事实问答、多模态推理或非 rule-based reward 任务同样有效。

## 16. Cross-Domain Fake News Detection on Unseen Domains via LLM-Based Domain-Aware User Modeling

**文件名**：`2602.01726v1.pdf`

**方法**：提出 DAUD，面向 unseen-domain cross-domain fake news detection。框架包含 LDAE，用 LLM 做新闻特征增强、用户行为建模、个性化 engagement augmentation；Domain-aware user agent，根据用户历史 engagement 迭代生成和修正用户画像，并预测用户是否会转发未交互新闻；DSRA，对原始数据特征和 LLM 生成特征做 domain-shared feature learning，并通过 relation-aware alignment 对新闻、用户、用户 engagement 三层信息进行对齐。

**贡献**：明确提出 unseen-domain CD-FND 设置；将 LLM 用于高层语义抽取，不只处理新闻文本，还对用户跨域行为偏好建模；用 DSRA 缓解 LLM 生成特征的 hallucination/noise 和 domain-specific bias，通过原始特征与 LLM 特征关系建模学习稳定共享表征。

**局限性**：论文未来工作明确指出，当前框架依赖 common users 提供跨域行为桥梁；作者计划扩展到没有 common users 的场景。推断：LDAE 多次调用 LLM 生成新闻摘要、用户画像、预测 engagement 和评论，计算成本与延迟较高；用户画像和生成评论可能继承 LLM 偏差；实验域为 Politics/Entertainment/COVID-19，尚未覆盖更细粒度、快速演化或低资源语言环境。

## 17. PAMAS: Self-Adaptive Multi-Agent System with Perspective Aggregation for Misinformation Detection

**文件名**：`2602.03158v1.pdf`

**方法**：提出 PAMAS，一个 LLM 多智能体 misinformation detection 框架，用 hierarchical perspective-aware aggregation 缓解 information drowning。Auditors 只观察特定 feature subset，捕捉局部异常线索；Coordinators 聚合下级 Auditor/Coordinator 的判断，维护 trust weights；Decision-Maker 结合完整上下文、协调器输出和自演化 memory 做最终判断。自适应机制包括 topology adaptation、targeted correction 和 confidence-guided routing。

**贡献**：将 LLM-empowered MAS 系统化引入 misinformation detection，并指出传统 MAS 在该任务中的 information-drowning 问题；通过 feature decomposition 和分层聚合保留弱异常信号；同时优化准确率、解释性和 token efficiency；case study 展示可从最终决策回溯到具体异常线索。

**局限性**：论文未来工作明确指出，尚未扩展到多模态 misinformation；作者也提出可加入 adaptive human-in-the-loop 以提升高风险审核场景的问责性。推断：框架需要为不同数据集设计特征维度和 Auditor profile，跨任务迁移时存在人工配置成本；最终决策仍可能在 Coordinator 分歧或 Decision-Maker 覆盖多数意见时误判；系统维护复杂度高于单模型方法。

## 18. Retrieval-Augmented Multimodal Model for Fake News Detection

**文件名**：`2604.18112v2.pdf`

**方法**：提出 RAMM，用于 multimodal multi-domain fake news detection。框架包含 MLLM backbone；Abstract Narrative Alignment，用 LLM 抽取新闻的 abstract narrative，再用文本 embedding 检索同叙事的 in-domain 和 out-of-domain 新闻；Narrative-enhanced fusion + CIBL，通过 attention 从候选 homogeneous news 合成正样本，并用 Common Information Bottleneck Loss 做 alignment、reconstruction、compression；Semantic Representation Alignment，用 CLIP 检索最相似训练样本作为 demonstration，让模型做 instance-based analogical reasoning。

**贡献**：针对“同一虚假叙事以不同图文样本集群传播”的问题，显式建模跨实例 narrative consistency；将 domain-specific knowledge gap 转化为 retrieval-augmented reasoning；提出 CIBL，避免普通 contrastive learning 强行拉近 noisy features 导致个性化信息损失；在多个公开数据集上整体优于相关多模态假新闻检测方法。

**局限性**：推断：ANA 依赖 LLM 抽取 abstract narrative；如果 LLM 摘要遗漏关键 claim、stance 或引入幻觉，后续同叙事检索会被放大影响。SRA/ANA 都依赖训练集检索相似样本；对训练集中缺少同叙事或相似案例的新事件，收益可能下降。框架包含 MLLM、CLIP、LLM narrative generation、embedding retrieval、CIBL 多重模块，训练和推理成本较高。检索只基于训练集也意味着实时新闻中的外部事实更新没有被直接纳入。

## 19. Unseen Fake News Detection Through Causal Debiasing

**文件名**：`3701716.3715517.pdf`

**方法**：提出 FNDCD，用于 zero-shot unseen-domain fake news detection。基于结构因果模型，将传播图观测特征视为由 causal variable 和 environment-biased variable 共同生成。用 RoBERTa 编码新闻、评论和转发文本，再用两层 GCN + MLP 分类；同时训练 classification model、structure estimator 和 posterior inference，并用 environment-independent 概率作为训练 reweight。

**贡献**：将 unseen fake news detection 形式化为 OOD generalization，而不是依赖目标域少量标注的 domain adaptation；从因果角度解释训练集中的 event/domain bias 如何破坏跨域泛化，并用 posterior reweighting 降低 biased samples 的影响；在 Twitter/Weibo open-domain 到 COVID-19 unseen-domain 的跨域、跨语言实验中超过多个 baseline。

**局限性**：论文明确指出，需要预定义 prior ratio `p(e)`，可改为动态估计 pseudo-environment variable；文本内容与环境变量之间的依赖建模仍不充分，作者建议利用 LLM 作为新闻内容处理器或辅助 agent；未来还需探索可扩展、实时的假新闻检测，并最好与 Twitter/X 等平台协作。推断：当前模型使用较简单的两层 GCN 和结构估计假设，可能难以刻画复杂传播行为、跨平台扩散或多模态内容。

## 20. Triple-R: Iterative Query Rewriting and Refinement for Retrieval-Augmented Fake News Detection

**文件名**：`3774904.3792246.pdf`

**方法**：提出 Triple-R：Rewriting, Retrieval, and iterative Refinement。用 T5-large 作为可训练 query rewriter，将原始新闻改写为检索 query；训练方式是用 frozen LLM-based fake news detector 的验证反馈作为 reward，通过强化学习优化 query rewriter；检索源包括 Bing Web search 和 Wikipedia/Contriever-MS MARCO；迭代 refinement 根据上一轮检索证据检查相关性，若不足则更新 query；最终将原始新闻与多轮检索证据合并，由验证模块输出真假判断。

**贡献**：将 retrieval-augmented fake news detection 的重点从“改 verifier”转向“改 query”，指出新闻原文和所需证据之间存在检索意图鸿沟；用小模型 rewriter + LLM detector reward，使 query rewriting 与下游真假检测目标对齐；证明 trainable rewriter 优于 frozen LLM rewriter 和 direct retrieval；query refinement 在 3-4 轮附近效果最好。

**局限性**：论文未来工作明确指出，计划将 Triple-R 扩展到其他 retrieval-augmented web mining 任务；当前验证范围主要是假新闻检测。推断：方法依赖外部搜索引擎和 Wikipedia 检索质量；reward 来自 frozen LLM-based detector，如果 detector 本身偏置或错误，rewriter 会被优化到错误检索偏好；多轮 refinement 超过 3-4 轮后性能下降，说明迭代检索容易引入冗余或噪声证据。

## 21. A Fact-Checking Framework with Denoising Evidence Retrieval and LLM-Based Debate Verification

**文件名**：`3774904.3792285.pdf`

**方法**：提出 SLED，包含自监督去噪证据检索和 LLM 增强的多智能体辩论验证。检索阶段从 Wikipedia、Google Web 检索、LLM 生成证据三类来源获得候选证据，训练 BERT verifier，并用 credibility 与 necessity 分数筛掉噪声证据。验证阶段设置 supporting agent 和 refuting agent，分别从支持/反驳立场构造论证；辩论包含 Argument、Questioning、Answering 三阶段，并可多轮迭代；Judge agent 对双方辩论结果评分，最后将 LLM judge 判断和 gold label 蒸馏到轻量 BERT 模型。

**贡献**：提出基于 credibility + necessity 的自监督证据选择机制，针对多源检索中的噪声证据进行过滤；将 claim verification 建模为支持方与反驳方的多轮辩论，缓解 LLM 因遗漏关键事实细节产生的幻觉；不只依赖 LLM judge，而是把辩论过程和标签蒸馏进小模型，提高分类稳定性。

**局限性**：明示/实验错误分析指出，剩余错误主要来自证据检索失败、NEI 倾向歧义、语义误解和推理错误；其中检索失败是最主要来源。推断：框架依赖 Google/Wikipedia/LLM 生成证据，多源检索质量会直接限制上限；多智能体辩论和多轮问答增加推理成本，训练阶段仍依赖 LLM 生成高质量辩论轨迹。

## 22. Semi-Supervised Fake News Detection with Mixture of Experts

**文件名**：`3774904.3792312.pdf`

**方法**：提出 S2MOE-F，用于半监督假新闻检测。使用 GNN expert 建模传播结构，识别 structural camouflage；使用 LLM expert 建模文本语义，识别 semantic camouflage。通过 Disentangled Masked Transformer 对 GNN/LLM embedding 做双向 masked attention，降低专家冗余。训练目标包括 OCC loss、InfoNCE loss 和 Covariance loss；用专家对 OCC center 的距离一致性生成高置信伪标签，利用无标签数据；用 RL-based router 动态调整 GNN/LLM 专家权重。

**贡献**：将 GNN 和 LLM 作为两个相对独立的防线，而不是简单把 LLM embedding 塞进 GNN，保留语义与结构判断的可交叉验证性；面向标签稀缺场景，用 OCC 减少对假新闻标签的依赖，并通过伪标签利用无标签样本；提出 disentangled masked Transformer 缓解专家冗余，并提出 RL router 在有限监督下学习专家主导权重。

**局限性**：推断：OCC 假设“真新闻形成紧凑分布，假新闻在边界外”，但真实新闻主题、平台和语言跨度很大时，真新闻分布可能并不紧凑；伪标签依赖 GNN/LLM 对 OCC center 距离的共识，如果两个专家在同一偏差方向上达成一致，错误伪标签可能被迭代放大；方法需要传播图结构和文本内容同时可用；RL router 是 batch-aware 而非严格 sample-level routing。

## 23. IConMoE: Modeling Intents of Misinformation using Concept Activation Vector-based Mixture of Experts

**文件名**：`3774904.3792582.pdf`

**方法**：提出 IConMoE，用意图建模增强 claim veracity prediction。定义 6 类 misinformation intent，使用 Gemini 离线生成每类 intent 的正例和风格匹配负例，训练线性分类器得到 Concept Activation Vector。CAV 固定后，在训练和推理时不再需要 LLM 调用。用 CAV-guided cross-attention 让每个 intent CAV 查询 claim embedding，得到 intent-aligned 表示；每个 intent 对应一个轻量 MLP expert，router 以 softmax 分配多意图权重，并加入 shared expert 捕获未覆盖意图或真实声明中的普通传播目的。

**贡献**：首次将 CAV 用于 misinformation intent 建模，并避免每个样本反复调用 LLM 推断 intent；从单意图分类转向多意图混合建模，适合政治讽刺、金融煽动、社会操纵等意图共存的声明；IConMoE 可作为 plug-and-play 模块接入 T5 等预训练 backbone；消融验证 CAV、router、contrastive/KL loss 的作用。

**局限性**：明示/未来工作指出，作者计划进一步研究 emotion 与 intent 的相互作用，说明当前模型主要建模 intent，未显式建模情绪因素；作者计划扩展到多模态 misinformation，当前 IConMoE 主要面向文本 claim。明示/错误分析显示，模型会在政治化但真实的声明上过度激活 manipulation intents，导致 false positive。推断：6 类 intent taxonomy 对新型、领域特定或文化相关的 misinformation motive 覆盖有限。

## 24. Navigating Truth in Multimodal Fact-checking via Retrieval- and Reasoning-Enhanced Large Language Models

**文件名**：`3774904.3792706.pdf`

**方法**：提出 FACTCOMPASS，面向多模态 fact-checking。Semantic-enhanced Retrieval Module 基于 CLIP 冻结图文编码器和 MLP adapter，构建 claim-evidence image pair，并使用 dual Fact NCE loss 对齐 claim/evidence 的图文语义。Knowledge-enhanced Retrieval Module 用命名实体识别、知识图谱和 Wikidata 等结构化知识库扩展文本证据，并用 SearchAgent/RefineAgent 检索与精炼证据。LLM reasoning training pipeline 包含 GPT-4o 冷启动生成 reasoning 轨迹、rejection sampling 筛选和 GRPO 优化。

**贡献**：将图像证据检索、文本知识增强和 LLM 深度推理训练整合成统一多模态 fact-checking 框架；针对新闻图像语义对齐不足，构建 claim-evidence image pair 并提出 Fact NCE 训练检索模块；用 GRPO 替代僵硬模板化推理，增强模型生成可解释推理链的能力；在多个中英文多模态数据集上验证效果。

**局限性**：明示：由于可用训练数据限制，论文将任务形式化为二分类；AVeriTeC、Factify、MOCHEG、CLAIMREVIEW2024+ 等因标签体系或任务格式差异未直接纳入主比较。推断：冷启动依赖 GPT-4o 生成 reasoning 数据，训练质量和偏差会受到教师模型影响；框架包含图像检索、知识图谱检索、agent 精炼和 RL fine-tuning，系统复杂度和部署成本较高；知识库覆盖不足、实体链接错误或图像证据难以检索时性能可能下降。

## 25. Knowledge-Enhanced Multimodal Fake News Detection: Semantic Visual and Priority Fusion

**文件名**：`3774904.3792729.pdf`

**方法**：提出 SVPF-Net，核心是 semantic visual representation 和 modality-priority fusion。文本特征由 RoBERTa 提取；视觉空间特征由 FND-Conformer 提取，结合 CNN/ResNet-like 分支捕获局部伪造痕迹，ViT 分支捕获全局结构和长距离语义异常，并通过 Dual Fusion Enhancement Module 融合；频域特征通过 DCT 和 CNN 提取处理痕迹；优先级渐进融合先以文本为语义锚点与图像空间特征做双向 co-attention，再与图像频域特征做第二层 co-attention。

**贡献**：提出面向假新闻图像的 FND-Conformer，同时捕获局部篡改线索和全局结构异常；将文本作为语义主导模态，引导视觉空间与频域信息逐步融合，避免简单拼接导致的信息交互不足；将空间域和频域视觉线索同时纳入多模态 fake news detection。

**局限性**：明示/未来工作指出，作者计划纳入 video sequences 和 social network propagation graphs，说明当前模型主要覆盖文本 + 图像，未处理视频和传播结构；作者计划探索模型压缩和轻量化，说明当前结构可能存在计算开销和资源消耗问题。推断：方法依赖 RoBERTa、CNN、ViT、DCT-CNN、多层 co-attention，模块多且训练复杂，对小数据集可能有过拟合风险；文本被设为语义锚点，当文本极短、含讽刺或故意误导，而图像更可靠时，固定的模态优先级可能限制模型判断。

## 26. HCSL: Rumor Detection by Integrating Intra-Sample Curriculum Learning and Hierarchical Semantic Learning

**文件名**：`3774904.3793016.pdf`

**方法**：提出 HCSL，结合 intra-sample curriculum learning 和 hierarchical semantic learning。输入事件包含 source post、propagation graph 和 source user 的 historical posts。ISCL 在单个传播图内部按节点深度构造 curriculum，初期只保留靠近 root 的浅层节点，随训练推进逐步增加传播深度，并用 GCN 编码传播子图。HSL 用 LDA 从用户历史帖子中过滤与事件主题相关的历史文本，使用两个轻量 BERT encoder 分别学习 word-level 和 post-level 表示，通过 attention 聚合后与传播图表示拼接，并进行语义对齐。

**贡献**：将 curriculum learning 从 inter-sample 扩展到 propagation graph 内部，针对深层传播噪声设计训练路径；同时建模传播结构和 source user 历史发帖倾向，提升 rumor 表示完整性；用层级语义学习捕获 word-level 与 post-level 互补信息，缓解单一文本编码器的噪声和偏差。

**局限性**：明示/未来工作指出，作者计划结合 LLM 和 multimodal learning，说明当前模型尚未利用大模型能力，也未处理图像/视频等多模态谣言内容。推断：ISCL 以传播深度作为难度指标，但深层节点不一定更难或更噪，浅层节点也可能包含误导性强的内容；方法依赖 source user 历史帖子，匿名用户、新用户、隐私受限平台或历史数据缺失场景下，HSL 部分效果会受限；LDA 主题过滤面对短文本、俚语、跨语言或话题漂移时可能不稳定。

## 27. Coordinating Search-Informed Reasoning and Reasoning-Guided Search in Claim Verification

**文件名**：`Coordinating Search-Informed Reasoning and Reasoning-Guided Search in.pdf`

**方法**：提出 HARIS，将多跳声明验证拆成两个协作 LLM Agent：高层 Reasoning Agent 负责构建验证链、判断何时需要补充信息，并用 `<question>` 调用搜索；低层 Search Agent 负责围绕问题迭代检索、用 `<search>` 查询语料，并用 `<report>` 回传证据。两个 Agent 使用 GRPO 强化学习训练：Reasoning Agent 以最终二分类验证正确性为奖励；Search Agent 使用格式奖励和 LLM-as-a-Judge，将搜索报告与 GPT-4o 生成的伪答案比较。

**贡献**：显式建模“推理指导搜索”和“搜索反哺推理”的循环，而不是一次性分解 claim 或一次性检索；将复杂多跳事实核查中的隐含 bridging facts 发现交给层级 Agent 协作完成，提高可解释性；在 EX-FEVER 和 HOVER 上取得强表现，并展示 RL 训练对两个 Agent 协同的增益。

**局限性**：论文明确指出，受算力限制，仅训练 4B 模型，未验证更大模型上的上限；只处理二分类 claim verification，即 support/refute，未覆盖 neutral 或 Not Enough Info；只评估多跳声明验证，未扩展到开放域问答、反事实检测等更广义事实核查任务。

## 28. Towards Real-Time Fake News Detection under Evidence Scarcity

**文件名**：`EASE Towards Real-Time Fake News Detection underEvidence Scarcity .pdf`

**方法**：提出 EASE，面向实时新闻证据稀缺场景采用顺序式专家选择。首先由 evidence agent 使用 Serper/Google Search 和 Firecrawl 迭代检索网页，evidence evaluator 判断证据是否充分；若充分，则 evidence expert 结合新闻、证据和评价理由分类。若证据不足，转入 reasoning evaluator/expert，使用 LLM 内部世界知识生成推理并评估可靠性。若推理也不可靠，则启用 sentiment expert，基于情绪、主观性和文风线索判断。评价器使用 ChatGPT-4 生成伪标签和 rationale，再用 LoRA 微调 Qwen2.5-14B-Instruct。

**贡献**：将实时假新闻检测中的 evidence scarcity 作为核心问题，并提出“先证据、再推理、最后情绪”的分层 fallback；构造 RealTimeNews-25，包含 2024 年 6 月至 2025 年 9 月的 3,487 篇近期新闻，用于评估新近事件上的泛化；通过 evaluator-expert 结构，使模型不只是使用证据，还显式判断证据是否足以支撑结论。

**局限性**：论文明确指出，RealTimeNews-25 虽比旧数据集更接近证据稀缺，但稀缺程度仍“不显著高”，因为实时新闻很难同时获得可靠标签；sentiment fallback 对客观写作、情绪线索弱的新闻不稳，可能误导 sentiment agent/expert。推断：方法仍依赖外部搜索 API、网页抓取和 ChatGPT 伪监督，部署时会受可用性、成本和搜索质量影响。

## 29. Explainable Fake News Detection With Large Language Model via Defense Among Competing Wisdom

**文件名**：`Explainable Fake News Detection With Large Language Model.pdf`

**方法**：提出 L-Defense，用“竞争性群体智慧”做可解释假新闻检测。先把相关 raw reports 拆成句子，用 Transformer 编码 claim 和候选证据，并分别计算 false score 与 true score，抽取支持“假”和支持“真”的两组 top-k evidence。然后用 LLM 分别基于两组 evidence 生成面向 false/true 的解释。最后将 claim、false explanation、true explanation 拼接输入小模型 Transformer，通过 defense-based inference 判断哪一方解释更有信息量和说服力；最终解释随预测标签选择，若标签为 half，则合并两侧解释。

**贡献**：不再把群体意见的多数派直接当解释，而是区分支持/反驳两方，缓解 majority bias；用 LLM 将两组竞争证据压缩成自然语言 justification，再由小模型比较双方质量；不依赖人工 debunked reports 作为解释监督，仍能生成接近专家解释的 justification。

**局限性**：推断：依赖 raw reports 中存在足够多、且立场可区分的群众意见；对于早期新闻、低讨论度新闻或缺少评论/报道的 claim，竞争证据抽取会受限。LLM 生成的两侧解释可能继承证据噪声或产生幻觉，后续小模型比较的是“解释质量”，不等价于事实真实性。证据抽取监督主要来自 claim 标签和临时 veracity label，缺少句级证据真值，可能把相关但不可靠的句子排到前列。

## 30. Multi-Sourced, Multi-Agent Evidence Retrieval for Fact-Checking

**文件名**：`Multi-Sourced, Multi-Agent Evidence Retrieval for Fact-Checking.pdf`

**方法**：提出 WKGFC，将事实核查建模为开放世界、部分可观测条件下的证据获取 POMDP。系统先从 claim 抽取实体并映射到 Wikidata/DBpedia 等 KG 节点，通过 SPARQL、beam search 和 LLM pruning 做 expand-and-prune 子图检索。若 KG 证据不足，Agent 可选择继续 `expandKG()` 或触发 `webSearch()`；网页证据经 Serper/BM25 粗检索、LLM 细粒度一致性过滤后，转成 triplets/annotations 并对齐到 KG schema，形成 web-enhanced KG。最后 LLM 基于聚合证据输出 verdict 和 justification。策略改进通过 self-reflection 和 prompt optimization 完成，不微调模型参数。

**贡献**：将结构化 KG 和开放 Web 证据放在统一 Agentic retrieval 框架内，解决单纯文本 RAG 的语义相似但事实不相关问题；用 POMDP 建模“何时继续检索、何时停止并判定”，比一次性检索更贴近人工 fact-checking；在多个数据集上相较 HerO、FIRE、GraphCheck 等方法取得更高 balanced accuracy，尤其适合 KG 不完整的开放世界场景。

**局限性**：推断：强依赖实体识别、实体链接和 KG 覆盖率；claim 中实体模糊或 KG 缺失时，初始子图可能偏离正确证据链。Web evidence 被转成 KG triplets 时可能丢失语境、时间条件和来源可信度细节。多步 SPARQL、网页检索、LLM pruning 和 prompt optimization 带来较高延迟与工程复杂度，不一定适合强实时场景。

## 31. REFLEX: Self-Refining Explainable Fact-Checking via Verdict-Anchored Style Control

**文件名**：`Self-Refining Explainable Fact-Checking via.pdf`

**方法**：与 `2511.20233v3.pdf` 为同一篇 REFLEX。方法是通过模型内部 activation editing 对齐 verdict 与 explanation：将 fact-checking 重写成单轮对话式 QA，训练模型联合生成 verdict 和 explanation；比较 backbone 与 SFT 后模型的预测，选择 self-disagreement 样本；构造 steering vectors，并拆成 Inference Vectors 与 Knowledge Vectors，在解码层进行干预；解释 refinement 阶段用隐藏状态与最优向量的 cosine alignment 找出负向片段并压制冗余/噪声表达。

**贡献**：不依赖检索、外部 API 或多 Agent，而是通过内部 activation editing 对齐 verdict 与 explanation；用少量 self-refined 样本即可在 RAW-FC、LIAR-RAW、AVeriTeC 等数据上提升 verdict 和解释质量；提出区分 fact-sensitive 与 style-sensitive 信号的 KV/IV 机制，并观察到 direction divergence 越大，增益越明显。

**局限性**：论文明确指出，实验规模受磁盘和实验成本限制，仅覆盖 LLaMA-2、Qwen-3、Mistral-v0.1，且主要是 7B/8B 规模；在 LIAR-RAW 上采用 3-way label scheme，未处理如 “pants-on-fire” 这类更细粒度标签；内部知识可能过时；未系统分析或缓解开源模型预训练数据中的社会偏见。

## 32. The Truth Becomes Clearer Through Debate! Multi-Agent Systems with Large Language Models Unmask Fake News

**文件名**：`The Truth Becomes Clearer Through Debate! Multi-Agent.pdf`

**方法**：提出 TruEDebate（TED），把假新闻检测设计成多 Agent 结构化辩论。DebateFlow Agents 分成 Proponents 和 Opponents 两队，分别主张新闻为真/假，并按 Lincoln-Douglas debate 流程执行 opening statement、cross-examination/rebuttal、closing statement。所有辩论内容形成 debate log。InsightFlow Agents 包括 Synthesis Agent 和 Analysis Agent：前者总结双方论点，后者用 role-aware encoder 编码不同角色发言，构造 debate graph，并用 graph attention 和 interactive attention 融合新闻内容与辩论结构，输出最终真假判断和解释。

**贡献**：将 LLM 推理能力组织成正式辩论流程，而不是直接 prompt LLM 给出单一结论；通过支持方/反对方、多阶段质询和反驳，让检测过程天然生成可解释的 debate log 与 summary；在 ARG-EN 和 ARG-CN 上优于传统模型和单独 LLM，并展示不同 LLM backbone 下的鲁棒性。

**局限性**：推断：不调用外部检索或工具，意味着辩论主要依赖 LLM 内部知识和新闻文本；对需要最新事实或外部证据的 claim，可能产生自洽但事实错误的辩论。多 Agent、多轮辩论会增加推理成本和延迟，难以直接用于大规模实时审核。辩论角色和流程预定义，可能让模型为了立场而生成强行支持/反驳的论点，放大 hallucination 或 confirmation bias。
