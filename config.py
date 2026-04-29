"""
TruEDebate (TED) 全局配置文件
"""

import os
from pathlib import Path

# ──────────────────────────────── 路径配置 ────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

# 自动创建必要目录
OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ──────────────────────────────── OpenAI 配置 ────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://yunwu.ai/v1")  # 第三方 API 地址，例如 "https://api.example.com/v1"
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_MAX_TOKENS = 1024
OPENAI_TEMPERATURE = 0.7
OPENAI_STAGE_MAX_TOKENS = {
    "claims": 320,
    "opening": 512,
    "questioner": 896,
    "closing": 640,
    "synthesis": 896,
}
OPENAI_STAGE_RETRY_MAX_TOKENS = {
    "claims": 448,
    "opening": 640,
    "questioner": 1152,
    "closing": 768,
    "synthesis": 1280,
}
OPENAI_MAX_RETRIES_ON_LENGTH = 2

# ──────────────────────────────── 辩论角色 ID 编码 ────────────────────────────
# 论文中原始新闻不作为图节点，而是通过独立 Encoder(F) 编码后与图表示做 Interactive Attention
ROLE_IDS = {
    "proponent_opening":  0,
    "opponent_opening":   1,
    "proponent_questioner": 2,
    "opponent_questioner":  3,
    "proponent_closing":  4,
    "opponent_closing":   5,
    "synthesis":          6,
}
NUM_ROLES = len(ROLE_IDS)  # 7
RTED_ROLE_IDS = {
    "proponent_opening": 0,
    "opponent_opening": 1,
    "proponent_questioner": 2,
    "opponent_questioner": 3,
    "proponent_closing": 4,
    "opponent_closing": 5,
    "claim_summary": 6,
    "td_rationale": 7,
    "cs_rationale": 8,
}
RTED_NUM_ROLES = len(RTED_ROLE_IDS)  # 9
RTED_MAX_CLAIMS = 5

RTED_NODE_TYPE_IDS = {
    "news": 0,
    "claim": 1,
    "td_rationale": 2,
    "cs_rationale": 3,
    "argument": 4,
    "role": 5,
    "source": 6,
    "time": 7,
    "synthesis": 8,
}
RTED_NUM_NODE_TYPES = len(RTED_NODE_TYPE_IDS)

# ──────────────────────────────── 辩论图边索引 ────────────────────────────────
# 【V4 重要修正】论文 Algorithm 1 第 18 行明确：图 V 只含辩论交互节点(6个)
# Synthesis (S) 仅作为可解释性理由 R = {S, D}，不是图节点
# Dataset 加载时会自动从 7 节点 JSON 中分离出 6 辩论节点 + Synthesis 独立字段
EDGE_LIST = [
    # 时序边: Stage 1 → Stage 2
    (0, 2), (0, 3),  # Pro-Opening → Pro/Opp-Questioner
    (1, 2), (1, 3),  # Opp-Opening → Pro/Opp-Questioner
    # 时序边: Stage 2 → Stage 3
    (2, 4), (2, 5),  # Pro-Questioner → Pro/Opp-Closing
    (3, 4), (3, 5),  # Opp-Questioner → Pro/Opp-Closing
    # 对抗边 (双向)
    (0, 1), (1, 0),  # Pro-Opening ↔ Opp-Opening
    (2, 3), (3, 2),  # Pro-Questioner ↔ Opp-Questioner
    (4, 5), (5, 4),  # Pro-Closing ↔ Opp-Closing
    # 反向时序边 (增强: 双向时序消息传递，让后期发言影响前期理解)
    (2, 0), (3, 0),  # Questioner → Opening
    (2, 1), (3, 1),
    (4, 2), (5, 2),  # Closing → Questioner
    (4, 3), (5, 3),
]  # 共 22 条边，仅在 6 个辩论节点之间
RTED_EDGE_LIST = EDGE_LIST + [
    # claim <-> debate
    (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5),
    (0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6),
    # rationale <-> claim
    (7, 6), (6, 7),
    (8, 6), (6, 8),
    # rationale <-> key debate stages
    (7, 0), (7, 2), (7, 4),
    (0, 7), (2, 7), (4, 7),
    (8, 1), (8, 3), (8, 5),
    (1, 8), (3, 8), (5, 8),
    # rationale interaction
    (7, 8), (8, 7),
]

RTED_EDGE_TYPE_IDS = {
    "news_contains_claim": 0,
    "claim_in_news": 1,
    "claim_supported_by_td": 2,
    "td_supports_claim": 3,
    "claim_supported_by_cs": 4,
    "cs_supports_claim": 5,
    "argument_cites_claim": 6,
    "claim_cited_by_argument": 7,
    "argument_cites_td": 8,
    "td_cited_by_argument": 9,
    "argument_cites_cs": 10,
    "cs_cited_by_argument": 11,
    "argument_supports_claim": 12,
    "claim_supported_by_argument": 13,
    "argument_refutes_claim": 14,
    "claim_refuted_by_argument": 15,
    "argument_attacks_argument": 16,
    "argument_attacked_by_argument": 17,
    "role_generates_argument": 18,
    "argument_generated_by_role": 19,
    "news_from_source": 20,
    "source_of_news": 21,
    "news_published_at_time": 22,
    "time_of_news": 23,
    "synthesis_mentions_claim": 24,
    "claim_mentioned_by_synthesis": 25,
    "synthesis_uses_td": 26,
    "td_used_by_synthesis": 27,
    "synthesis_uses_cs": 28,
    "cs_used_by_synthesis": 29,
    "synthesis_reviews_argument": 30,
    "argument_reviewed_by_synthesis": 31,
    "role_interacts_role": 32,
}
RTED_NUM_EDGE_TYPES = len(RTED_EDGE_TYPE_IDS)

# ──────────────────────────────── BERT 配置 ────────────────────────────────
# 本地模型目录 (若已手动下载 BERT 模型，将路径改为本地绝对路径)
# 例如: BERT_LOCAL_DIR = ROOT_DIR / "models"
# 放置方式: models/bert-base-uncased/  和  models/chinese-bert-wwm-ext/
BERT_LOCAL_DIR = ROOT_DIR / "models"

BERT_MODELS = {
    "en": "bert-base-uncased",
    "zh": "hfl/chinese-bert-wwm-ext",
    "cn": "hfl/chinese-bert-wwm-ext",  # 兼容别名
}
BERT_MAX_LENGTH = 512
BERT_HIDDEN_DIM = 768
BERT_FREEZE_LAYERS = 6  # 建议冻结前 6 层，平衡稳定性与可塑性

# ──────────────────────────────── 标签映射 ────────────────────────────────
# 中文数据集使用字符串标签 "real"/"fake"，需要映射为整数
LABEL_MAP = {
    "real": 0, "Real": 0, "REAL": 0, "true": 0, "True": 0,
    "fake": 1, "Fake": 1, "FAKE": 1, "false": 1, "False": 1,
    0: 0, 1: 1,  # 英文数据集已经是整数，直接透传
}

# ──────────────────────────────── 模型超参数 ────────────────────────────────
# 【V4 修正】回归稳定配置，避免过激的容量提升引入噪声
ROLE_EMBED_DIM = 32
ROLE_PROJ_DIM = BERT_HIDDEN_DIM  # 论文 Eq.6: Wrole ∈ R^(dh×dr)
GAT_HIDDEN_DIM = 128  # 回退到论文原始值
GAT_HEADS = 4
GAT_LAYERS = 2
GAT_DROPOUT = 0.2
PROJ_DIM = 128  # 回退到论文原始值
MHA_HEADS = 4
CLASSIFIER_DROPOUT = 0.2

# ──────────────────────────────── 训练超参数 ────────────────────────────────
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
EPOCHS = 30
GRAD_ACCUM_STEPS = 4  # 有效 batch_size = 16
USE_AMP = True
BERT_LR_FACTOR = 0.1
WARMUP_RATIO = 0.1
MIN_LR_RATIO = 0.01
EARLY_STOPPING_PATIENCE = 7
LABEL_SMOOTHING = 0.1
USE_CLASS_WEIGHT = True
GRAD_CLIP_MAX_NORM = 1.0
RTED_RELIABILITY_LOSS_WEIGHT = 0.0
RTED_CONSISTENCY_LOSS_WEIGHT = 0.0
EVITED_STRUCTURE_LOSS_WEIGHT = 0.02
EVITED_USE_CAUSAL_DEBIAS = True
EVITED_CAUSAL_KL_WEIGHT = 0.005
EVITED_CAUSAL_PRIOR = 0.5
EVITED_CAUSAL_CONF_WEIGHT = 0.5
EVITED_CAUSAL_STRUCT_WEIGHT = 0.5
EVITED_CAUSAL_MIN_WEIGHT = 0.5
EVITED_CAUSAL_MAX_WEIGHT = 1.5
SOURCE_EMBED_BUCKETS = 4096
TIME_FEATURE_DIM = 5
TEACHER_FEATURE_DIM = 6

# ──────────────────────────────── EviTED 证据配置 ────────────────────────────────
EVIDENCE_DIR = ROOT_DIR / "evidence"
EVIDENCE_DIR.mkdir(exist_ok=True)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_SEARCH_URL = os.getenv("TAVILY_SEARCH_URL", "https://api.tavily.com/search")
EVITED_MAX_EVIDENCE_PER_SAMPLE = 8
EVITED_MAX_EVIDENCE_PER_CLAIM = 2
EVITED_NUM_EDGE_TYPES = 31

# 【V4 调整】默认关闭 Focal Loss（V3 实验显示 Focal 在此任务反而更差）
# 使用 CrossEntropy + 平方根逆频率类别权重，更适合此数据规模
USE_FOCAL_LOSS = False  # 改回 CE + class_weight (V3 实验显示 Focal 反而过拟合)
FOCAL_GAMMA = 2.0       # Focal Loss 聚焦参数 (仅在 USE_FOCAL_LOSS=True 时生效)

# ──────────────────────────────── 生成配置 ────────────────────────────────
MAX_WORKERS = 4  # 多线程并发数
