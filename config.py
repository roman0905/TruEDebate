"""
TruEDebate (TED/PAMD) 全局配置文件。
"""

import os
from pathlib import Path


# ──────────────────────────────── 路径配置 ────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
BERT_LOCAL_DIR = ROOT_DIR / "models"

OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)


# ──────────────────────────────── OpenAI 配置 ────────────────────────────────

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-w8D4s4Z69HMmgUqp1FKpD0Ozz37iW2pGJHEAI9WDand20X1d")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://yunwu.ai/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "1024"))
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))


# ──────────────────────────────── 图角色配置 ────────────────────────────────

ROLE_IDS = {
    # 原始 TED 角色，保持旧输出兼容。
    "proponent_opening": 0,
    "opponent_opening": 1,
    "proponent_questioner": 2,
    "opponent_questioner": 3,
    "proponent_closing": 4,
    "opponent_closing": 5,
    "synthesis": 6,
    # Perspective-Adaptive Multi-Agent Debate 新增角色。
    "perspective_planner": 7,
    "perspective_factual_consistency": 8,
    "perspective_causal_reasoning": 9,
    "perspective_temporal_reasoning": 10,
    "perspective_emotional_manipulation": 11,
    "perspective_intent_analysis": 12,
    "perspective_linguistic_style": 13,
    "perspective_coordinator": 14,
    "self_reflective_judge": 15,
    "role_reversal_judge": 16,
    "final_judge": 17,
}
NUM_ROLES = max(ROLE_IDS.values()) + 1

# 节点角色分组：用于分层池化分类头。
# 0=planner(背景/规划), 1=perspective(证据), 2=coordinator(综合), 3=judge(裁判)
NODE_GROUP_PLANNER = 0
NODE_GROUP_PERSPECTIVE = 1
NODE_GROUP_COORDINATOR = 2
NODE_GROUP_JUDGE = 3
NUM_NODE_GROUPS = 4

ROLE_TO_GROUP = {
    # 旧 TED 角色映射
    0: NODE_GROUP_PERSPECTIVE,  # proponent_opening
    1: NODE_GROUP_PERSPECTIVE,  # opponent_opening
    2: NODE_GROUP_PERSPECTIVE,  # proponent_questioner
    3: NODE_GROUP_PERSPECTIVE,  # opponent_questioner
    4: NODE_GROUP_COORDINATOR,  # proponent_closing
    5: NODE_GROUP_COORDINATOR,  # opponent_closing
    6: NODE_GROUP_JUDGE,        # synthesis
    # PAMD 角色映射
    7: NODE_GROUP_PLANNER,      # perspective_planner
    8: NODE_GROUP_PERSPECTIVE,  # factual_consistency
    9: NODE_GROUP_PERSPECTIVE,  # causal_reasoning
    10: NODE_GROUP_PERSPECTIVE, # temporal_reasoning
    11: NODE_GROUP_PERSPECTIVE, # emotional_manipulation
    12: NODE_GROUP_PERSPECTIVE, # intent_analysis
    13: NODE_GROUP_PERSPECTIVE, # linguistic_style
    14: NODE_GROUP_COORDINATOR, # perspective_coordinator
    15: NODE_GROUP_JUDGE,       # self_reflective_judge
    16: NODE_GROUP_JUDGE,       # role_reversal_judge
    17: NODE_GROUP_JUDGE,       # final_judge
}

# 原始 TED 边结构，用于兼容旧模式和旧 JSON。
EDGE_LIST = [
    (0, 2), (0, 3), (1, 2), (1, 3),
    (2, 4), (2, 5), (3, 4), (3, 5),
    (0, 1), (1, 0), (2, 3), (3, 2), (4, 5), (5, 4),
    (0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6),
]

GRAPH_SCHEMA_VERSION = 3
EDGE_TYPE_IDS = {
    "temporal": 0,
    "support": 1,
    "attack": 2,
    "synthesis": 3,
    "self": 4,
    "plan": 5,
    "judge": 6,
    "consistency": 7,
    "cross_perspective": 8,
}
NUM_EDGE_TYPES = len(EDGE_TYPE_IDS)
EDGE_TYPE_EMBED_DIM = 32
TYPED_EDGE_LIST = [
    (0, 2, "temporal"), (0, 3, "temporal"), (1, 2, "temporal"), (1, 3, "temporal"),
    (2, 4, "temporal"), (2, 5, "temporal"), (3, 4, "temporal"), (3, 5, "temporal"),
    (0, 4, "support"), (2, 4, "support"), (1, 5, "support"), (3, 5, "support"),
    (0, 1, "attack"), (1, 0, "attack"), (2, 3, "attack"), (3, 2, "attack"),
    (4, 5, "attack"), (5, 4, "attack"),
    (0, 6, "synthesis"), (1, 6, "synthesis"), (2, 6, "synthesis"),
    (3, 6, "synthesis"), (4, 6, "synthesis"), (5, 6, "synthesis"),
]


# ──────────────────────────────── 多视角辩论配置 ────────────────────────────────

PERSPECTIVE_AGENT_DEFINITIONS = {
    "factual_consistency": {
        "role_name": "Factual Consistency Agent",
        "role_id": ROLE_IDS["perspective_factual_consistency"],
        "focus": "检查文本内部事实一致性、可验证细节、实体关系和来源可信度。",
    },
    "causal_reasoning": {
        "role_name": "Causal Reasoning Agent",
        "role_id": ROLE_IDS["perspective_causal_reasoning"],
        "focus": "检查因果跳跃、错误归因、相关性被包装成因果性的问题。",
    },
    "temporal_reasoning": {
        "role_name": "Temporal Reasoning Agent",
        "role_id": ROLE_IDS["perspective_temporal_reasoning"],
        "focus": "检查时间线、旧闻新炒、事件顺序、时间表达和年龄/日期一致性。",
    },
    "emotional_manipulation": {
        "role_name": "Emotional Manipulation Agent",
        "role_id": ROLE_IDS["perspective_emotional_manipulation"],
        "focus": "检查煽动性、恐慌性、羞辱性、情绪攻击和过度道德化表述。",
    },
    "intent_analysis": {
        "role_name": "Intent Agent",
        "role_id": ROLE_IDS["perspective_intent_analysis"],
        "focus": "判断是否存在政治操控、商业诱导、社会心理操纵或流量诱导意图。",
    },
    "linguistic_style": {
        "role_name": "Linguistic Style Agent",
        "role_id": ROLE_IDS["perspective_linguistic_style"],
        "focus": "检查标题党、夸张表达、模糊来源、断言式语气和非新闻化写法。",
    },
}
DEFAULT_PERSPECTIVE_TOP_K = int(os.getenv("PERSPECTIVE_TOP_K", "4"))
DEFAULT_DEBATE_MODE = os.getenv("DEBATE_MODE", "perspective")


# ──────────────────────────────── 模型与训练配置 ────────────────────────────────

BERT_MODELS = {
    "en": "bert-base-uncased",
    "zh": "hfl/chinese-bert-wwm-ext",
    "cn": "hfl/chinese-bert-wwm-ext",
}
BERT_MAX_LENGTH = 512
BERT_HIDDEN_DIM = 768
# 优化：增加冻结层从 6 → 9，减少可训练参数（缓解 3884 样本对应 45M 参数过拟合）。
BERT_FREEZE_LAYERS = 9

LABEL_MAP = {
    "real": 0, "Real": 0, "REAL": 0, "true": 0, "True": 0, "TRUE": 0,
    "fake": 1, "Fake": 1, "FAKE": 1, "false": 1, "False": 1, "FALSE": 1,
    0: 0, 1: 1,
}

ROLE_EMBED_DIM = 32
ROLE_PROJ_DIM = BERT_HIDDEN_DIM
GAT_HIDDEN_DIM = 256
GAT_HEADS = 4
GAT_LAYERS = 2
# 优化：GAT dropout 0.1 → 0.2，加强消息传播正则化。
GAT_DROPOUT = 0.2
PROJ_DIM = 256
MHA_HEADS = 4
# 优化：分类器 dropout 0.1 → 0.3，缓解过拟合。
CLASSIFIER_DROPOUT = 0.3
# 优化：数值特征从 8 维扩充到 16 维。
NUMERIC_FEATURE_DIM = 16
NUMERIC_FEATURE_PROJ_DIM = 64
SANITIZE_FINAL_LABEL_TEXT = True

# ──────────────────────────────── 创新点：训练正则化 ────────────────────────────────

# 节点 dropout：训练时随机 mask 部分 perspective 节点的特征，模拟 perspective 缺失。
NODE_DROPOUT_P = 0.15
# 边 dropout：训练时随机 drop 部分边，正则化 GAT。
EDGE_DROPOUT_P = 0.1
# 是否在分类头里加入 perspective 节点辅助分类损失（multi-task）。
USE_AUX_LOSS = True
# V5 修复：从 0.3 降到 0.15，减轻 aux 对主任务的扰动。
AUX_LOSS_WEIGHT = 0.15

# Focal Loss：V4 实验显示 Focal × class_weight 叠加导致梯度过冲，5 epoch 即过拟合。
# V5 默认关闭，退回 CE+class_weight+label_smoothing。
USE_FOCAL_LOSS = False
FOCAL_LOSS_GAMMA = 2.0

# 类别权重模式：
#   "inverse"  —— total/(2*counts)，会给少数类放大 ~2.86×，与 Focal 叠加会过冲；
#   "sqrt"     —— sqrt(total/counts)/normalize，温和的少数类加权 (~1.4×)；
#   "balanced" —— 直接 [1.0, 1.0]，禁用类权重，让 Focal 单独处理。
# V4 训练/测试分布差 (train fake 26% vs val/test 19%) 下，sqrt 更稳。
CLASS_WEIGHT_MODE = "sqrt"

# R-Drop 一致性损失：默认关闭，需要 2x 前向。
USE_RDROP = False
RDROP_ALPHA = 0.5

# SWA：V4 配置失效（start_epoch=18, early stop@13 → 未触发）。
# V5 把 start_ratio 改到 0.3，patience 改到 12，确保 SWA 真正能用上。
USE_SWA = True
SWA_START_RATIO = 0.3

# EMA（V5 新增）：训练全程维护影子模型，结束后与 best/SWA 三选一。
USE_EMA = True
EMA_DECAY = 0.999

# Manifold Mixup（V5 新增）：在分类器输入做 mixup，强正则化。
USE_MIXUP = True
MIXUP_ALPHA = 0.4
MIXUP_PROB = 0.5

# 阈值调优：每 epoch 都搜，并把 tuned macF1 作为 best 选择依据。
USE_THRESHOLD_TUNING = True
THRESHOLD_SEARCH_RANGE = (0.20, 0.80)
THRESHOLD_SEARCH_STEP = 0.01

BATCH_SIZE = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
# V5：non-BERT 参数（GAT/分类头/池化）weight_decay 单独提高，缓解头部过拟合。
WEIGHT_DECAY_OTHER = 0.05
EPOCHS = 30
GRAD_ACCUM_STEPS = 4
USE_AMP = True
BERT_LR_FACTOR = 0.1
WARMUP_RATIO = 0.15
MIN_LR_RATIO = 0.01
# V5：与 SWA 配合放宽到 12，避免 SWA 还没启动就 early stop。
EARLY_STOPPING_PATIENCE = 12
# V5：Focal 关闭后启用温和 label smoothing。
LABEL_SMOOTHING = 0.05
USE_CLASS_WEIGHT = True
GRAD_CLIP_MAX_NORM = 1.0
MAX_WORKERS = 4
SEED = 42
