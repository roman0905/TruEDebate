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
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")  # 第三方 API 地址，例如 "https://api.example.com/v1"
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_MAX_TOKENS = 1024
OPENAI_TEMPERATURE = 0.7

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
BERT_MAX_LENGTH = 256
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

# 【V4 调整】默认关闭 Focal Loss（V3 实验显示 Focal 在此任务反而更差）
# 使用 CrossEntropy + 平方根逆频率类别权重，更适合此数据规模
USE_FOCAL_LOSS = False  # 改回 CE + class_weight (V3 实验显示 Focal 反而过拟合)
FOCAL_GAMMA = 2.0       # Focal Loss 聚焦参数 (仅在 USE_FOCAL_LOSS=True 时生效)

# ──────────────────────────────── 生成配置 ────────────────────────────────
MAX_WORKERS = 4  # 多线程并发数
