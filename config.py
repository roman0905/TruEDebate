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
OPENAI_MAX_TOKENS = 512
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
# 有向边列表: (src, dst)
# 论文图结构: 6 个辩论发言节点 (0-5) + 1 个 Synthesis 节点 (6)
# 原始新闻不在图中，通过 Interactive Attention 与图表示交互 (Eq.10)
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
    # 综合边: 全部发言节点 → Synthesis
    (0, 6), (1, 6),
    (2, 6), (3, 6),
    (4, 6), (5, 6),
]

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
ROLE_EMBED_DIM = 32
ROLE_PROJ_DIM = BERT_HIDDEN_DIM  # 论文 Eq.6: Wrole ∈ R^(dh×dr)，投影到与 BERT 隐层同维度
GAT_HIDDEN_DIM = 128 # 论文中 GAT 隐层维度最原始设置 dh = 128
GAT_HEADS = 4
GAT_LAYERS = 2 # 论文中 GAT最原始设置 层数 L = 2
GAT_DROPOUT = 0.2  # 降低 dropout，从 0.3 → 0.2，减少过拟合但保留学习能力
PROJ_DIM = 128
MHA_HEADS = 4
CLASSIFIER_DROPOUT = 0.2  # 降低 dropout，从 0.3 → 0.2，论文原始为 0.1

# ──────────────────────────────── 训练超参数 ────────────────────────────────
BATCH_SIZE = 4
LEARNING_RATE = 2e-4  # 提高学习率从 1e-4 → 2e-4，加快收敛
WEIGHT_DECAY = 1e-2
EPOCHS = 30
GRAD_ACCUM_STEPS = 4  # 梯度累积步数 (有效 batch_size = BATCH_SIZE * GRAD_ACCUM_STEPS)
USE_AMP = True  # 混合精度训练
BERT_LR_FACTOR = 0.1
WARMUP_RATIO = 0.1
MIN_LR_RATIO = 0.01
EARLY_STOPPING_PATIENCE = 7  # 增加 patience 从 5 → 7，给模型更多恢复机会
LABEL_SMOOTHING = 0.1  # 增加 label smoothing 从 0.05 → 0.1，缓解类别不平衡
USE_CLASS_WEIGHT = True
GRAD_CLIP_MAX_NORM = 1.0

# ──────────────────────────────── 生成配置 ────────────────────────────────
MAX_WORKERS = 4  # 多线程并发数
