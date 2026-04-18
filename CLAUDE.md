# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TruEDebate (TED) is a fake news detection system that combines multi-agent debate simulation with graph neural networks. The project has two main phases:

1. **Phase 1 (DebateFlow)**: Generate debate records using LLM-powered agents via OpenAI API
2. **Phase 2 (InsightFlow)**: Train a BERT + GAT + MHA classifier on the generated debates

## Key Commands

### Generate Debate Records
```bash
# Generate debates for English dataset (train split)
python main_generate.py --dataset en --split train --max_workers 4

# Generate debates for Chinese dataset (validation split)
python main_generate.py --dataset zh --split val --max_workers 2

# Debug mode (process only first 10 samples)
python main_generate.py --dataset en --split train --max_samples 10
```

### Train the Classifier
```bash
# Train on English dataset
python main_train.py --dataset en --epochs 30 --batch_size 4

# Train on Chinese dataset with CPU
python main_train.py --dataset zh --epochs 20 --batch_size 2 --device cpu

# Custom hyperparameters
python main_train.py --dataset en --epochs 30 --lr 1e-4 --weight_decay 1e-2 --grad_accum 4 --freeze_layers 6
```

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set OpenAI API credentials (required for Phase 1)
export OPENAI_API_KEY="your-key-here"
export OPENAI_BASE_URL="https://api.example.com/v1"  # Optional: for third-party APIs
```

## Architecture

### Two-Phase Pipeline

**Phase 1: DebateFlow Agents** (`debate_flow/`)
- Uses Mesa framework to simulate multi-agent debate
- 6 debate agents: Proponent/Opponent × {Opening, Questioner, Closing}
- 1 Synthesis agent: generates summary report
- Outputs: JSON files in `output/{dataset}_{split}/` containing debate graph structure

**Phase 2: InsightFlow Agents** (`insight_flow/`)
- Loads debate JSONs and constructs PyG graph datasets
- TEDClassifier network: Role-aware Encoder → GAT → Debate-News MHA → Classifier
- Outputs: trained model checkpoints in `checkpoints/`

### Debate Graph Structure

The debate graph has 7 nodes (6 debate speeches + 1 synthesis):
- Node 0-5: Debate speeches (proponent_opening, opponent_opening, proponent_questioner, opponent_questioner, proponent_closing, opponent_closing)
- Node 6: Synthesis summary

Edges follow temporal flow (Stage 1→2→3), adversarial connections (Pro↔Opp), and synthesis aggregation (all→synthesis). See `config.EDGE_LIST` for full edge definition.

**Important**: The original news text is NOT a graph node. It's encoded separately via BERT and interacts with the graph representation through Multi-Head Attention (Eq.10 in paper).

### Network Architecture (TEDClassifier)

1. **Role-aware Encoder**: BERT encodes node text → concatenate with role embedding projection
2. **GAT Layers**: Multi-layer GATConv with LayerNorm, ELU activation, dropout
3. **Global Pooling**: `global_mean_pool` aggregates node features to graph-level representation
4. **Debate-News Interactive Attention**: 
   - News encoded via BERT → project to `proj_dim`
   - Graph representation → project to `proj_dim`
   - MHA with Query=news, Key/Value=graph_repr
5. **Classifier**: Concatenate [graph_proj; mha_output] → Linear(2) → logits

## Configuration

All hyperparameters are centralized in `config.py`:

- **Paths**: `DATA_DIR`, `OUTPUT_DIR`, `CHECKPOINT_DIR`, `BERT_LOCAL_DIR`
- **OpenAI**: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`
- **BERT Models**: English uses `bert-base-uncased`, Chinese uses `hfl/chinese-bert-wwm-ext`
- **Graph Structure**: `ROLE_IDS`, `EDGE_LIST` define the debate graph topology
- **Model Hyperparameters**: GAT dimensions, MHA heads, dropout rates
- **Training Hyperparameters**: Learning rates, batch size, gradient accumulation, early stopping

### Local BERT Models

To avoid repeated downloads, place BERT models in `models/` directory:
```
models/
├── bert-base-uncased/
│   ├── config.json
│   ├── tokenizer_config.json
│   └── pytorch_model.bin
└── chinese-bert-wwm-ext/
    ├── config.json
    ├── tokenizer_config.json
    └── pytorch_model.bin
```

The code automatically detects local models and falls back to HuggingFace download if not found.

## Training Strategy

### Optimizer Configuration
- Uses AdamW with decoupled weight decay
- Differential learning rates: BERT layers use `lr * BERT_LR_FACTOR` (default 0.1×), other layers use full `lr`
- No weight decay on bias and LayerNorm parameters (see `_build_optimizer` in `train.py`)

### Learning Rate Scheduling
- Linear warmup for first 10% of training steps
- Cosine annealing decay to `min_lr_ratio * lr`
- Step-based scheduling (updates per gradient accumulation step, not per epoch)

### Regularization
- Gradient clipping: max_norm=1.0
- Early stopping: patience=5 epochs based on validation macro F1
- Label smoothing: 0.05
- Class weights: computed from training set label distribution
- BERT layer freezing: freeze first N layers (default 6) to prevent overfitting

### Mixed Precision Training
- Enabled by default on CUDA devices via `torch.amp.autocast` and `GradScaler`
- Disable with `--no_amp` flag if encountering numerical issues

## Data Format

### Input Data (Phase 1)
Located in `data/{en,zh}/{train,val,test}.json`:
```json
[
  {
    "content": "news text here...",
    "label": 0,  // 0=real, 1=fake (or "real"/"fake" strings)
    "id": 0
  }
]
```

### Generated Debate Records (Phase 1 Output)
Located in `output/{dataset}_{split}/{dataset}_{split}_{id:06d}.json`:
```json
{
  "id": 0,
  "news_text": "original news...",
  "label": 0,
  "dataset": "en",
  "split": "train",
  "nodes": [
    {"text": "speech text", "role_id": 0, "role_name": "proponent_opening"},
    ...
  ],
  "edge_index": [[src...], [dst...]],
  "synthesis": "summary text"
}
```

## Known Issues & Optimization Notes

See `TED_优化分析文档.md` for detailed analysis. Key points:

1. **Overfitting**: Current results show train loss dropping to 0.18 while val loss rises to 0.55. Mitigated by learning rate scheduling, early stopping, and BERT layer freezing.

2. **Class Imbalance**: Real:Fake ratio ~3:1 in ARG-EN dataset. Use class weights and label smoothing to improve F1(Fake) score.

3. **Interactive Attention Implementation**: Current code uses node-level K/V in MHA, while paper Eq.10 uses graph-level aggregated representation. The graph-level approach is implemented in the current version.

4. **Synthesis Node**: Node 6 (synthesis) aggregates information from all 6 debate nodes via incoming edges in the graph structure.

## Development Notes

- **Windows Compatibility**: DataLoader uses `num_workers=0` on Windows to avoid multiprocessing issues. On Linux, increase to 4+ for faster data loading.
- **Gradient Accumulation**: Effective batch size = `BATCH_SIZE * GRAD_ACCUM_STEPS`. Adjust based on GPU memory.
- **Checkpoint Management**: Best model saved to `checkpoints/best_model.pt` based on validation macro F1 score.
- **Logging**: Training logs written to `train.log`, generation logs to `generate.log`.
