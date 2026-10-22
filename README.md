# Mini-GPT: a Transformer Language Model from Scratch

A small GPT-style (decoder-only) transformer written from scratch in PyTorch. It learns to predict the next character of a text, then generates new text in the same style. No pretrained models or high-level transformer libraries are used.

## Results

Trained on Tiny Shakespeare (about 1.1M characters, 65-character vocabulary) on a Colab T4 GPU.

| Metric | Value |
|---|---|
| Parameters | 3.23M |
| Initial val loss | 4.17 (= ln 65, random guessing) |
| **Final val loss** | **1.63** |
| Training steps | 3,000 |

A val loss of 1.63 means the model gives the correct next character about 20% probability on average (e^-1.63), compared with about 1.5% (1/65) for random guessing.

## Architecture

```
characters → token embedding + position embedding
           → [ LayerNorm → Multi-Head Self-Attention → + residual
               LayerNorm → Feed-Forward (4x expand, GELU)  → + residual ] × 4 layers
           → LayerNorm → Linear → next-character scores
```

| Setting | Value |
|---|---|
| Layers | 4 |
| Attention heads | 4 |
| Embedding size | 256 |
| Context length | 128 characters |
| Dropout | 0.2 |
| Optimizer | AdamW, lr 3e-4, gradient clipping 1.0 |

Key pieces implemented by hand in [`mini_gpt.py`](mini_gpt.py):
- **Multi-head self-attention** with scaled dot-product attention: softmax(QKᵀ / √dₖ) · V
- **Causal mask** so each position can only attend to earlier positions
- **Learned positional embeddings**
- **Pre-LayerNorm residual blocks**
- **Autoregressive generation** with temperature and top-k sampling

## How to run

**Google Colab (recommended):** upload [`mini_gpt.ipynb`](mini_gpt.ipynb), set Runtime → Change runtime type → T4 GPU, and run all cells (about 10–15 min).

**Locally:**
```bash
pip install torch
python mini_gpt.py                    # downloads Tiny Shakespeare and trains
python mini_gpt.py --data my_text.txt # train on your own text
python mini_gpt.py --smoke            # tiny settings, quick CPU check
```

Run `python mini_gpt.py --help` for all options (layers, heads, context length, and more).
