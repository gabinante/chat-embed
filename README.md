# ThreadEmbed

A chat-native embedding model for threaded, multi-turn retrieval. Fine-tuned from [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) on 200K conversation pairs across Discord, Slack, and IRC to outperform general-purpose embedding models on chat-specific retrieval tasks.

This isn't intended to be a premier embedding model — it's an argument for a chat-first embedding approach, paired with [chat-bench](https://github.com/gabinante/chat-bench), a benchmark for evaluating embedding models on conversational data.

## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("gabinante/thread-embed-v1")

# Encode chat messages
messages = [
    "[2024-01-15 10:32] alice: Has anyone tried the new API?",
    "[2024-01-15 10:33] bob: Yeah, the auth flow changed completely",
    "[2024-01-15 10:35] alice: What do you mean? The OAuth endpoints?",
]
embeddings = model.encode(messages, normalize_embeddings=True)

# Use truncated dimensions for faster retrieval (Matryoshka)
embeddings_256d = model.encode(messages, normalize_embeddings=True)[:, :256]
```

### Matryoshka Dimensions

Trained with [MatryoshkaLoss](https://arxiv.org/abs/2205.13147) — truncate embeddings to trade accuracy for speed/memory:

| Dimension | Thread MRR@10 | Thread NDCG@10 | Response MRR@10 | Response NDCG@10 |
|-----------|---------------|----------------|-----------------|------------------|
| 768 (full) | 0.4501 | 0.4981 | 0.7249 | 0.7641 |
| 512 | 0.4424 | 0.4893 | 0.7148 | 0.7520 |
| 256 | 0.4301 | 0.4818 | 0.6848 | 0.7284 |
| 128 | 0.4029 | 0.4531 | 0.6613 | 0.7074 |
| 64 | 0.3556 | 0.4114 | 0.6017 | 0.6537 |

*Results on internal eval set (500 queries each). See [chat-bench](https://github.com/gabinante/chat-bench) for independent benchmarks.*

## Intended Uses

- **Thread retrieval:** Find which conversation thread a message belongs to
- **Response matching:** Retrieve relevant responses given a conversation prefix
- **Chat search:** Semantic search over Discord, Slack, IRC, and similar platforms
- **Conversation clustering:** Group related messages across channels or platforms

## Limitations

- Trained primarily on English-language chat data
- Optimized for informal, multi-turn conversation — may underperform on formal documents
- Training data overlaps with the internal eval set (no holdout split); use [chat-bench](https://github.com/gabinante/chat-bench) for unbiased evaluation
- Cross-platform transfer varies: strong on Slack/Ubuntu IRC, weaker on raw IRC disentanglement

## Evaluation

For benchmark results, see [chat-bench](https://github.com/gabinante/chat-bench) — an independent benchmark for evaluating embedding models on conversational retrieval tasks including thread retrieval, response matching, and cross-platform transfer.

## Methodology

### Base Model

[BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) (110M parameters, 768-dim embeddings). Chosen for its strong general-purpose retrieval performance at a manageable size.

### Training Data

~1.1M training pairs generated from 7 preprocessed conversation datasets spanning three platforms:

| Source | Platform | Scale |
|---|---|---|
| [Discord Dialogues](https://huggingface.co/datasets/mookiezi/Discord-Dialogues) | Discord | 7.3M exchanges |
| [Discord Unveiled](https://huggingface.co/datasets/SaisExperiments/Discord-Unveiled-Compressed) | Discord | 1% sample of 2B+ messages |
| [IRC Disentangle](https://github.com/jkkummerfeld/irc-disentangle) | IRC | ~77K annotated messages |
| [Ubuntu IRC (common-pile)](https://huggingface.co/datasets/EleutherAI/common-pile) | IRC | 564K conversations, 44.8M messages |
| [Flyte Slack](https://huggingface.co/datasets/unionai/flyte-slack-data) | Slack | 28K Q&A pairs |
| [Flyte Slack (long)](https://huggingface.co/datasets/Samhita/slack-data-long-responses) | Slack | 24.5K long responses |
| [Slack Dev Chats](https://github.com/nicksonymern/Disentangle) | Slack | 5 channels, 4 communities |

### Pair Generation Strategies

Three strategies produce complementary training signal, totaling 200K pairs (sampled from 1.1M):

- **Thread-based (121K):** Anchor and positive drawn from different windows of the same conversation. Teaches the model that messages in the same thread are related even when lexically different.
- **Query-response (500K):** Question message paired with its response. Teaches retrieval of answers given questions.
- **Temporal adjacency (500K):** Consecutive message windows from the same conversation. Teaches temporal coherence.

### Loss Function

[MatryoshkaLoss](https://arxiv.org/abs/2205.13147) wrapping MultipleNegativesRankingLoss, enabling flexible dimensionality at inference time:

- **Matryoshka dimensions:** 64, 128, 256, 512, 768
- Smaller dimensions trade accuracy for speed/memory — useful for large-scale retrieval with re-ranking

### Training Configuration

| Parameter | Value |
|---|---|
| Batch size | 16 (per-device) |
| Gradient accumulation | 4 (effective batch = 64) |
| Learning rate | 2e-5 (linear warmup 10%, then decay) |
| Epochs | 1 |
| Max sequence length | 512 tokens |
| Precision | bf16 (MPS) |

Final training loss: **4.089** (from 9.457 at step 0).

## Quick Start

```bash
# Install
uv sync

# Download training data
python scripts/download_data.py --priority P0

# Preprocess
python scripts/preprocess_data.py --source discord_dialogues

# Build training pairs
python scripts/build_pairs.py --all

# Train
python scripts/train.py --base-model BAAI/bge-base-en-v1.5 --epochs 1

# Evaluate
python scripts/evaluate.py --model models/thread-embed-v1
```

### Evaluating API models

The eval script also supports OpenAI and Voyage AI models:

```bash
python scripts/evaluate.py --model "openai:text-embedding-3-small"
python scripts/evaluate.py --model "voyage:voyage-3-large"
```

Requires `OPENAI_API_KEY` or `VOYAGE_API_KEY` in `.env`.

## Project Structure

```
src/thread_embed/
  data/           # Downloaders, preprocessors, pair builders, schemas
  training/       # Training config, loss functions, training loop
  eval/           # Benchmark runner, metrics
  utils/          # Tokenization utilities
scripts/
  download_data.py
  preprocess_data.py
  build_pairs.py
  build_eval.py
  train.py
  evaluate.py
data/
  eval/           # Benchmark datasets and results
models/
  thread-embed-v1/  # Trained model (gitignored, ~1.6GB)
```

## Related

- [chat-bench](https://github.com/gabinante/chat-bench) — Independent benchmark for evaluating embedding models on chat data
- [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) — Base model
- [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) — Flexible dimensionality embeddings

## License

Apache-2.0
