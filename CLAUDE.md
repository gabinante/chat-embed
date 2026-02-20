# ThreadEmbed

## Repositories
- **Embedding model:** `git@github.com:gabinante/chat-embed.git` (this repo)
- **Benchmark:** `https://github.com/gabinante/chat-bench` (separate repo)

## Project Goal
Build a fine-tuned embedding model optimized for chat/conversational data — threaded, multi-turn, timestamp-aware retrieval. Should outperform general-purpose embedding models (text-embedding-3-small, bge-base) on chat-specific retrieval benchmarks.

## Current Phase
**Session 1: Scaffolding + Data Download** — repo structure created, download scripts ready, preprocessing pipeline implemented.

## Architecture
- **Base model:** `BAAI/bge-base-en-v1.5` (110M params)
- **Training framework:** `sentence-transformers` v3+
- **Loss:** `MultipleNegativesRankingLoss` with Matryoshka wrapper
- **Matryoshka dims:** [64, 128, 256, 512, 768]
- **Max seq length:** 512 tokens

## Data Sources

### P0 (highest priority)
- `mookiezi/Discord-Dialogues` — 7.3M exchanges, 16M turns (HuggingFace, ChatML)
- `jkkummerfeld/irc-disentangle` — ~70k annotated IRC messages with conversation labels (HF)

### P1 (good value)
- `SaisExperiments/Discord-Unveiled-Compressed` — 2B+ messages, sample 1% (HF)
- `preethac/Software-related-Slack-Chats-with-Disentangled-Conversations` — 5 channels, 4 communities, with conversation_id annotations (GitHub, XML)
- `unionai/flyte-slack-data` — 28k Flyte community Slack Q&A pairs (HF)
- `Samhita/slack-data-long-responses` — 24.5k long-form Slack responses (HF)
- `GaiwanTeam/clojurians-log-v2` — ~2M Clojurians Slack messages (GitHub)

### P2 (supplementary)
- `Conversational-Reasoning/Topical-Chat` — ~11k conversations (HF)
- `HuggingFaceH4/ultrachat_200k` — 200k dialogues, synthetic (HF)

## Training Strategy
1. **Stage 1:** Thread discrimination — Strategy A (thread-based) + C (temporal) pairs, ~3M pairs, 2-3 epochs
2. **Stage 2:** Retrieval fine-tuning — Strategy B (query-response) + D (synthetic summaries), ~500k-1M pairs, lower LR (5e-6)
3. **Stage 3:** Hard negative refinement (optional) — mine hard negatives from Stage 2 model

## Pair Generation Strategies
- **A: Thread-based** — anchor/positive from same conversation, different windows
- **B: Query-response** — question message → response messages
- **C: Temporal adjacency** — consecutive message windows
- **D: Synthetic summaries** — LLM-generated summary → source conversation

## Evaluation Benchmark (novel contribution)
- **Task 1:** Thread Retrieval (query → correct thread)
- **Task 2:** Response Retrieval (prefix → next window)
- **Task 3:** Summary-to-Thread Matching (description → conversation)
- **Task 4:** Cross-Platform Transfer (Discord/IRC → Slack)
- **Metrics:** MRR@10, Recall@1/5/10, NDCG@10

## Key Commands
```bash
# Download data
python scripts/download_data.py --priority P0

# Preprocess
python scripts/preprocess_data.py --source discord_dialogues

# Build training pairs
python scripts/build_pairs.py --all

# Train
python scripts/train.py --base-model BAAI/bge-base-en-v1.5 --epochs 3

# Evaluate
python scripts/evaluate.py --model models/thread-embed-v1

# Run tests
pytest tests/
```

## Package Management
Using `uv` with `pyproject.toml`. Install: `uv sync` or `uv pip install -e ".[dev,eval]"`
