# ThreadEmbed: Conversational Embedding Model for Threaded Multi-Turn Retrieval

## Project Overview

Build a fine-tuned embedding model optimized for chat/conversational data — threaded, multi-turn, timestamp-aware retrieval. The model should outperform general-purpose embedding models (text-embedding-3-small, bge-base) on chat-specific retrieval benchmarks.

**Target model name:** `thread-embed` (or similar)
**Base model candidates:** `bge-base-en-v1.5`, `gte-base-en-v1.5`, `all-MiniLM-L6-v2` (start small, scale up)
**Training framework:** `sentence-transformers`
**Eval framework:** Custom chat retrieval benchmark + MTEB subset for regression testing

---

## Phase 0: Project Scaffolding

Set up the repo structure and dev environment.

```
thread-embed/
├── CLAUDE.md                  # Project context for Claude Code sessions
├── README.md
├── pyproject.toml             # uv/pip project config
├── src/
│   └── thread_embed/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── downloaders.py     # HuggingFace dataset downloaders
│       │   ├── preprocessors.py   # Raw chat → clean conversations
│       │   ├── pair_builders.py   # Conversations → training pairs
│       │   └── schemas.py         # Pydantic models for data structures
│       ├── training/
│       │   ├── __init__.py
│       │   ├── config.py          # Training hyperparameters
│       │   ├── losses.py          # Custom loss functions if needed
│       │   └── train.py           # Main training loop
│       ├── eval/
│       │   ├── __init__.py
│       │   ├── benchmark.py       # Chat retrieval benchmark runner
│       │   ├── metrics.py         # MRR, NDCG, Recall@k
│       │   └── baselines.py       # Compare against base models
│       └── utils/
│           ├── __init__.py
│           └── tokenization.py
├── scripts/
│   ├── download_data.py
│   ├── build_pairs.py
│   ├── train.py
│   ├── evaluate.py
│   └── export_model.py
├── data/                      # .gitignored, local data storage
│   ├── raw/
│   ├── processed/
│   ├── pairs/
│   └── eval/
├── models/                    # .gitignored, checkpoints
├── notebooks/                 # Exploration and analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_pair_quality_analysis.ipynb
│   └── 03_eval_results.ipynb
└── tests/
    ├── test_preprocessors.py
    ├── test_pair_builders.py
    └── test_eval.py
```

### CLAUDE.md Contents (for Claude Code context)

The CLAUDE.md should include:
- Project goal: chat-native embedding model for threaded retrieval
- Current phase and what's being worked on
- Key architectural decisions made so far
- Data sources and their status
- Training configuration choices and rationale
- Eval benchmark design
- Links to relevant papers/resources

---

## Phase 1: Data Collection & Preprocessing

### 1.1 Download Source Datasets

Priority order by value/effort ratio:

| Dataset | Source | Size | Format | Priority |
|---------|--------|------|--------|----------|
| Discord-Dialogues | `mookiezi/Discord-Dialogues` on HF | 7.3M exchanges, 16M turns | ChatML/parquet | **P0** |
| IRC Disentanglement | `jkkummerfeld/irc_disentangle` on HF | ~70k annotated messages | Text + annotations | **P0** |
| Discord Unveiled (sample) | `SaisExperiments/Discord-Unveiled-Compressed` on HF | 2B+ messages (sample 1%) | JSON | **P1** |
| Slack Dev Chats | `preethac/Software-related-Slack-Chats` on GitHub | ~2 years, 5 channels | XML | **P1** |
| Topical-Chat | `Conversational-Reasoning/Topical-Chat` on HF | ~11k conversations | JSON | **P2** |
| UltraChat 200k | `HuggingFaceH4/ultrachat_200k` on HF | 200k dialogues | Parquet | **P2** |

**Task: `scripts/download_data.py`**
- Use `datasets` library to pull from HuggingFace
- Clone GitHub repos for non-HF sources
- Store raw data in `data/raw/{source_name}/`
- Log download stats (message counts, size, date range)

### 1.2 Preprocessing Pipeline

**Goal:** Normalize all sources into a common conversation format.

```python
# Target schema (src/thread_embed/data/schemas.py)
class Message(BaseModel):
    id: str
    author_id: str               # anonymized
    content: str                 # cleaned text
    timestamp: datetime | None   # preserve if available
    reply_to: str | None         # parent message id if threaded

class Conversation(BaseModel):
    id: str
    source: str                  # "discord", "irc", "slack", etc.
    messages: list[Message]
    metadata: dict               # channel, topic, etc.
```

**Preprocessing steps per source:**

**Discord-Dialogues:**
- Already structured as exchanges — map to Conversation objects
- Strip ChatML formatting tokens (`<|im_start|>`, `<|im_end|>`)
- Preserve turn order as implicit thread structure

**IRC Disentanglement:**
- Use the disentanglement annotations to group messages into conversations
- Parse timestamps from IRC log format
- Handle multi-party conversations (>2 participants)

**Slack Dev Chats:**
- Parse XML format
- Use `conversation_id` annotations for threading
- Strip XML metadata, keep speaker + content + timestamp

**Discord Unveiled (if sampling):**
- Parse JSON message objects
- Group by channel + time proximity (messages within N minutes = same conversation)
- Heavier filtering needed — lots of noise, bot messages, single-word reactions

**Common cleaning (all sources):**
- Remove URLs (replace with `[URL]` token)
- Remove bot/automated messages
- Strip platform-specific formatting (Discord markdown, Slack mrkdwn)
- Normalize unicode, collapse whitespace
- Remove conversations shorter than 3 messages
- Remove conversations where >50% of messages are under 3 words
- Anonymize any remaining usernames → `speaker_1`, `speaker_2`, etc.
- Language filter: English only (use fasttext `lid.176.bin` or similar)

**Output:** `data/processed/{source_name}/conversations.jsonl`

---

## Phase 2: Training Pair Construction

This is the most critical phase. The quality of your contrastive pairs determines model quality.

### 2.1 Pair Generation Strategies

**Strategy A: Thread-Based Positives (primary)**
- Anchor: a window of N messages (e.g., 3-8 messages) from a conversation
- Positive: a different window from the SAME conversation
- Negative: a window from a DIFFERENT conversation
- This teaches the model that messages in the same thread are semantically related

**Strategy B: Query-Response Positives**
- Anchor: a question or request message
- Positive: the response/answer message(s)
- Negative: responses from unrelated threads
- This teaches retrieval-style matching (question → relevant answer)

**Strategy C: Temporal Adjacency Positives**
- Anchor: message window at time T
- Positive: message window at time T+1 (immediately following)
- Hard negative: message window from same channel but different time (topically similar but different conversation)
- This teaches temporal coherence

**Strategy D: Summary-to-Conversation Positives (synthetic)**
- Use an LLM (Claude API or local model) to generate summaries of conversations
- Anchor: the summary
- Positive: the source conversation window
- Negative: conversations on different topics
- This bridges the gap to the "summarize then embed" use case

### 2.2 Hard Negative Mining

Easy negatives (random conversations) are less useful for training. Hard negatives improve model discrimination.

**Hard negative strategies:**
1. **Same-channel, different-thread:** Messages from the same channel/topic area but different conversations. Forces the model to distinguish topically similar but contextually separate discussions.
2. **BM25 negatives:** For each anchor, retrieve top BM25 matches that are NOT from the same conversation. These are keyword-similar but semantically distinct.
3. **Cross-batch negatives:** Use in-batch negatives during training (free hard negatives from the batch itself).

### 2.3 Pair Construction Pipeline

**Task: `scripts/build_pairs.py`**

```python
# Target output format
class TrainingPair(BaseModel):
    anchor: str           # concatenated message window
    positive: str         # related message window
    negative: str | None  # explicit hard negative (optional, can use in-batch)
    strategy: str         # which strategy generated this pair
    source: str           # dataset source
    metadata: dict        # conversation_id, window positions, etc.
```

**Target volumes:**
- Strategy A (thread-based): ~2-5M pairs (bulk of training data)
- Strategy B (query-response): ~500k-1M pairs (filtered for Q&A-like exchanges)
- Strategy C (temporal): ~1-2M pairs
- Strategy D (synthetic summaries): ~100-500k pairs (expensive, LLM-generated)
- Total: ~5-8M training pairs

**Message windowing:**
- Window size: 3-8 messages (randomly sampled per pair for robustness)
- Format each window as:
  ```
  speaker_1: message content here
  speaker_2: response content here
  speaker_1: follow up here
  ```
- Optionally prepend relative timestamps: `[+0s] speaker_1: ...` / `[+45s] speaker_2: ...`

**Output:** `data/pairs/{strategy_name}/pairs.jsonl`

---

## Phase 3: Model Training

### 3.1 Training Configuration

```python
# src/thread_embed/training/config.py
@dataclass
class TrainingConfig:
    # Model
    base_model: str = "BAAI/bge-base-en-v1.5"
    max_seq_length: int = 512
    pooling: str = "mean"  # mean pooling over tokens
    
    # Training
    epochs: int = 3
    batch_size: int = 256       # larger = more in-batch negatives
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    fp16: bool = True
    
    # Loss
    loss: str = "MultipleNegativesRankingLoss"  # InfoNCE variant
    # Alternative: "TripletLoss" if using explicit negatives
    
    # Matryoshka (optional, for flexible dimensionality)
    matryoshka_dims: list[int] | None = [64, 128, 256, 512, 768]
    
    # Data
    max_train_samples: int | None = None  # None = use all
    eval_steps: int = 1000
    save_steps: int = 5000
```

### 3.2 Training Script

**Task: `scripts/train.py`**

```python
# High-level training loop (sentence-transformers v3+)
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.trainer import SentenceTransformerTrainer

# 1. Load base model
model = SentenceTransformer(config.base_model)

# 2. Load training data as InputExamples
train_dataset = load_pairs("data/pairs/")

# 3. Configure loss
loss = losses.MultipleNegativesRankingLoss(model)
# If using Matryoshka:
# loss = losses.MatryoshkaLoss(model, loss, matryoshka_dims=config.matryoshka_dims)

# 4. Train
args = SentenceTransformerTrainingArguments(
    output_dir="models/thread-embed-v1",
    num_train_epochs=config.epochs,
    per_device_train_batch_size=config.batch_size,
    learning_rate=config.learning_rate,
    warmup_ratio=config.warmup_ratio,
    fp16=config.fp16,
    eval_strategy="steps",
    eval_steps=config.eval_steps,
    save_steps=config.save_steps,
    logging_steps=100,
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
)
trainer.train()
```

### 3.3 Training Stages

**Stage 1: Thread discrimination (bulk training)**
- Use Strategy A + C pairs (thread-based + temporal)
- ~3M pairs, 2-3 epochs
- Goal: learn that same-thread messages cluster together

**Stage 2: Retrieval fine-tuning**
- Use Strategy B + D pairs (query-response + synthetic summaries)
- ~500k-1M pairs, 1-2 epochs
- Lower learning rate (5e-6)
- Goal: optimize for the actual retrieval task

**Stage 3: Hard negative refinement (optional)**
- Mine hard negatives using the Stage 2 model itself
- Re-train with hard negatives for 1 epoch
- Goal: sharpen decision boundaries

---

## Phase 4: Evaluation

### 4.1 Build a Chat Retrieval Benchmark

This is as important as the model itself. No public benchmark exists for this.

**Benchmark design:**

**Task 1: Thread Retrieval**
- Given a query message, retrieve the correct conversation thread from a corpus of threads
- Metric: MRR@10, Recall@1, Recall@5, Recall@10
- Corpus size: 10k-50k conversation threads

**Task 2: Response Retrieval**
- Given a conversation prefix (first N messages), retrieve the next message window
- Metric: MRR@10, Recall@5
- Tests temporal coherence understanding

**Task 3: Summary-to-Thread Matching**
- Given a natural language description/summary, retrieve the matching conversation
- Metric: MRR@10, NDCG@10
- Tests the practical "find the conversation about X" use case

**Task 4: Cross-Platform Transfer**
- Evaluate on held-out Slack data when trained primarily on Discord/IRC
- Tests generalization across chat platforms

### 4.2 Baselines to Compare Against

| Model | Type | Dims | Notes |
|-------|------|------|-------|
| `text-embedding-3-small` | OpenAI API | 1536 | Commercial baseline |
| `text-embedding-3-large` | OpenAI API | 3072 | Upper bound commercial |
| `BAAI/bge-base-en-v1.5` | Open, unfinetuned | 768 | Our base model (before training) |
| `BAAI/bge-large-en-v1.5` | Open, unfinetuned | 1024 | Larger open baseline |
| `thenlper/gte-base-en-v1.5` | Open, unfinetuned | 768 | Alternative base |
| `all-MiniLM-L6-v2` | Open, unfinetuned | 384 | Lightweight baseline |
| BM25 | Lexical | N/A | Non-neural baseline |

### 4.3 Evaluation Script

**Task: `scripts/evaluate.py`**
- Run all benchmark tasks against all baselines + trained model
- Output comparison table with confidence intervals
- Generate plots for the README / paper
- Also run a subset of MTEB to check for regression on general tasks

---

## Phase 5: Packaging & Release

### 5.1 Model Export
- Save to HuggingFace model hub format
- Include proper model card with:
  - Training data sources and sizes
  - Benchmark results
  - Recommended use cases
  - Limitations (English only, primarily informal/technical chat)
  - License (Apache 2.0)

### 5.2 Benchmark Release
- Package the chat retrieval benchmark as a standalone dataset on HuggingFace
- Submit to MTEB as a new task category (stretch goal)

### 5.3 Integration Testing
- Test with Weaviate (your production target)
- Test with ChromaDB, Qdrant (broad compatibility)
- Provide example code for common retrieval patterns

---

## Development Session Plan for Claude Code

### Session 1: Scaffolding + Data Download
- Initialize repo, pyproject.toml, CLAUDE.md
- Write download scripts for P0 datasets (Discord-Dialogues, IRC)
- Verify data loads correctly, explore structure
- **Deliverable:** Raw data downloaded, initial exploration notebook

### Session 2: Preprocessing Pipeline
- Build preprocessors for each data source
- Implement common cleaning pipeline
- Write tests for preprocessors
- **Deliverable:** Normalized conversations in common schema

### Session 3: Pair Construction
- Implement Strategy A (thread-based) and Strategy C (temporal)
- Build windowing logic
- Analyze pair quality (sample and inspect)
- **Deliverable:** Initial training pairs, quality analysis notebook

### Session 4: Training V1
- Set up sentence-transformers training loop
- Train Stage 1 model on thread discrimination pairs
- Quick smoke-test evaluation
- **Deliverable:** First checkpoint, initial eval numbers

### Session 5: Evaluation Framework
- Build benchmark dataset from held-out data
- Implement all metrics
- Run baselines
- Compare v1 model against baselines
- **Deliverable:** Benchmark suite, comparison table

### Session 6: Iterate
- Analyze failure cases from eval
- Build Strategy B (query-response) pairs
- Train Stage 2
- Hard negative mining if Stage 2 results warrant it
- **Deliverable:** Improved model, updated eval numbers

### Session 7: Polish & Release
- Model card, README
- HuggingFace upload
- Weaviate integration test
- Benchmark dataset release
- **Deliverable:** Public release

---

## Key Decision Points

These are decisions to make as you go, not upfront:

1. **Base model size:** Start with `bge-base` (110M params). Only scale to `bge-large` if the base model plateaus and you have evidence that capacity is the bottleneck, not data quality.

2. **Window size:** Start with 5 messages. Experiment with 3 and 8 to see what the eval prefers.

3. **Timestamp inclusion:** Try both with and without relative timestamps in the windowed text. Hypothesis: timestamps help for temporal retrieval but might hurt for topical retrieval.

4. **Matryoshka training:** Include this from the start — it's nearly free and gives users flexibility in choosing embedding dimensionality vs. performance tradeoff.

5. **Synthetic summary pairs:** Defer Strategy D until you have eval results from A+B+C. If the "summary → thread" retrieval task is already good, skip the LLM generation cost.

6. **Multi-GPU / distributed:** Probably not needed. bge-base fine-tuning on 5M pairs fits comfortably on a single GPU with 24GB VRAM in a few hours.

---

## Hardware Requirements

- **Training:** Single GPU with 16-24GB VRAM (RTX 3090/4090, A5000, or cloud equivalent). Batch size 256 with fp16 on bge-base uses ~18GB.
- **Data processing:** 32GB+ RAM recommended for Discord Unveiled sampling. Other datasets are small enough to process on any machine.
- **Evaluation:** CPU is fine for embedding + retrieval benchmarks. GPU speeds it up but isn't required.

---

## References

- [Sentence-Transformers Training Overview](https://www.sbert.net/docs/training/overview.html)
- [MTEB Benchmark](https://huggingface.co/spaces/mteb/leaderboard)
- [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147)
- [BGE Paper](https://arxiv.org/abs/2309.07597)
- [IRC Disentanglement Paper](https://arxiv.org/abs/1810.11118)
- [MultipleNegativesRankingLoss](https://www.sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss)
