# ThreadEmbed

A chat-native embedding model for threaded multi-turn retrieval. Fine-tuned to outperform general-purpose embedding models on chat-specific tasks like thread retrieval, response matching, and conversation search across Slack, Discord, and IRC.

## Quick Start

```bash
# Install
uv sync

# Download training data
python scripts/download_data.py --priority P0

# Train
python scripts/train.py

# Evaluate
python scripts/evaluate.py --model models/thread-embed-v1
```

## Project Structure

```
src/thread_embed/
├── data/           # Downloaders, preprocessors, pair builders, schemas
├── training/       # Training config, loss functions, training loop
├── eval/           # Benchmark runner, metrics, baselines
└── utils/          # Tokenization utilities
```

## License

Apache-2.0
