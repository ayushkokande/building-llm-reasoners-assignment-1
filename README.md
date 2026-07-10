# Transformer Language Model from Scratch

A from-scratch Transformer language model — byte-level BPE tokenizer, RoPE, SwiGLU, RMSNorm, and AdamW, all implemented without Hugging Face `transformers` — trained on the TinyStories dataset.

Project scaffolding and test suite adapted from Stanford CS336 ([original repository](https://github.com/stanford-cs336/)); credit to the Stanford course staff.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

The tests are wired to the implementation through the adapter
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data:

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

cd ..
```

