#!/usr/bin/env bash
# End-to-end pipeline: download data -> train BPE -> tokenize -> train LM -> sample.
# Run once on a fresh RunPod pod (in the repo root). Idempotent-ish: re-running
# skips downloads/artifacts that already exist.
#
#   bash run_all.sh
#
# Override defaults via env vars, e.g.:
#   MAX_ITERS=50000 CTX=256 BATCH=64 bash run_all.sh
#   D_MODEL=256 LAYERS=4 HEADS=4 D_FF=512 bash run_all.sh   # smaller/faster
#
# Defaults below = the "great" config: 100k iters, ctx 256, batch 64,
# ~25M-param model (d_model 512 / 6 layers). ~1.6B tokens seen (~2.8 epochs).

set -euo pipefail

VOCAB_SIZE="${VOCAB_SIZE:-10000}"
MAX_ITERS="${MAX_ITERS:-100000}"
NPROC="${NPROC:-$(nproc 2>/dev/null || echo 8)}"
CTX="${CTX:-256}"
BATCH="${BATCH:-64}"
D_MODEL="${D_MODEL:-512}"
LAYERS="${LAYERS:-6}"
HEADS="${HEADS:-8}"
D_FF="${D_FF:-1344}"
ART="${ART:-artifacts}"
DATA="${DATA:-data}"
DEVICE="${DEVICE:-cuda}"

TRAIN_TXT="$DATA/TinyStoriesV2-GPT4-train.txt"
VAL_TXT="$DATA/TinyStoriesV2-GPT4-valid.txt"
BASE="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"

echo "==> Config: vocab=$VOCAB_SIZE iters=$MAX_ITERS ctx=$CTX batch=$BATCH"
echo "           d_model=$D_MODEL layers=$LAYERS heads=$HEADS d_ff=$D_FF nproc=$NPROC device=$DEVICE"

echo "==> [0/4] Sync deps"
uv sync

echo "==> [0/4] Download TinyStories"
mkdir -p "$DATA"
[ -f "$TRAIN_TXT" ] || wget -q --show-progress -O "$TRAIN_TXT" "$BASE/TinyStoriesV2-GPT4-train.txt"
[ -f "$VAL_TXT" ]   || wget -q --show-progress -O "$VAL_TXT"   "$BASE/TinyStoriesV2-GPT4-valid.txt"

echo "==> [1/4] Train BPE tokenizer"
if [ ! -f "$ART/vocab.json" ]; then
  uv run python -m student.train_bpe \
    --input_path "$TRAIN_TXT" --vocab_size "$VOCAB_SIZE" \
    --out_dir "$ART" --num_processes "$NPROC"
else
  echo "    (skip: $ART/vocab.json exists)"
fi

echo "==> [2/4] Tokenize train + val -> .npy"
[ -f "$ART/train.npy" ] || uv run python -m student.tokenize_dataset \
  --input_path "$TRAIN_TXT" --vocab_json "$ART/vocab.json" --merges_txt "$ART/merges.txt" \
  --output_path "$ART/train.npy" --num_processes "$NPROC"
[ -f "$ART/val.npy" ] || uv run python -m student.tokenize_dataset \
  --input_path "$VAL_TXT" --vocab_json "$ART/vocab.json" --merges_txt "$ART/merges.txt" \
  --output_path "$ART/val.npy" --num_processes "$NPROC"

echo "==> [3/4] Train Transformer LM"
uv run python -m student.train \
  --train_data "$ART/train.npy" --val_data "$ART/val.npy" \
  --vocab_size "$VOCAB_SIZE" --context_length "$CTX" --batch_size "$BATCH" \
  --d_model "$D_MODEL" --num_layers "$LAYERS" --num_heads "$HEADS" --d_ff "$D_FF" \
  --max_iters "$MAX_ITERS" --cosine_cycle_iters "$MAX_ITERS" \
  --checkpoint_dir "$ART/checkpoints" --device "$DEVICE"

echo "==> [4/4] Sample from final checkpoint"
CKPT="$ART/checkpoints/checkpoint_${MAX_ITERS}.pt"
uv run python -m student.decode \
  --checkpoint "$CKPT" --vocab_json "$ART/vocab.json" --merges_txt "$ART/merges.txt" \
  --prompt "Once upon a time" --max_new_tokens 200 --temperature 0.8 --top_p 0.95 \
  --device "$DEVICE"

echo "==> Done. Checkpoint: $CKPT"
