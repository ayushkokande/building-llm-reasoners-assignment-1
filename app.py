#!/usr/bin/env python3
"""
Gradio web UI: a text-generation playground for a trained Transformer LM.

Loads a checkpoint + BPE tokenizer once, then serves a browser UI with a
prompt box and temperature / top-p / max-tokens controls. Generation reuses
the exact sampling logic from `student/decode.py`, streaming tokens live.

Example (run on the pod after training, or locally with artifacts/):
    uv run python app.py \
        --checkpoint artifacts/checkpoints/checkpoint_100000.pt \
        --vocab_json artifacts/vocab.json \
        --merges_txt artifacts/merges.txt \
        --share
"""

from __future__ import annotations

import argparse

import gradio as gr
import torch

from student.decode import (
    _get_end_token_id,
    _infer_hparams_from_state_dict,
    _sample_next_token,
)
from student.lm import TransformerLM
from student.tokenizer import Tokenizer

# Loaded once at startup (see load_model).
MODEL: TransformerLM | None = None
TOKENIZER: Tokenizer | None = None
END_ID: int | None = None
DEVICE: str = "cpu"


def load_model(checkpoint: str, vocab_json: str, merges_txt: str, end_token: str, device: str) -> None:
    """Load tokenizer + model into module globals. Hparams inferred from the checkpoint."""
    global MODEL, TOKENIZER, END_ID, DEVICE
    DEVICE = device

    TOKENIZER = Tokenizer.from_files(
        vocab_filepath=vocab_json,
        merges_filepath=merges_txt,
        special_tokens=[end_token],
    )
    END_ID = _get_end_token_id(TOKENIZER, end_token)

    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    assert isinstance(state, dict), "Checkpoint must be a state_dict or a dict containing 'model'"

    h = _infer_hparams_from_state_dict(state)
    model = TransformerLM(**h).to(device)
    model.load_state_dict(state)
    model.eval()
    MODEL = model

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model: {h}")
    print(f"Params: {n_params / 1e6:.1f}M | device: {device}")


def generate(prompt: str, max_new_tokens: int, temperature: float, top_p: float):
    """Stream generated text token-by-token. Yields the full prompt+completion each step."""
    assert MODEL is not None and TOKENIZER is not None

    ids = TOKENIZER.encode(prompt)
    if not ids:
        yield ""
        return

    idx = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

    with torch.no_grad():
        for _ in range(int(max_new_tokens)):
            idx_cond = idx[:, -MODEL.context_length :]
            logits = MODEL(idx_cond)
            next_logits = logits[0, -1, :]
            next_id = _sample_next_token(next_logits, temperature=temperature, top_p=top_p)
            idx = torch.cat([idx, next_id.view(1, 1)], dim=1)

            if END_ID is not None and int(next_id.item()) == END_ID:
                break

            yield TOKENIZER.decode(idx[0].tolist())


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="TinyStories LM Playground") as demo:
        gr.Markdown(
            "# 📖 TinyStories LM Playground\n"
            "Generate text from your from-scratch Transformer LM. "
            "Lower temperature = safer/repetitive, higher = more creative."
        )
        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(
                    label="Prompt",
                    value="Once upon a time",
                    lines=3,
                    placeholder="Start a story...",
                )
                generate_btn = gr.Button("Generate", variant="primary")
            with gr.Column(scale=2):
                max_new_tokens = gr.Slider(16, 512, value=200, step=8, label="Max new tokens")
                temperature = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="Temperature (0 = greedy)")
                top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p (1.0 = off)")

        output = gr.Textbox(label="Generated text", lines=14, show_copy_button=True)

        inputs = [prompt, max_new_tokens, temperature, top_p]
        generate_btn.click(fn=generate, inputs=inputs, outputs=output)
        prompt.submit(fn=generate, inputs=inputs, outputs=output)

        gr.Examples(
            examples=[
                ["Once upon a time, there was a little", 200, 0.8, 0.95],
                ["The dragon looked at the boy and said", 200, 0.9, 0.95],
                ["Lily found a shiny key in the garden.", 256, 0.7, 0.9],
            ],
            inputs=inputs,
        )
    return demo


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gradio playground for a trained Transformer LM.")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt")
    p.add_argument("--vocab_json", type=str, required=True, help="vocab.json from train_bpe.py")
    p.add_argument("--merges_txt", type=str, required=True, help="merges.txt from train_bpe.py")
    p.add_argument("--end_token", type=str, default="<|endoftext|>", help="Stop token (if in vocab)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--host", type=str, default="0.0.0.0", help="Bind address (0.0.0.0 for RunPod proxy)")
    p.add_argument("--port", type=int, default=7860, help="Port to serve on")
    p.add_argument("--share", action="store_true", help="Create a public gradio.live link")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_model(args.checkpoint, args.vocab_json, args.merges_txt, args.end_token, args.device)
    demo = build_ui()
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
