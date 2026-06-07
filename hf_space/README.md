---
title: TinyStories LM Playground
emoji: 📖
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
short_description: Generate text from a from-scratch Transformer LM
---

# 📖 TinyStories LM Playground

A from-scratch Transformer language model (byte-level BPE tokenizer, RoPE,
SwiGLU, RMSNorm — no Hugging Face `transformers`), trained on the TinyStories
dataset and served as an interactive text-generation playground.

Type a prompt, tune temperature / top-p / length, and watch it write a short
story token-by-token. Built as part of the NYU "Building LLM Reasoners" course
(adapted from Stanford CS336 Assignment 1).
