# Deploy the playground to HuggingFace Spaces (free, permanent, embeddable)

End result: a permanent URL `https://<you>-tinystories-lm.hf.space` you can
iframe into your portfolio. Free CPU tier — the model is ~28.9M params, fast
enough on CPU.

## 0. Prereqs
- A HuggingFace account: https://huggingface.co/join
- Your trained artifacts from RunPod: `checkpoint_100000.pt`, `vocab.json`, `merges.txt`

## 1. Shrink the checkpoint (optional but recommended)
The training checkpoint also stores optimizer state (~3x bigger). For inference
you only need the model weights. Strip them out:

```sh
python -c "import torch; c=torch.load('checkpoint_100000.pt', map_location='cpu'); torch.save(c['model'] if 'model' in c else c, 'model.pt')"
```

This turns ~300MB into ~100MB.

## 2. Create the Space
1. Go to https://huggingface.co/new-space
2. Owner = you, Space name = `tinystories-lm`
3. SDK = **Gradio**, Hardware = **CPU basic (free)**, Visibility = **Public**
4. Create. Note the git URL it shows.

## 3. Assemble the Space repo
Clone the (empty) Space and copy in the app + model. From this project root:

```sh
git clone https://huggingface.co/spaces/<you>/tinystories-lm
cd tinystories-lm

# app + model code (the student/ package is imported by app.py)
cp ../app.py .
cp -r ../student .

# Space config + deps (these versions live in hf_space/)
cp ../hf_space/README.md .
cp ../hf_space/requirements.txt .

# model files in a model/ dir (app.py auto-loads model/{model.pt,vocab.json,merges.txt})
mkdir -p model
cp /path/to/model.pt model/model.pt
cp /path/to/vocab.json model/vocab.json
cp /path/to/merges.txt model/merges.txt
```

## 4. Push (model.pt needs Git LFS)
```sh
git lfs install
git lfs track "model/*.pt"
git add .gitattributes app.py student model README.md requirements.txt
git commit -m "TinyStories LM playground"
git push
```

The Space builds automatically (~2-3 min). When it goes green, your app is live at:
`https://huggingface.co/spaces/<you>/tinystories-lm`

## 5. Embed in your portfolio
The direct app URL (for an iframe) is:
`https://<you>-tinystories-lm.hf.space`

```html
<iframe
  src="https://<you>-tinystories-lm.hf.space"
  width="100%" height="640" frameborder="0"></iframe>
```

That's it — permanent, free, always-on.

## Notes
- No CLI args needed on Spaces: `app.py` detects `SPACE_ID` and auto-loads from `model/`.
- To override paths without renaming files, set Space *Variables*: `MODEL_CKPT`, `VOCAB_JSON`, `MERGES_TXT`.
- Free CPU generates ~200 tokens in a few seconds. Fine for a demo.
