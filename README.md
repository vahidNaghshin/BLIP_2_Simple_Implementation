# Minimal Educational BLIP-2

A self-contained, lightweight, and educational PyTorch implementation of **BLIP-2** (Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models). This codebase demonstrates the core multi-stage alignment training paradigm using a frozen CLIP ViT-B/32, a custom Q-Former, and a frozen GPT-2.

## 🚀 Project Overview

BLIP-2 bridges the gap between frozen vision-only backbones and frozen language models using a querying Transformer (Q-Former) trained in two stages:

### **Stage 1: Vision-Language Representation Learning**
Aligns the Q-Former query outputs with the text representation space by optimizing three joint objectives simultaneously over the same Q-Former parameters using specific attention masks:
1. **Image-Text Contrastive (ITC) Loss**: Aligns visual queries and textual features using a unimodal mask.
2. **Image-Text Matching (ITM) Loss**: Evaluates whether an image-text pair is matched or not using a bidirectional mask.
3. **Image-Text Generation (ITG) Loss**: Trains the Q-Former to generate text conditioned on the query features using a causal mask.

### **Stage 2: Vision-to-Language Generative Learning**
Connects the Q-Former outputs directly to a frozen Large Language Model (LLM).
* Query outputs from Stage 1 are projected into the GPT-2 embedding space.
* These visual tokens are prepended to the text embeddings as prefixes.
* The model is optimized using standard Language Modeling (LM) loss while the LLM remains completely frozen.

## 🛠️ Architecture Details

- **Vision Encoder**: Frozen `openai/clip-vit-base-patch32` (second-to-last layer features).
- **Querying Transformer (Q-Former)**: Custom implementation supporting self-attention, cross-attention to visual features (in alternating layers), and multiple attention masking strategies.
- **Language Model**: Frozen `gpt2` with a learnable linear projection layer.

## 📊 Dataset

- **Real Mode**: Attempts to load the `flickr30k` dataset via Hugging Face `datasets` for realistic image-to-caption learning.
- **Synthetic Mode (Fallback)**: If Hugging Face datasets are unavailable, a dynamic synthetic generator creates geometric shapes (circles, squares, triangles) of varying colors (red, green, blue, yellow, purple, orange) on the fly, demonstrating immediate proof-of-concept convergence.
