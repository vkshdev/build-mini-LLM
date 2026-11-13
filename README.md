<div align="center">

## Building a Generative LLM from Scratch - Decoder-Only Transformer
* Tracked with Weights & Biases (W&B)*

<p align="center">
  <img src="assets/neural-forward-pass.gif" width="55%" />
</p>

<br>

</div>

> It serves as an open, accessible learning reference for students, engineers, and researchers wanting to understand how Large Language Models operates. Architecture refinements, pre-training runs, and loss convergence tracking are executed on Kaggle GPUs with real-time telemetry and reporting in Weights & Biases.

---

## What Is This Project?

Rather than importing an off-the-shelf model or fine-tuning an existing checkpoint with two lines of code, this project is an end-to-end educational journey of building a **generative Large Language Model from zero**.

The main goal is to **demystify the black box of modern autoregressive Transformers**. Every matrix multiplication, attention score, residual connection and training loop is implemented from first principles in PyTorch, designed to teach how deep learning models actually train under real-world constraints.

---

## How It Works

We train the model in two steps, the same way modern foundation models are developed:
```
[ Raw Text Data ] ---> [ Stage 1: Pre-training ] ---> [ Foundational Model ]
                                                            |
                                                            v
[ Conversational Chatbot ] <--- [ Stage 2: SFT / Chat ] <---+
```

---
