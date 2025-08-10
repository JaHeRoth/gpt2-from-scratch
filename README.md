# gpt2-from-scratch
The main purpose of this project was for me to better grasp all the details in the end-to-end process of training a system such as GPT-2.
However, I hope this project can also be of use to others, if only by providing an example implementation of this end-to-end process.

## Contents
- `train.py` contains the entry point for training this GPT-2 like model on a subset of FineWeb-Edu, across all GPUs of the machine.
- `inference.ipynb` contains code for loading and streaming from the model trained in `train.py` (or the model I have pre-trained for you).
- `outputs/tokenizer` and `outputs/model.pt` contain the result of my training run on a 1024-length sequence packed encoding of the _sample-10BT_ subset of _FineWeb-Edu_, for just below 30k updates (1.5 epochs) of batch size 512 each.
- `utilities` contains our model and optimizer architectures, as well as all the helper code we use for loading and preprocessing data and performing our training runs.

## Prerequisites
- [pixi](https://pixi.sh/latest/). Installation guide: `curl -fsSL https://pixi.sh/install.sh | sh`
- [git lfs](https://git-lfs.com/). Installation guide:
  - Linux: `sudo apt-get install git-lfs ; git lfs install`
  - macOS: `brew install git-lfs ; git lfs install`

## Setup
```bash
git clone https://github.com/JaHeRoth/gpt2-from-scratch.git
cd gpt2-from-scratch
pixi install
pixi shell
```

## Learnings
- Exact network architectures of GPT and GPT-2 (down to the level of every individual nn.Parameter)
- Inner workings of AdamW optimizer
- LLM sampling tricks (and implementing temperature and nucleus sampling)
- Sequence packing
- HuggingFace tokenizers, datasets and Hub
- The PyTorch stack
- GPU tricks (kernel fusion through torch.compile, optimizing tensor sizes)
- Mixed-precision training
- Distributed training with DistributedDataParallel

## Future directions
- Optimizations of newer models: MoE, MLA, ROPE, YaRN, SWiGLU, QKNorm
- SFT, to turn into chatbot
- Reinforcement learning and allowing "thinking" stage (although I doubt model is smart enough to benefit from chain of thought)

## Sources
- [Attention Is All You Need (Transformer paper)](https://arxiv.org/abs/1706.03762)
- [Generating Wikipedia by Summarizing Long Sequences (Decoder-only transformer paper)](https://arxiv.org/abs/1801.10198)
- [Improving Language Understanding by Generative Pre-Training (GPT paper)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [Language Models are Unsupervised Multitask Learners (GPT-2 paper)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Gaussian Error Linear Units (GELU paper)](https://arxiv.org/abs/1606.08415)
- [Decoupled Weight Decay Regularization (AdamW paper)](https://arxiv.org/abs/1711.05101)
- [Using the Output Embedding to Improve Language Models (Weight tying paper)](https://arxiv.org/abs/1608.05859v3)
- [The Curious Case of Neural Text Degeneration (Nucleus sampling paper)](https://arxiv.org/abs/1904.09751v2)
- [Andrej Karpathy’s Lectures and Tutorials](https://www.youtube.com/@AndrejKarpathy)
