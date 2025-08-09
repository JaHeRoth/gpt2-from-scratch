- Did this project for learning, but hope others can also benefit from seeing a simple implementation.
- This repository contains a script for multi-GPU training GPT2 on a subset of FineWeb-Edu and a notebook for loading and streaming from the resulting model.
- How to run this code (Linux and MacOS are supported):
  - Install pixi if you haven't already:
    - On Linux: `curl -fsSL https://pixi.sh/install.sh | sh`
    - On MacOS: TODO
  - Install git lfs if you haven't already:
    - On Linux: `sudo apt-get install git-lfs ; git lfs install`
    - On MacOS: TODO

Then restart your terminal, clone this repo, navigate into it and activate the pixi environment:
```bash
git clone https://github.com/JaHeRoth/reimplementing.git
cd reimplementing
pixi install
pixi shell
```

Finally, run `run.py` like you would any other python script:
```bash
python run.py
```
- Learnings:
  - Exact network architectures of GPT and GPT2 (down to the level of every individual nn.Parameter)
  - Inner workings of AdamW optimizer
  - LLM sampling tricks (and implementing temperature and nucleus sampling)
  - Sequence packing
  - Using HuggingFace tokenizers and datasets
  - The PyTorch stack
  - GPU tricks (kernel fusion through torch.compile, optimizing tensor sizes)
  - Mixed-precision training
  - Distributed training with DistributedDataParallel
- Future directions:
  - Optimizations of newer models: MoE, MLA, ROPE, YaRN, SWiGLU, QKNorm?
  - SFT, to turn into chatbot
  - Reinforcement learning and allowing "thinking" stage (although I doubt model is smart enough to benefit from chain of thought)
- Sources: Attention is all you need, GPT paper, GPT2 paper, GELU paper, AdamW paper, weight tying paper (1608.05859v3), nucleus sampling paper (1904.09751v2), Karpathy's lectures and tutorials, etc?