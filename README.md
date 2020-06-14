# WAPPO in PyTorch
This is a PyTorch implementation of WAPPO[[1]](#references). I tried to make it easy for readers to understand the algorithm. Please let me know if you have any questions.

## Setup
If you are using Anaconda, first create the virtual environment.

```bash
conda create -n wappo python=3.8 -y
conda activate wappo
```

You can install Python liblaries using pip.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## References
[[1]](https://arxiv.org/abs/2006.03465) Roy, Josh, and George Konidaris. "Visual Transfer for Reinforcement Learning via Wasserstein Domain Confusion."