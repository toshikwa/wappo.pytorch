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

If you're using other than CUDA 10.2, you need to install PyTorch for the proper version of CUDA. See [instructions](https://pytorch.org/get-started/locally/) for more details.

## Example

### VisualCartpole

I trained WAPPO and PPO on `cartpole-visual-v1` as below. Following the WAPPO paper, results are averaged over 5 seeds. This graph corresponds to Figure 2 in the paper.

```bash
python train.py --cuda --wappo --env_id cartpole-visual-v1 --config config/cartpole.yaml --trial 0
```

<img src="https://user-images.githubusercontent.com/37267851/85052188-76f1df00-b1d3-11ea-8ccf-252d69bbb609.png" title="graph" width=600>

## References
[[1]](https://arxiv.org/abs/2006.03465) Roy, Josh, and George Konidaris. "Visual Transfer for Reinforcement Learning via Wasserstein Domain Confusion."