# WAPPO in PyTorch
This is a PyTorch implementation of Wasserstein Adversarial Proximal Policy Optimization (WAPPO)[[1]](#references). I tried to make it easy for readers to understand the algorithm. Please let me know if you have any questions.

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

I trained WAPPO and PPO on `cartpole-visual-v1` as below. Following the WAPPO paper, results are averaged over 5 trials. a graph below corresponds to Figure 2 in the paper. Source and target tasks in my experiment are also shown below.

Note that I changed some hyperparameters from the paper. I set 128 for `rollout_length` instead of 256, and 2 for `num_initial_blocks` instead of 1. Please refer to `config/cartpole.yaml` for details.

```bash
python train.py --cuda --wappo --env_id cartpole-visual-v1 --config config/cartpole.yaml --trial 0
```

<img src="https://user-images.githubusercontent.com/37267851/85052188-76f1df00-b1d3-11ea-8ccf-252d69bbb609.png" title="graph" width=500> <img src="https://user-images.githubusercontent.com/37267851/85111758-bd851f00-b24f-11ea-9f7f-e5e9e66e92c1.png" title="graph" width=218>

## References
[[1]](https://arxiv.org/abs/2006.03465) Roy, Josh, and George Konidaris. "Visual Transfer for Reinforcement Learning via Wasserstein Domain Confusion."
