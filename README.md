# openboost

GPU-accelerated gradient boosting, written in Python.

## Building Blocks

[numba-cuda](https://github.com/NVIDIA/numba-cuda) compiles Python to CUDA kernels:

```python
@cuda.jit
def _histogram_kernel(binned, grad, hess, hist_grad, hist_hess):
    feature_idx = cuda.blockIdx.x
    local_grad = cuda.shared.array(256, dtype=float32)
    # ...
```

## The Problem

[GBDT research](https://github.com/jxucoder/awesome-gradient-boosting-machines) is active:

- [DART](http://proceedings.mlr.press/v38/korlakaivinayak15.pdf) — dropout for trees
- [NGBoost](https://arxiv.org/abs/1910.03225) — probabilistic predictions
- [GAMLSS](https://arxiv.org/abs/2304.03271) — distributional regression
- [GOSS](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree) — gradient-based sampling
- [GBDT-PL](https://www.ijcai.org/Proceedings/2019/0476.pdf) — linear leaves
- [Oblivious Trees](https://arxiv.org/abs/1706.09516) — symmetric splits

OpenBoost lets you implement these in Python.

## Goal

GPU GBDT core in Python. Extend it to build variants.

## Install

```bash
pip install openboost
```

## Status

WIP.

## License

Apache-2.0
