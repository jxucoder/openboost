# openboost

[![Unit Tests](https://github.com/jxucoder/openboost/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/jxucoder/openboost/actions/workflows/unit-tests.yml)
[![PyPI version](https://img.shields.io/pypi/v/openboost.svg)](https://pypi.org/project/openboost/)
[![Python](https://img.shields.io/pypi/pyversions/openboost.svg)](https://pypi.org/project/openboost/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

GPU-accelerated gradient boosting, written in Python.

## Building Blocks

[numba-cuda](https://github.com/NVIDIA/numba-cuda) compiles Python to CUDA kernels.

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
