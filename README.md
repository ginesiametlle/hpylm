# hpylm

### About this repository

This repository provides a Python implementation of a Hierarchical Bayesian Language Model based on the Pitman-Yor Process (HPYLM). It is largely based on the work of Prof. Dr. Y. W. Teh (see References).

The code here provided is meant rather as a proof of concept than an efficient implementation. You should look [elsewhere](https://github.com/kpu/kenlm/) if you want to estimate language models on a large corpus.

### Dependencies

Reasonably recent versions of `python3`, `nltk`, `numpy` and `pickle` are expected.

### Traning a Hierarchical Pitman-Yor Language Model (HPYLM)

```$ python3 src/train.py [-h] --train TRAIN --out OUT [--order ORDER] [--niter NITER]```

The [train.py](src/train.py) script accepts the following arguments: 

  * `-h` : shows usage information.
  
  * `--train TRAIN` : specifies a training file in text form (system path).
  
  * `--out OUT` : specifies the output model file (system path).
  
  * `--order ORDER` : length of contexts (default: 3).
  
  * `--niter NITER` : number of iterations over the training data (default: 10).
  
### Evaluating a trained model

```$ python3 src/eval.py [-h] --test TEST --model MODEL```

The [eval.py](src/eval.py) script accepts the following arguments: 

  * `-h` : shows usage information.
  
  * `--test TEST` : specifies a test file in text form (system path).
  
  * `--model MODEL` : specifies the model file produced by [train.py](src/train.py) (system path).
  
### References

1. Y. W. Teh. [_A Hierarchical Bayesian Language Model based on Pitman-Yor Processes_](http://www.gatsby.ucl.ac.uk/~ywteh/research/compling/acl2006.pdf). In Proceedings of the 21st International Conference on Computational Linguistics and 44th Annual Meeting of the Association for Computational Linguistics, pages 985–992, 2006.

2. Y. W. Teh. [_A Bayesian Interpretation of Interpolated Kneser-Ney. Technical Report TRA2/06_](https://www.stats.ox.ac.uk/~teh/research/compling/hpylm.pdf), School of Computing, National University of Singapore, 2006.
