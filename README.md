# Error-feedback SGD

We present here the code of the experimental parts of the paper [Error Feedback Fixes SignSGD and other Gradient Compression Schemes](https://arxiv.org/abs/1901.09847).

The implementation is based on [this repository](https://github.com/kuangliu/pytorch-cifar)'s code and uses PyTorch.



## Requirements

The following packages were used for the experiments. Newer versions are also likely to work.

- torchvision==0.2.1
- numpy==1.15.4
- torch==0.4.1
- pandas==0.23.4
- scikit_learn==0.20.3

To install them automatically: `pip install -r requirements.txt`

## Organization

- `notebooks/` contains jupyter notebook files with plotted results and experiments.
- `optimizers/` contains the custom optimizer, namely ErrorFeedbackSGD.
- `models/` contains the deep net architectures. Only VGG and Resnet were experimented.
- `results/` contains the results of the experiments in pickle files.
- `utils/` contains utility functions for saving/loading objects, convex optimization, progress bar...
- `checkpoints/` contains the saved models' checkpoints with all the nets parameters. The folder is empty here as those files are very large.

## Notations

A few notations in the code don't match the notations from the paper. In particular,

- What is called signSGD in the paper is the scaled sign SGD in the code, as the gradients are rescaled by their norm.
- What is called ef-signSGD in the paper is scaled sign SGD with memory in the code. The `memory` parameter can also be used with other compressions besides the sign.

## Usage

- `main.py` can be called from the command line to run a single network training and testing. It can take a variety of optional arguments. Type `python main.py --help` for further details.
- `utils.hyperparameters.py` facilitate the definition of all the hyper-parameters of the experiments.
- `tune_lr.py` allows to tune the learning rate for a network architecture/data set/optimizer configuration.
- `main_experiments.py` contains the experiments presented in the paper, section 6.

# Reference
If you use this code, please cite the following [paper](http://proceedings.mlr.press/v97/karimireddy19a/karimireddy19a-supp.pdf)

    @inproceedings{karimireddy19a,
      title = 	 {Error Feedback Fixes {SignSGD} and other Gradient Compression Schemes},
      author = 	 {Karimireddy, Sai Praneeth and Rebjock, Quentin and Stich, Sebastian U. and Jaggi, Martin},
      booktitle = 	 {ICML - Proceedings of the 36th International Conference on Machine Learning},
      pages = 	 {3252--3261},
      year = 	 {2019}
    }
