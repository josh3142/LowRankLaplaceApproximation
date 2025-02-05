# Optimal Subspace Inference for the Laplace Approximation of Bayesian Neural Networks
Subspace inference for neural networks assumes that a subspace of their parameter space suffices to produce a reliable uncertainty quantification. In this repository, various subspace models are implemented to evaluate their approximation quality. For more information see the preprint [Optimal Subspace Inference for the Laplace Approximation of Bayesian Neural Networks](https://doi.org/10.48550/arXiv.2502.02345).

## Requirements

### Packages
To run the code install the required packages. 

Create a virtual environment with `python=3.11` and install all the packages with `pip` from `requirementsLaplaceRed.txt`. However, one can also create the predefined environment from `create_env_LaplaceRed.yml`. 
```setup
conda env create -f utils/create_env_LaplaceRed.yml
conda activate LaplaceRed
pip install -r utils/requirementsLaplaceRed.txt
pip install laplace-torch==0.2.1
```

## Introduction with a notebook

A short introduction and an illustrative example of the paper and the repository is presented in `ShowCaseLaplaceRed.ipynb`. This `jupyter` notebook walks through the main steps of the computation for a simple one-dimensional toy example.

## Running the script

Some experiments are precoded in the `bash` scripts that are contained in the folder `scripts`. However, note that they require the weights of the neural networks to run, but these are not provided in this repository. To use these scripts own nets have to be trained and the trained weights have to be put in the corresponding folders `ckpt`. (But for the toy example in `ShowCaseLaplaceRed.ipynb` everything is provided.)

Nevertheless, these scripts illustrate how to get the scripts running with user defined adjustments. Each script runs one experiment for different seeds. Other options can be easily selected. To manage different configurations [hydra](https://hydra.cc/docs/intro/) is used. 
```
bash scripts/redwine.sh
```
computes the covariance matrix of the predictive distribution for different dimensions $s$ and computes the relative error, logarithm of the trace, trace and negative log-likelihood for various subspace models. All these results are stored in a dataframe and plotted.


## Disclaimer
This software was developed at Physikalisch-Technische Bundesanstalt
(PTB). The software is made available "as is" free of cost. PTB assumes
no responsibility whatsoever for its use by other parties, and makes no
guarantees, expressed or implied, about its quality, reliability, safety,
suitability or any other characteristic. In no event will PTB be liable
for any direct, indirect or consequential damage arising in connection

## License
MIT License

Copyright (c) 2023 Datenanalyse und Messunsicherheit, working group 8.42, PTB Berlin

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Reference
If you find this repository useful, please cite our preprint:

[1] **Optimal Subspace Inference for the Laplace Approximation of Bayesian Neural Networks**. Josua Faller and Jörg Martin. *arXiv Preprint* [arXiv:2502.02345](https://doi.org/10.48550/arXiv.2502.02345).
