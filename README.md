# Overview
This repository investigates tensor methods for multivariate density estimation on noisy data. Through this we hope to be able to recover the conformational heterogenity in cryo-EM datasets more reliably.

In this, we build on the tensor train density estimation (TT-DE) code found [here](https://github.com/stat-ml/TTDE/), which uses the algorith put forth in [this paper](https://proceedings.mlr.press/v161/novikov21a/novikov21a.pdf).

# Additions
Aside from adding a few new datasets and providing code to show how to use the model once it is fitted by TT-DE, we have also provided two new loss functions: (i) the $L_2$-error and (ii) the convolved negative log likelihood (NLL).

(i) In Novikov et al., they mention using this error, but they ultimately decided against it in their code. Making use of the TT form of our density estimator we can use this equation: \
$$
\mathcal{L}  = \|p-q\|_{L_2}^2 
= \int q(\boldsymbol{x})^2 \, d \boldsymbol{x} - 2 \,\mathbb{E}_{\boldsymbol{x} \sim p} q(\boldsymbol{x}) + \text{const.},
$$

# Getting started
For a more thorough explanation of this section please refer to the aforementioned repository. 
To get started with this Python package, run the following:
``` 
git clone git clone https://github.com/stat-ml/TTDE.git
cd TTDE
pip install -e .
```
Then add/replace the files from this repository into that `TTDE` directory.

# Running it
We will show how to run this code on some example cryo-EM data, which will highlight some of the key changes we have made to the existing code. The example cryo-EM dataset we use can be found [here](https://www.dropbox.com/scl/fi/simc0vv9h9bhexbbhdefc/cryoem_test.joblib?rlkey=8rxva5boicaq08ukp3zcxlqz9&st=bxzi1s67&dl=0).

## Training the model

## Loading the model & results
