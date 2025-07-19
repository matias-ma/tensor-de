# Overview
This repository investigates tensor methods for multivariate density estimation on noisy data. Through this we hope to be able to recover the conformational heterogenity in cryo-EM datasets more reliably.

In this, we build on the tensor train density estimation (TT-DE) code found [here](https://github.com/stat-ml/TTDE/), which uses the algorith put forth in [this paper](https://proceedings.mlr.press/v161/novikov21a/novikov21a.pdf).

# Additions
Aside from adding a few new datasets and providing code to show how to use the model once it is fitted by TT-DE, we have also provided two new loss functions: (i) the $L_2$-error and (ii) the convolved negative log likelihood (NLL).

(i) In Novikov et al., they mention using this error, but they ultimately decided against it in their code. Making use of the TT form of our density estimator we can use this equation:
```math
\mathcal{L}  = \|p-q\|_{\mathrm{L}_2}^2 = \int q(\boldsymbol{x})^2 \, d \boldsymbol{x} - 2 \, \mathbb{E}_{\boldsymbol{x} \sim p} q(\boldsymbol{x}) + \text{const.},
```
where $p$ is the underlying density and $q$ is the density we are trying to estimate, which written in terms of our basis looks like:
```math
q(\boldsymbol{x}) := \big\langle \alpha, \Phi(\boldsymbol{x}) \big\rangle \implies
\int q(\boldsymbol{x})^2 \, d \boldsymbol{x} 
= \int \big\langle \alpha, \Phi(\boldsymbol{x}) \big\rangle^2 \, d \boldsymbol{x}
= \big \langle \alpha, \mathcal{D} \circ \alpha \big \rangle,
```
```math
\text{where } \big[ D^{(k)} \big]_{i,j} := \int \phi_i \big( x_k \big) \phi_j \big( x_k \big) \, dx_k \in \mathbb{R}^{m \times m}, \quad 
\mathcal{D} = \bigotimes_{i=1}^d D^{(i)}
```
Here $\phi$ are our univariate basis functions, $m$ is the number of these basis functions, and $\alpha$ is our coefficient tensor.
Because of the low TT-rank structure of these tensors we can calculate all of this in a time linear in dimension. 

(ii) For cryo-EM data this loss function is the one that should be used. Building on the original NLL loss function, we define the convolved one to be:
```math
\mathcal{L} = - \frac{1}{N} \sum_{i=1}^N \log \Big[ (U * q)\big({z^*}^{(i)} \big) \Big],
```
where $U$ is the PDF of our noise and $\big \{ {z^*}^{(i)}  \in \mathbb{R}^d \big \}_{i=1}^N$ is the collection of our noisy data points, i.e. ${z^*}^{(i)} \overset{\mathrm{iid}}{\sim} (U*p)$. This convolution is approximated using Monte Carlo integration, i.e.:
```math
(U*q)(\boldsymbol{x}) = \mathbb{E}_{\varepsilon \sim U} \big[ q(\boldsymbol{x} - \varepsilon) \big]
\approx \frac{1}{M} \sum_{k=1}^M q \big(\boldsymbol{x}-\varepsilon^{(k)} \big),
\text{ where } \varepsilon^{(k)} \overset{\mathrm{iid}}{\sim} U
```
There are clearly many future improvements that can be made on this. A straightforward one would be to instead convolve the $L_2$ loss function. To address the problems of Monte Carlo integration (i.e. very slow convergence especially in high dimensions), one should investigate, e.g., [this paper](https://www.sciencedirect.com/science/article/pii/S0377042710000750) . To use these algorithms, one needs the additional assumption that the noise kernel is (approximately) low TT rank (this is not a very big assumption).


# Getting started
For a more thorough explanation of this section please refer to the aforementioned repository. 
To get started with this Python package, run the following:
``` 
git clone https://github.com/stat-ml/TTDE.git
cd TTDE
pip install -e .
```
Then add/replace the files from this repository into that `TTDE` directory.

## Requirements 
This code is meant to be run on a GPU but can run on a CPU (it will be very slow).

Be sure to have run this in your conda environment:
```
pip install jax==0.4.8
pip install jaxlib==0.4.7+cuda11.cudnn86 \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install nvidia-cudnn-cu11==8.6.0.163
```

If working on a cluster that does not have internet access, also be sure to run:
```
export WANDB_MODE=offline
```

# Running it
We will show how to run this code on some example cryo-EM data, which will highlight some of the key changes we have made to the existing code. The example cryo-EM dataset we use can be found [here](https://www.dropbox.com/scl/fi/simc0vv9h9bhexbbhdefc/cryoem_test.joblib?rlkey=8rxva5boicaq08ukp3zcxlqz9&st=bxzi1s67&dl=0). Note this algorithm is very computationally expensive (we ran it on a 80Gb A100 GPU). 

## Training the model
```
python -m ttde.train --dataset cryoem --q 2 --m 16 --rank 16 --n-comps 32 --em-steps 10 --noise 0.01 --batch-sz 512 --train-noise 0.01 --lr 0.001 --train-steps 100 --data-dir ~/data_dir --work-dir ~/work_dir --dim 4 --loss-func ConvLLLoss --num-mc 128
```
where you need to have `~/data_dir/cryoEM/cryoem_test.joblib` and `~/work_dir` is where the model will be exported.

## Loading the model & results
Then to load this model and get some visuals (e.g. 2D marginal) run this:
```
python model_eval.py --dataset cryoEM --data-dir ~/data_dir --work-dir ~/work_dir --dim 4 --vis-dims 0,1
```
If you just want the actual density function, load in the model (see `model_eval.py` for how to do this) and then run `p_est = batched_vmap(lambda x: model.apply(params, x, method=model.p), batch_sz)`.
