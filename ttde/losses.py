from typing import Optional

from flax import struct, linen as lenin
import jax.numpy as jnp
import jax
from jax import vmap
from jax.scipy.special import logsumexp

from ttde.dl_routine import batched_vmap


@struct.dataclass
class LLLoss:
    def __call__(self, model: lenin.Module, params, xs: jnp.ndarray, batch_sz: Optional[int] = None) -> float:
        def log_p(x):
            return model.apply(params, x, method=model.log_p)

        if batch_sz is None:
            log_ps = vmap(log_p)(xs)
        else:
            log_ps = batched_vmap(log_p, batch_sz)(xs)

        return -log_ps.mean()

@struct.dataclass
class L2Loss:
    def __call__(
        self,
        model: lenin.Module,
        params,
        xs: jnp.ndarray,
        batch_sz: Optional[int] = None
    ) -> float:
        # —— DATA TERM: -2 * E_p[q(x)]
        def q(x):
            # model.p(x) should return q_theta(x)
            return model.apply(params, x, method=model.p)

        if batch_sz is None:
            qs = vmap(q)(xs)
        else:
            qs = batched_vmap(q, batch_sz)(xs)

        data_term = -2.0 * qs.mean()

        # —— MODEL TERM:  ∫ q^2(x) dx  via an analytic log‐integral
        # we assume the model has a method log_int_q2() that returns
        #    log ∫ q^2(x) dx
        log_int_q2 = model.apply(params, method=model.log_int_q2)
        model_term = jnp.exp(log_int_q2)

        # —— final loss (we drop any additive constant)
        return model_term + data_term
    

@struct.dataclass
class ConvLLLoss:
    cvrnc: jnp.ndarray
    num_mc: int = 128           # Monte‑Carlo samples per data point
    seed:   int = 0             # RNG seed

    def __call__(self,
                 model: lenin.Module,
                 params,
                 xs:    jnp.ndarray,     # shape (B, d)
                ) -> float:
        """
        Estimates 
            -1/N ∑ᵢ log [ ∫ q_θ(x) ϕσ(x - yᵢ) dx ]
        by sampling num_mc points from N(yᵢ, σ²I) for each yᵢ,
        and using a log‑mean‑exp estimator.
        """
        key = jax.random.PRNGKey(self.seed)
        B, D = xs.shape

        # 1) draw eps ~ N(0,σ²I), shape (B, num_mc, D)
        key, sub = jax.random.split(key)
        eps = jax.random.multivariate_normal(key, jnp.zeros(D), self.cvrnc, shape=(B, self.num_mc))
        # 2) form samples: yᵢ + epsᵢⱼ
        samp = xs[:, None, :] + eps      # (B, num_mc, D)
        flat = samp.reshape((-1, D))     # (B*num_mc, D)

        # 3) eval log_p at all samples
        def log_p(x):
            return model.apply(params, x, method=model.log_p)
        logps_flat = vmap(log_p)(flat)   # (B*num_mc,)

        # 4) reshape and compute log‑mean‑exp
        logps = logps_flat.reshape((B, self.num_mc))
        # log [ (1/K) ∑ⱼ exp(logps_{i,j}) ]
        log_conv = logsumexp(logps, axis=1) - jnp.log(self.num_mc)

        # 5) return -mean log_conv
        return - jnp.mean(log_conv)