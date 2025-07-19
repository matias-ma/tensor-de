import joblib
from pathlib import Path

import numpy as np
import jax.numpy as jnp
from ttde.dl_routine import TensorDatasetX



class CRYOEM:
    class Data:
        def __init__(self, x: np.ndarray):
            # store as float32 and number of examples
            self.x = x.astype(np.float32)
            self.N = x.shape[0]

    def __init__(self, root_path: Path, dim: int):
        """
        root_path is expected to be the directory containing data.npy,
        """
        # load, split, normalize
        trn, val, tst, cvrnc = load_data(root_path, dim)
        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(tst)

        self.cvrnc = cvrnc

        # dimensionality of the data
        self.n_dims = self.trn.x.shape[1]


def load_data(root_path: Path, dim: int):
    """
    raw loading and splitting:
      - loads root_path / 'data.npy'
      - last 10% → test
      - next 10% → validation
      - remaining 80% → train
    """
    embedding = joblib.load(root_path / 'cryoem_test.joblib')

    assert dim <= 10, f'dimension {dim} is too high; maximum allowed is 10'

    if dim > 4:
        load_dim = 10
    else:
        load_dim = 4

    zdim_key = f'{load_dim}_noreg'
    points = embedding['zs'][zdim_key]
    covariances_inv = embedding['cov_zs'][zdim_key]

    points = points[:,:dim]
    covariances_inv = covariances_inv[:, :dim, :dim]

    ## Throw away bad points
    covariances = np.linalg.inv(covariances_inv)

    #Throw away the top 1% with larger variances
    variances = np.trace(covariances, axis1=1, axis2=2)
    variances_threshold = np.percentile(variances, 99)

    covariances = covariances[variances < variances_threshold]
    # Average covariance to use as the kernel
    covariances_mean = jnp.array(np.mean(covariances, axis=0))

    ## These are the samples of the NOISY distribution
    points = points[variances < variances_threshold]

    N = points.shape[0]
    n_test = int(0.1 * N)
    n_val  = int(0.1 * (N - n_test))

    data_test  = points[-n_test:]
    data_rem   = points[: -n_test]
    data_val  = data_rem[-n_val:]
    data_train = data_rem[: -n_val]

    return data_train, data_val, data_test, covariances_mean


def load_cryoem_dataset(path: Path, dim: int, to_jax: bool = True):
    """
    Factory function for NAME_TO_DATASET.
    path should be the same Path passed via --data-dir.
    """
    ds = CRYOEM(path / 'cryoem', dim)
    data_train = TensorDatasetX(ds.trn.x)
    data_val   = TensorDatasetX(ds.val.x)
    covariance = ds.cvrnc

    if to_jax:
        data_train = data_train.to_jax()
        data_val   = data_val.to_jax()

    return data_train, data_val, covariance
