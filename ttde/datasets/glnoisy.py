from pathlib import Path

import numpy as np
from ttde.dl_routine import TensorDatasetX


class GLNOISY:
    class Data:
        def __init__(self, x: np.ndarray):
            # store as float32 and number of examples
            self.x = x.astype(np.float32)
            self.N = x.shape[0]

    def __init__(self, root_path: Path):
        """
        root_path is expected to be the directory containing data.npy,
        """
        # load, split, normalize
        trn, val, tst = load_data_normalised(root_path)
        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(tst)

        # dimensionality of the data
        self.n_dims = self.trn.x.shape[1]


def load_data(root_path: Path):
    """
    raw loading and splitting:
      - loads root_path / 'data.npy'
      - last 10% → test
      - next 10% → validation
      - remaining 80% → train
    """
    data = np.load(root_path / 'data.npy')
    N = data.shape[0]
    n_test = int(0.1 * N)
    n_val  = int(0.1 * (N - n_test))

    data_test  = data[-n_test:]
    data_rem   = data[: -n_test]
    data_val   = data_rem[-n_val:]
    data_train = data_rem[: -n_val]

    return data_train, data_val, data_test


def load_data_normalised(root_path: Path):
    """
    like load_data, but also z-score normalises
    (mean/std computed over train+val), then splits again.
    """
    trn, val, tst = load_data(root_path)
    # fit on train+val
    stacked = np.vstack((trn, val))
    mu, sigma = stacked.mean(axis=0), stacked.std(axis=0)

    trn_n = (trn - mu) / sigma
    val_n = (val - mu) / sigma
    tst_n = (tst - mu) / sigma

    return trn_n, val_n, tst_n


def load_glnoisy_dataset(path: Path, to_jax: bool = True):
    """
    Factory function for NAME_TO_DATASET.
    path should be the same Path passed via --data-dir.
    """
    ds = GLNOISY(path / 'glnoisy')
    data_train = TensorDatasetX(ds.trn.x)
    data_val   = TensorDatasetX(ds.val.x)

    if to_jax:
        data_train = data_train.to_jax()
        data_val   = data_val.to_jax()

    return data_train, data_val
