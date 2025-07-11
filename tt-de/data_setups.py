from pathlib import Path
from dataclasses import dataclass, field

from ttde.datasets.bsds300 import load_bsds300_dataset
from ttde.datasets.gas import load_gas_dataset
from ttde.datasets.hepmass import load_hepmass_dataset
from ttde.datasets.miniboone import load_miniboone_dataset
from ttde.datasets.power import load_power_dataset
from ttde.datasets.ginzburglandau import load_ginzburglandau_dataset
from ttde.datasets.glnoisy import load_glnoisy_dataset
from ttde.datasets.umixture import load_umixture_dataset

from ttde.score.experiment_setups.base import Base

@dataclass
class WithPath(Base):
    path: Path = field(repr=False)


@dataclass
class Power(WithPath):
    pass


@dataclass
class Gas(WithPath):
    pass


@dataclass
class Hepmass(WithPath):
    pass


@dataclass
class Miniboone(WithPath):
    pass


@dataclass
class BSDS300(WithPath):
    pass


@dataclass
class GinzburgLandau(WithPath):
    pass


@dataclass
class GLnoisy(WithPath):
    pass


@dataclass
class Umixture(WithPath):
    pass


TABLE_DATASETS = [Power, Gas, Hepmass, Miniboone, BSDS300, GinzburgLandau, GLnoisy, Umixture]
NAME_TO_DATASET = {dataset.__name__: dataset for dataset in TABLE_DATASETS}


def load_dataset(dataset):
    if isinstance(dataset, Power):
        return load_power_dataset(dataset.path)
    if isinstance(dataset, Gas):
        return load_gas_dataset(dataset.path)
    elif isinstance(dataset, Hepmass):
        return load_hepmass_dataset(dataset.path)
    elif isinstance(dataset, Miniboone):
        return load_miniboone_dataset(dataset.path)
    elif isinstance(dataset, BSDS300):
        return load_bsds300_dataset(dataset.path)
    elif isinstance(dataset, GinzburgLandau):
        return load_ginzburglandau_dataset(dataset.path)
    elif isinstance(dataset, GLnoisy):
        return load_glnoisy_dataset(dataset.path)
    elif isinstance(dataset, Umixture):
        return load_umixture_dataset(dataset.path)
    else:
        assert False, f'Unknown dataset {dataset}'
