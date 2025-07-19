import click
import optax

from ttde.all_imports import *
from ttde.score.all_imports import *
from ttde.score.experiment_setups.data_setups import NAME_TO_DATASET

jnp.set_printoptions(precision=4, linewidth=float('inf'))
np.set_printoptions(precision=4, linewidth=float('inf'))

from jax.config import config
config.update("jax_enable_x64", True)


@click.command()
@click.option(
    '--dataset',
    type=click.Choice(NAME_TO_DATASET.keys(), case_sensitive=False),
    required=True,
    help=f'Name of the dataset. Choose one of {", ".join(NAME_TO_DATASET.keys())}'
)
@click.option('--q', type=int, required=True, help='degree of splines')
@click.option('--m', type=int, required=True, help='number of basis functions')
@click.option('--rank', type=int, required=True, help='rank of tensor-train decomposition')
@click.option('--n-comps', type=int, required=True, help='number of components in the mixture')
@click.option('--em-steps', type=int, required=True, help='number of EM steps for model initializaion')
@click.option('--noise', type=float, required=True, help='magnitude of Gaussian noise for model initializatoin')
@click.option('--batch-sz', type=int, required=True, help='batch size')
@click.option('--train-noise', type=float, required=True, help='Gaussian noise to add to samples during training')
@click.option('--lr', type=float, required=True, help='learning rate for Adam optimizer')
@click.option('--train-steps', type=int, required=True, help='number of train steps')
@click.option('--data-dir', type=Path, required=True, help='directory with MAF datasets')
@click.option('--work-dir', type=Path, required=True, help='directory where to store checkpoints')
@click.option('--loss-func', type=str, required=True, help='loss function, options LLLoss, ConvLLLoss, L2Loss')
@click.option('--num-mc', type=int, default=128, show_default=True, help='number of MC samples for ConvLLLoss')
@click.option('--dim', type=int, default=4, show_default=True, help='dimension for cryo-EM data, int in [2,10]')

def main(
    dataset: str,
    q: int,
    m: int,
    rank: int,
    n_comps: int,
    em_steps: int,
    noise: float,
    batch_sz: int,
    train_noise: float,
    lr: float,
    train_steps: int,
    data_dir: Path,
    work_dir: Path,
    loss_func: str,
    num_mc: int,
    dim: int
):
    DATASET = NAME_TO_DATASET[dataset](data_dir)

    if dataset != 'cryoEM':
        data_train, data_val = data_setups.load_dataset(DATASET, dim)
    else:
        data_train, data_val, covariance = data_setups.load_dataset(DATASET, dim)
    
    print(data_train.X.shape, data_val.X.shape)

    if loss_func == 'LLLoss':
        loss_func = LLLoss()
    elif loss_func == 'ConvLLLoss':
        loss_func = ConvLLLoss(num_mc=num_mc, cvrnc=covariance)
    elif loss_func == 'L2Loss':
        loss_func = L2Loss()
    else:
        raise ValueError(f'Invalid loss function: {loss_func}')

    MODEL = model_setups.PAsTTSqrOpt(q=q, m=m, rank=rank, n_comps=n_comps)
    INIT = init_setups.CanonicalRankK(em_steps=em_steps, noise=noise)
    TRAINER = trainer_setups.Trainer(batch_sz=batch_sz, lr=lr, noise=train_noise)

    WORK_DIR = Path(work_dir / f'{DATASET}/{MODEL}/{INIT}/{TRAINER}')

    model = MODEL.create(KEY_0, data_train.X)
    init_params = INIT(model, KEY_0, data_train.X)

    optimizer = riemannian_optimizer.FlaxWrapper.create(optax.adam(learning_rate=TRAINER.lr), target=init_params)

    trainer = Trainer(
        model=model,
        optim_state=optimizer,
        loss_fn=loss_func,
        post_processing=MODEL.postprocessing,
        data_train=data_train,
        data_val=data_val,
        batch_sz=TRAINER.batch_sz,
        noise=TRAINER.noise,
        work_dir=utils.suffix_with_date(WORK_DIR),
    )

    trainer.fit(KEY_0, train_steps)


if __name__ == '__main__':
    main()
