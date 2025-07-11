import re
from pathlib import Path
import click
import optax
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from ttde.all_imports import *
from ttde.score.all_imports import *
from ttde.score.experiment_setups.data_setups import NAME_TO_DATASET
from ttde.dl_routine import batched_vmap


def plot_2d_marginal(
    data,
    model,
    params,
    dims=(0, 1),
    n_points=200,
    n_rest=1000,
    rest_limits=(-2.5, 2.5),
    batch_size=4096,
    key_seed=0,
    dataset_name='',
    **metrics
):
    """
    Approximate and plot the marginal p(x_i, x_j) for a D-dimensional density.
    """
    D = metrics.get('D')
    i, j = dims

    # build the 2-D grid
    xmin, xmax = min(data[:,i]), max(data[:,i])
    ymin, ymax = min(data[:,j]), max(data[:,j])
    x_vals = jnp.linspace(xmin, xmax, n_points)
    y_vals = jnp.linspace(ymin, ymax, n_points)
    XX, YY = jnp.meshgrid(x_vals, y_vals, indexing='xy')
    grid_xy = jnp.stack([XX.ravel(), YY.ravel()], axis=-1)

    # sample the other D-2 dims
    key = jax.random.PRNGKey(key_seed)
    rest_dim = D - 2
    c, d = rest_limits
    rest_samples = jax.random.uniform(key, (n_rest, rest_dim), minval=c, maxval=d)

    # create full points
    def make_full_points(xy):
        out = jnp.zeros((n_rest, D))
        idxs = [k for k in range(D) if k not in dims]
        out = out.at[:, idxs].set(rest_samples)
        out = out.at[:, i].set(xy[0])
        out = out.at[:, j].set(xy[1])
        return out

    batched_make = jax.vmap(make_full_points, in_axes=(0,))
    all_points = batched_make(grid_xy)
    flat_points = all_points.reshape(-1, D)

    # eval log-prob
    logp_fn = lambda x: model.apply(params, x, method=model.log_p)
    batched_logp = batched_vmap(logp_fn, batch_size)
    logps = batched_logp(flat_points)
    ps = jnp.exp(logps)

    # average over rest dims
    marginals = jnp.mean(ps.reshape(n_points**2, n_rest), axis=1)
    Z = jnp.array(marginals).reshape(n_points, n_points)

    # plot
    plt.figure(figsize=(6, 5))
    plt.imshow(Z, extent=(xmin, xmax, ymin, ymax), origin='lower', aspect='auto')
    plt.colorbar(label=f'Approx. p(x_{i}, x_{j})')
    plt.xlabel(f'x_{i}')
    plt.ylabel(f'x_{j}')
    title = (f"{dataset_name} marginal dims {i},{j} | " +
             ", ".join([f"{k}={v}" for k, v in metrics.items() if k != 'D']))
    plt.title(title, fontsize=4)
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_results.png", dpi=300)
    click.echo(f"Saved marginal plot to {dataset_name}_results.png")

    plt.clf()
    plt.hist2d(data[:,i], data[:,j], bins=100)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.colorbar(label='Counts')
    plt.xlabel(f'x_{i}')
    plt.ylabel(f'x_{j}')
    plt.title(f'Histogram of dims {i} and {j}')
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_hist.png', dpi=300)
    click.echo(f'Saved histogram to {dataset_name}_hist.png')


@click.command()
@click.option("--dataset", required=True, help="Name of the dataset (must match a key in NAME_TO_DATASET)")
@click.option("--data-dir", "-d", required=True, type=Path, help="Path to the data directory")
@click.option("--work-dir", "-w", required=True, type=Path, help="Base work directory containing experiment runs")
def main(dataset: str, data_dir: Path, work_dir: Path):
    # Load dataset
    DATASET_CLS = NAME_TO_DATASET[dataset]
    if DATASET_CLS is None:
        raise click.ClickException(f"Unknown dataset '{dataset}'. Available: {list(NAME_TO_DATASET)}")
    DATASET = DATASET_CLS(data_dir)
    data_train, data_val = data_setups.load_dataset(DATASET)
    click.echo(f'Train shape: {data_train.X.shape}, Val shape: {data_val.X.shape}')

    # Locate checkpoint
    base = work_dir / dataset
    if not base.exists():
        raise click.ClickException(f"Work path {base} does not exist")
    cpts_dirs = list(base.rglob('cpts'))
    if not cpts_dirs:
        raise click.ClickException(f"No 'cpts' folder found under {base}")
    ckpt_dir = max(cpts_dirs, key=lambda p: len(str(p)))
    click.echo(f"Using checkpoint directory: {ckpt_dir}")

    # Extract params
    parts = [p.name for p in ckpt_dir.parents]
    patterns = {
        'q': r'q=(?P<q>\d+)',
        'm': r'm=(?P<m>\d+)',
        'rank': r'rank=(?P<rank>\d+)',
        'n_comps': r'n_comps=(?P<n_comps>\d+)',
        'em_steps': r'em_steps=(?P<em_steps>\d+)',
        'noise': r'noise=(?P<noise>[0-9.]+)',
        'batch_sz': r'batch_sz=(?P<batch_sz>\d+)',
        'train_noise': r'Trainer_batch_sz=[0-9]+_noise=(?P<train_noise>[0-9.]+)',
        'lr': r'lr=(?P<lr>[0-9.eE-]+)'
    }
    params = {}
    for part in parts:
        for key, pat in patterns.items():
            m = re.search(pat, part)
            if m:
                params[key] = float(m.group(key)) if '.' in m.group(key) or 'e' in m.group(key).lower() else int(m.group(key))
    # Destructure
    q = params['q']; m = params['m']; rank = params['rank']; n_comps = params['n_comps']
    em_steps = params['em_steps']; noise = params['noise']
    batch_sz = params['batch_sz']; train_noise = params['train_noise']; lr = params['lr']

    click.echo(
        f"Extracted: q={q}, m={m}, rank={rank}, n_comps={n_comps}, em_steps={em_steps}, "
        f"batch_sz={batch_sz}, noise={noise}, train_noise={train_noise}, lr={lr}"
    )

    # Build
    MODEL = model_setups.PAsTTSqrOpt(q=q, m=m, rank=rank, n_comps=n_comps)
    INIT = init_setups.CanonicalRankK(em_steps=em_steps, noise=noise)
    TRAINER = trainer_setups.Trainer(batch_sz=batch_sz, lr=lr, noise=train_noise)

    key = jax.random.PRNGKey(0)
    model = MODEL.create(key, data_train.X)
    init_params = INIT(model, key, data_train.X)

    optimizer = riemannian_optimizer.FlaxWrapper.create(optax.adam(learning_rate=TRAINER.lr), target=init_params)
    restored_state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=optimizer, parallel=False)
    restored_state = jax.tree_map(jnp.array, restored_state)

    params = restored_state.target
    log_norm = model.apply(params, method=model.tt_log_sqr_norm)
    click.echo(f'Log square norm: {log_norm:.4e}')

    # Plot a marginal (dims and sampling params can be adjusted)
    all_data = jnp.vstack([data_train.X, data_val.X])
    plot_2d_marginal(
        all_data,
        model,
        params,
        dims=(0, 1),
        n_points=200,
        n_rest=2000,
        rest_limits=(-3, 3),
        batch_size=4096,
        key_seed=0,
        dataset_name=dataset,
        D=data_train.X.shape[1],
        q=q, m=m, rank=rank, n_comps=n_comps,
        em_steps=em_steps, batch_sz=batch_sz,
        noise=noise, train_noise=train_noise, lr=lr
    )

if __name__ == '__main__':
    main()
