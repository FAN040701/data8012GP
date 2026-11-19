"""Paired patient/seed summaries shared by the new exploratory analyses."""
import numpy as np


def interval(values, seed=2026111503):
    values = np.asarray(values, dtype=float)
    assert values.ndim == 1 and len(values) > 0 and np.isfinite(values).all()
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(10000, len(values)))].mean(1)
    lo, hi = np.quantile(means, [.025, .975])
    return float(values.mean()), float(lo), float(hi)


def effect(series, bootstrap_seed=2026111503):
    matrix = series.unstack('seed').sort_index().sort_index(axis=1)
    assert matrix.shape[1] == 3 and not matrix.isna().any().any()
    values = matrix.to_numpy()
    mean, lo, hi = interval(values.mean(1), bootstrap_seed)
    rng = np.random.default_rng(bootstrap_seed)
    patients = rng.integers(len(values), size=(10000, len(values)))
    seeds = rng.integers(3, size=(10000, 3))
    crossed = values[patients[:, :, None], seeds[:, None, :]].mean((1, 2))
    low2, high2 = np.quantile(crossed, [.025, .975])
    seedmeans = values.mean(0)
    return {'mean': mean, 'ci_low': lo, 'ci_high': hi,
            'crossed_ci_low': float(low2), 'crossed_ci_high': float(high2),
            'patients': len(values), 'seeds': 3, 'seed_min': float(seedmeans.min()),
            'seed_max': float(seedmeans.max()), 'seed_sd': float(seedmeans.std(ddof=1))}, matrix


def component_contrasts():
    base = {'full_context_gain': {'context': 1, 'repeat': -1},
            'slope_gain': {'slope': 1, 'repeat': -1},
            'curvature_gain': {'curvature': 1, 'repeat': -1},
            'curvature_given_slope': {'context': 1, 'slope': -1},
            'slope_given_curvature': {'context': 1, 'curvature': -1},
            'component_interaction': {'context': 1, 'slope': -1, 'curvature': -1, 'repeat': 1}}
    contrasts = {}
    for name, weights in base.items():
        contrasts[name+'_without_aug'] = weights
        contrasts[name+'_with_aug'] = {arm+'_aug': weight for arm, weight in weights.items()}
        contrasts['augmentation_x_'+name] = {**{arm: -weight for arm, weight in weights.items()},
                                              **{arm+'_aug': weight for arm, weight in weights.items()}}
    for arm in ['repeat', 'context', 'slope', 'curvature']:
        contrasts['augmentation_'+arm] = {arm+'_aug': 1, arm: -1}
    return contrasts
