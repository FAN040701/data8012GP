"""Descriptive training/development checks; never select checkpoints on test data."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/mechanism/training_diagnostics'
ARMS = ['repeat', 'context', 'slope', 'curvature']
LABELS = ['Repeated centre', 'Full neighbours', 'Slope only', 'Second difference only']
COLORS = ['#666666', '#0072B2', '#009E73', '#D55E00']


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    validation, losses, summaries, sources = [], [], [], {}
    for base in ARMS:
        for suffix in ['', '_aug']:
            arm = base+suffix
            root = ROOT/('results/mechanism/runs' if base in ['slope', 'curvature'] else 'results/continuation/b40000')
            for seed in [11, 22, 33]:
                run = root/f'{arm}_seed{seed}'
                complete = json.loads((run/'complete.json').read_text())
                trace = pd.read_json(run/'train.jsonl', lines=True)
                vals = pd.read_json(run/'validation.jsonl', lines=True).set_index('step')
                assert complete['training_steps'] == 40000 and trace.step.max() == 40000
                assert len(vals) == 20 and np.isfinite(trace[['loss', 'grad_norm']]).all().all()
                assert complete['best_step'] == vals.macro_dice.idxmax()
                for file in ['train.jsonl', 'validation.jsonl', 'complete.json']:
                    p = run/file; sources[str(p.relative_to(ROOT))] = sha(p)
                for step, row in vals.iterrows():
                    validation.append({'arm': arm, 'seed': seed, 'step': step, 'dice': row.macro_dice,
                                       'selected': step == complete['best_step']})
                trace['window_end'] = ((trace.step-1)//1000+1)*1000
                for step, rows in trace.groupby('window_end'):
                    losses.append({'arm': arm, 'seed': seed, 'step': step,
                                   'logged_loss_mean': rows.loss.mean(), 'logged_batches': len(rows)})
                summaries.append({'arm': arm, 'seed': seed, 'best_step': complete['best_step'],
                                  'best_development_dice': complete['best_validation_macro_dice'],
                                  'final_development_dice': vals.loc[40000, 'macro_dice'],
                                  'development_gain_30k_to_40k': vals.loc[40000, 'macro_dice']-vals.loc[30000, 'macro_dice'],
                                  'maximum_logged_gradient_norm': trace.grad_norm.max(),
                                  'all_logged_loss_and_gradients_finite': True})
    validation, losses, summaries = map(pd.DataFrame, [validation, losses, summaries])
    validation.to_csv(OUT/'development_curves.csv', index=False)
    losses.to_csv(OUT/'logged_loss_windows.csv', index=False)
    summaries.to_csv(OUT/'run_summary.csv', index=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for col, suffix in enumerate(['', '_aug']):
        for base, label, color in zip(ARMS, LABELS, COLORS):
            arm = base+suffix
            for row, table, field, scale in [(0, validation, 'dice', 100), (1, losses, 'logged_loss_mean', 1)]:
                values = table[table.arm == arm].pivot(index='step', columns='seed', values=field)*scale
                assert list(values.columns) == [11, 22, 33]
                x = values.index.to_numpy()/1000
                axes[row, col].plot(x, values.mean(axis=1), color=color, label=label)
                axes[row, col].fill_between(x, values.min(axis=1), values.max(axis=1), color=color, alpha=.12)
        axes[0, col].set_title('With degradation augmentation' if suffix else 'Without degradation augmentation')
        axes[0, col].set_ylabel('Development macro Dice (%)')
        axes[1, col].set_ylabel('Mean of logged training losses')
        for ax in axes[:, col]:
            ax.set_xlabel('Training updates (thousands)'); ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Matched 40k training schedules | mean and range across three seeds\n'
                 'Development scores use the 3 mm model grid; loss windows include logged batches only', fontsize=12)
    artifacts = []
    for ext in ['png', 'svg']:
        p = OUT/f'component_training_trajectories.{ext}'
        fig.savefig(p, dpi=180, bbox_inches='tight', pad_inches=.15)
        artifacts.append({'path': str(p.relative_to(ROOT)), 'sha256': sha(p)})
    plt.close(fig)
    result = {'completed_utc': datetime.now(timezone.utc).isoformat(), 'runs': len(summaries),
              'training_log_rows_all_finite': True,
              'best_checkpoints_at_40k': int((summaries.best_step == 40000).sum()),
              'interpretation': 'Development trajectories can reveal optimization problems but do not establish convergence or identify information content; no test-driven checkpoint selection.',
              'sources_sha256': sources, 'source_sha256': sha(Path(__file__)),
              'artifacts': artifacts, 'report_generated': False}
    (OUT/'complete.json').write_text(json.dumps(result, indent=2))
    print(summaries.groupby('arm')[['best_development_dice', 'development_gain_30k_to_40k']].mean().to_string(), flush=True)


if __name__ == '__main__': main()
