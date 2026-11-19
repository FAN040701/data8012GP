# Experimental design

| Comparison | Cohort | What varies | Training |
|---|---|---|---|
| Architecture | Original 40 | Single-section, adjacent-section or volumetric U-Net | 20k updates, three seeds |
| Neighbour information | Replication 40 | Repeated centre or ±3 mm, plain or degradation-augmented training | 40k updates, six seeds |
| Slice layout | Replication 40 | ±3, ±6, ±9, (−3,+9) or (−9,+3) mm around a fixed centre | 40k updates, six seeds |
| Direction sensitivity | Original 40; replication 40 | Frozen development-selected mirrors on original cases; fixed 40k checkpoint on replication cases | Same fitted weights |

All fits share 80 fitting and 20 development patients. The replication and original cohorts are disjoint. Standard inputs use a 3-mm isotropic grid. Degradation factors act on native axial spacing before interpolation. Native-space probabilities are reconstructed before labels and metrics are computed. The primary geometry comparison averages the scores of the two asymmetric layouts and subtracts the symmetric ±6-mm score at equal outer span. The mirror follow-up compares (−3,+9) minus (−9,+3) under augmented versus plain training.

## Code map

- `prepare_data.py`, `models.py`, `experiment.py`, `evaluate.py` implement preprocessing, U-Nets, slab training and native-grid scoring. `proposal_quality.py` computes the maximum of the two directed HD95 values for the original architecture comparison.
- `slice_selection.py`, `train_layout.py`, `eval_slice_selection.py` implement training and inference for the four new layouts. The standalone trainer checks initialization and every sampled crop against the original seed-specific digests. Historical training entry points are retained alongside it.
- `analyze_offsets.py` and `analyze_final_study.py` compute paired contrasts and resample patients and seeds. Positive direction is always near-inferior minus near-superior. Positive augmentation interaction means this contrast increases with augmentation.
- `diagnose_offset_inputs.py`, `final_axial_errors.py` and `diagnose_seed_components.py` quantify neighbour label correspondence, regional FP/FN contributions and the effects of keeping an organ's largest component.

For the regional diagnostic after a full run:

~~~bash
python src/diagnose_offset_inputs.py
python src/final_axial_errors.py first3
python src/final_axial_errors.py new3
python src/final_axial_errors.py summarize
~~~

The fit and evaluation scripts write weights, native predictions and per-organ records under `results/`. The public tables contain summary statistics; acquire the labelled scans from the original data release.

A clean build reproduced all 760 input and label arrays for the 200 cases in the release manifest. Independent seed-11 fits of all three architectures and one asymmetric layout changed mean degraded-input Dice by +0.05 to +1.03 percentage points. The training code uses cuDNN autotuning and mixed precision. [Refit measurements](results_summary/reproduction_models.json)
