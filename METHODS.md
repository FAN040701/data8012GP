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

Running the portable evaluator on two frozen adjacent-slice controls (plain, seed 66 and augmented, seed 11) reproduced all 640 organ Dice scores and predicted voxel counts. [Control evaluation checks](results_summary/reproduction_controls.json)

## Same-seed, channel-order and capacity controls

Use the prepared checkout and the original architecture, control and mirrored-layout results from the README. The commands below create separate fits in `results/review_controls/`. The factorial study repeats seed 11 twice, with four arms per repetition. The capacity study trains 3D width 9 at seeds 11, 22 and 33. The swap study recomputes ordinary and exchanged-channel predictions for both frozen mirrors, plain and augmented, on the 40 replication cases at f3. The 3D comparison uses standard input on the original 40 cases.

~~~bash
python src/review_refits.py factorial
python src/review_refits.py capacity
python src/review_evaluate.py factorial
python src/review_evaluate.py factorial --checkpoint last
python src/review_evaluate.py capacity
python src/review_evaluate.py swap
python src/proposal_latency.py
python src/review_latency.py
python src/review_analysis.py all
~~~

Run the timing scripts on an otherwise idle H100. GPU-grid timing starts with preprocessed intensities in device memory and ends with model-grid probabilities in device memory. Follow the protocol and see the aggregate results in [the review control tables](results_summary/review_controls/). Patient-level records are produced by the evaluation scripts.

The two report figures for model capacity and channel order can also be rebuilt directly from the public aggregate tables.

~~~bash
python src/course_figures.py review
~~~
