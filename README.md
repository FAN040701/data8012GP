# Spatial context in abdominal CT segmentation

A DATA8012 course project using AMOS CT to segment the liver, spleen and both kidneys. We compare 2D, three-slice 2.5D and 3D U-Nets, then hold the planar architecture fixed to study slice distance, direction and training augmentation. Offsets are in millimetres on a common 3-mm model grid.

## Results

On the original 40 scans, standard-input mean Dice was **91.96%** for 2D, **91.93%** for adjacent-slice 2.5D and **93.64%** for 3D. On a separate 40 scans under threefold axial downsampling, degradation augmentation increased the gain from adjacent sections by **0.94 Dice points** (95% interval **0.31 to 1.59**). The development-selected asymmetric layout (−3, 0, +9 mm) gained **0.08 points** over immediate adjacency across six seeds (−0.26 to +0.43). The mirror-direction interaction was **+0.37 points** in the three additional seeds (−0.32 to +1.07). [Aggregate tables](results_summary/)

## Reproduce

Use Linux, Python 3.11 and a CUDA-capable GPU with bfloat16 support. The experiments used PyTorch 2.5.1, CUDA 12.4 and NVIDIA H100 GPUs. From the repository root:

~~~bash
python3.11 -m venv .venv
source environment/activate.sh
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
~~~

Download and extract the [AMOS22 release](https://zenodo.org/records/7262581). Pass the directory containing the extracted `amos22/` folder. Preparation verifies the selected image and label hashes and creates patient splits, normalised volumes and degradation conditions.

~~~bash
python src/course_workflow.py prepare --source /path/to/extracted-directory
python src/course_workflow.py baseline --kind 2p5d --seed 11
~~~

The second command trains and evaluates one baseline. Use `2d`, `2p5d` and `3d` with seeds 11, 22 and 33 for the architecture comparison, then run `python src/course_workflow.py architecture` to recompute Dice and directional HD95 from the nine sets of native predictions. For the complete planar study, run the following on a prepared checkout:

~~~bash
for seed in 11 22 33 44 55 66; do
    python src/course_workflow.py controls --seed "$seed"
    python src/course_workflow.py offsets --seed "$seed"
    python src/course_workflow.py sensitivity --seed "$seed"
done
python src/course_workflow.py analyze
~~~

The four new offset policies each use the same 40,000-update schedule with and without degradation augmentation. The sensitivity stage evaluates mirrored layouts on the original cohort and at the final checkpoint. Main paired contrasts are written to `results/final_study/contrasts.csv`; group labels distinguish the initial three seeds, additional three and all six. [Experimental design and script map](METHODS.md)
