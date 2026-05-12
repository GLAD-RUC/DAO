<!-- <p align="center"><img src="assets/dao_logo.png" alt="DAO Logo" width="400"></p> -->

# DAO: Siamese Foundation Models for Crystal Structure Prediction

<!-- > Figure 1 (PDF): [assets/Figure 1.pdf](assets/Figure%201.pdf) -->

<p align="center">
  <a href="https://www.nature.com/articles/s41467-026-72362-3"><img src="https://img.shields.io/badge/Paper-Nature%20Comms-green.svg"></a>
  <a href="https://arxiv.org/abs/2503.10471"><img src="https://img.shields.io/badge/arXiv-2503.10471-b31b1b.svg"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
  <a href="https://www.python.org/downloads/release/python-380/"><img src="https://img.shields.io/badge/Python-3.8-blue.svg"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-1.10-red.svg"></a>
</p>

**DAO** (**D**iffusion-based cryst**A**l **O**mni) presents a pair of Siamese foundation models for material science:
*   **DAO-G**: A generative model for **stable crystal structure prediction (CSP)**, capable of generating diverse polymorphic structures.
*   **DAO-P**: A predictive model for **energy and property prediction**, which acts as an energy guider for DAO-G to steer generation towards thermodynamic stability.

Both models are built upon **Crysformer**, an equivariant graph transformer, and are pretrained on **CrysDB** (940K entries) via a novel two-stage pretraining strategy involving unstable structure relaxation.

> 🌐 Website: https://glad-ruc.github.io/DAO/ 

## Table of Contents
- [DAO: Siamese Foundation Models for Crystal Structure Prediction](#dao-siamese-foundation-models-for-crystal-structure-prediction)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Environment variables (optional)](#environment-variables-optional)
  - [Data and Checkpoints](#data-and-checkpoints)
    - [1. Datasets](#1-datasets)
    - [2. Checkpoints](#2-checkpoints)
  - [Quick Start](#quick-start)
    - [1. Crystal Structure Prediction (DAO-G)](#1-crystal-structure-prediction-dao-g)
    - [2. Structure Evaluation](#2-structure-evaluation)
    - [3. Property Prediction (DAO-P)](#3-property-prediction-dao-p)
  - [Superconductors](#superconductors)
      - [**Generate Superconductor Structures:**](#generate-superconductor-structures)
      - [**Critical Temperature (Tc) Prediction:**](#critical-temperature-tc-prediction)
  - [Training](#training)
    - [Finetuning](#finetuning)
    - [Pretraining](#pretraining)
  - [Repository Structure](#repository-structure)
  - [Citation](#citation)
  - [Contact](#contact)
  - [License](#license)

## Installation

We recommend using **Conda** to manage the environment to ensure compatibility with the specific PyTorch and CUDA versions used in our experiments. The repo ships a one-shot `setup.sh` that:

- creates a conda env (default name: `dao`) with Python 3.8.18,
- installs `pytorch==1.10.0` + `cudatoolkit=11.3` via conda,
- installs the rest of the stack pinned in `pyproject.toml` via `pip install -e .`.

```bash
# 1. Run the setup script (creates conda env "dao")
bash setup.sh
conda activate dao

# 2. Verify installation
python -m dao doctor
```

To use a different env name (e.g., `my_dao`):

```bash
ENV_NAME=my_dao bash setup.sh
conda activate my_dao
```

**Note**: `pyproject.toml` pins prebuilt PyG wheels (`torch-scatter`, `torch-sparse`, `torch-cluster`) for Linux x86_64 + Python 3.8 + CUDA 11.3. On other OS/CUDA combinations you will need to adjust those URLs manually.

### Environment variables (optional)

The scripts/CLI rely on a few environment variables to locate the repo and decide where to write outputs. You usually do not need to set these manually (the CLI sets reasonable defaults):

- `PROJECT_ROOT`: repository root (used to find `conf/`).
- `HYDRA_JOBS`: Hydra run directory root (default: `outputs/hydra`).
- `WANDB_DIR`: W&B run directory root (default: `outputs/wandb`).

## Data and Checkpoints

To replicate our results or use the models, you need to download the datasets and pretrained checkpoints.

### 1. Datasets
Download the datasets (MP-20, MPTS-52, SuperCon, etc.) and place them in the `data/` directory:
- [Download Link (Google Drive)](https://drive.google.com/drive/folders/1SOOvLycBhsOKKp3qX_l6SkASjfwf7-7A?usp=drive_link)

or use `gdown` to download directly:
```bash
pip install gdown

gdown https://drive.google.com/drive/folders/1SOOvLycBhsOKKp3qX_l6SkASjfwf7-7A?usp=drive_link --output ./data --folder
```



Expected structure (examples):
```text
DAO/
  data/
    mp_20/
    mpts_52/
    super_conductors/
      supercon3d/
      real_world/
      supercon_rest/
    ...
```

### 2. Checkpoints
Download the pretrained and finetuned checkpoints and place them in the `ckpts/` directory:
- [Download Link (Google Drive)](https://drive.google.com/drive/folders/1msp-D3uWD0fJrwE7qrbRXok1u-FoOAdc?usp=drive_link)

or use `gdown` to download directly:
```bash
pip install gdown

gdown https://drive.google.com/drive/folders/1msp-D3uWD0fJrwE7qrbRXok1u-FoOAdc?usp=drive_link --output ./ckpts --folder
```



Expected structure (examples):
```text
DAO/
  ckpts/
    dao_g/
    dao_p/
    finetune_mp_20/
    finetune_mpts_52/
    finetune_gen_supercon/
    ...
```

Notes:
- The sampling/inference scripts expect the checkpoint *directory* to also contain the saved scalers (e.g. `prop_scaler.pt`).
- If you finetune/train yourself, Hydra will write run artifacts under `outputs/hydra/` by default.

## Quick Start

We provide a CLI via `python -m dao` (and also an installed console script `dao`) for easy interaction with the models.

### 1. Crystal Structure Prediction (DAO-G)

Generate candidate crystal structures using a finetuned DAO-G model. You can optionally enable **energy guidance** using DAO-P to improve stability.

**Basic Generation (Single GPU):**
```bash
python -m dao csp generate \
  --dataset mp_20 \
  --model-path ckpts/finetune_mp_20 \
  --num-evals 1 \
  --num-gpus 1
```

**Multi-GPU Generation:**
To speed up generation, you can shard the workload across multiple GPUs.
```bash
# Example: Running on 4 GPUs
python -m dao csp generate \
  --dataset mp_20 \
  --model-path ckpts/finetune_mp_20 \
  --num-evals 1 \
  --num-gpus 4 \
  --base-gpu 0
```
This will automatically split the test set batches among the available GPUs (range: `[base_gpu, base_gpu + num_gpus)`).


**Energy-Guided Generation:**
Using DAO-P (`--energy-model-path`) to guide the diffusion process towards lower-energy structures.

```bash
python -m dao csp generate \
  --dataset mp_20 \
  --model-path ckpts/finetune_mp_20 \
  --energy-guidance \
  --energy-model-path ckpts/dao_p/last.ckpt \
  --num-evals 1 \
  --num-gpus 1
```

### 2. Structure Evaluation

Evaluate the generated structures against the ground truth using Match Rate (MR) and RMSD metrics.

```bash
python -m dao csp evaluate \
  --dataset mp_20 \
  --root-path ckpts/finetune_mp_20 \
  --num-evals 1 \
  --label 1_all
```
*   `--root-path`: The directory where generation results (`eval_diff_*.pt`) are saved.
*   `--label`: The suffix of the generated file (e.g., `1_all` for `eval_diff_1_all.pt`).

### 3. Property Prediction (DAO-P)

Use DAO-P to predict properties (e.g., energy above hull, band gap) for datasets or generated structures.
The command prints MAE and also saves predicted properties to a `.npy` file for quick inspection.

**Predict on a Dataset:**
```bash
python -m dao prop predict \
  --mode dataset \
  --ori-path data/mp_20/test_ori.pt \
  --model-path ckpts/dao_p_dedup/last.ckpt \
  --pred-energy
```
This saves `pred_<prop>_of_<ori_file>.npy` next to `--ori-path`.

**Predict on Generated Structures:**

Additionally provide the `eval-path` argument to predict on generated structures, turn the `mode` to "generated" and provide the `sample-size` argument to control the number of samples per structure.

```bash
python -m dao prop predict \
  --mode generated \
  --ori-path data/mp_20/test_ori.pt \
  --eval-path ckpts/finetune_mp_20/eval_diff_1_all.pt \
  --model-path ckpts/dao_p_dedup/last.ckpt \
  --pred-energy \
  --sample-size 1
```
This saves `pred_<prop>_of_<eval_file>.npy` next to `--eval-path`.

>**Note:** To predict a property **other than energy**, specify the `--ori-path` argument with the path to the **finetuned** DAO-P model, provide the target property using `--prop`, and omit the `--pred-energy` flag. [Critical temperature prediction](#critical-temperature-tc-prediction) is an example.

## Superconductors

DAO demonstrates significant potential in discovering superconductors.

#### **Generate Superconductor Structures:**  

For ordered superconductors without experimentally resolved structures in SuperCon dataset (`supercon_rest`, 748 entries):

```bash
python -m dao supercon generate \
  --dataset supercon_rest \
  --model-path ckpts/finetune_gen_supercon \
  --energy-guidance \
  --energy-model-path ckpts/dao_p/last.ckpt \
  --num-evals 1 \
  --gpu 0
```

For three real-world superconductors, just replace "supercon_rest" with "supercon_realworld" and set `--num-evals 20` in the command above.


#### **Critical Temperature (Tc) Prediction:**

For three real-world superconductors not in SuperCon3D dataset (`supercon_real`, 3 entries):

```bash
  for fold in {0..4};do
    model_path="ckpts/finetuned_tc_model_fold_${fold}/last.ckpt"
    python -m dao prop predict \
      --mode dataset \
      --ori-path data/super_conductors/real_world/output_ori.pt \
      --model-path $model_path \
      --prop logtc
  done
```

Then average the results of the five folds to get the final predictions.

## Training

### Finetuning

To adapt a pretrained DAO-G model to a specific downstream dataset (e.g., MP-20):

```bash
python -m dao csp finetune \
  --dataset mp_20 \
  --pretrain-ckpt ckpts/dao_g/last.ckpt \
  --epochs 1000 \
  --lr 2e-5 \
  --weight-decay 1e-5 \
  --gpus 1
```

The finetune outputs are written by Hydra to `outputs/hydra/singlerun/<date>/finetune_<dataset>/` by default. Use that directory as `--model-path` for subsequent `csp generate`/`csp evaluate`.

### Pretraining

To reproduce the pretraining of DAO-G (Stage I & II) and DAO-P, please refer to the scripts in `scripts/run/`:

```bash
# Example: Run pretraining
bash scripts/run/run_pretrain.sh
```

## Repository Structure

*   `dao/`: Source code for models and CLI.
    *   `pl_modules/`: PyTorch Lightning modules (Crysformer, Diffusion, etc.).
    *   `pl_data/`: Data loading logic.
*   `conf/`: Hydra configuration files.
*   `scripts/`: Helper scripts used by the CLI.
    *   `scripts/run/`: Generation/finetune launchers.
    *   `scripts/eval/`: Evaluation utilities.
    *   `scripts/infer/`: Property/energy inference utilities.
*   `data/`: Datasets (CSV + cached `*_ori.pt`).
*   `ckpts/`: Pretrained/finetuned checkpoints (plus scalers).
*   `outputs/`: Hydra/W&B run artifacts (created automatically).
*   `assets/`: Figures and diagrams.

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@article{wu2026dao,
  title = {Siamese foundation models for crystal structure prediction},
  issn = {2041-1723},
  doi = {10.1038/s41467-026-72362-3},
  journal = {Nature Communications},
  author = {Wu, Liming and Huang, Wenbing and Jiao, Rui and Huang, Jianxing and Liu, Liwei and Zhou, Yipeng and Sun, Hao and Liu, Yang and Sun, Fuchun and Ren, Yuxiang and Wen, Ji-Rong},
  year = {2026},
}
```


## Contact

If you have any questions, feedback, or collaboration ideas, feel free to reach out: 📧 wlm155@126.com

## License

This project is licensed under the [MIT License](LICENSE).


