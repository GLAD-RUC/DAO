# README

This repository provides the code and resources for running pretraining, finetuning, and evaluation of the DAO-G and DAO-P models. Below is a detailed guide on how to set up the environment, download the data, use the pretrained models, and perform various tasks.

---

## 1. Environment Setup

The project requires specific dependencies to run. You can set up the environment using the provided `environment.yml` file.

To create the environment, run the following commands:

```bash
conda env create -f environment.yml
conda activate dao
```



Rename the `.env.template` file into `.env` and specify the following variables (from DiffCSP).

```
PROJECT_ROOT: the absolute path of this repo
HYDRA_JOBS: the absolute path to save hydra outputs
WABDB_DIR: the absolute path to save wabdb outputs
```



---

## 2. Data Download

### 2.1 Downstream Task Data

The data for downstream tasks is included in the code repository. No additional steps are required for these datasets.

### 2.2 Pretraining Dataset CrysDB （Optional）

**Note: This step can be skipped, as we have already provided the pretrained model parameters. There is no need to pretrain the model separately.**



The pretraining dataset is hosted on Google Drive. You can download it using the link provided below:

[Download pretraining Dataset](https://drive.google.com/drive/folders/1_oM_Ow3w_fQHlNOR4yD87C6oGRRUUImt?usp=drive_link)

After downloading, place the dataset in the following directory:

```bash
/path/to/DAO/data/scaling_data/full
```




---

## 3. Model Weights

Pretrained model checkpoints for DAO-G and DAO-P are provided in `ckpts` folder for your convenience. Additionally, finetuned DAO-G on `mp_20`, `mpts_52`, and `supercon` are also available for direct generation inference.



**Therefore, if you want to quickly get started and verify the performance of our model, you can directly use the provided model weights.**


---



## 4. Pretraining (Optional)

If you wish to pretrain the model by yourself, you can do so by running the following script:

```bash
bash run_pretrain.sh
```

This script will execute the pretraining process using the dataset specified in the configuration. Note that this step is optional, as pretrained weights are already provided.



The pretrained models can be finetuned on two downstream tasks: crystal structure prediction (CSP) and property prediction.

To finetune the model, follow these steps:

1. Ensure the pretrained weights are placed in the correct directory (see Section 3).
2. Run the finetuning script for the desired task, which will be introduced in the following.



---

## 5. Crystal Structure Prediction (CSP)

The CSP task involves three main steps: finetuning the pretrained DAO-G, generating structures (sampling), and evaluating the generated structures using Match Rate (MR) and Root Mean Square Error (RMSE). Below is a detailed breakdown of each step:

### 5.1 Finetuning for Structure Prediction (Optional)

**Note: This step is optional, because we have provided the finetuned model for direct usage.**



Before generating structures, the model needs to be finetuned on a dataset specifically designed for crystal structure prediction. This step ensures that the model learns the patterns and relationships in the data required for accurate structure generation.

To finetune the model, run the following command:

```bash
bash finetune_csp.sh
```

In the scripts, the variable `DATA` can be set to one of the following datasets: `mp_20`, `mpts_52`, or `supercon`, depending on the task you wish to perform.



After this step, the finetuned model will be saved in `hydra/singlerun/yyyy-mm-dd/finetune_${DATA}` directory.

### 5.2 Structure Generation (Sampling)

Once the model is finetuned, you can use it to generate crystal structures. This process involves sampling from the model's learned distribution to produce candidate structures.

To generate structures, run the following command:

```bash
bash structure_generation.sh
```

You can customize the `DATA` and `NUM_EVALS` by modifying the arguments in the command. The generated structures will be saved in the same directory as the finetuned model, with the filename `eval_diff_xx_all.pt`.



### 5.3 Evaluation of the Generated Structure

After generating the structures, you can evaluate their quality by comparing them to the ground truth structures. The evaluation metrics used are Match Rate (MR) and Root Mean Square Error (RMSE).

To compute these metrics, run the following command:

```bash
bash structure_evaluation.sh
```

You just need to modify the `LABEL` variable to choose different generated structures for evaluation.



---

## 6. Crystal Property Prediction

The pretrained DAO-P model can be finetuned and tested on eight downstream property prediction tasks. These tasks involve predicting various material properties, such as bandgap, ehull, and more. To do this, run the following command:

```bash
bash finetune_prop.sh
```

In this script, we have consolidated the finetuning code for all datasets into a single file. Before running the script, you can select the dataset you wish to finetune on, by commenting out the sections corresponding to other datasets.



Once the script finishes running, it will display the Mean Absolute Error (MAE) on the test set, which serves as the evaluation metric for the model's performance in predicting material properties. The MAE value provides insight into the model's accuracy, with lower values indicating better performance.



---

## 7. Superconductivity Experiments

This section covers experiments related to superconductivity, including finetuning and prediction on the `supercon` dataset, as well as testing on three real-world datasets.

### 7.1 Finetuning DAO-G on Supercon3D (optional)

To finetune the DAO-G model on the `supercon3d` dataset, run:
```bash
## set DATA=supercon3d_gen  before running
bash finetune_csp.sh 
```

Similarly, this step is also optional, you can use our provied checkpoints in the folder `ckpts`.



### 7.2 Generating Structures for Superconductors without structures

After finetuning DAO-G on Supercon3D, it possess the ability to generate structures for those no-structure superconductors, by running:

```bash
## set DATA=supercon_rest  before running
bash supercon_generation.sh
```



Then you can run `data/supercon3d/split_k_fold.py` and `data/supercon3d/aug_k_fold.py` to split the original supercon3D dataset into 5 folds and then augment the training data for each fold.



### 7.3 Experiments on Three Real-World Superconductors

By using DAO-G and DAO-P models finetuned on the Supercon3D dataset, we can generate structures and predict Critical Temperatures ($T_c$) for three recently discovered superconductors, which are complex to analyze with conventional calculation software, such as DFT.

#### 7.3.1 Structure Generation via DAO-G

One can generate the structures conditioned on their formulas (CSP). Use the following command:

```bash
## set DATA=supercon_real  before running
bash supercon_generation.sh
```

#### 7.3.2 Critical Temperature Prediction by DAO-P

For $T_c$ prediction, one can run:

```bash
bash pred_tc.sh
```



---

## Conclusion

This README provides a comprehensive guide to setting up and using the DAO-G and DAO-P models. Whether you want to pretrain, finetune, or evaluate the models, the steps outlined above should help you get started. If you encounter any issues or have questions, feel free to open an issue in the repository.

Happy coding!

