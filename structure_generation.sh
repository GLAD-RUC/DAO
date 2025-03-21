#### mp_20 / mpts_52
DATA=mp_20

### deduped DAO-P model
energy_model_path="/path/to/DAO_P.ckpt"

### dir of finetuned DAO-G model
MODEL_PATH="/dir/to/finetune_DAO_G.ckpt"
NUM_EVALS=1

GPU=0
LABEL="${NUM_EVALS}_all"

echo "data: ${DATA}"    
echo "label: ${LABEL}"
echo "model: ${MODEL_PATH}"
echo "num_evals: ${NUM_EVALS}"

CUDA_VISIBLE_DEVICES=${GPU} python scripts/generate.py --energy_guidance \
       --dataset $DATA --num_evals ${NUM_EVALS} --label $LABEL \
       --model_path ${MODEL_PATH} --energy_model_path=${energy_model_path}