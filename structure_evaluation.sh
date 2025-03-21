### mp_20 / mpts_52 / supercon_real
DATA=mp_20

### dir of finetuned DAO-G model (that is, dir of the generated structures)
MODEL_PATH="/dir/to/finetune_DAO_G.ckpt"

### 1 or 20
NUM_EVALS=1
LABEL="${NUM_EVALS}_all"
RANK="False"
CLOSEST="True"
echo "data: ${DATA}"
echo "label: ${LABEL}"
echo "model: ${MODEL_PATH}"
echo "num_evals: ${NUM_EVALS}"
echo "rank_selected : ${RANK}"
echo "select closest : ${CLOSEST}"


#### compute metric: match_rate, rmsd
if [ ${NUM_EVALS} -eq 1 ]
then
    python scripts/evaluate_gen.py --root_path ${MODEL_PATH} --tasks csp --label $LABEL --gt_file data/${DATA}/test.csv
else
    python scripts/evaluate_gen.py --root_path ${MODEL_PATH} --tasks csp --label $LABEL --gt_file data/${DATA}/test.csv --multi_eval
fi
