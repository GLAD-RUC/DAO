DATA="mp_20"  # optional: mp_20/mpts_52
num_gpus=1    # your total number of GPUs available
base_gpu=0    # the base GPU ID to start from
NUM_EVALS=1   # number of generations per data, optional: 1/20

## Note: if using multiple GPUs, combine the results (by scripts/combine_eval_results.ipynb) after generation

### deduped DAO-P model
energy_model_path="/path/to/DAO_P.ckpt"

## set the total number of batches based on the dataset
if [[ "$DATA" == "mp_20" ]]; then
    total_batches=38    ## related to batch size
    ## dir of finetuned DAO-G model on mp_20 dataset
    MODEL_PATH="/dir/to/finetune_DAO_G_mp_20.ckpt"
elif [[ "$DATA" == "mpts_52" ]]; then
    total_batches=405
    ## dir of finetuned DAO-G model on mpts_52 dataset
    MODEL_PATH="/dir/to/finetune_DAO_G_mpts_52.ckpt"
else
    echo "Unknown dataset: $dataset"
    exit 1
fi

## allocate the total number of batches to each GPU
base_batch=$(( total_batches / num_gpus ))
remainder_batch=$(( total_batches % num_gpus ))

## initialize start_batch_idx
start_batch_idx=0

## start a loop to iterate over the number of GPUs
for (( i=0; i<num_gpus; i++ )); do
    # For the first `remainder_batch` GPUs, allocate one extra batch
    if (( i < remainder_batch )); then
        current_batch_count=$(( base_batch + 1 ))
    else
        current_batch_count=$base_batch
    fi

    end_batch_idx=$(( start_batch_idx + current_batch_count))

    gpu_id=$(( base_gpu + i ))
    
    ## if num_gpus is 1, set LABEL to "all"
    if (( num_gpus == 1 )); then
       LABEL="${NUM_EVALS}_all"
    else
       LABEL="${NUM_EVALS}_${i}"
    fi     

    echo "GPU ${gpu_id} (with label ${LABEL}): Batch ${start_batch_idx} to ${end_batch_idx}"

    # start a background process for each GPU
    (
       CUDA_VISIBLE_DEVICES=${gpu_id} python scripts/generate.py --energy_guidance \
              --dataset $DATA --num_evals ${NUM_EVALS} --label $LABEL \
              --model_path ${MODEL_PATH} --energy_model_path=${energy_model_path} \
              --start ${start_batch_idx} --end ${end_batch_idx}
    ) &

    # Update the start index for the next GPU
    start_batch_idx=$(( end_batch_idx))
done

# wait for all background tasks to complete
wait

echo "All tasks completed."