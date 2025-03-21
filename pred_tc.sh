ori_path=data/super_conductors/real_world/output_ori.pt

for idx in 0 1 2 3 4;do
    model_path="/path/to/finetuned_DAO_P_${idx}.ckpt"  ## five fold models
    echo ${model_path}
    python inference_prop_dataset.py --ori_path=ori_path --model_path=${model_path} --prop=logtc
done
