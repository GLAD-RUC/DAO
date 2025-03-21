
mode='lbfgs'
steps=5
step_size=1.

label='_full_'

cfg_dir='/dir/to/DAO_G_Stage1.ckpt'
model_dir='/dir/to/DAO_P.ckpt'
    

### --return_relax_only
python scripts/relax_dataset.py \
        --model_dir=${model_dir} --label=${label}  --start=$start --end=$end \
        --step_size=${step_size} --num_steps=${steps} --threshold=0.5 \
        --relax_mode=${mode} --cfg_dir=${cfg_dir} 