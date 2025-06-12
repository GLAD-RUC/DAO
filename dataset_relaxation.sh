
mode='lbfgs'
steps=5
step_size=1.
start=-1
end=-1


cfg_dir='/dir/to/DAO_G_Stage1.ckpt'
model_dir='/dir/to/DAO_P.ckpt'


return_stable=0
return_unrelax=0

if (( return_stable && return_unrelax )); then
    label="_full"
elif (( return_stable )); then
    label="_stable_relax"
elif (( return_unrelax )); then
    label="_relax_unrelax"
else
    label="_relax"
fi
    
echo ${label}

CUDA_VISIBLE_DEVICES="3" python scripts/relax_dataset.py \
        --model_dir=${model_dir} --label=${label}  --start=$start --end=$end \
        --step_size=${step_size} --num_steps=${steps} --threshold=0.5 \
        --relax_mode=${mode} --cfg_dir=${cfg_dir} \
        --return_stable=${return_stable} --return_unrelax=${return_unrelax}
