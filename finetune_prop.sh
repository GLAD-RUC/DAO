## shear (lr=0.0001, decay=0.00005, EPOCHS=500, GPUs=2)
## bulk (lr=0.0003, decay=0., EPOCHS=500, GPUs=2)
DATA=shear 
EPOCHS=500
echo $EPOCHS

lr=0.0001
decay=0.00005
GPU="0,1"   
echo $DATA
FINETUNE_MODE='pred'
echo "pretrain_mode: ${PRETRAIN_MODE}"
echo "finetune_mode: ${FINETUNE_MODE}"
CKPT='/path/to/DAO_P.ckpt'  ### change this to the path of the pretrained DAO_P (no deduplication)

echo ${GPU}

CUDA_VISIBLE_DEVICES="$GPU" python diffcsp/finetune.py train.finetune_mode=${FINETUNE_MODE} \
        data=$DATA model=finetune expname=finetune_${DATA} "model.pretrain_repr=${CKPT}" \
        optim.lr_scheduler_cos.T_0=$EPOCHS data.train_max_epochs=$EPOCHS \
        optim.optimizer.lr=$lr optim.optimizer.weight_decay=$decay \
        optim.warm_epochs=20  train.pl_trainer.gpus=2


### gap_pbe (DATA=mp_bench, prop=gap_pbe, lr=0.00002, decay=0., epochs=300, GPUs=3)
### ehull (DATA=jdft3d, prop=ehull, lr=0.0001, decay=0.0002, epochs=500, GPUs=3)
### mbj_bandgap (DATA=jdft3d, prop=mbj_bandgap, lr=0.0001, decay=0.0002, epochs=300, GPUs=2)
EPOCHS=300
echo $EPOCHS
DATA=mp_bench 
prop=gap_pbe
GPU='0,1,2'
lr=0.00002
decay=0.

echo "${DATA}, ${prop}"
echo $GPU

CUDA_VISIBLE_DEVICES="$GPU" python diffcsp/finetune.py train.finetune_mode=${FINETUNE_MODE} \
    data=${DATA} data.prop=${prop} model=finetune expname=finetune_${prop} "model.pretrain_repr=${CKPT}" \
    optim.lr_scheduler_cos.T_0=$EPOCHS data.train_max_epochs=$EPOCHS \
    optim.optimizer.lr=$lr optim.optimizer.weight_decay=$decay \
    optim.warm_epochs=20  train.pl_trainer.gpus=3



### jdft2d (lr=0.00008, decay=0.00005, epochs=300, GPUs=1)
### kvrh (lr=0.00008, decay=0.00001, epochs=200, GPUs=1)
### dielectric (lr=0.00005, decay=0.00005, epochs=300, GPUs=2)
EPOCHS=300
echo $EPOCHS
DATA=jdft2d 
lr=0.00008
decay=0.00005
GPU="0"   
echo $DATA
FINETUNE_MODE='pred'

echo ${GPU}

for fold in 0 1 2 3 4
do  
     echo "fold ${fold}"
     CUDA_VISIBLE_DEVICES="$GPU" python diffcsp/finetune.py train.finetune_mode=${FINETUNE_MODE} \
        data=${DATA} data=$DATA data.fold=$fold expname=finetune_${DATA}_${fold} \
        model=finetune "model.pretrain_repr=${CKPT}" \
        optim.lr_scheduler_cos.T_0=$EPOCHS data.train_max_epochs=$EPOCHS \
        optim.optimizer.lr=$lr optim.optimizer.weight_decay=$decay \
        optim.warm_epochs=20  train.pl_trainer.gpus=1
done



### supercon
EPOCHS=300
echo $EPOCHS
DATA=supercon3d_fold
lr=0.0004
decay=0.00005
GPU="0,1"   
echo $DATA
FINETUNE_MODE='pred'


echo ${GPU}
echo "$lr, $decay"

for fold in {0..0};do
       train_data_path='data/super_conductors/supercon3d/fold_${fold}/train_ori_aug_guidance.pt'
       echo "fold ${fold}"
       CUDA_VISIBLE_DEVICES="$GPU" python diffcsp/finetune.py train.finetune_mode=${FINETUNE_MODE} data=$DATA data.fold=$fold \
              expname=finetune_${DATA}/fold_${fold} model=finetune "model.pretrain_repr=${CKPT}" \
              train.has_val=False train.monitor_metric='train_loss' \
              data.datamodule.datasets.train.save_path=${train_data_path} \
              optim.optimizer.lr=${lr} optim.optimizer.weight_decay=${decay} optim.warm_epochs=20 \
              data.train_max_epochs=$EPOCHS optim.warm_up=True optim.lr_scheduler_cos.T_0=$EPOCHS \
              train.pl_trainer.gpus=2
done
