#### mp_20 / mpts_52 / supercon3d_gen
DATA=mp_20
EPOCHS=1000    
lr=0.00002
decay=0.00001
echo "data: ${DATA}"
echo "epochs: ${EPOCHS}"

CKPT='/path/to/DAO_G_Stage2.ckpt'

## for supercon3d_gen, use 2 GPUs
python diffcsp/finetune.py data=$DATA expname="finetune_${DATA}" \
     model=finetune "model.pretrain_repr=${CKPT}" \
     optim.lr_scheduler_cos.T_0=$EPOCHS data.train_max_epochs=$EPOCHS \
     optim.optimizer.lr=${lr}  optim.optimizer.weight_decay=${decay} \
     train.pl_trainer.gpus=8
     

