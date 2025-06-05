### DAO_G_Stage1
EPOCHS=800
echo $EPOCHS
CKPT=''

CUDA_VISIBLE_DEVICES='0,1,2' python diffcsp/run.py expname=DAO_G_Stage1 \
    data=pretrain_daog_data data.train_max_epochs=$EPOCHS train.pretrain=True \
    optim.lr_scheduler_cos.T_0=$EPOCHS optim.optimizer.lr=0.0003  optim.warm_epochs=50\
    +load_state_dict_only=True +ckpt_path=$CKPT\
    model.only_diffusion=True train.pl_trainer.gpus=3

### DAO_G_Stage2
EPOCHS=500
echo $EPOCHS
CKPT='/path/to/DAO_G_Stage1.ckpt'
relax_data_path='/path/to/relax_data.pt'

CUDA_VISIBLE_DEVICES='0,1,2' python diffcsp/run.py expname=DAO_G_Stage2 \
    data=pretrain_daog_data data.train_max_epochs=$EPOCHS train.pretrain=True \
    optim.lr_scheduler_cos.T_0=$EPOCHS optim.optimizer.lr=0.0003  optim.warm_epochs=50\
    +load_state_dict_only=True +ckpt_path=$CKPT\
    data.datamodule.datasets.train.save_path=${relax_data_path} \
    model.only_diffusion=True train.pl_trainer.gpus=3

### DAO_P
EPOCHS=800
echo $EPOCHS
CKPT=''
CUDA_VISIBLE_DEVICES='0,1' python diffcsp/run.py expname=DAO_P \
    data=pretrain_daop_data data.train_max_epochs=$EPOCHS train.pretrain=True \
    optim.lr_scheduler_cos.T_0=$EPOCHS optim.optimizer.lr=0.0002  optim.warm_epochs=50\
    +load_state_dict_only=True +ckpt_path=$CKPT\
    model.only_diffusion=False train.pl_trainer.gpus=2