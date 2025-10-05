# Task setting
TASKCFG=task_configs/biomarker/BCNB_ER.yaml
DATASETCSV=dataset_csv/biomarker/BCNB_ER.csv

ROOTPATH=/data4/embedding/BCNB/Gigapath_tile
PRESPLITDIR=dataset_split/BCNB_ER/
MAX_WSI_SIZE=250000  # Maximum WSI size in pixels for the longer side (width or height).
TILE_SIZE=256
# Model settings
MODELARCH=gigapath_slide_enc12l768d
TILEEMBEDSIZE=1536
LATENTDIM=768
# Training settings
EPOCH=5
GC=32
BLR=0.002
WD=0.05
LD=0.95
FEATLAYER="11"
DROPOUT=0.1
# Output settings
WORKSPACE=outputs/BCNB_ER
SAVEDIR=$WORKSPACE
EXPNAME=run_epoch-${EPOCH}_blr-${BLR}_wd-${WD}_ld-${LD}_feat-${FEATLAYER}
ark_pretrained_ckpt='/home/yuhaowang/project/PathARK/Pretraining/outputs/ark_pretrain/ark_pretrain_e149.pth.tar'
echo "Data directory set to $ROOTPATH"

python main.py --task_cfg_path ${TASKCFG} \
               --dataset_csv $DATASETCSV \
               --root_path $ROOTPATH \
               --model_arch $MODELARCH \
               --blr $BLR \
               --layer_decay $LD \
               --optim_wd $WD \
               --dropout $DROPOUT \
               --drop_path_rate 0.0 \
               --val_r 0.1 \
               --epochs $EPOCH \
               --input_dim $TILEEMBEDSIZE \
               --latent_dim $LATENTDIM \
               --feat_layer $FEATLAYER \
               --warmup_epochs 1 \
               --gc $GC \
               --ark_pretrained_ckpt $ark_pretrained_ckpt \
               --model_select last_epoch \
               --lr_scheduler cosine \
               --folds 1 \
               --dataset_csv $DATASETCSV \
               --pre_split_dir $PRESPLITDIR \
               --save_dir $SAVEDIR \
               --report_to tensorboard \
               --exp_name $EXPNAME \
               --max_wsi_size $MAX_WSI_SIZE