export CUDA_VISIBLE_DEVICES=3
export NGPUS=1
#export CUDA_LAUNCH_BLOCKING=1
export PYTHONWARNINGS="ignore"

OUTPUT_DIR=/storageStudents/danhnt/camo_transformer/EVA/EVA-02/det/output/
config=configs/CIS_PVTv2B2Li.yaml
WEIGHT=weights/osformer-pvt.pth

cfg_MODEL='
SOLVER.IMS_PER_BATCH 1
DATALOADER.NUM_WORKERS 0
'

python tools/train_net.py --num-gpus ${NGPUS} --dist-url auto --config-file ${config} --num-machines 1\
      --opts MODEL.WEIGHTS ${WEIGHT} OUTPUT_DIR ${OUTPUT_DIR} ${cfg_MODEL}
  
#python tools/TUAN_lazyconfig_train_net.py --num-gpus 1 --config-file projects/ViTDet/configs/eva2_mim_to_coco/TUAN_coco_cascade_mask_rcnn_vitdet_b_4attn_1024_lrd0p7.py --eval-only
