export CUDA_VISIBLE_DEVICES=0
export NGPUS=1
#export CUDA_LAUNCH_BLOCKING=1
export PYTHONWARNINGS="ignore"

MODEL_NAME='test'
OUTPUT_DIR=checkpoints/camopp/osformer_${MODEL_NAME}_r50
config=configs/CIS_R50.yaml #done


OUTPUT_DIR=checkpoints/camopp/osformer_${MODEL_NAME}_swint
config=configs/CIS_SWINT.yaml
WEIGHT=weights/osformer-swin.pth #done without tensorboard







OUTPUT_DIR=checkpoints/camopp/osformer_${MODEL_NAME}_pvt
config=configs/CIS_PVTv2B2Li.yaml
WEIGHT=weights/osformer-pvt.pth

cfg_MODEL='
SOLVER.IMS_PER_BATCH 1
DATALOADER.NUM_WORKERS 0
'

python tools/train_net.py --num-gpus ${NGPUS} --dist-url auto --config-file ${config} --num-machines 1\
      --opts MODEL.WEIGHTS ${WEIGHT} OUTPUT_DIR ${OUTPUT_DIR} ${cfg_MODEL}
  

OUTPUT_DIR=checkpoints/camopp/osformer_${MODEL_NAME}_rt
config=configs/CIS_RT.yaml
WEIGHT=weights/osformer-rt.pth
  
python tools/train_net.py --num-gpus ${NGPUS} --dist-url auto --config-file ${config} --num-machines 1\
      --opts MODEL.WEIGHTS ${WEIGHT} OUTPUT_DIR ${OUTPUT_DIR} ${cfg_MODEL}
      
      

OUTPUT_DIR=checkpoints/camopp/osformer_${MODEL_NAME}_r101
config=configs/CIS_R101.yaml
WEIGHT=weights/osformer-r101.pth
  
python tools/train_net.py --num-gpus ${NGPUS} --dist-url auto --config-file ${config} --num-machines 1\
      --opts MODEL.WEIGHTS ${WEIGHT} OUTPUT_DIR ${OUTPUT_DIR} ${cfg_MODEL}
      


OUTPUT_DIR=checkpoints/camopp/osformer_${MODEL_NAME}_swint_rerun
config=configs/CIS_SWINT.yaml
WEIGHT=weights/osformer-swin.pth

python tools/train_net.py --num-gpus ${NGPUS} --dist-url auto --config-file ${config} --num-machines 1\
      --opts MODEL.WEIGHTS ${WEIGHT} OUTPUT_DIR ${OUTPUT_DIR} ${cfg_MODEL}
      

OUTPUT_DIR=checkpoints/camopp/osformer_${MODEL_NAME}_r50_rerun
config=configs/CIS_R50.yaml
WEIGHT=weights/osformer-r50.pth

python tools/train_net.py --num-gpus ${NGPUS} --dist-url auto --config-file ${config} --num-machines 1\
      --opts MODEL.WEIGHTS ${WEIGHT} OUTPUT_DIR ${OUTPUT_DIR} ${cfg_MODEL}