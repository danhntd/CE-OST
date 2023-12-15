export CUDA_VISIBLE_DEVICES=7
export NGPUS=1
#export CUDA_LAUNCH_BLOCKING=1
export PYTHONWARNINGS="ignore"

MODEL_NAME='cod10k'


OUTPUT_DIR=checkpoints/camopp/osformer_${MODEL_NAME}_pvt
config=configs/CIS_PVTv2B2Li_cod10k.yaml
WEIGHT=weights/osformer-pvt.pth

cfg_MODEL='
SOLVER.IMS_PER_BATCH 1
DATALOADER.NUM_WORKERS 0
'

python tools/train_net.py --num-gpus ${NGPUS} --resume --dist-url auto --config-file ${config} --num-machines 1\
      --opts MODEL.WEIGHTS ${WEIGHT} OUTPUT_DIR ${OUTPUT_DIR} ${cfg_MODEL}
  

OUTPUT_DIR=checkpoints/camopp/osformer_${MODEL_NAME}_rt
config=configs/CIS_RT_cod10k.yaml
WEIGHT=weights/osformer-rt.pth
  
python tools/train_net.py --num-gpus ${NGPUS} --dist-url auto --config-file ${config} --num-machines 1\
      --opts MODEL.WEIGHTS ${WEIGHT} OUTPUT_DIR ${OUTPUT_DIR} ${cfg_MODEL}
      
      

OUTPUT_DIR=checkpoints/camopp/osformer_${MODEL_NAME}_r101
config=configs/CIS_R101_cod10k.yaml
WEIGHT=weights/osformer-r101.pth
  
python tools/train_net.py --num-gpus ${NGPUS} --dist-url auto --config-file ${config} --num-machines 1\
      --opts MODEL.WEIGHTS ${WEIGHT} OUTPUT_DIR ${OUTPUT_DIR} ${cfg_MODEL}
      


OUTPUT_DIR=checkpoints/camopp/osformer_${MODEL_NAME}_swint
config=configs/CIS_SWINT_cod10k.yaml
WEIGHT=weights/osformer-swin.pth

python tools/train_net.py --num-gpus ${NGPUS} --dist-url auto --config-file ${config} --num-machines 1\
      --opts MODEL.WEIGHTS ${WEIGHT} OUTPUT_DIR ${OUTPUT_DIR} ${cfg_MODEL}
      

OUTPUT_DIR=checkpoints/camopp/osformer_${MODEL_NAME}_r50
config=configs/CIS_R50_cod10k.yaml
WEIGHT=weights/osformer-r50.pth

python tools/train_net.py --num-gpus ${NGPUS} --dist-url auto --config-file ${config} --num-machines 1\
      --opts MODEL.WEIGHTS ${WEIGHT} OUTPUT_DIR ${OUTPUT_DIR} ${cfg_MODEL}