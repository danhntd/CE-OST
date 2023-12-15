export CUDA_VISIBLE_DEVICES=4
export NGPUS=1
#export CUDA_LAUNCH_BLOCKING=1
export PYTHONWARNINGS="ignore"

MODEL_NAME='test_more_epochs'
cfg_MODEL='
SOLVER.IMS_PER_BATCH 1
DATALOADER.NUM_WORKERS 0
'


OUTPUT_DIR=checkpoints/camopp/osformer_${MODEL_NAME}_pvt
config=configs/CIS_PVTv2B2Li.yaml
WEIGHT=checkpoints/camopp/osformer_test_pvt/model_final.pth #weights/osformer-pvt.pth 

python tools/train_net.py --num-gpus ${NGPUS} --resume --dist-url auto --config-file ${config} --num-machines 1\
      --opts MODEL.WEIGHTS ${WEIGHT} OUTPUT_DIR ${OUTPUT_DIR} ${cfg_MODEL}
  

OUTPUT_DIR=checkpoints/camopp/osformer_${MODEL_NAME}_rt
config=configs/CIS_RT.yaml
WEIGHT=checkpoints/camopp/osformer_test_rt/model_final.pth #weights/osformer-rt.pth
  
python tools/train_net.py --num-gpus ${NGPUS} --resume --dist-url auto --config-file ${config} --num-machines 1\
      --opts MODEL.WEIGHTS ${WEIGHT} OUTPUT_DIR ${OUTPUT_DIR} ${cfg_MODEL}
      
      

OUTPUT_DIR=checkpoints/camopp/osformer_${MODEL_NAME}_r101
config=configs/CIS_R101.yaml
WEIGHT=checkpoints/camopp/osformer_test_r101/model_final.pth #weights/osformer-r101.pth
   
python tools/train_net.py --num-gpus ${NGPUS} --resume --dist-url auto --config-file ${config} --num-machines 1\
      --opts MODEL.WEIGHTS ${WEIGHT} OUTPUT_DIR ${OUTPUT_DIR} ${cfg_MODEL}
      


OUTPUT_DIR=checkpoints/camopp/osformer_${MODEL_NAME}_swint
config=configs/CIS_SWINT.yaml
WEIGHT=checkpoints/camopp/osformer_test_swint_rerun/model_final.pth #weights/osformer-swin.pth

python tools/train_net.py --num-gpus ${NGPUS} --resume --dist-url auto --config-file ${config} --num-machines 1\
      --opts MODEL.WEIGHTS ${WEIGHT} OUTPUT_DIR ${OUTPUT_DIR} ${cfg_MODEL}
      

OUTPUT_DIR=checkpoints/camopp/osformer_${MODEL_NAME}_r50
config=configs/CIS_R50.yaml
WEIGHT=checkpoints/camopp/osformer_test_r50_rerun/model_final.pth #weights/osformer-r50.pth

python tools/train_net.py --num-gpus ${NGPUS} --resume --dist-url auto --config-file ${config} --num-machines 1\
      --opts MODEL.WEIGHTS ${WEIGHT} OUTPUT_DIR ${OUTPUT_DIR} ${cfg_MODEL}