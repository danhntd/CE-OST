export CUDA_VISIBLE_DEVICES=7
export NGPUS=1
#export CUDA_LAUNCH_BLOCKING=1
export PYTHONWARNINGS="ignore"

python ./tools/visualize_json_results.py --root "/storageStudents/danhnt/camo_transformer/OSFormer/checkpoints/camopp/osformer_test_pvt/"
python ./tools/visualize_json_results.py --root "/storageStudents/danhnt/camo_transformer/OSFormer/checkpoints/camopp/osformer_test_r101/"
python ./tools/visualize_json_results.py --root "/storageStudents/danhnt/camo_transformer/OSFormer/checkpoints/camopp/osformer_test_r50_rerun/"
python ./tools/visualize_json_results.py --root "/storageStudents/danhnt/camo_transformer/OSFormer/checkpoints/camopp/osformer_test_rt/"
python ./tools/visualize_json_results.py --root "/storageStudents/danhnt/camo_transformer/OSFormer/checkpoints/camopp/osformer_test_swint_rerun/"

python ./tools/visualize_json_results.py --root "/storageStudents/danhnt/camo_transformer/OSFormer/checkpoints/camopp/osformer_hed_contrast_pvt/"
python ./tools/visualize_json_results.py --root "/storageStudents/danhnt/camo_transformer/OSFormer/checkpoints/camopp/osformer_hed_contrast_r101/"
python ./tools/visualize_json_results.py --root "/storageStudents/danhnt/camo_transformer/OSFormer/checkpoints/camopp/osformer_hed_contrast_r50/"
python ./tools/visualize_json_results.py --root "/storageStudents/danhnt/camo_transformer/OSFormer/checkpoints/camopp/osformer_hed_contrast_rt/"
python ./tools/visualize_json_results.py --root "/storageStudents/danhnt/camo_transformer/OSFormer/checkpoints/camopp/osformer_hed_contrast_swint/"

python ./tools/visualize_json_results.py --root "/storageStudents/danhnt/camo_transformer/OSFormer/checkpoints/camopp/osformer_hed_white_pvt/"
python ./tools/visualize_json_results.py --root "/storageStudents/danhnt/camo_transformer/OSFormer/checkpoints/camopp/osformer_hed_white_r101/"
python ./tools/visualize_json_results.py --root "/storageStudents/danhnt/camo_transformer/OSFormer/checkpoints/camopp/osformer_hed_white_r50/"
python ./tools/visualize_json_results.py --root "/storageStudents/danhnt/camo_transformer/OSFormer/checkpoints/camopp/osformer_hed_white_rt/"
python ./tools/visualize_json_results.py --root "/storageStudents/danhnt/camo_transformer/OSFormer/checkpoints/camopp/osformer_hed_white_swint/"

python ./tools/visualize_json_results.py --root "/storageStudents/danhnt/camo_transformer/OSFormer/checkpoints/camopp/osformer_test_more_epochs_pvt/"
python ./tools/visualize_json_results.py --root "/storageStudents/danhnt/camo_transformer/OSFormer/checkpoints/camopp/osformer_test_more_epochs_r101/"
python ./tools/visualize_json_results.py --root "/storageStudents/danhnt/camo_transformer/OSFormer/checkpoints/camopp/osformer_test_more_epochs_r50/"
python ./tools/visualize_json_results.py --root "/storageStudents/danhnt/camo_transformer/OSFormer/checkpoints/camopp/osformer_test_more_epochs_rt/"
python ./tools/visualize_json_results.py --root "/storageStudents/danhnt/camo_transformer/OSFormer/checkpoints/camopp/osformer_test_more_epochs_swint/"


python ./tools/visualize_json_results.py --root "/storageStudents/danhnt/camo_transformer/OSFormer/checkpoints/camopp/osformer_cod10k_pvt/"
python ./tools/visualize_json_results.py --root "/storageStudents/danhnt/camo_transformer/OSFormer/checkpoints/camopp/osformer_cod10k_r101/"
python ./tools/visualize_json_results.py --root "/storageStudents/danhnt/camo_transformer/OSFormer/checkpoints/camopp/osformer_cod10k_r50/"
python ./tools/visualize_json_results.py --root "/storageStudents/danhnt/camo_transformer/OSFormer/checkpoints/camopp/osformer_cod10k_rt/"
python ./tools/visualize_json_results.py --root "/storageStudents/danhnt/camo_transformer/OSFormer/checkpoints/camopp/osformer_cod10k_swint/"

python ./tools/visualize_json_results.py --root "/storageStudents/danhnt/camo_transformer/OSFormer/checkpoints/camopp/osformer_cod10k_contrast_hed_pvt/"
python ./tools/visualize_json_results.py --root "/storageStudents/danhnt/camo_transformer/OSFormer/checkpoints/camopp/osformer_cod10k_contrast_hed_r101/"
python ./tools/visualize_json_results.py --root "/storageStudents/danhnt/camo_transformer/OSFormer/checkpoints/camopp/osformer_cod10k_contrast_hed_r50/"
python ./tools/visualize_json_results.py --root "/storageStudents/danhnt/camo_transformer/OSFormer/checkpoints/camopp/osformer_cod10k_contrast_hed_rt/"
python ./tools/visualize_json_results.py --root "/storageStudents/danhnt/camo_transformer/OSFormer/checkpoints/camopp/osformer_cod10k_contrast_hed_swint/"

python ./tools/visualize_json_results.py --root "/storageStudents/danhnt/camo_transformer/OSFormer/checkpoints/camopp/osformer_cod10k_white_hed_pvt/"
python ./tools/visualize_json_results.py --root "/storageStudents/danhnt/camo_transformer/OSFormer/checkpoints/camopp/osformer_cod10k_white_hed_r101/"
python ./tools/visualize_json_results.py --root "/storageStudents/danhnt/camo_transformer/OSFormer/checkpoints/camopp/osformer_cod10k_white_hed_r50/"
python ./tools/visualize_json_results.py --root "/storageStudents/danhnt/camo_transformer/OSFormer/checkpoints/camopp/osformer_cod10k_white_hed_rt/"
python ./tools/visualize_json_results.py --root "/storageStudents/danhnt/camo_transformer/OSFormer/checkpoints/camopp/osformer_cod10k_white_hed_swint/"

