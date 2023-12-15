import os
from detectron2.data.datasets.coco import load_coco_json
from detectron2.data import MetadataCatalog, DatasetCatalog

DATASET_ROOT = '/storageStudents/danhnt/camo_data/COD10K-v3'
#ANN_ROOT = os.path.join(DATASET_ROOT, 'annotations')
TRAIN_PATH = os.path.join(DATASET_ROOT, 'Train_Image_CAM')
TEST_PATH = os.path.join(DATASET_ROOT, 'Test_Image_CAM')
TRAIN_JSON = os.path.join(DATASET_ROOT, 'train_instance.json')
TEST_JSON = os.path.join(DATASET_ROOT, 'test2026.json')

NC4K_ROOT = '/storageStudents/danhnt/camo_data/NC4K'
NC4K_PATH = os.path.join(NC4K_ROOT, 'test/image')
NC4K_JSON = os.path.join(NC4K_ROOT, 'nc4k_test.json')

COD_CONTRAST_HED_TRAIN_PATH = os.path.join(DATASET_ROOT, 'Train_Image_CAM_contrast_hed')
COD_CONTRAST_HED_TRAIN_JSON = TRAIN_JSON
COD_WHITE_HED_TRAIN_PATH = os.path.join(DATASET_ROOT, 'Train_Image_CAM_white_hed')
COD_WHITE_HED_TRAIN_JSON = TRAIN_JSON

CAMO_ROOT = '/storageStudents/danhnt/detectron2/datasets/camopp/Public/' 
CAMO_PATH = os.path.join(CAMO_ROOT, 'Images')
CAMO_TRAIN_JSON = os.path.join(CAMO_ROOT, 'Annotations/camo_train_new.json')
CAMO_TEST_JSON = os.path.join(CAMO_ROOT, 'Annotations/camo_test_new.json')

CAMO_CONTRAST_HED_ROOT = '/storageStudents/danhnt/detectron2/projects/bem_mask_rcnn_tune_cfg/camopp_bd_contrast_grid/Public/' 
CAMO_CONTRAST_HED_PATH = os.path.join(CAMO_CONTRAST_HED_ROOT, 'Images')
CAMO_CONTRAST_HED_TRAIN_JSON = os.path.join(CAMO_CONTRAST_HED_ROOT, 'Annotations/camo_train_new.json')
CAMO_CONTRAST_HED_TEST_JSON = os.path.join(CAMO_CONTRAST_HED_ROOT, 'Annotations/camo_test_new.json') #no use because we do not apply HED on test image

CAMO_WHITE_HED_ROOT = '/storageStudents/danhnt/detectron2/projects/bem_mask_rcnn_tune_cfg/camopp_bd_white_grid/Public/' 
CAMO_WHITE_HED_PATH = os.path.join(CAMO_WHITE_HED_ROOT, 'Images')
CAMO_WHITE_HED_TRAIN_JSON = os.path.join(CAMO_WHITE_HED_ROOT, 'Annotations/camo_train_new.json')
CAMO_WHITE_HED_TEST_JSON = os.path.join(CAMO_WHITE_HED_ROOT, 'Annotations/camo_test_new.json') #no use because we do not apply HED on test image


CLASS_NAMES = ["foreground"]

PREDEFINED_SPLITS_DATASET = {
    "my_data_train_coco_cod_style": (TRAIN_PATH, TRAIN_JSON),
    "my_data_test_coco_cod_style": (TEST_PATH, TEST_JSON),
    "my_data_test_coco_nc4k_style": (NC4K_PATH, NC4K_JSON),
    
    "my_data_train_coco_cod_contrast_hed_style": (COD_CONTRAST_HED_TRAIN_PATH, COD_CONTRAST_HED_TRAIN_JSON),
    "my_data_train_coco_cod_white_hed_style": (COD_WHITE_HED_TRAIN_PATH, COD_WHITE_HED_TRAIN_JSON),
    
    
    
    "my_data_train_coco_camo_style": (CAMO_PATH, CAMO_TRAIN_JSON),
    "my_data_test_coco_camo_style": (CAMO_PATH, CAMO_TEST_JSON),

    "my_data_train_coco_camo_contrast_hed_style": (CAMO_CONTRAST_HED_PATH, CAMO_CONTRAST_HED_TRAIN_JSON),
    "my_data_test_coco_camo_contrast_hed_style": (CAMO_CONTRAST_HED_PATH, CAMO_CONTRAST_HED_TEST_JSON),

    "my_data_train_coco_camo_white_hed_style": (CAMO_WHITE_HED_PATH, CAMO_WHITE_HED_TRAIN_JSON),
    "my_data_test_coco_camo_white_hed_style": (CAMO_WHITE_HED_PATH, CAMO_WHITE_HED_TEST_JSON),
}


def register_dataset():
    """
    purpose: register all splits of dataset with PREDEFINED_SPLITS_DATASET
    """
    for key, (image_root, json_file) in PREDEFINED_SPLITS_DATASET.items():
        register_dataset_instances(name=key,
                                   json_file=json_file,
                                   image_root=image_root)


def register_dataset_instances(name, json_file, image_root):
    """
    purpose: register dataset to DatasetCatalog,
             register metadata to MetadataCatalog and set attribute
    """
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(json_file=json_file,
                                  image_root=image_root,
                                  evaluator_type="coco")