#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import json
import numpy as np
import os
from collections import defaultdict
import cv2
import tqdm
from fvcore.common.file_io import PathManager

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

from adet.data.datasets.cis import register_dataset

def create_instances(predictions, image_size):
    ret = Instances(image_size)

    score = np.asarray([x["score"] for x in predictions])
    chosen = (score > args.conf_threshold).nonzero()[0]
    score = score[chosen]
    bbox = np.asarray([predictions[i]["bbox"] for i in chosen])
    if bbox.shape[0] > 0: 
      bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)

    labels = np.asarray([dataset_id_map(predictions[i]["category_id"]) for i in chosen])

    ret.scores = score
    ret.pred_boxes = Boxes(bbox)
    ret.pred_classes = labels

    try:
        ret.pred_masks = [predictions[i]["segmentation"] for i in chosen]
    except KeyError:
        pass
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that visualizes the json predictions from COCO or LVIS dataset."
    )
    parser.add_argument("--root", default=None, help="root folder to visualize")
    #parser.add_argument("--input", default=root+"inference/coco_instances_results.json", help="JSON file produced by the model")
    #parser.add_argument("--output", default=root+"/visualization/", help="output directory")
    parser.add_argument("--dataset", help="name of the dataset", default="my_data_test_coco_camo_style")
    parser.add_argument("--conf-threshold", default=0.5, type=float, help="confidence threshold")
    args = parser.parse_args()
    
    
    register_dataset()
    
    input_ = args.root + "/inference/coco_instances_results.json"
    output_ = args.root+"/visualization/"
    
    total_vis = []
    
    logger = setup_logger()

    with PathManager.open(input_, "r") as f:
        predictions = json.load(f)

    pred_by_image = defaultdict(list)
        
    for p in predictions:
        pred_by_image[p["image_id"]].append(p)

    dicts = list(DatasetCatalog.get(args.dataset))
    metadata = MetadataCatalog.get(args.dataset)
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):

        def dataset_id_map(ds_id):
            return metadata.thing_dataset_id_to_contiguous_id[ds_id]

    elif "lvis" in args.dataset:
        # LVIS results are in the same format as COCO results, but have a different
        # mapping from dataset category id to contiguous category id in [0, #categories - 1]
        def dataset_id_map(ds_id):
            return ds_id - 1

    else:
        raise ValueError("Unsupported dataset: {}".format(args.dataset))

    os.makedirs(output_, exist_ok=True)

    for dic in tqdm.tqdm(dicts):
        
        img = cv2.imread(dic["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]
        basename = os.path.basename(dic["file_name"])

        predictions = create_instances(pred_by_image[dic["image_id"]], img.shape[:2])
        
        if len(predictions.pred_classes) > 0:
            vis = Visualizer(img, metadata)
            vis_pred = vis.draw_instance_predictions(predictions).get_image()
    
            vis = Visualizer(img, metadata)
            vis_gt = vis.draw_dataset_dict(dic).get_image()
    
            #concat = np.concatenate((img, vis_pred, vis_gt), axis=1)
            #cv2.imwrite(os.path.join(output_, basename), concat[:, :, ::-1])
            
            cv2.imwrite(os.path.join(output_, basename[:-4]+"_gt.jpg"), vis_gt[:, :, ::-1])
            cv2.imwrite(os.path.join(output_, basename[:-4]+"_pred.jpg"), vis_pred[:, :, ::-1])
            cv2.imwrite(os.path.join(output_, basename[:-4]+"_img.jpg"), img[:, :, ::-1])
            
            total_vis.append(dic["file_name"])
            
            print(dic["file_name"])
            print('predictions.scores', predictions.scores)
            print('predictions.pred_boxes', predictions.pred_boxes)
            print('predictions.pred_classes', predictions.pred_classes)
            print('predictions.pred_masks[i][size]', [predictions.pred_masks[i]['size'] for i in range(len(predictions.pred_masks))])
            print('_____')
            #break
            
    with open(args.root+'vis_img.txt', 'w+') as f:
        for item in total_vis:
            f.write("%s\n" % item)