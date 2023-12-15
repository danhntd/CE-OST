from detectron2.data import (
    DatasetMapper,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
    DatasetCatalog,
)
from detectron2.data.datasets import register_coco_instances
import os

#!/usr/bin/env python

import torch

import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys
import cv2
from matplotlib.image import imread
import numpy as np
import torch.nn as nn


class HED(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
        self.netVggOne = torch.nn.Sequential(
          torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
          torch.nn.ReLU(inplace=False),
          torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
          torch.nn.ReLU(inplace=False)
        )
    
        self.netVggTwo = torch.nn.Sequential(
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
          torch.nn.ReLU(inplace=False),
          torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
          torch.nn.ReLU(inplace=False)
        )
    
        self.netVggThr = torch.nn.Sequential(
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
          torch.nn.ReLU(inplace=False),
          torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
          torch.nn.ReLU(inplace=False),
          torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
          torch.nn.ReLU(inplace=False)
        )
    
        self.netVggFou = torch.nn.Sequential(
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
          torch.nn.ReLU(inplace=False),
          torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
          torch.nn.ReLU(inplace=False),
          torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
          torch.nn.ReLU(inplace=False)
        )
    
        self.netVggFiv = torch.nn.Sequential(
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
          torch.nn.ReLU(inplace=False),
          torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
          torch.nn.ReLU(inplace=False),
          torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
          torch.nn.ReLU(inplace=False)
        )
    
        self.netScoreOne = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreTwo = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreThr = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFou = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFiv = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
    
        self.netCombine = torch.nn.Sequential(
          torch.nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
          torch.nn.Sigmoid()
        )
    
        self.load_state_dict({ 
            strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(
                url='http://content.sniklaus.com/github/pytorch-hed/network-' + arguments_strModel + '.pytorch', file_name='hed-' + arguments_strModel).items() })
        # end

    def forward(self, tenInput):
        tenVggOne = self.netVggOne(tenInput)
        tenVggTwo = self.netVggTwo(tenVggOne)
        tenVggThr = self.netVggThr(tenVggTwo)
        tenVggFou = self.netVggFou(tenVggThr)
        tenVggFiv = self.netVggFiv(tenVggFou)
    
        tenScoreOne = self.netScoreOne(tenVggOne)
        tenScoreTwo = self.netScoreTwo(tenVggTwo)
        tenScoreThr = self.netScoreThr(tenVggThr)
        tenScoreFou = self.netScoreFou(tenVggFou)
        tenScoreFiv = self.netScoreFiv(tenVggFiv)
    
        tenScoreOne = torch.nn.functional.interpolate(input=tenScoreOne, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=True)
        tenScoreTwo = torch.nn.functional.interpolate(input=tenScoreTwo, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=True)
        tenScoreThr = torch.nn.functional.interpolate(input=tenScoreThr, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=True)
        tenScoreFou = torch.nn.functional.interpolate(input=tenScoreFou, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=True)
        tenScoreFiv = torch.nn.functional.interpolate(input=tenScoreFiv, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=True)
    
        return self.netCombine(torch.cat([ tenScoreOne, tenScoreTwo, tenScoreThr, tenScoreFou, tenScoreFiv ], 1))




def estimate(tenInput):
    netNetwork = HED().cuda()
    return netNetwork(tenInput.cuda()).cpu()


if __name__ == '__main__':
    arguments_strModel = 'bsds500' # only 'bsds500' for now
    
    data_root = "/your/root/to/camo/data/"

    # register data
    register_coco_instances("cod_train", {}, data_root + "/COD10K-v3/train_instance.json", data_root + "/COD10K-v3/Train_Image_CAM/")
    camopp_train_metadata = MetadataCatalog.get("cod_train")
    camopp_train_dataset_dicts = DatasetCatalog.get("cod_train")
    
    # print(camopp_train_dataset_dicts[0]['file_name'])
    # print(camopp_train_metadata)

    from tqdm import tqdm
    import gc
    from PIL import Image
    from itertools import product
    from matplotlib.image import imread
    from matplotlib import cm
    
    apply_hed = []
    non_apply_hed = []
    
    def check_grid(img):
        d_h = int(img.shape[0]/5)
        d_w = int(img.shape[1]/5)
        img = img[:,:,0]
        img = Image.fromarray(img.astype('uint8'), 'L')
        w, h = img.size
        grid = product(range(0, h-h%d_h, d_h), range(0, w-w%d_w, d_w))
        grid_value = []
        for i, j in grid:
            box = (j, i, j+d_w, i+d_h)
            grid_value_i = np.count_nonzero(img.crop(box))
            grid_value.append(grid_value_i)

        no_grid_to_eliminate = len(list(np.where(np.array(grid_value)>0.5*d_h*d_w)[0]))
        print("no_grid_to_eliminate ", no_grid_to_eliminate)
        if no_grid_to_eliminate < 12:
            return True
        else:
            return False
    
    for i in tqdm(camopp_train_dataset_dicts):
        img_path = data_root + "/COD10K-v3/Train_Image_CAM/" + i['file_name'].split("/")[-1]
        save_path = data_root + "/COD10K-v3/Train_Image_CAM_contrast_hed/" + i['file_name'].split("/")[-1]
        # print('save_path', save_path)
        
        img = imread(img_path)

        if os.path.getsize(img_path) > 1000000 or img.shape[0] > 1000 or img.shape[1] > 1000:     
            w = img.shape[1]
            h = img.shape[0]
            dim = (int(500), int(500*h/w))
            resized = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            root_temp = "/storageStudents/danhnt/camo_transformer/OSFormer/contour_emphasis/contrast_temp/"
            temp_path = root_temp + save_path.split("/")[-1][:-4] + "_temp.jpg"
            cv2.imwrite(temp_path, resized)
            print("RESIZED: ", img_path, img.shape)
            
            del img
            del resized
            gc.collect()
        
            img = imread(temp_path)
            img = torch.from_numpy(img)
            img = img.permute(2, 0, 1).unsqueeze(0)
            tenInput = img*1.0  
            tenOutput = estimate(tenInput)
        
            img = imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
            bd = (tenOutput.permute(2,3,0,1).detach().numpy().squeeze(2))*255/2
            bd2 = cv2.resize(bd, (w,h), interpolation = cv2.INTER_CUBIC)
            bd2 = np.expand_dims(bd2, axis=2)
            if check_grid(bd2.copy()):
                img[:,:,0][bd2[:,:,0] > 75] = 255 - img[:,:,0][bd2[:,:,0] > 75]
                img[:,:,1][bd2[:,:,0] > 75] = 255 - img[:,:,1][bd2[:,:,0] > 75]
                img[:,:,2][bd2[:,:,0] > 75] = 255 - img[:,:,2][bd2[:,:,0] > 75]
                
                cv2.imwrite(save_path, img+0.25*bd2)
                apply_hed.append(save_path)
            else:
                non_apply_hed.append(save_path)
            del img
            gc.collect()
            torch.cuda.empty_cache()
        
        else:
            img = torch.from_numpy(img)
            img = img.permute(2, 0, 1).unsqueeze(0)
            tenInput = img*1.0  
            tenOutput = estimate(tenInput)
        
            img = imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            bd = (tenOutput.permute(2,3,0,1).detach().numpy().squeeze(2))*255/2
            if check_grid(bd.copy()):
                img[:,:,0][bd[:,:,0] > 75] = 255 - img[:,:,0][bd[:,:,0] > 75]
                img[:,:,1][bd[:,:,0] > 75] = 255 - img[:,:,1][bd[:,:,0] > 75]
                img[:,:,2][bd[:,:,0] > 75] = 255 - img[:,:,2][bd[:,:,0] > 75]
                
                cv2.imwrite(save_path, img+0.25*bd)
                apply_hed.append(save_path)
            else:
                non_apply_hed.append(save_path)
            del img
            gc.collect()
            torch.cuda.empty_cache()
    
    print(camopp_train_dataset_dicts[0]['file_name'])
    print(camopp_train_metadata)

    # with open('./CE-OST/contour_emphasis/contrast/apply_hed.txt', 'w') as f1:
    #     for item in apply_hed:
    #         f1.write("%s\n" % item)
        
    # with open('./CE-OST/contour_emphasis/contrast/non_apply_hed.txt', 'w') as f2:
    #     for item in non_apply_hed:
    #         f2.write("%s\n" % item)