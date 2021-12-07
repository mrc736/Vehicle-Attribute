from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
import matplotlib.pyplot as plt
from darknet_util import count_parameters as count
from darknet_util import convert2cpu as cpu
from PIL import Image, ImageDraw
import os
import shutil

def create_test():
    import glob
    test_path = r'./vehicle_test'
    ori_path = r'./vehicle_new/*'
    list_floders=glob.glob(ori_path)
    for floder in list_floders:
        ima_list=os.listdir(floder)
        if len(ima_list)>2:
            rand_list=np.random.choice(ima_list,len(ima_list)//3)
            floder_name=floder.split('\\')[1]
            root = os.path.join(test_path, floder_name)
            if not os.path.exists(root):
               os.mkdir(root)
            for file in rand_list:
                try:
                    shutil.move(os.path.join(r'./vehicle_new/%s'%floder_name,file),os.path.join(root,file))
                except:
                    print(file)
# create_test()

def create_floder():

    ori_path=r'C:\Data\Github\Vehicle-Car-detection-and-multilabel-classification-master\output_new'
    newpath = r'./vehicle_new'
    # rand_seed=np.random.randint(1,10,1)
    # print(rand_seed)
    # if rand_seed<=6:
    #     newpath=r'./vehicle_train'
    # else:
    #     newpath = r'./vehicle_test'
    for filename in os.listdir(ori_path):
        rand_seed = np.random.randint(1, 10, 1)
        print(filename)
        # if rand_seed <= 6:
        #     newpath = r'./vehicle_train_new'
        # else:
        #     newpath = r'./vehicle_test_new'
        label=filename.split('_')[-1].split('.')[0].replace(' ','_')
        newfloder=os.path.join(newpath,label)
        if not os.path.exists(newfloder):
            os.mkdir(newfloder)
        # shutil.copy(os.path.join(ori_path,filename),os.path.join(newfloder,filename))
# create_floder()

def letterbox_image(img, inp_dim):
    '''
    resize image with unchanged aspect ratio using padding
    '''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)
    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

    return canvas


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.
    Returns a Tensor or Variable 
    """
    orig_im = cv2.imread(img)
    dim = orig_im.shape[1], orig_im.shape[0]  # 图像原始宽高
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()  # BGR->RGB and WxHxchans => chansxWxH
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim


def process_img(img, inp_dim):
    """
    input PIL img, return processed img
    """
    dim = img.width, img.height
    img = (letterbox_image(np.asarray(img), (inp_dim, inp_dim)))
    img_ = img.transpose((2, 0, 1)).copy()  # WxHxchans => chansxWxH
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0) 
    return img_


def prep_image_pil(img, network_dim):
    orig_im = Image.open(img)
    img = orig_im.convert('RGB')
    dim = img.size
    img = img.resize(network_dim)
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    img = img.view(*network_dim, 3).transpose(0, 1).transpose(0, 2).contiguous()
    img = img.view(1, 3, *network_dim)
    img = img.float().div(255.0)
    return (img, orig_im, dim)


def inp_to_image(inp):
    inp = inp.cpu().squeeze()
    inp = inp * 255
    try:
        inp = inp.data.numpy()
    except RuntimeError:
        inp = inp.numpy()
    inp = inp.transpose(1, 2, 0)

    inp = inp[:, :, ::-1]
    return inp
