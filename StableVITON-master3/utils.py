import json
import argparse

import numpy as np
import torch.nn.functional as F
import torch
from PIL import Image

def save_args(args, to_path):
    with open(to_path, "w") as f:
        json.dump(args.__dict__, f, indent=2)
def load_args(from_path, is_test=True):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    with open(from_path, "r") as f:
        args.__dict__ = json.load(f)
    args.is_test = is_test
    if "E_name" not in args.__dict__.keys():
        args.E_name = "basic"
    return args   
def tensor2img(x):
    '''
    x : [BS x c x H x W] or [c x H x W]
    '''
    if x.ndim == 3:
        x = x.unsqueeze(0)
    BS, C, H, W = x.shape
    x = x.permute(0,2,3,1).reshape(-1, W, C).detach().cpu().numpy()
    # x = (x+1)/2
    # x = np.clip(x, 0, 1)
    x = np.clip(x, -1, 1)
    x = (x+1)/2
    x = np.uint8(x*255.0)
    if x.shape[-1] == 1:  # gray sclae
        x = np.concatenate([x,x,x], axis=-1)
    return x
def tensor2img(img_tensor):
    img = img_tensor.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = (img + 1.0) * 127.5
    img = np.clip(img, 0, 255)
    img = np.uint8(img)
    return img

def preprocess_image(image_path, target_size=(384, 512)):
    """이미지를 모델 입력에 맞게 전처리합니다.
    
    Args:
        image_path (str): 이미지 파일 경로
        target_size (tuple): 목표 이미지 크기 (width, height)
        
    Returns:
        torch.Tensor: 전처리된 이미지 텐서 [C, H, W] 형태, 값 범위 [-1, 1]
    """
    # 이미지 로드 및 RGB 변환
    image = Image.open(image_path).convert('RGB')
    
    # 크기 조정 (width, height)
    image = image.resize(target_size, Image.LANCZOS)
    
    # numpy 배열로 변환
    image = np.array(image)
    
    # [0, 255] -> [-1, 1] 범위로 정규화
    image = image.astype(np.float32) / 127.5 - 1.0
    
    # [H, W, C] -> [C, H, W] 변환
    image = np.transpose(image, (2, 0, 1))
    
    # torch tensor로 변환
    image = torch.from_numpy(image)
    
    return image


def resize_mask(m, shape):
    m = F.interpolate(m, shape)
    m[m > 0.5] = 1
    m[m < 0.5] = 0
    return m