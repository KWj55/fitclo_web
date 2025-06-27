import os
from os.path import join as opj
from omegaconf import OmegaConf
from importlib import import_module
import argparse

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from cldm.plms_hacked import PLMSSampler
from cldm.model import create_model
from utils import tensor2img

def build_args():
    # 스크립트 실행 시 필요한 인자들을 정의하고 파싱합니다.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default = "./configs/VITONHD.yaml")
    parser.add_argument("--model_load_path", type=str, default = "./ckpts/VITONHD.ckpt")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--data_root_dir", type=str, default="./dataset")
    parser.add_argument("--repaint", action="store_true")
    parser.add_argument("--unpair", action="store_true")
    parser.add_argument("--save_dir", type=str, default="./samples")

    parser.add_argument("--denoise_steps", type=int, default=50)
    parser.add_argument("--img_H", type=int, default=512)
    parser.add_argument("--img_W", type=int, default=384)
    parser.add_argument("--eta", type=float, default=0.0)
    args = parser.parse_args()
    return args


@torch.no_grad()
def main(args):
    # 인자로 입력받은 변수들을 지역 변수로 할당합니다.
    batch_size = args.batch_size
    img_H = args.img_H
    img_W = args.img_W

    # 설정 파일(YAML)을 로드하고, 이미지 크기 정보를 업데이트합니다.
    config = OmegaConf.load(args.config_path)
    config.model.params.img_H = args.img_H
    config.model.params.img_W = args.img_W
    params = config.model.params # 모델 파라미터들을 쉽게 접근하기 위해 변수에 저장

    # 모델 생성 및 체크포인트 로드
    # 1. 설정 파일(config)을 기반으로 모델 구조를 생성합니다.
    model = create_model(config_path=None, config=config)
    # 2. 지정된 경로(args.model_load_path)에서 체크포인트 파일을 로드합니다. map_location="cpu"는 GPU 메모리 부족 시 CPU로 먼저 로드하기 위함입니다.
    load_cp = torch.load(args.model_load_path, map_location="cpu")
    # 3. 체크포인트 파일이 딕셔너리 형태이고 'state_dict' 키를 포함하면 해당 값을 사용하고, 그렇지 않으면 로드된 전체 딕셔너리를 사용합니다.
    #    (PyTorch Lightning 등으로 학습된 모델은 종종 'state_dict' 키 아래에 모델 가중치를 저장합니다.)
    load_cp = load_cp["state_dict"] if "state_dict" in load_cp.keys() else load_cp
    # 4. 생성된 모델 구조에 로드된 가중치를 적용합니다.
    model.load_state_dict(load_cp)
    # 5. 모델을 사용 가능한 GPU로 이동시킵니다.
    model = model.cuda()
    # 6. 모델을 평가 모드(evaluation mode)로 설정합니다. 이는 Dropout이나 BatchNorm과 같은 레이어들의 동작을 추론에 맞게 변경합니다.
    model.eval()

    # PLMS 샘플러를 초기화합니다. 이 샘플러는 확산 모델의 노이즈 제거 과정을 수행합니다.
    sampler = PLMSSampler(model)
    # 데이터셋을 로드합니다. config 파일에 정의된 dataset_name을 사용하여 dataset.py에서 해당 클래스를 동적으로 가져옵니다.
    dataset = getattr(import_module("dataset"), config.dataset_name)(
        data_root_dir=args.data_root_dir, # 데이터셋 루트 디렉토리
        img_H=img_H,                      # 이미지 높이
        img_W=img_W,                      # 이미지 너비
        is_paired=not args.unpair,        # 이미지 쌍으로 구성된 데이터인지 여부 (unpair 플래그의 반대)
        is_test=True,                     # 테스트 모드인지 여부
        is_sorted=True                    # 데이터를 정렬할지 여부
    )
    # DataLoader를 생성하여 배치 단위로 데이터를 효율적으로 로드할 수 있도록 합니다.
    dataloader = DataLoader(dataset, num_workers=4, shuffle=False, batch_size=batch_size, pin_memory=True)

    # 생성될 이미지의 잠재 공간(latent space)에서의 형태(shape)를 정의합니다.
    # 일반적으로 VAE의 다운샘플링 팩터(예: 8)로 원본 이미지 크기를 나눈 값입니다. 채널 수는 4로 고정 (Stable Diffusion VAE 기준).
    shape = (4, img_H//8, img_W//8) 
    # 결과 이미지를 저장할 디렉토리를 설정합니다. unpair 모드 여부에 따라 하위 디렉토리를 구분합니다.
    save_dir = opj(args.save_dir, "unpair" if args.unpair else "pair")
    os.makedirs(save_dir, exist_ok=True)

    # 데이터로더를 순회하며 각 배치에 대해 추론을 수행합니다.
    for batch_idx, batch in enumerate(dataloader):
        print(f"{batch_idx}/{len(dataloader)}")
        # 모델의 get_input 함수를 사용하여 배치 데이터로부터 잠재 벡터(z)와 컨디셔닝 정보(c)를 추출합니다.
        z, c = model.get_input(batch, params.first_stage_key)
        bs = z.shape[0] # 현재 배치의 크기
        c_crossattn = c["c_crossattn"][0][:bs] # 크로스 어텐션에 사용될 컨디셔닝 정보
        if c_crossattn.ndim == 4: # 컨디셔닝 정보가 이미지 형태(4차원 텐서)일 경우
            # 모델의 get_learned_conditioning 함수를 통해 학습된 컨디셔닝 표현으로 변환합니다.
            c_crossattn = model.get_learned_conditioning(c_crossattn)
            c["c_crossattn"] = [c_crossattn]
        # 조건 없는(unconditional) 컨디셔닝을 생성합니다. 이는 Classifier-Free Guidance 기법에 사용됩니다.
        uc_cross = model.get_unconditional_conditioning(bs)
        uc_full = {"c_concat": c["c_concat"], "c_crossattn": [uc_cross]}
        uc_full["first_stage_cond"] = c["first_stage_cond"]
        # 배치 내의 모든 텐서들을 GPU로 이동시킵니다.
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.cuda()
        sampler.model.batch = batch

        ts = torch.full((1,), 999, device=z.device, dtype=torch.long)
        start_code = model.q_sample(z, ts)     
        # PLMS 샘플러를 사용하여 노이즈 제거 과정을 수행하여 이미지를 생성합니다.
        samples, _, _ = sampler.sample(
            args.denoise_steps, # 노이즈 제거 스텝 수
            bs,                 # 배치 크기
            shape,              # 생성될 잠재 공간의 형태
            c,                  # 컨디셔닝 정보 (예: 옷, 포즈 등)
            x_T=start_code,     # 샘플링을 시작할 초기 노이즈 코드
            verbose=False,      # 상세 로그 출력 여부
            eta=args.eta,       # DDIM 샘플링 시 사용되는 파라미터 (0이면 DDIM, 1이면 DDPM과 유사)
            unconditional_conditioning=uc_full, # Classifier-Free Guidance를 위한 조건 없는 컨디셔닝
        )

        # 생성된 잠재 공간의 샘플(samples)을 VAE 디코더를 통해 실제 이미지로 복원합니다.
        x_samples = model.decode_first_stage(samples)
        # 배치 내 각 생성된 이미지에 대해 후처리 및 저장을 수행합니다.
        for sample_idx, (x_sample, fn,  cloth_fn) in enumerate(zip(x_samples, batch['img_fn'], batch["cloth_fn"])):
            # 텐서를 이미지로 변환합니다 (값 범위: [0, 255], 타입: uint8).
            x_sample_img = tensor2img(x_sample)  # [0, 255]
            if args.repaint: # repaint 옵션이 활성화된 경우, 특정 영역을 원본 이미지로 복원합니다.
                # 원본 사람 이미지 (agnostic 이미지, 즉 옷이 없는 사람 이미지)
                repaint_agn_img = np.uint8((batch["image"][sample_idx].cpu().numpy()+1)/2 * 255)   # [0,255]
                # 옷이 없는 영역을 나타내는 마스크
                repaint_agn_mask_img = batch["agn_mask"][sample_idx].cpu().numpy()  # 0 or 1
                # 생성된 이미지에서 옷이 없는 영역은 원본 사람 이미지로, 옷 영역은 생성된 이미지로 합성합니다.
                x_sample_img = repaint_agn_img * repaint_agn_mask_img + x_sample_img * (1-repaint_agn_mask_img)
                x_sample_img = np.uint8(x_sample_img)
            # 최종 결과 이미지를 저장할 경로를 설정합니다.
            to_path = opj(save_dir, f"{fn.split('.')[0]}_{cloth_fn.split('.')[0]}.jpg")
            # 이미지를 파일로 저장합니다. OpenCV는 BGR 순서로 저장하므로, RGB 이미지를 BGR로 변환 (x_sample_img[:,:,::-1])하여 저장합니다.
            cv2.imwrite(to_path, x_sample_img[:,:,::-1])

if __name__ == "__main__":
    # 스크립트 실행 시 인자를 파싱합니다.
    args = build_args()
    # 메인 함수를 실행합니다.
    main(args)
