import os
import argparse
import torch
from os.path import join as opj
import datetime
from importlib import import_module
from omegaconf import OmegaConf

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, ConcatDataset

from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from utils import save_args


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="VITONHD")

    #필요에 따라 데이터셋 경로 수정
    #parser.add_argument("--data_root_dir", type=str, default="./data/zalando-hd-resized")
    parser.add_argument("--data_root_dir", type=str, default="./dataset")

    parser.add_argument("--category", type=str, default=None, choices=["upper", "lower_body", "dresses"])
    parser.add_argument("--vae_load_path", type=str, default="./ckpts/VITONHD_VAE_finetuning.ckpt")
    parser.add_argument("--batch_size", "-bs",  type=int, default=1)
    parser.add_argument("--transform_size", default=None, nargs="+", choices=["crop", "hflip", "shiftscale", "shiftscale2", "shiftscale3", "resize"])
    parser.add_argument("--transform_color", default=None, nargs="+", choices=["hsv", "bright_contrast", "colorjitter", "resize"])
    parser.add_argument("--use_atv_loss", action="store_true")
    parser.add_argument("--valid_epoch_freq", type=int, default=20)
    parser.add_argument("--save_every_n_epochs", type=int, default=20)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--save_root_dir", type=str, default="./logs")
    parser.add_argument("--save_name", type=str, default="dummy")

    parser.add_argument("--use_validation", action="store_false")
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--accum_iter", type=int, default=1)
    parser.add_argument("--img_H", type=int, default=512)
    parser.add_argument("--img_W", type=int, default=384)
    parser.add_argument("--logger_freq", type=int, default=5000)  # 성능 최적화: 1000 -> 5000
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--sd_unlocked", action="store_true")
    parser.add_argument("--all_unlocked", action="store_true")
    parser.add_argument("--only_mid_control", action="store_true")
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--num_sanity_val_steps", type=int, default=0)
    parser.add_argument("--pbe_train_mode", action="store_true")

    parser.add_argument("--lambda_simple", type=float, default=1.0)
    parser.add_argument("--control_scales", nargs="+", type=float, default=None)
    parser.add_argument("--imageclip_trainable", action="store_false")
    parser.add_argument("--no_strict_load", action="store_true", default=True)    
    
    args = parser.parse_args()
    
    # GPU 설정 및 CUDA 환경 확인
    import torch
    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    
    print(f"CUDA available: {cuda_available}")
    print(f"CUDA device count: {cuda_device_count}")
    
    if cuda_available and cuda_device_count > 0:
        # GPU가 실제로 사용 가능한 경우
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            n_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        else:
            n_gpus = cuda_device_count
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(cuda_device_count))
    else:
        # GPU가 사용 불가능한 경우
        n_gpus = 0
        print("GPU not available, will use CPU for training")
    
    args.config_path = opj("./configs", f"{args.config_name}.yaml")
    args.n_gpus = n_gpus
    args.devices = [i for i in range(args.n_gpus)] if n_gpus > 0 else [0]
    # strategy는 trainer에서 동적으로 설정
    args.sd_locked = not args.sd_unlocked
    args.no_validation = not args.use_validation
    
    args.valid_real_dir = opj(args.data_root_dir, "test", "image")
    args.save_dir = opj(args.save_root_dir, f"{datetime.datetime.now().strftime('%Y%m%d')}_" + args.save_name)
    args.img_save_dir = opj(args.save_dir, "images")
    args.model_save_dir = opj(args.save_dir, "models")
    args.tb_save_dir = opj(args.save_dir, "tb")
    args.valid_img_save_dir = opj(args.save_dir, "validation_sampled_images")
    args.args_save_path = opj(args.save_dir, "args.json")
    args.config_save_path = opj(args.save_dir, "config.yaml")
    os.makedirs(args.img_save_dir, exist_ok=True)
    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.tb_save_dir, exist_ok=True)
    os.makedirs(args.valid_img_save_dir, exist_ok=True)
    
    return args
def build_config(args, config_path=None):
    if config_path is None: 
        config_path = args.config_path
    config = OmegaConf.load(config_path)
    config.model.params.setdefault("use_VAEdownsample", False)
    config.model.params.setdefault("use_imageCLIP", False)
    config.model.params.setdefault("use_lastzc", False)
    config.model.params.setdefault("use_pbe_weight", False)
    if args is not None:
        for k, v in vars(args).items():
            config.model.params.setdefault(k, v)
    if not config.model.params.get("validation_config", None):
        config.model.params.validation_config = OmegaConf.create()
    config.model.params.validation_config.ddim_steps = config.model.params.validation_config.get("ddim_steps", 20)  # 성능 최적화: 50->20
    config.model.params.validation_config.eta = config.model.params.validation_config.get("eta", 0.0)
    config.model.params.validation_config.scale = config.model.params.validation_config.get("scale", 1.0)
    if args is not None:
        config.model.params.unet_config.params.use_atv_loss = args.use_atv_loss
        config.model.params.validation_config.img_save_dir = args.valid_img_save_dir
        config.model.params.validation_config.real_dir = args.valid_real_dir
        
        if args.use_atv_loss:
            config.model.params.use_attn_mask = True
    return config
    
def main_worker(args):
    config = build_config(args)
    OmegaConf.save(config, args.config_save_path)
    model = create_model(args.config_path, config=config).cpu()
    if args.resume_path is not None:
        if not args.no_strict_load:
            model.load_state_dict(load_state_dict(args.resume_path, location="cpu"))
        else:
            model.load_state_dict(load_state_dict(args.resume_path, location="cpu"), strict=False)
    elif config.resume_path is not None:
        if not args.no_strict_load:
            model.load_state_dict(load_state_dict(config.resume_path, location="cpu"))
        else:
            model.load_state_dict(load_state_dict(config.resume_path, location="cpu"), strict=False)
        
    # finetuned vae load
    if args.vae_load_path is not None:
        state_dict = load_state_dict(args.vae_load_path, location="cpu")
        new_state_dict = {}
        for k, v in state_dict.items():
            if "loss." not in k:
                new_state_dict[k] = v.clone()
        model.first_stage_model.load_state_dict(new_state_dict)

    model.learning_rate = args.learning_rate
    model.sd_locked = args.sd_locked
    model.only_mid_control = args.only_mid_control

    train_dataset = getattr(import_module("dataset"), config.dataset_name)(
        data_root_dir=args.data_root_dir, 
        img_H=args.img_H, 
        img_W=args.img_W, 
        transform_size=args.transform_size, 
        transform_color=args.transform_color, 
    )
    valid_paired_dataset = getattr(import_module("dataset"), config.dataset_name)(
        data_root_dir=args.data_root_dir, 
        img_H=args.img_H, 
        img_W=args.img_W, 
        is_test=True, 
        is_paired=True, 
        is_sorted=True, 
    )
    valid_unpaired_dataset = getattr(import_module("dataset"), config.dataset_name)(
        data_root_dir=args.data_root_dir, 
        img_H=args.img_H, 
        img_W=args.img_W, 
        is_test=True, 
        is_paired=False, 
        is_sorted=True, 
    )
      
    train_dataloader = DataLoader(
        train_dataset,
        num_workers=4, 
        batch_size=max(args.batch_size//args.n_gpus, 1), 
        shuffle=True, 
        pin_memory=True
    )
    valid_paired_dataloader = DataLoader(
        valid_paired_dataset, 
        num_workers=4, 
        batch_size=max(args.batch_size//args.n_gpus, 1), 
        shuffle=False, 
        pin_memory=True
    )
    valid_unpaired_dataloader = DataLoader(
        valid_unpaired_dataset, 
        num_workers=4, 
        batch_size=max(args.batch_size//args.n_gpus, 1), 
        shuffle=False, 
        pin_memory=True
    )
    
    #### trainer >>>>
    img_logger = ImageLogger(
        batch_frequency=args.logger_freq,
        max_images=2,  # 성능 최적화: 기본 4개에서 2개로 줄임
        save_dir=args.img_save_dir,
        log_images_kwargs=config.get("log_images_kwargs", None)
    )
    tb_logger = TensorBoardLogger(args.tb_save_dir)
    cp_callback = ModelCheckpoint(
        dirpath=args.model_save_dir, 
        filename="[Train]_[{epoch}]_[{train_loss_epoch:.04f}]", 
        save_top_k=-1, 
        every_n_epochs=args.save_every_n_epochs, 
        save_last=False, 
        save_on_train_epoch_end=True
    )

    # GPU/CPU 사용 결정
    if torch.cuda.is_available() and args.n_gpus > 0:
        accelerator = "gpu"
        devices = min(args.n_gpus, torch.cuda.device_count())  # 실제 사용 가능한 GPU 수로 제한
        if devices > 1:
            #strategy = "ddp"
            strategy = None
        else:
            strategy = None  # 단일 GPU일 때는 strategy를 None으로 설정
        print(f"Using GPU acceleration with {devices} GPU(s)")
    else:
        accelerator = "cpu"
        devices = 1
        strategy = None  # CPU일 때도 strategy를 None으로 설정
        print("Using CPU for training")

    trainer = pl.Trainer(
        precision=args.precision, 
        callbacks=[img_logger, cp_callback], 
        logger=tb_logger, 
        devices=devices,
        accelerator=accelerator, 
        strategy=strategy, 
        max_epochs=args.max_epochs, 
        accumulate_grad_batches=args.accum_iter, 
        check_val_every_n_epoch=args.valid_epoch_freq,
        num_sanity_val_steps=args.num_sanity_val_steps
    )
    #### trainer <<<<
    
    if not args.no_validation:
        trainer.fit(model, train_dataloader, [valid_paired_dataloader, valid_unpaired_dataloader])
    else:
        trainer.fit(model, train_dataloader)

if __name__ == "__main__":
    args = build_args()
    print(args)
    save_args(args, args.args_save_path)
    main_worker(args)
    print("Done")