import sys
import logging
from datetime import datetime
import os
from utils.logger import setup_logger

# 로거 설정
log_file = setup_logger()

# 테스트 출력
print("이 메시지는 콘솔과 파일에 동시에 기록됩니다.")
logging.info("이 메시지도 로그에 기록됩니다.")

import os
import glob
import shutil
from datetime import datetime
import time
import warnings

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import Dataset, DataLoader

import torchvision

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

import torchOptics.optics as tt
import torchOptics.metrics as tm

from env import BinaryHologramEnv

IPS = 256  #이미지 픽셀 사이즈
CH = 8  #채널
RW = 800  #보상

class BinaryNet(nn.Module):
    def __init__(self, num_hologram, final='Sigmoid', in_planes=3,
                 channels=[32, 64, 128, 256, 512, 1024, 2048, 4096],
                 convReLU=True, convBN=True, poolReLU=True, poolBN=True,
                 deconvReLU=True, deconvBN=True):
        super(BinaryNet, self).__init__()

        def CRB2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, relu=True, bn=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            if relu:
                layers += [nn.Tanh()]
            if bn:
                layers += [nn.BatchNorm2d(num_features=out_channels)]

            cbr = nn.Sequential(*layers)  # *으로 list unpacking

            return cbr

        def TRB2d(in_channels, out_channels, kernel_size=2, stride=2, bias=True, relu=True, bn=True):
            layers = []
            layers += [nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                          kernel_size=2, stride=2, padding=0,
                                          bias=True)]
            if bn:
                layers += [nn.BatchNorm2d(num_features=out_channels)]
            if relu:
                layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)  # *으로 list unpacking

            return cbr

        self.enc1_1 = CRB2d(in_planes, channels[0], relu=convReLU, bn=convBN)
        self.enc1_2 = CRB2d(channels[0], channels[0], relu=convReLU, bn=convBN)
        self.pool1 = CRB2d(channels[0], channels[0], stride=2, relu=poolReLU, bn=poolBN)

        self.enc2_1 = CRB2d(channels[0], channels[1], relu=convReLU, bn=convBN)
        self.enc2_2 = CRB2d(channels[1], channels[1], relu=convReLU, bn=convBN)
        self.pool2 = CRB2d(channels[1], channels[1], stride=2, relu=poolReLU, bn=poolBN)

        self.enc3_1 = CRB2d(channels[1], channels[2], relu=convReLU, bn=convBN)
        self.enc3_2 = CRB2d(channels[2], channels[2], relu=convReLU, bn=convBN)
        self.pool3 = CRB2d(channels[2], channels[2], stride=2, relu=poolReLU, bn=poolBN)

        self.enc4_1 = CRB2d(channels[2], channels[3], relu=convReLU, bn=convBN)
        self.enc4_2 = CRB2d(channels[3], channels[3], relu=convReLU, bn=convBN)
        self.pool4 = CRB2d(channels[3], channels[3], stride=2, relu=poolReLU, bn=poolBN)

        self.enc5_1 = CRB2d(channels[3], channels[4], relu=convReLU, bn=convBN)
        self.enc5_2 = CRB2d(channels[4], channels[4], relu=convReLU, bn=convBN)

        self.deconv4 = TRB2d(channels[4], channels[3], relu=deconvReLU, bn=deconvBN, stride=2)
        self.dec4_1 = CRB2d(channels[4], channels[3], relu=convReLU, bn=convBN)
        self.dec4_2 = CRB2d(channels[3], channels[3], relu=convReLU, bn=convBN)

        self.deconv3 = TRB2d(channels[3], channels[2], relu=deconvReLU, bn=deconvBN, stride=2)
        self.dec3_1 = CRB2d(channels[3], channels[2], relu=convReLU, bn=convBN)
        self.dec3_2 = CRB2d(channels[2], channels[2], relu=convReLU, bn=convBN)

        self.deconv2 = TRB2d(channels[2], channels[1], relu=deconvReLU, bn=deconvBN, stride=2)
        self.dec2_1 = CRB2d(channels[2], channels[1], relu=convReLU, bn=convBN)
        self.dec2_2 = CRB2d(channels[1], channels[1], relu=convReLU, bn=convBN)

        self.deconv1 = TRB2d(channels[1], channels[0], relu=deconvReLU, bn=deconvBN, stride=2)
        self.dec1_1 = CRB2d(channels[1], channels[0], relu=convReLU, bn=convBN)
        self.dec1_2 = CRB2d(channels[0], channels[0], relu=convReLU, bn=convBN)

        self.classifier = CRB2d(channels[0], num_hologram, relu=False, bn=False)

    def forward(self, x):
        # Encoder
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)
        enc5_2 = self.enc5_2(enc5_1)

        deconv4 = self.deconv4(enc5_2)
        concat4 = torch.cat((deconv4, enc4_2), dim=1)
        dec4_1 = self.dec4_1(concat4)
        dec4_2 = self.dec4_2(dec4_1)

        deconv3 = self.deconv3(dec4_2)
        concat3 = torch.cat((deconv3, enc3_2), dim=1)
        dec3_1 = self.dec3_1(concat3)
        dec3_2 = self.dec3_2(dec3_1)

        deconv2 = self.deconv2(dec3_2)
        concat2 = torch.cat((deconv2, enc2_2), dim=1)
        dec2_1 = self.dec2_1(concat2)
        dec2_2 = self.dec2_2(dec2_1)

        deconv1 = self.deconv1(dec2_2)
        concat1 = torch.cat((deconv1, enc1_2), dim=1)
        dec1_1 = self.dec1_1(concat1)
        dec1_2 = self.dec1_2(dec1_1)

        # Final classifier
        out = self.classifier(dec1_2)
        out = nn.Sigmoid()(out)
        return out


model = BinaryNet(num_hologram=CH, in_planes=1, convReLU=False,
                  convBN=False, poolReLU=False, poolBN=False,
                  deconvReLU=False, deconvBN=False).cuda()
test = torch.randn(1, 1, IPS, IPS).cuda()
out = model(test)
print(out.shape)

class Dataset512(Dataset):
    def __init__(self, target_dir, meta, transform=None, isTrain=True, padding=0):
        self.target_dir = target_dir
        self.transform = transform
        self.meta = meta
        self.isTrain = isTrain
        self.target_list = sorted(glob.glob(target_dir+'*.png'))
        self.center_crop = torchvision.transforms.CenterCrop(IPS)
        self.random_crop = torchvision.transforms.RandomCrop((IPS, IPS))
        self.padding = padding

    def __len__(self):
        return len(self.target_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        target = tt.imread(self.target_list[idx], meta=self.meta, gray=True).unsqueeze(0)
        if target.shape[-1] < IPS or target.shape[-2] < IPS:
            target = torchvision.transforms.Resize(IPS)(target)
        if self.isTrain:
            target = self.random_crop(target)
            target = torchvision.transforms.functional.pad(target, (self.padding, self.padding, self.padding, self.padding))
        else:
            target = self.center_crop(target)
            target = torchvision.transforms.functional.pad(target, (self.padding, self.padding, self.padding, self.padding))
        # 데이터와 파일 경로를 함께 반환
        return target, self.target_list[idx]


# 환경 정의
class BinaryHologramEnv(gym.Env):
    def __init__(self, target_function, trainloader, max_steps=100000000000000, T_PSNR=300):
        super(BinaryHologramEnv, self).__init__()

        self.observation_space = spaces.Dict({
            "state": spaces.Box(low=0, high=1, shape=(1, CH, IPS, IPS), dtype=np.int8),
            "pre_model": spaces.Box(low=0, high=1, shape=(1, CH, IPS, IPS), dtype=np.float32),
            "target_image": spaces.Box(low=0, high=1, shape=(1, IPS, IPS), dtype=np.float32),
        })

        # 행동 공간: 픽셀 하나를 선택하는 인덱스 (CH * IPS * IPS)
        self.num_pixels = CH * IPS * IPS
        self.action_space = spaces.Discrete(self.num_pixels)

        # 타겟 함수와 데이터 로더 설정
        self.target_function = target_function
        self.trainloader = trainloader

        # 환경 설정
        self.max_steps = max_steps
        self.T_PSNR = T_PSNR

        # 학습 상태 초기화
        self.state = None
        self.state_record = None
        self.observation = None
        self.steps = 0
        self.psnr_sustained_steps = 0
        self.flip_count = 0
        self.start_time = 0

        # 최고 PSNR_DIFF 추적 변수
        self.max_psnr_diff = float('-inf')  # 가장 높은 PSNR_DIFF를 추적

        # PSNR 저장 변수
        self.previous_psnr = None

        # 데이터 로더에서 첫 배치 설정
        self.data_iter = iter(self.trainloader)
        self.target_image = None

        # 에피소드 카운트
        self.episode_num_count = 0

    def reset(self, seed=None, options=None, z=2e-3):
        torch.cuda.empty_cache()

        self.episode_num_count += 1  # Increment episode count at the start of each reset

        self.start_time = time.time()

        # 이터레이터에서 다음 데이터를 가져옴
        try:
            self.target_image, self.current_file = next(self.data_iter)
        except StopIteration:
            # 데이터셋 끝에 도달하면 이터레이터를 다시 생성하고 처음부터 다시 시작
            print(f"\033[40;93m[INFO] Reached the end of dataset. Restarting from the beginning.\033[0m")
            raise SystemExit("Dataset exhausted. Program terminated.")

        print(f"\033[40;93m[Episode Start] Currently using dataset file: {self.current_file}, Episode count: {self.episode_num_count}\033[0m")

        self.target_image = self.target_image.cuda()

        with torch.no_grad():
            model_output = self.target_function(self.target_image)
        self.observation = model_output.cpu().numpy()  # (1, CH, IPS, IPS)

        # 매 에피소드마다 초기화
        self.max_psnr_diff = float('-inf')
        self.steps = 0
        self.flip_count = 0
        self.psnr_sustained_steps = 0

        self.state = (self.observation >= 0.5).astype(np.int8)  # 초기 Binary state

        binary = torch.tensor(self.state, dtype=torch.float32).cuda()  # (1, CH, IPS, IPS)
        binary = tt.Tensor(binary, meta={'dx': (7.56e-6, 7.56e-6), 'wl': 515e-9})  # meta 정보 포함

        # 시뮬레이션
        sim = tt.simulate(binary, z).abs()**2
        result = torch.mean(sim, dim=1, keepdim=True)

        # MSE 및 PSNR 계산
        mse = tt.relativeLoss(result, self.target_image, F.mse_loss).detach().cpu().numpy()
        self.initial_psnr = tt.relativeLoss(result, self.target_image, tm.get_PSNR)  # 초기 PSNR 저장
        self.previous_psnr = self.initial_psnr # 초기 PSNR 저장

        state = self.state
        pre_model = self.observation

        return {
            "state":state,
            "pre_model": pre_model,
            "target_image": self.target_image,
            "current_file": self.current_file,
        }

# 최적화 함수
def optimize_with_random_pixel_flips(env, z=2e-3):
    total_start_time = time.time()

    obs = env.reset()  # 환경 초기화
    current_state = obs["state"]
    pre_model = obs["pre_model"]
    target_image = obs["target_image"]
    current_file = obs["current_file"]
    initial_psnr = env.initial_psnr  # 초기 PSNR
    previous_psnr = initial_psnr
    steps = 0
    flip_count = 0
    psnr_after = 0

    # 다음 출력 기준 PSNR 값 리스트 설정 (0.5 단위로 증가)
    next_print_thresholds = [initial_psnr + i * 0.5 for i in range(1, 21)]  # 최대 10.0 상승까지 출력

    # 픽셀 크기 정보 가져오기
    num_channels, img_height, img_width = current_state.shape[1:]
    all_pixels = np.arange(num_channels * img_height * img_width)
    np.random.shuffle(all_pixels)  # 랜덤 순서로 픽셀 섞기

    print(f"Starting pixel flip optimization with initial PSNR: {initial_psnr:.6f}")

    # 모든 픽셀에 대해 한 번씩 시도
    for attempt, pixel in enumerate(all_pixels):
        channel = pixel // (img_height * img_width)
        pixel_index = pixel % (img_height * img_width)
        row = pixel_index // img_width
        col = pixel_index % img_width

        # 현재 상태의 픽셀 값을 플립
        current_state[0, channel, row, col] = 1 - current_state[0, channel, row, col]

        steps += 1

        # 시뮬레이션 수행
        binary_after = torch.tensor(current_state, dtype=torch.float32).cuda()
        binary_after = tt.Tensor(binary_after, meta={'dx': (7.56e-6, 7.56e-6), 'wl': 515e-9})
        sim_after = tt.simulate(binary_after, z).abs()**2
        result_after = torch.mean(sim_after, dim=1, keepdim=True)
        psnr_after = tt.relativeLoss(result_after, target_image, tm.get_PSNR)

        # PSNR이 개선되었는지 확인
        if psnr_after > previous_psnr:
            flip_count += 1
            # 현재 PSNR 값이 출력 기준을 충족했는지 확인
            while next_print_thresholds and psnr_after >= next_print_thresholds[0]:
                psnr_change = psnr_after - previous_psnr
                psnr_diff = psnr_after - initial_psnr
                success_ratio = flip_count / steps
                data_processing_time = time.time() - total_start_time

                print(
                    f"Step: {steps}"
                    f"\nPSNR Before: {previous_psnr:.6f} | PSNR After: {psnr_after:.6f} | Change: {psnr_change:.6f} | Diff: {psnr_diff:.6f}"
                    f"\nSuccess Ratio: {success_ratio:.6f} | Flip Count: {flip_count}"
                    f"\nFlip Pixel: Channel={channel}, Row={row}, Col={col}"
                    f"\nTime taken for this data: {data_processing_time:.2f} seconds"
                )

                # 출력된 기준값 제거
                next_print_thresholds.pop(0)

            previous_psnr = psnr_after
        else:
            # PSNR이 개선되지 않았으면 플립 롤백
            current_state[0, channel, row, col] = 1 - current_state[0, channel, row, col]

    # 최종 결과 출력
    psnr_diff = psnr_after - initial_psnr
    data_processing_time = time.time() - total_start_time
    print(
        f"Step: {steps}"
        f"\nPSNR Before: {previous_psnr:.6f} | PSNR After: {psnr_after:.6f} | Change: {psnr_change:.6f} | Diff: {psnr_diff:.6f}"
        f"\nSuccess Ratio: {success_ratio:.6f} | Flip Count: {flip_count}"
        f"\nFlip Pixel: Channel={channel}, Row={row}, Col={col}"
        f"\nTime taken for this data: {data_processing_time:.2f} seconds"
    )
    print(f"{current_file} Optimization completed. Final Diff: {psnr_diff:.6f}")
    print(f"Time taken for this data: {data_processing_time:.2f} seconds\n")


batch_size = 1
#target_dir = 'dataset1/'
target_dir = '/nfs/dataset/DIV2K/DIV2K_train_HR/DIV2K_train_HR/'
valid_dir = '/nfs/dataset/DIV2K/DIV2K_valid_HR/DIV2K_valid_HR/'
meta = {'wl': (515e-9), 'dx': (7.56e-6, 7.56e-6)}  # 메타 정보
padding = 0

# Dataset512 클래스 사용
train_dataset = Dataset512(target_dir=target_dir, meta=meta, isTrain=False, padding=padding) #센터크롭
#train_dataset = Dataset512(target_dir=target_dir, meta=meta, isTrain=True, padding=padding) #랜덤크롭
valid_dataset = Dataset512(target_dir=valid_dir, meta=meta, isTrain=False, padding=padding)

# DataLoader 생성
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
#train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# BinaryNet 모델 로드
model = BinaryNet(num_hologram=CH, in_planes=1, convReLU=False, convBN=False,
                  poolReLU=False, poolBN=False, deconvReLU=False, deconvBN=False).cuda()
model.load_state_dict(torch.load('result_v/2024-12-19 20:37:52.499731_pre_reinforce_8_0.002/2024-12-19 20:37:52.499731_pre_reinforce_8_0.002'))
model.eval()

# 환경 생성에 새로운 데이터 로더 적용
env = BinaryHologramEnv(
    target_function=model,
    trainloader=train_loader,
)

optimize_with_random_pixel_flips(env)