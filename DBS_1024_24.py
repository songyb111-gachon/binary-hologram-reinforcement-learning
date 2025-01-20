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

import matplotlib.pyplot as plt

from env_1024_24 import BinaryHologramEnv

IPS = 1024  #이미지 픽셀 사이즈
CH = 24  #채널
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


model = BinaryNet(num_hologram=CH, in_planes=3, convReLU=False,
                  convBN=False, poolReLU=False, poolBN=False,
                  deconvReLU=False, deconvBN=False).cuda()
test = torch.randn(1, 3, IPS, IPS).cuda()
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
        target = tt.imread(self.target_list[idx], meta=self.meta)
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

import numpy as np
from collections import defaultdict

def optimize_with_random_pixel_flips(env, z=2e-3, pixel_pitch=7.56e-6):
    db_num = 800
    max_datasets = 800  # 최대 데이터셋 처리 개수
    output_bins = np.linspace(0, 1.0, 11)  # pre-model output 값의 범위 설정
    bin_counts = defaultdict(int)  # 각 범위에 해당하는 전체 픽셀 수
    improved_bin_counts = defaultdict(int)  # PSNR이 개선된 픽셀 수
    psnr_improvements = defaultdict(list)  # 각 범위에서 PSNR 개선량 저장

    while db_num <= max_datasets:
        try:
            obs, info = env.reset()
            db_num += 1
        except Exception as e:
            print(f"An error occurred during reset: {e}")
            break

        total_start_time = time.time()

        current_state = obs["state"]
        target_image = obs["target_image"]
        target_image_cuda = torch.tensor(target_image, dtype=torch.float32).cuda()
        initial_psnr = env.initial_psnr  # 초기 PSNR
        previous_psnr = initial_psnr
        steps = 0
        flip_count = 0
        psnr_after = 0

        meta = {'wl': (638e-9, 515e-9, 450e-9), 'dx': (pixel_pitch, pixel_pitch)}
        rmeta = {'wl': (638e-9), 'dx': (pixel_pitch, pixel_pitch)}
        gmeta = {'wl': (515e-9), 'dx': (pixel_pitch, pixel_pitch)}
        bmeta = {'wl': (450e-9), 'dx': (pixel_pitch, pixel_pitch)}

        rgbchannel = current_state.shape[1]

        rchannel = int(rgbchannel / 3)
        gchannel = int(rgbchannel * 2 / 3)

        red = current_state[:, :rchannel, :, :]
        green = current_state[:, rchannel:gchannel, :, :]
        blue = current_state[:, gchannel:, :, :]

        red = tt.Tensor(red, meta=rmeta)
        green = tt.Tensor(green, meta=gmeta)
        blue = tt.Tensor(blue, meta=bmeta)

        rsim = tt.simulate(red, z).abs() ** 2
        gsim = tt.simulate(green, z).abs() ** 2
        bsim = tt.simulate(blue, z).abs() ** 2

        rmean = torch.mean(rsim, dim=1, keepdim=True)
        gmean = torch.mean(gsim, dim=1, keepdim=True)
        bmean = torch.mean(bsim, dim=1, keepdim=True)

        rgb = torch.cat([rmean, gmean, bmean], dim=1)
        rgb = tt.Tensor(rgb, meta=meta)

        # Pre-model output 계산
        pre_model_output = obs["pre_model"].squeeze()  # 필요 시 차원 축소

        # 초기화: 특정 범위 값의 픽셀 개수와 PSNR 개선량 저장
        bin_counts = defaultdict(int)  # 각 범위에 해당하는 전체 픽셀 수
        improved_bin_counts = defaultdict(int)  # PSNR이 개선된 픽셀 수
        psnr_improvements = defaultdict(list)  # 각 범위에서 PSNR 개선량 저장

        # 각 범위 값의 픽셀 개수 계산
        for i in range(len(output_bins) - 1):
            bin_counts[i] = np.logical_and(
                pre_model_output >= output_bins[i],
                pre_model_output < output_bins[i + 1]
            ).sum()

        # 다음 출력 기준 PSNR 값 리스트 설정
        next_print_thresholds = [initial_psnr + i * 0.01 for i in range(1, 101)]  # 최대 10.0 상승까지 출력

        print(f"Starting pixel flip optimization for file {db_num}.png with initial PSNR: {initial_psnr:.6f}")

        # 픽셀 크기 정보 가져오기
        num_channels, img_height, img_width = current_state.shape[1:]
        all_pixels = np.arange(num_channels * img_height * img_width)
        np.random.shuffle(all_pixels)  # 랜덤 순서로 픽셀 섞기

        for attempt, pixel in enumerate(all_pixels):
            channel = pixel // (img_height * img_width)
            pixel_index = pixel % (img_height * img_width)
            row = pixel_index // img_width
            col = pixel_index % img_width

            # 현재 상태의 픽셀 값을 플립
            current_state[0, channel, row, col] = 1 - current_state[0, channel, row, col]
            steps += 1

            meta = {'wl': (638e-9, 515e-9, 450e-9), 'dx': (pixel_pitch, pixel_pitch)}
            rmeta = {'wl': (638e-9), 'dx': (pixel_pitch, pixel_pitch)}
            gmeta = {'wl': (515e-9), 'dx': (pixel_pitch, pixel_pitch)}
            bmeta = {'wl': (450e-9), 'dx': (pixel_pitch, pixel_pitch)}

            rgbchannel = current_state.shape[1]

            rchannel = int(rgbchannel / 3)
            gchannel = int(rgbchannel * 2 / 3)

            # 조건에 따라 연산 수행
            if channel < rchannel:
                # Red 채널 범위일 때, rmean만 갱신
                red = current_state[:, :rchannel, :, :]
                red = tt.Tensor(red, meta=rmeta)
                rsim = tt.simulate(red, z).abs() ** 2
                rmean = torch.mean(rsim, dim=1, keepdim=True)
                rgb = torch.cat([rmean, gmean, bmean], dim=1)
                rgb = tt.Tensor(rgb, meta=meta)
                psnr_after = tt.relativeLoss(rgb, target_image_cuda, tm.get_PSNR)  # 초기 PSNR 저장
                psnr_change = psnr_after - previous_psnr
                print(f"r_change: {psnr_change:.8f}, steps: {steps}, Flip Pixel: Channel={channel}, Row={row}, Col={col}")

            elif rchannel <= channel < gchannel:
                # Green 채널 범위일 때, gmean만 갱신
                green = current_state[:, rchannel:gchannel, :, :]
                green = tt.Tensor(green, meta=gmeta)
                gsim = tt.simulate(green, z).abs() ** 2
                gmean = torch.mean(gsim, dim=1, keepdim=True)
                rgb = torch.cat([rmean, gmean, bmean], dim=1)
                rgb = tt.Tensor(rgb, meta=meta)
                psnr_after = tt.relativeLoss(rgb, target_image_cuda, tm.get_PSNR)  # 초기 PSNR 저장
                psnr_change = psnr_after - previous_psnr
                print(f"g_change: {psnr_change:.8f}, steps: {steps}, Flip Pixel: Channel={channel}, Row={row}, Col={col}")

            elif gchannel <= channel:
                # Blue 채널 범위일 때, bmean만 갱신
                blue = current_state[:, gchannel:, :, :]
                blue = tt.Tensor(blue, meta=bmeta)
                bsim = tt.simulate(blue, z).abs() ** 2
                bmean = torch.mean(bsim, dim=1, keepdim=True)
                rgb = torch.cat([rmean, gmean, bmean], dim=1)
                rgb = tt.Tensor(rgb, meta=meta)
                psnr_after = tt.relativeLoss(rgb, target_image_cuda, tm.get_PSNR)  # 초기 PSNR 저장
                psnr_change = psnr_after - previous_psnr
                print(f"b_change: {psnr_change:.8f}, steps: {steps}, Flip Pixel: Channel={channel}, Row={row}, Col={col}")

            red = current_state[:, :rchannel, :, :]
            green = current_state[:, rchannel:gchannel, :, :]
            blue = current_state[:, gchannel:, :, :]

            red = tt.Tensor(red, meta=rmeta)
            green = tt.Tensor(green, meta=gmeta)
            blue = tt.Tensor(blue, meta=bmeta)

            rsim = tt.simulate(red, z).abs() ** 2
            gsim = tt.simulate(green, z).abs() ** 2
            bsim = tt.simulate(blue, z).abs() ** 2

            rmean = torch.mean(rsim, dim=1, keepdim=True)
            gmean = torch.mean(gsim, dim=1, keepdim=True)
            bmean = torch.mean(bsim, dim=1, keepdim=True)

            # RGB 결합
            rgb = torch.cat([rmean, gmean, bmean], dim=1)
            rgb = tt.Tensor(rgb, meta=meta)

            psnr_after = tt.relativeLoss(rgb, target_image_cuda, tm.get_PSNR)  # 초기 PSNR 저장

            psnr_change = psnr_after - previous_psnr
            print(f"rgb_change: {psnr_change:.8f}, steps: {steps}, Flip Pixel: Channel={channel}, Row={row}, Col={col}/n")

            # PSNR이 개선되었는지 확인
            if psnr_after > previous_psnr:
                flip_count += 1

                # 현재 PSNR 값이 출력 기준을 충족했는지 확인
                while next_print_thresholds and psnr_after >= next_print_thresholds[0]:
                    threshold = next_print_thresholds.pop(0)
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
                    total_improved_pixels = sum(improved_bin_counts.values())
                    for i in range(len(output_bins) - 1):
                        total_count = bin_counts[i]
                        improved_count = improved_bin_counts[i]
                        improved_ratio = improved_count / total_count if total_count > 0 else 0
                        range_improved_ratio = improved_count / total_improved_pixels if total_improved_pixels > 0 else 0
                        total_psnr_improvement = sum(psnr_improvements[i]) if improved_count > 0 else 0
                        avg_psnr_improvement = total_psnr_improvement / improved_count if improved_count > 0 else 0

                        print(f"Range {output_bins[i]:.1f}-{output_bins[i + 1]:.1f}: "
                              f"Total Pixels = {total_count}, Improved Pixels = {improved_count}, "
                              f"Improvement Ratio (in range) = {improved_ratio:.6f}, "
                              f"Improvement Ratio (to total improved) = {range_improved_ratio:.6f}, "
                              f"Total PSNR Improvement = {total_psnr_improvement:.6f}, "
                              f"Average PSNR Improvement = {avg_psnr_improvement:.8f}")

                    print("\n")

                # 플립 성공 픽셀의 pre-model output 값 확인
                pre_value = pre_model_output[channel, row, col]

                # 범위에 따른 카운트 증가
                for i in range(len(output_bins) - 1):
                    if output_bins[i] <= pre_value < output_bins[i + 1]:
                        bin_counts[i] += 1  # 해당 범위 픽셀 수 증가
                        if psnr_after > previous_psnr:
                            improved_bin_counts[i] += 1  # 개선된 픽셀 수 증가
                            psnr_improvements[i].append(psnr_after - previous_psnr)  # PSNR 개선량 저장
                        break

                previous_psnr = psnr_after

            else:
                # PSNR이 개선되지 않았으면 플립 롤백
                current_state[0, channel, row, col] = 1 - current_state[0, channel, row, col]

        # 성공 비율 계산
        success_ratio = flip_count / steps if steps > 0 else 0

        # 최종 결과 출력
        psnr_diff = psnr_after - initial_psnr
        data_processing_time = time.time() - total_start_time
        print(
            f"Step: {steps}"
            f"\nPSNR Before: {previous_psnr:.6f} | PSNR After: {psnr_after:.6f} | Change: {psnr_diff:.6f}"
            f"\nSuccess Ratio: {success_ratio:.6f} | Flip Count: {flip_count}"
            f"\nFlip Pixel: Channel={channel}, Row={row}, Col={col}"
            f"\nTime taken for this data: {data_processing_time:.2f} seconds"
        )
        print(f"{db_num}.png Optimization completed. Final PSNR improvement: {psnr_diff:.6f}")
        print(f"Time taken for this data: {data_processing_time:.2f} seconds\n")
        print("Pre-model output range statistics:")

        # Reconstructed RGB 이미지를 matplotlib에 맞게 정규화
        rgb_np = rgb.cpu().numpy().squeeze().transpose(1, 2, 0)  # (H, W, C)로 변환
        rgb_np = np.clip(rgb_np, 0, 1)  # 값이 [0, 1] 범위 내에 있도록 제한

        target_np = target_image.cpu().numpy().squeeze().transpose(1, 2, 0) # (H, W, C)로 변환
        if target_np.max() > 1.0:  # 만약 target 이미지 값이 [0, 255] 범위라면
            target_np = np.clip(target_np / 255.0, 0, 1)  # [0, 1]로 정규화

        # 이미지 출력 및 저장
        plt.figure(figsize=(12, 6))

        # Reconstructed RGB 이미지 출력
        plt.subplot(1, 2, 1)
        plt.imshow(rgb_np)  # Reconstructed RGB 이미지 출력
        plt.title("Reconstructed RGB Image")
        plt.axis("off")

        # Target RGB 이미지 출력
        plt.subplot(1, 2, 2)
        plt.imshow(target_np)  # Target RGB 이미지 출력
        plt.title("Target RGB Image")
        plt.axis("off")

        # 화면에 출력
        plt.show()

        # 이미지 저장
        save_path = f"output_images/episode_{db_num}_comparison.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 저장 디렉토리 생성
        plt.savefig(save_path, bbox_inches='tight')
        print(f"이미지가 저장되었습니다: {save_path}")

        total_improved_pixels = sum(improved_bin_counts.values())
        for i in range(len(output_bins) - 1):
            total_count = bin_counts[i]
            improved_count = improved_bin_counts[i]
            improved_ratio = improved_count / total_count if total_count > 0 else 0
            range_improved_ratio = improved_count / total_improved_pixels if total_improved_pixels > 0 else 0
            total_psnr_improvement = sum(psnr_improvements[i]) if improved_count > 0 else 0
            avg_psnr_improvement = total_psnr_improvement / improved_count if improved_count > 0 else 0

            print(f"Range {output_bins[i]:.1f}-{output_bins[i + 1]:.1f}: "
                  f"Total Pixels = {total_count}, Improved Pixels = {improved_count}, "
                  f"Improvement Ratio (in range) = {improved_ratio:.6f}, "
                  f"Improvement Ratio (to total improved) = {range_improved_ratio:.6f}, "
                  f"Total PSNR Improvement = {total_psnr_improvement:.6f}, "
                  f"Average PSNR Improvement = {avg_psnr_improvement:.8f}")

        print("\n")

batch_size = 1
#target_dir = 'dataset1/'
target_dir = '/nfs/dataset/DIV2K/DIV2K_train_HR/DIV2K_train_HR/'
valid_dir = '/nfs/dataset/DIV2K/DIV2K_valid_HR/DIV2K_valid_HR/'
meta = {'wl' : (638e-9, 515e-9, 450e-9), 'dx':(7.56e-6, 7.56e-6)}
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
model = BinaryNet(num_hologram=CH, in_planes=3, convReLU=False, convBN=False,
                  poolReLU=False, poolBN=False, deconvReLU=False, deconvBN=False).cuda()
model.load_state_dict(torch.load('result_v/2024-10-17 14_27_01.569309_1018_proposed_7.56e-6_24_0.002'))
model.eval()

# 환경 생성에 새로운 데이터 로더 적용
env = BinaryHologramEnv(
    target_function=model,
    trainloader=valid_loader,
)

optimize_with_random_pixel_flips(env)