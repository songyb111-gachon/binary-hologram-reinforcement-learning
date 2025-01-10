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

IPS = 256  #이미지 픽셀 사이즈
CH = 8  #채널
RW = 800  #보상

warnings.filterwarnings('ignore')

# 현재 날짜와 시간을 가져와 포맷 지정
current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# BinaryHologramEnv 클래스
class BinaryHologramEnv(gym.Env):
    def __init__(self, target_function, trainloader, max_steps=10000, T_PSNR=30, T_steps=1, T_PSNR_DIFF=0.1):
        super(BinaryHologramEnv, self).__init__()

        # 관찰 공간 정보
        self.observation_space = spaces.Dict({
            "state_record": spaces.Box(low=0, high=1, shape=(1, CH, IPS, IPS), dtype=np.int8), #이걸 기반으로 마스크를 해 말아
            "state": spaces.Box(low=0, high=1, shape=(1, CH, IPS, IPS), dtype=np.int8),
            "pre_model": spaces.Box(low=0, high=1, shape=(1, CH, IPS, IPS), dtype=np.float32),
            "recon_image": spaces.Box(low=0, high=1, shape=(1, IPS, IPS), dtype=np.float32),
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
        self.T_steps = T_steps
        self.T_PSNR_DIFF = T_PSNR_DIFF

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
            self.data_iter = iter(self.trainloader)
            self.target_image, self.current_file = next(self.data_iter)

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
        self.state_record = np.zeros_like(self.state)  # 플립 횟수를 저장하기 위한 배열 초기화

        binary = torch.tensor(self.state, dtype=torch.float32).cuda()  # (1, CH, IPS, IPS)
        binary = tt.Tensor(binary, meta={'dx': (7.56e-6, 7.56e-6), 'wl': 515e-9})  # meta 정보 포함

        # 시뮬레이션
        sim = tt.simulate(binary, z).abs()**2
        result = torch.mean(sim, dim=1, keepdim=True)

        # MSE 및 PSNR 계산
        mse = tt.relativeLoss(result, self.target_image, F.mse_loss).detach().cpu().numpy()
        self.initial_psnr = tt.relativeLoss(result, self.target_image, tm.get_PSNR)  # 초기 PSNR 저장
        self.previous_psnr = self.initial_psnr # 초기 PSNR 저장

        state_record = self.state_record
        state = self.state
        pre_model = self.observation
        target_image_np = self.target_image.cpu().numpy()
        result_np = result.cpu().numpy()

        obs = {"state_record": state_record, "state": state, "pre_model": pre_model, "recon_image": result_np, "target_image": target_image_np}

        current_time = datetime.now().strftime("%H:%M:%S")
        print(
            f"\033[92mInitial PSNR: {self.initial_psnr:.6f} | Time: {current_time}"
            f"\nInitial MSE: {mse:.6f}\033[0m"
        )

        return obs, {"state": self.state}

    def step(self, action, z=2e-3):
        psnr_before = self.previous_psnr

        # 행동을 기반으로 픽셀 좌표 계산
        channel = action // (IPS * IPS)
        pixel_index = action % (IPS * IPS)
        row = pixel_index // IPS
        col = pixel_index % IPS

        # 상태 변경
        pre_model_Value = self.observation[0, channel, row, col]
        self.state[0, channel, row, col] = 1 - self.state[0, channel, row, col]
        self.state_record[0, channel, row, col] = self.state_record[0, channel, row, col] + 1

        self.flip_count += 1  # 플립 증가

        # 시뮬레이션 수행
        binary_after = torch.tensor(self.state, dtype=torch.float32).cuda()
        binary_after = tt.Tensor(binary_after, meta={'dx': (7.56e-6, 7.56e-6), 'wl': 515e-9})
        sim_after = tt.simulate(binary_after, z).abs()**2
        result_after = torch.mean(sim_after, dim=1, keepdim=True)
        psnr_after = tt.relativeLoss(result_after, self.target_image, tm.get_PSNR)

        # 시뮬레이션 결과를 NumPy로 변환
        state_record = self.state_record
        state = self.state
        pre_model = self.observation
        target_image_np = self.target_image.cpu().numpy()
        result_np = result_after.cpu().numpy()

        obs = {"state_record": state_record, "state": state, "pre_model": pre_model, "recon_image": result_np, "target_image": target_image_np}
        # PSNR 변화량 계산
        psnr_change = psnr_after - psnr_before
        psnr_diff = psnr_after - self.initial_psnr

        # 보상 계산
        reward = psnr_change * RW  # PSNR 변화량(psnr_change)에 기반한 보상

        self.steps += 1

        # psnr_change가 음수인 경우 상태 롤백 수행
        if psnr_change < 0:

            self.state[0, channel, row, col] = 1 - self.state[0, channel, row, col]
            self.flip_count -= 1

            success_ratio = self.flip_count / self.steps if self.steps > 0 else 0

            # 출력 추가 (100 스텝마다 출력)
            if self.steps % 100 == 0:
                print(
                    f"Step: {self.steps:<6}"
                    f"\nPSNR Before: {psnr_before:.6f} | PSNR After: {psnr_after:.6f} | Change: {psnr_change:.6f} | Diff: {psnr_diff:.6f}"
                    f"\nReward: {reward:.2f} | Success Ratio: {success_ratio:.6f} | Flip Count: {self.flip_count}"
                    f"\nPre model Value: {pre_model_Value:.6f} | New State Value: {self.state[0, channel, row, col]}"
                    f"\nFlip Pixel: Channel={channel}, Row={row}, Col={col}"
                )

            # 실패 정보 생성
            info = {
                "psnr_before": psnr_before,
                "psnr_after": psnr_after,
                "psnr_change": psnr_change,
                "psnr_diff": psnr_diff,
                "pre_model_Value": pre_model_Value,
                "state_before": self.state.copy(),  # 행동 이전 상태
                "state_after": None,  # 실패한 경우에는 상태를 업데이트하지 않음
                "observation_before": self.observation.copy(),  # 행동 이전 관찰값
                "observation_after": obs,
                "failed_action": action,  # 실패한 행동
                "flip_count": self.flip_count,  # 현재까지의 플립 횟수
                "success_ratio": success_ratio,
                "reward": reward,
                "target_image": self.target_image.cpu().numpy(),  # 타겟 이미지
                "simulation_result": result_np,  # 현재 시뮬레이션 결과
                "step": self.steps,  # 현재 스텝
            }
            return obs, reward, False, False, info

        self.max_psnr_diff = max(self.max_psnr_diff, psnr_diff)  # 최고 PSNR_DIFF 업데이트

        success_ratio = self.flip_count / self.steps if self.steps > 0 else 0

        # 출력 추가 (100 스텝마다 출력)
        if self.steps % 100 == 0:
            print(
                f"Step: {self.steps:<6}"
                f"\nPSNR Before: {psnr_before:.6f} | PSNR After: {psnr_after:.6f} | Change: {psnr_change:.6f} | Diff: {psnr_diff:.6f}"
                f"\nReward: {reward:.2f} | Success Ratio: {success_ratio:.6f} | Flip Count: {self.flip_count}"
                f"\nPre-model Value: {pre_model_Value:.6f} | New State Value: {self.state[0, channel, row, col]}"
                f"\nFlip Pixel: Channel={channel}, Row={row}, Col={col}"
            )

        self.previous_psnr = psnr_after

        # 성공 종료 조건: PSNR >= T_PSNR 또는 PSNR_DIFF >= T_PSNR_DIFF
        terminated = self.steps >= self.max_steps or self.psnr_sustained_steps >= self.T_steps
        truncated = self.steps >= self.max_steps

        if psnr_diff >= self.T_PSNR_DIFF or (psnr_after >= self.T_PSNR and psnr_diff < 0.1):
            current_time = datetime.now().strftime("%H:%M:%S")
            print(
                f"Step: {self.steps:<6} | Time: {current_time}"
                f"\nPSNR Before: {psnr_before:.6f} | PSNR After: {psnr_after:.6f} | Change: {psnr_change:.6f} | Diff: {psnr_diff:.6f}"
                f"\nReward: {reward:.2f} | Success Ratio: {success_ratio:.6f} | Flip Count: {self.flip_count}"
                f"\nPre-model Value: {pre_model_Value:.6f} | New State Value: {self.state[0, channel, row, col]}"
                f"\nFlip Pixel: Channel={channel}, Row={row}, Col={col}"
                f"\n[Time Check] 에피소드{self.episode_num_count} 소요 시간: {time.time() - self.start_time:.6f} seconds"
            )
            self.psnr_sustained_steps += 1

            if self.psnr_sustained_steps >= self.T_steps and psnr_diff >= self.T_PSNR_DIFF:  # 성공 에피소드 조건
                # Goal-Reaching Reward or Penalty 함수
                # 1 = +300, 1/2 = +100, 1/4 = -100, 1/8 = -300
                reward += (
                    1828.57 * (success_ratio ** 3)
                    - 3733.33 * (success_ratio ** 2)
                    + 2800 * success_ratio
                    - 595.2
                )

        if self.steps >= self.max_steps:
            current_time = datetime.now().strftime("%H:%M:%S")
            print(
                f"Step: {self.steps:<6} | Time: {current_time}"
                f"\nPSNR Before: {psnr_before:.6f} | PSNR After: {psnr_after:.6f} | Change: {psnr_change:.6f} | Diff: {psnr_diff:.6f}"
                f"\nReward: {reward:.2f} | Success Ratio: {success_ratio:.6f} | Flip Count: {self.flip_count}"
                f"\nPre-model Value: {pre_model_Value:.6f} | New State Value: {self.state[0, channel, row, col]}"
                f"\nFlip Pixel: Channel={channel}, Row={row}, Col={col}"
                f"\n[Time Check] 에피소드{self.episode_num_count} 소요 시간: {time.time() - self.start_time:.6f} seconds"
            )
            # Goal-Reaching Reward or Penalty 함수
            # 1 = +300, 1/2 = +100, 1/4 = -100, 1/8 = -300
            reward += (
                    1828.57 * (success_ratio ** 3)
                    - 3733.33 * (success_ratio ** 2)
                    + 2800 * success_ratio
                    - 595.24
                )

        # 관찰값 업데이트
        info = {
            "psnr_before": psnr_before,
            "psnr_after": psnr_after,
            "psnr_change": psnr_change,
            "psnr_diff": psnr_diff,
            "pre_model_Value": pre_model_Value,
            "state_before": self.state.copy(),  # 행동 이전 상태
            "state_after": self.state.copy() if psnr_change >= 0 else None,  # 행동 성공 시 상태
            "observation_before": self.observation.copy(),  # 행동 이전 관찰값
            "observation_after": obs,
            "failed_action": action if psnr_change < 0 else None,  # 실패한 행동
            "flip_count": self.flip_count,  # 현재까지의 플립 횟수
            "success_ratio": success_ratio,
            "reward": reward,
            "target_image": self.target_image.cpu().numpy(),  # 타겟 이미지
            "simulation_result": result_np,  # 현재 시뮬레이션 결과
            "action_coords": (channel, row, col),  # 행동한 좌표
            "step": self.steps  # 현재 스텝
        }

        return obs, reward, terminated, truncated, info

# 최적화 함수
def optimize_with_random_pixel_flips(env, z=2e-3):
    total_start_time = time.time()
    db_num = 0
    max_datasets = 800  # 최대 데이터셋 처리 개수

    # 데이터셋 전체 순회
    while db_num <= max_datasets:
        try:
            obs, info = env.reset()  # 환경 초기화 (튜플 언패킹)
            db_num += 1
        except Exception as e:
            print(f"An error occurred during reset: {e}")
            break

        current_state = obs["state"]
        target_image = obs["target_image"]
        current_file = obs["current_file"]
        initial_psnr = env.initial_psnr  # 초기 PSNR
        previous_psnr = initial_psnr
        steps = 0
        flip_count = 0
        psnr_after = 0

        # 픽셀 크기 정보 가져오기
        num_channels, img_height, img_width = current_state.shape[1:]
        all_pixels = np.arange(num_channels * img_height * img_width)
        np.random.shuffle(all_pixels)  # 랜덤 순서로 픽셀 섞기

        print(f"Starting pixel flip optimization for file {current_file} with initial PSNR: {initial_psnr:.6f}")

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
                previous_psnr = psnr_after
                if steps % 1000 == 0:
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
            else:
                # PSNR이 개선되지 않았으면 플립 롤백
                current_state[0, channel, row, col] = 1 - current_state[0, channel, row, col]

        # 최종 결과 출력
        psnr_diff = psnr_after - initial_psnr
        data_processing_time = time.time() - total_start_time
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