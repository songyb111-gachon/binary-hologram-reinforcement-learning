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
from collections import defaultdict

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
            "state_record": spaces.Box(low=0, high=1, shape=(1, CH, IPS, IPS), dtype=np.int8),
            "state": spaces.Box(low=0, high=1, shape=(1, CH, IPS, IPS), dtype=np.int8),
            "pre_model": spaces.Box(low=0, high=1, shape=(1, CH, IPS, IPS), dtype=np.float32),
            "recon_image": spaces.Box(low=0, high=1, shape=(1, 1, IPS, IPS), dtype=np.float32),
            "target_image": spaces.Box(low=0, high=1, shape=(1, 1, IPS, IPS), dtype=np.float32),
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
        self.steps = None
        self.psnr_sustained_steps = None
        self.flip_count = None
        self.start_time = None
        self.next_print_thresholds = 0
        self.total_start_time = None
        self.target_image_np = None
        self.initial_psnr = None

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
        self.target_image_np = self.target_image.cpu().numpy()

        with torch.no_grad():
            model_output = self.target_function(self.target_image)
        self.observation = model_output.cpu().numpy()  # (1, CH, IPS, IPS)

        # 매 에피소드마다 초기화
        self.max_psnr_diff = float('-inf')
        self.steps = 0
        self.flip_count = 0
        self.psnr_sustained_steps = 0
        self.next_print_thresholds = 0

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

        obs = {"state_record": self.state_record,
               "state": self.state,
               "pre_model": self.observation,
               "recon_image": result.cpu().numpy(),
               "target_image": self.target_image_np,
               }

        print(
            f"\033[92mInitial PSNR: {self.initial_psnr:.6f}\033[0m"
            f"\nInitial MSE: {mse:.6f}\033[0m"
        )

        # 다음 출력 기준 PSNR 값 리스트 설정 (0.01 단위로 증가)
        self.next_print_thresholds = [self.initial_psnr + i * 0.5 for i in range(1, 21)]  # 최대 0.1 상승까지 출력

        self.output_bins = np.linspace(0, 1.0, 11)  # pre-model output 값의 범위 설정

        # Pre-model output 계산
        self.pre_model_output = self.observation.squeeze()  # 필요 시 차원 축소

        # 초기화: 특정 범위 값의 픽셀 개수와 PSNR 개선량 저장
        self.bin_counts = defaultdict(int)  # 각 범위에 해당하는 전체 픽셀 수
        self.improved_bin_counts = defaultdict(int)  # PSNR이 개선된 픽셀 수
        self.psnr_improvements = defaultdict(list)  # 각 범위에서 PSNR 개선량 저장

        # 각 범위 값의 픽셀 개수 계산
        for i in range(len(self.output_bins) - 1):
            self.bin_counts[i] = np.logical_and(
                self.pre_model_output >= self.output_bins[i],
                self.pre_model_output < self.output_bins[i + 1]
            ).sum()

        self.total_start_time = time.time()

        return obs, {"state": self.state}

    def step(self, action, z=2e-3):
        self.steps += 1

        # 행동을 기반으로 픽셀 좌표 계산
        channel = action // (IPS * IPS)
        pixel_index = action % (IPS * IPS)
        row = pixel_index // IPS
        col = pixel_index % IPS

        # 상태 변경
        self.state[0, channel, row, col] = 1 - self.state[0, channel, row, col]
        self.state_record[0, channel, row, col] = self.state_record[0, channel, row, col] + 1

        self.flip_count += 1  # 플립 증가

        # 시뮬레이션 수행
        binary_after = torch.tensor(self.state, dtype=torch.float32).cuda()
        binary_after = tt.Tensor(binary_after, meta={'dx': (7.56e-6, 7.56e-6), 'wl': 515e-9})
        sim_after = tt.simulate(binary_after, z).abs()**2
        result_after = torch.mean(sim_after, dim=1, keepdim=True)
        psnr_after = tt.relativeLoss(result_after, self.target_image, tm.get_PSNR)

        obs = {"state_record": self.state_record,
               "state": self.state,
               "pre_model": self.observation,
               "recon_image": result_after.cpu().numpy(),
               "target_image": self.target_image_np,
               }

        # PSNR 변화량 계산
        psnr_change = psnr_after - self.previous_psnr
        psnr_diff = psnr_after - self.initial_psnr

        # 보상 계산
        reward = psnr_change * RW  # PSNR 변화량(psnr_change)에 기반한 보상

        # psnr_change가 음수인 경우 상태 롤백 수행
        if psnr_change < 0:

            self.state[0, channel, row, col] = 1 - self.state[0, channel, row, col]
            self.flip_count -= 1

            return obs, reward, False, False, {}

        self.max_psnr_diff = max(self.max_psnr_diff, psnr_diff)  # 최고 PSNR_DIFF 업데이트

        success_ratio = self.flip_count / self.steps if self.steps > 0 else 0

        # 출력 추가 (0.01 PSNR 상승마다 출력)
        while self.next_print_thresholds and psnr_after >= self.next_print_thresholds[0]:
            self.next_print_thresholds.pop(0)
            data_processing_time = time.time() - self.total_start_time
            print(
                f"Step: {self.steps:<6} | Initial PSNR: {self.initial_psnr:.6f}"
                f"\nPSNR After: {psnr_after:.6f} | Change: {psnr_change:.6f} | Diff: {psnr_diff:.6f}"
                f"\nReward: {reward:.2f} | Success Ratio: {success_ratio:.6f} | Flip Count: {self.flip_count}"
                f"\nFlip Pixel: Channel={channel}, Row={row}, Col={col}"
                f"\nTime taken for this data: {data_processing_time:.2f} seconds"
            )

        # 플립 성공 픽셀의 pre-model output 값 확인
        pre_value = self.pre_model_output[channel, row, col]

        # 범위에 따른 카운트 증가
        for i in range(len(self.output_bins) - 1):
            if self.output_bins[i] <= pre_value < self.output_bins[i + 1]:
                self.bin_counts[i] += 1  # 해당 범위 픽셀 수 증가
                if psnr_after > self.previous_psnr:
                    self.improved_bin_counts[i] += 1  # 개선된 픽셀 수 증가
                    self.psnr_improvements[i].append(psnr_after - self.previous_psnr)  # PSNR 개선량 저장
                break

        self.previous_psnr = psnr_after

        if psnr_diff >= self.T_PSNR_DIFF or (psnr_after >= self.T_PSNR and psnr_diff < 0.1):
            data_processing_time = time.time() - self.total_start_time
            print(
                f"Step: {self.steps:<6} | Initial PSNR: {self.initial_psnr:.6f}"
                f"\nPSNR After: {psnr_after:.6f} | Change: {psnr_change:.6f} | Diff: {psnr_diff:.6f}"
                f"\nReward: {reward:.2f} | Success Ratio: {success_ratio:.6f} | Flip Count: {self.flip_count}"
                f"\nFlip Pixel: Channel={channel}, Row={row}, Col={col}"
                f"\nTime taken for this data: {data_processing_time:.2f} seconds"
            )

            total_improved_pixels = sum(self.improved_bin_counts.values())

            for i in range(len(self.output_bins) - 1):
                total_count = self.bin_counts[i]
                improved_count = self.improved_bin_counts[i]
                improved_ratio = improved_count / total_count if total_count > 0 else 0
                range_improved_ratio = improved_count / total_improved_pixels if total_improved_pixels > 0 else 0
                total_psnr_improvement = sum(self.psnr_improvements[i]) if improved_count > 0 else 0
                avg_psnr_improvement = total_psnr_improvement / improved_count if improved_count > 0 else 0

                print(f"Range {self.output_bins[i]:.1f}-{self.output_bins[i + 1]:.1f}: "
                      f"Total Pixels = {total_count}, Improved Pixels = {improved_count}, "
                      f"Improvement Ratio (in range) = {improved_ratio:.6f}, "
                      f"Improvement Ratio (to total improved) = {range_improved_ratio:.6f}, "
                      f"Total PSNR Improvement = {total_psnr_improvement:.6f}, "
                      f"Average PSNR Improvement = {avg_psnr_improvement:.6f}")

            print("\n")

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
            # 현재 PSNR 값이 출력 기준을 충족했는지 확인
            data_processing_time = time.time() - self.total_start_time
            print(
                f"Step: {self.steps:<6} | Initial PSNR: {self.initial_psnr:.6f}"
                f"\nPSNR After: {psnr_after:.6f} | Change: {psnr_change:.6f} | Diff: {psnr_diff:.6f}"
                f"\nReward: {reward:.2f} | Success Ratio: {success_ratio:.6f} | Flip Count: {self.flip_count}"
                f"\nFlip Pixel: Channel={channel}, Row={row}, Col={col}"
                f"\nTime taken for this data: {data_processing_time:.2f} seconds"
            )
            # Goal-Reaching Reward or Penalty 함수
            # 1 = +300, 1/2 = +100, 1/4 = -100, 1/8 = -300
            reward += (
                    1828.57 * (success_ratio ** 3)
                    - 3733.33 * (success_ratio ** 2)
                    + 2800 * success_ratio
                    - 595.24
                )

        # 성공 종료 조건: PSNR >= T_PSNR 또는 PSNR_DIFF >= T_PSNR_DIFF
        terminated = self.steps >= self.max_steps or self.psnr_sustained_steps >= self.T_steps
        truncated = self.steps >= self.max_steps

        return obs, reward, terminated, truncated, {}  # 빈 딕셔너리 반환