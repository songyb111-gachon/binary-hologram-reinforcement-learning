import logging
from utils.logger import setup_logger

# 로거 설정
log_file = setup_logger()

# 테스트 출력
print("이 메시지는 콘솔과 파일에 동시에 기록됩니다.")
logging.info("이 메시지도 로그에 기록됩니다.")

import os
import glob
import time

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader

import torchvision

import torchOptics.optics as tt
import torchOptics.metrics as tm
from env_1024_24_128 import BinaryHologramEnv

IPS = 1024  #이미지 픽셀 사이즈
CH = 24  #채널

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

def optimize_with_random_pixel_flips(env, z=2e-3, pixel_pitch=7.56e-6, crop_margin=64):
    db_num = 0
    max_datasets = 1  # 최대 데이터셋 처리 개수
    output_bins = np.round(np.linspace(0, 1.0, 11), decimals=10)
    print(output_bins) # pre-model output 값의 범위 설정

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

        print(current_state.shape)
        print(target_image.shape)
        print(target_image_cuda.shape)
        # 64픽셀 크롭 적용
        cropped_state = current_state[:, :, crop_margin:-crop_margin, crop_margin:-crop_margin]
        print(cropped_state.shape)
        cropped_target_image = target_image[:, :, crop_margin:-crop_margin, crop_margin:-crop_margin]
        print(cropped_target_image.shape)
        cropped_target_image_cuda = torch.tensor(cropped_target_image, dtype=torch.float32).cuda()
        print(cropped_target_image_cuda.shape)

        initial_psnr = env.initial_psnr  # 초기 PSNR
        previous_psnr = initial_psnr
        steps = 0
        flip_count = 0
        psnr_after = 0

        meta = {'wl': (638e-9, 515e-9, 450e-9), 'dx': (pixel_pitch, pixel_pitch)}
        rmeta = {'wl': (638e-9), 'dx': (pixel_pitch, pixel_pitch)}
        gmeta = {'wl': (515e-9), 'dx': (pixel_pitch, pixel_pitch)}
        bmeta = {'wl': (450e-9), 'dx': (pixel_pitch, pixel_pitch)}

        rgbchannel = cropped_state.shape[1]

        rchannel = int(rgbchannel / 3)
        gchannel = int(rgbchannel * 2 / 3)

        red = cropped_state[:, :rchannel, :, :]
        green = cropped_state[:, rchannel:gchannel, :, :]
        blue = cropped_state[:, gchannel:, :, :]

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
        print(pre_model_output.shape)
        cropped_pre_model_output = pre_model_output[:, crop_margin:-crop_margin, crop_margin:-crop_margin]
        print(cropped_pre_model_output.shape)

        # 초기화: 특정 범위 값의 픽셀 개수와 PSNR 개선량 저장
        bin_counts = defaultdict(int)  # 각 범위에 해당하는 전체 픽셀 수
        improved_bin_counts = defaultdict(int)  # PSNR이 개선된 픽셀 수
        psnr_improvements = defaultdict(list)  # 각 범위에서 PSNR 개선량 저장

        # 저장 경로 설정
        dbs_folder = "DBS"
        os.makedirs(dbs_folder, exist_ok=True)  # 폴더가 없으면 생성

        a, imgname = next(iter(env.trainloader))

        # imgname에서 파일 이름 추출
        if isinstance(imgname, list) or isinstance(imgname, tuple):
            imgname = imgname[0]  # imgname이 리스트나 튜플인 경우 첫 번째 요소 선택

        # 파일 이름만 추출
        file_name = os.path.basename(imgname)  # 경로에서 파일 이름 추출
        file_name = os.path.splitext(file_name)[0]  # 확장자 제거 (ex: "0814")

        # Reconstructed RGB 이미지를 numpy 배열로 변환 및 저장
        rgb_np = rgb.cpu().numpy()  # RGB 데이터를 NumPy 배열로 변환
        save_path_rgb = os.path.join(dbs_folder, f"episode_{file_name}png_rgb_before.npy")  # 저장 경로 설정

        # NumPy 배열로 저장
        np.save(save_path_rgb, rgb_np)
        print(f"RGB data saved to {save_path_rgb}")

        # 각 범위 값의 픽셀 개수 계산
        for i in range(len(output_bins) - 1):
            if i == len(output_bins) - 2:  # 마지막 범위
                bin_counts[i] = np.logical_and(
                    cropped_pre_model_output >= output_bins[i],
                    cropped_pre_model_output <= output_bins[i + 1]
                ).sum()
                print("i")
                print(output_bins[i])
            else:
                bin_counts[i] = np.logical_and(
                    cropped_pre_model_output >= output_bins[i],
                    cropped_pre_model_output < output_bins[i + 1]
                ).sum()
                print("e")
                print(output_bins[i])

        print(f"Starting pixel flip optimization for file {file_name}.png with initial PSNR: {initial_psnr:.6f}")

        # 픽셀 크기 정보 가져오기
        num_channels, img_height, img_width = cropped_state.shape[1:]
        all_pixels = np.arange(num_channels * img_height * img_width)
        np.random.shuffle(all_pixels)  # 랜덤 순서로 픽셀 섞기

        for attempt, pixel in enumerate(all_pixels):
            channel = pixel // (img_height * img_width)
            pixel_index = pixel % (img_height * img_width)
            row = pixel_index // img_width
            col = pixel_index % img_width

            # 현재 상태의 픽셀 값을 플립
            cropped_state[0, channel, row, col] = 1 - cropped_state[0, channel, row, col]
            steps += 1

            # 조건에 따라 연산 수행
            if channel < 8:
                # Red 채널 범위일 때, rmean만 갱신
                red_after = cropped_state[:, :rchannel, :, :]
                red_after = tt.Tensor(red_after, meta=rmeta)
                rsim_after = tt.simulate(red_after, z).abs() ** 2
                rmean_after = torch.mean(rsim_after, dim=1, keepdim=True)
                rgb_after = torch.cat([rmean_after, gmean, bmean], dim=1)
                rgb_after = tt.Tensor(rgb_after, meta=meta)
                psnr_after = tt.relativeLoss(rgb_after, cropped_target_image_cuda, tm.get_PSNR)  # 초기 PSNR 저장

            elif 8 <= channel < 16:
                # Green 채널 범위일 때, gmean만 갱신
                green_after = cropped_state[:, rchannel:gchannel, :, :]
                green_after = tt.Tensor(green_after, meta=gmeta)
                gsim_after = tt.simulate(green_after, z).abs() ** 2
                gmean_after = torch.mean(gsim_after, dim=1, keepdim=True)
                rgb_after = torch.cat([rmean, gmean_after, bmean], dim=1)
                rgb_after = tt.Tensor(rgb_after, meta=meta)
                psnr_after = tt.relativeLoss(rgb_after, cropped_target_image_cuda, tm.get_PSNR)  # 초기 PSNR 저장

            elif 16 <= channel:
                # Blue 채널 범위일 때, bmean만 갱신
                blue_after = cropped_state[:, gchannel:, :, :]
                blue_after = tt.Tensor(blue_after, meta=bmeta)
                bsim_after = tt.simulate(blue_after, z).abs() ** 2
                bmean_after = torch.mean(bsim_after, dim=1, keepdim=True)
                rgb_after = torch.cat([rmean, gmean, bmean_after], dim=1)
                rgb_after = tt.Tensor(rgb_after, meta=meta)
                psnr_after = tt.relativeLoss(rgb_after, cropped_target_image_cuda, tm.get_PSNR)  # 초기 PSNR 저장

            cropped_state[0, channel, row, col] = 1 - cropped_state[0, channel, row, col]

            # PSNR이 개선되었는지 확인
            if psnr_after > previous_psnr:
                flip_count += 1

                # 플립 성공 픽셀의 pre-model output 값 확인
                pre_value = cropped_pre_model_output[channel, row, col]

                # 범위에 따른 카운트 증가
                for i in range(len(output_bins) - 1):
                    if i == len(output_bins) - 2: # 마지막 범위
                        if output_bins[i] <= pre_value <= output_bins[i + 1]:  # `1.0` 포함
                            if psnr_after > previous_psnr:
                                improved_bin_counts[i] += 1  # 개선된 픽셀 수 증가
                                psnr_improvements[i].append(psnr_after - previous_psnr)  # PSNR 개선량 저장
                            break
                    else:
                        if output_bins[i] <= pre_value < output_bins[i + 1]:
                            if psnr_after > previous_psnr:
                                improved_bin_counts[i] += 1  # 개선된 픽셀 수 증가
                                psnr_improvements[i].append(psnr_after - previous_psnr)  # PSNR 개선량 저장
                            break

            if steps % 5000 == 0:
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
                    f"\npre_value: {pre_value:.6f}"
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
        print(f"{file_name}.png Optimization completed. Final PSNR improvement: {psnr_diff:.6f}")
        print(f"Time taken for this data: {data_processing_time:.2f} seconds\n")
        print("Pre-model output range statistics:")

        # 저장 경로 설정
        dbs_folder = "DBS"
        os.makedirs(dbs_folder, exist_ok=True)  # 폴더가 없으면 생성

        # Reconstructed RGB 이미지를 numpy 배열로 변환 및 저장
        rgb_np = rgb.cpu().numpy()  # RGB 데이터를 NumPy 배열로 변환
        save_path_rgb = os.path.join(dbs_folder, f"episode_{file_name}_rgb_after.npy")  # 저장 경로 설정

        # NumPy 배열로 저장
        np.save(save_path_rgb, rgb_np)
        print(f"RGB data saved to {save_path_rgb}")

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
target_dir = '/nfs/dataset/DIV2K/DIV2K_train_HR/DIV2K_train_HR/'
valid_dir = '/nfs/dataset/DIV2K/DIV2K_valid_HR/DIV2K_valid_HR/'
meta = {'wl' : (638e-9, 515e-9, 450e-9), 'dx':(7.56e-6, 7.56e-6)}
padding = 0

# Dataset512 클래스 사용
train_dataset = Dataset512(target_dir=target_dir, meta=meta, isTrain=False, padding=padding) #센터크롭
valid_dataset = Dataset512(target_dir=valid_dir, meta=meta, isTrain=False, padding=padding)

# DataLoader 생성
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
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