import os
import io
import shutil
from PIL import Image
import tkinter as tk
from tkinter import filedialog

# Tkinter를 이용해 폴더 선택 다이얼로그 실행
root = tk.Tk()
root.withdraw()  # 메인 창 숨김
folder_path = filedialog.askdirectory(title="이미지가 있는 폴더를 선택하세요")
if not folder_path:
    print("폴더가 선택되지 않았습니다.")
    exit(1)

# 출력 폴더 생성 (원본 파일은 보존)
output_folder = os.path.join(folder_path, "compressed_images")
os.makedirs(output_folder, exist_ok=True)

# 목표 파일 크기: 3MB (바이트 단위)
target_size = 3 * 1024 * 1024  # 3MB

# 폴더 내의 모든 파일에 대해 처리
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    # 파일이 아닌 경우 스킵
    if not os.path.isfile(file_path):
        continue

    # 파일 크기가 3MB 이하이면 단순 복사
    file_size = os.path.getsize(file_path)
    if file_size <= target_size:
        output_file_path = os.path.join(output_folder, filename)
        shutil.copy2(file_path, output_file_path)
        print(f"{filename} ({file_size / 1024:.2f} KB)는 3MB 이하이므로 복제 완료")
        continue

    try:
        # 이미지 파일 열기
        img = Image.open(file_path)
    except Exception as e:
        print(f"{filename}은(는) 이미지 파일이 아니므로 스킵합니다.")
        continue

    # JPEG로 저장하기 위해 RGB 모드로 변환 (필요한 경우)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # 압축: 최대한 좋은 품질(높은 quality 값)을 유지하며 3MB 이하가 되는 quality 값을 이진 탐색으로 찾음
    low, high = 20, 95  # Pillow에서는 quality 값 최대 95를 권장합니다.
    best_quality = low
    best_buffer = None

    while low <= high:
        mid = (low + high) // 2
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=mid, optimize=True)
        size = buffer.tell()

        if size <= target_size:
            best_quality = mid
            best_buffer = buffer
            low = mid + 1  # 더 높은 quality를 시도
        else:
            high = mid - 1  # quality를 낮춰서 시도

    if best_buffer is None:
        # quality 20에서도 조건을 만족하지 않으면 quality 20으로 저장
        best_quality = 20
        best_buffer = io.BytesIO()
        img.save(best_buffer, format="JPEG", quality=best_quality, optimize=True)

    # 출력 파일명: 원본 파일명에서 확장자를 제거하고 .jpg 확장자로 저장
    base_name, _ = os.path.splitext(filename)
    output_file_path = os.path.join(output_folder, base_name + ".jpg")
    with open(output_file_path, 'wb') as f:
        f.write(best_buffer.getvalue())

    final_size_kb = best_buffer.tell() / 1024
    print(f"{filename} 압축 완료 - 최종 크기: {final_size_kb:.2f} KB, 사용 quality: {best_quality}")