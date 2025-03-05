import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image


def center_crop_image(image, crop_width, crop_height):
    # 이미지의 크기를 가져옵니다.
    width, height = image.size
    # 중앙에서 crop 영역의 좌표를 계산합니다.
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height
    return image.crop((left, top, right, bottom))


def process_images(folder_path):
    # 크롭한 이미지를 저장할 폴더 생성 (이미 존재하면 무시)
    cropped_folder = os.path.join(folder_path, "cropped")
    os.makedirs(cropped_folder, exist_ok=True)

    # 폴더 내의 모든 파일에 대해 처리
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".png"):
            file_path = os.path.join(folder_path, filename)
            try:
                # 이미지 열기
                with Image.open(file_path) as img:
                    # 이미지 크롭 (원본 이미지에 손상이 없도록 새 이미지로 만듭니다.)
                    cropped_img = center_crop_image(img, 256, 256)
                    # 크롭한 이미지를 PNG 형식으로 저장 (손실 없이)
                    output_path = os.path.join(cropped_folder, filename)
                    cropped_img.save(output_path, format="PNG")
                    print(f"Processed {filename}")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")


def main():
    # Tkinter GUI를 통해 폴더 선택
    root = tk.Tk()
    root.withdraw()  # 메인 창을 숨김
    folder_path = filedialog.askdirectory(title="PNG 파일이 있는 폴더를 선택하세요")
    if folder_path:
        process_images(folder_path)
    else:
        print("폴더가 선택되지 않았습니다.")


if __name__ == "__main__":
    main()
