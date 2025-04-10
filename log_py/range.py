import tkinter as tk
from tkinter import scrolledtext, messagebox
import re
import matplotlib.pyplot as plt


def plot_graphs():
    # 텍스트 위젯에서 전체 로그를 가져옴
    log_text = text_input.get("1.0", tk.END)

    # "Starting pixel flip optimization for file" 기준으로 에피소드별로 분리
    episodes = re.split(r"(?=Starting pixel flip optimization for file)", log_text)

    episode_count = 0  # 처리한 에피소드 수
    for ep in episodes:
        # "Pre-model output range statistics:"가 없는 블록은 건너뜀
        if "Pre-model output range statistics:" not in ep:
            continue

        # 에피소드 내에서 png 파일 이름 추출
        png_match = re.search(r"Starting pixel flip optimization for file\s+(\S+)", ep)
        if not png_match:
            continue
        png_name = png_match.group(1)

        # "Pre-model output range statistics:" 이후의 range 데이터 블록 추출
        ranges_match = re.search(r"Pre-model output range statistics:\s*((?:Range.*\n)+)", ep)
        if not ranges_match:
            continue
        ranges_text = ranges_match.group(1)

        # 각 range 라인에서 데이터 추출: 범위, Total Pixels, Improvement Ratio (in range)
        line_pattern = r"Range\s+([\d\.]+-[\d\.]+):.*?Total Pixels\s*=\s*(\d+),.*?Improvement Ratio \(in range\)\s*=\s*([\d\.]+)"
        matches = re.findall(line_pattern, ranges_text)

        if not matches:
            continue

        range_labels = []
        total_pixels = []
        improvement_ratios = []
        for rng, total, ratio in matches:
            range_labels.append(rng)
            total_pixels.append(int(total))
            improvement_ratios.append(float(ratio))

        # 에피소드별 Total Pixels 막대그래프 생성 (그래프 제목에 png 이름 포함)
        #plt.figure(figsize=(10, 5))
        #plt.bar(range_labels, total_pixels)
        #plt.title(f"{png_name} - Total Pixels")
        #plt.xlabel("Range")
        #plt.ylabel("Total Pixels")
        #plt.tight_layout()
        #plt.show()

        # 에피소드별 Improvement Ratio (in range) 막대그래프 생성 (그래프 제목에 png 이름 포함)
        plt.figure(figsize=(10, 5))
        plt.bar(range_labels, improvement_ratios)
        plt.title(f"{png_name} - Improvement Ratio (in range)")
        plt.xlabel("Range")
        plt.ylabel("Improvement Ratio (in range)")
        plt.tight_layout()
        plt.show()

        episode_count += 1

    if episode_count == 0:
        messagebox.showerror("데이터 없음", "로그에서 에피소드 데이터를 찾을 수 없습니다.")


# tkinter 메인 윈도우 생성
root = tk.Tk()
root.title("로그 데이터 시각화 (에피소드별)")

# 스크롤 가능한 텍스트 위젯 (로그 입력용)
text_input = scrolledtext.ScrolledText(root, width=100, height=30)
text_input.pack(padx=10, pady=10)

# 그래프 출력 버튼
plot_button = tk.Button(root, text="그래프 그리기", command=plot_graphs)
plot_button.pack(pady=10)

# GUI 실행
root.mainloop()