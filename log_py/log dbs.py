import tkinter as tk
from tkinter import scrolledtext, messagebox
import re
import numpy as np
import matplotlib.pyplot as plt


def plot_graph():
    # 텍스트 영역에서 전체 로그 데이터 읽기
    data = text_area.get("1.0", tk.END)
    if not data.strip():
        messagebox.showerror("Error", "데이터를 입력하세요.")
        return

    # "RGB data saved to" 로 시작하는 각 블럭을 분리 (각 블럭은 하나의 에피소드)
    blocks = re.split(r"(?=RGB data saved to)", data)

    if not blocks:
        messagebox.showerror("Error", "유효한 에피소드 블럭이 발견되지 않았습니다.")
        return

    # 각 블럭(에피소드)별로 처리
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # 블럭의 첫 줄에서 에피소드 번호 추출 (예: "episode_0801")
        header_match = re.search(r"episode_\d+", block)
        ep_id = header_match.group(0) if header_match else "Unknown Episode"

        # 블럭 내에서 초기 PSNR 값 추출 (첫번째 발생하는 값 사용)
        psnr_match = re.search(r"with initial PSNR:\s*([\d\.]+)", block)
        initial_psnr = psnr_match.group(1) if psnr_match else "N/A"

        range_labels = []
        imp_ratios = []

        # 각 블럭 내에서 "Range ..." 라인에서 Improvement Ratio (전체)를 추출
        pattern = (
            r"Range\s*([\d\.]+-[\d\.]+):\s*Total Pixels\s*=\s*\d+,\s*"
            r"Improved Pixels\s*=\s*\d+,\s*Attempted Pixels\s*=\s*\d+,\s*"
            r"Improvement Ratio\s*=\s*([\d\.]+),\s*"
            r"Improvement Ratio \(in range\)\s*=\s*[\d\.]+,\s*"
            r"Improvement Ratio \(to total improved\)\s*=\s*[\d\.]+,\s*"
            r"Total PSNR Improvement\s*=\s*[\d\.]+,\s*"
            r"Average PSNR Improvement\s*=\s*[\d\.]+"
        )

        for match in re.finditer(pattern, block):
            range_labels.append(match.group(1))
            imp_ratios.append(float(match.group(2)))

        if not range_labels:
            continue  # 해당 블럭에 Range 데이터가 없으면 건너뜀

        # 막대그래프 그리기 (에피소드별)
        x = np.arange(len(range_labels))
        width = 0.5

        plt.figure(figsize=(10, 5))
        plt.bar(x, imp_ratios, width, color='skyblue', edgecolor='black')
        plt.xlabel("Range")
        plt.ylabel("Improvement Ratio")
        plt.title(f"{ep_id} (Initial PSNR: {initial_psnr}) - Improvement Ratio per Range")
        plt.xticks(x, range_labels, rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()


# Tkinter GUI 설정
root = tk.Tk()
root.title("로그 데이터 그래프")

instruction = tk.Label(root, text="로그 파일의 데이터를 아래에 붙여넣고 [Plot Graph] 버튼을 클릭하세요.", font=("Arial", 12))
instruction.pack(pady=10)

text_area = scrolledtext.ScrolledText(root, width=100, height=25, font=("Courier", 10))
text_area.pack(padx=10, pady=10)

plot_button = tk.Button(root, text="Plot Graph", font=("Arial", 12), command=plot_graph)
plot_button.pack(pady=10)

root.mainloop()