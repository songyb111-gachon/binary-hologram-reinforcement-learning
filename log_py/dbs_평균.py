import tkinter as tk
from tkinter import scrolledtext, messagebox
import re
import numpy as np
import matplotlib.pyplot as plt

def plot_graph():
    # GUI 텍스트 위젯에서 데이터 가져오기
    data = text_area.get("1.0", tk.END)
    lines = data.strip().splitlines()

    # 각 Range별로 (Improvement Ratio * Average PSNR Improvement) 값을 누적할 딕셔너리
    data_dict = {}
    # 정규표현식 패턴:
    # 그룹1: Range (예: "0.0-0.1")
    # 그룹2: Total Pixels
    # 그룹3: Improved Pixels
    # 그룹4: Attempted Pixels
    # 그룹5: Improvement Ratio (전체)
    # 그룹6: Improvement Ratio (in range)
    # 그룹7: Improvement Ratio (to total improved)
    # 그룹8: Total PSNR Improvement
    # 그룹9: Average PSNR Improvement
    pattern = (
        r"Range\s*([\d\.]+-[\d\.]+):\s*Total Pixels\s*=\s*(\d+),\s*"
        r"Improved Pixels\s*=\s*(\d+),\s*Attempted Pixels\s*=\s*(\d+),\s*"
        r"Improvement Ratio\s*=\s*([\d\.]+),\s*"
        r"Improvement Ratio \(in range\)\s*=\s*([\d\.]+),\s*"
        r"Improvement Ratio \(to total improved\)\s*=\s*([\d\.]+),\s*"
        r"Total PSNR Improvement\s*=\s*([\d\.]+),\s*"
        r"Average PSNR Improvement\s*=\s*([\d\.]+)"
    )

    # 각 줄에서 정규표현식으로 매칭된 데이터 누적
    for line in lines:
        match = re.search(pattern, line)
        if match:
            range_label = match.group(1)
            imp_ratio = float(match.group(5))
            avg_psnr = float(match.group(9))
            product_value = imp_ratio * avg_psnr  # 두 값의 곱 계산
            if range_label not in data_dict:
                data_dict[range_label] = {"sum": 0.0, "count": 0}
            data_dict[range_label]["sum"] += product_value
            data_dict[range_label]["count"] += 1

    if not data_dict:
        messagebox.showerror("Error", "유효한 데이터가 발견되지 않았습니다.")
        return

    # Range 레이블을 오름차순(시작 값 기준)으로 정렬
    sorted_ranges = sorted(data_dict.keys(), key=lambda x: float(x.split('-')[0]))
    # 각 Range별 평균 (Improvement Ratio * Average PSNR Improvement) 계산
    avg_products = [data_dict[r]["sum"] / data_dict[r]["count"] for r in sorted_ranges]

    # 막대그래프 생성
    x = np.arange(len(sorted_ranges))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x, avg_products, width, label="Improvement Ratio * Average PSNR Improvement")
    plt.xlabel("Range")
    plt.ylabel("Product Value")
    plt.title("각 Range별 Improvement Ratio * Average PSNR Improvement (전체 데이터 평균)")
    plt.xticks(x, sorted_ranges, rotation=45)
    plt.legend()
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
