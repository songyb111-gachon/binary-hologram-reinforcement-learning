import tkinter as tk
from tkinter import scrolledtext, messagebox
import re
import numpy as np
import matplotlib.pyplot as plt


def plot_graph():
    # GUI 텍스트 위젯에서 데이터 가져오기
    data = text_area.get("1.0", tk.END)
    lines = data.strip().splitlines()

    # 각 범위와 3종류의 Improvement Ratio를 저장할 리스트 초기화
    range_labels = []
    improvement_ratios = []
    improvement_ratios_in_range = []
    improvement_ratios_to_total = []

    # 정규 표현식을 사용하여 값 추출
    pattern = (
        r"Range\s*([\d\.]+-[\d\.]+):.*?"
        r"Improvement Ratio\s*=\s*([\d\.]+).*?"
        r"Improvement Ratio \(in range\)\s*=\s*([\d\.]+).*?"
        r"Improvement Ratio \(to total improved\)\s*=\s*([\d\.]+)"
    )

    for line in lines:
        match = re.search(pattern, line)
        if match:
            range_labels.append(match.group(1))
            improvement_ratios.append(float(match.group(2)))
            improvement_ratios_in_range.append(float(match.group(3)))
            improvement_ratios_to_total.append(float(match.group(4)))

    if not range_labels:
        messagebox.showerror("Error", "유효한 데이터가 발견되지 않았습니다.")
        return

    # x축 좌표와 막대 너비 설정
    x = np.arange(len(range_labels))
    width = 0.25

    # matplotlib를 이용한 그룹화된 막대 그래프 생성
    plt.figure(figsize=(12, 6))
    plt.bar(x - width, improvement_ratios, width, label="Improvement Ratio")
    #plt.bar(x, improvement_ratios_in_range, width, label="Improvement Ratio (in range)")
    #plt.bar(x + width, improvement_ratios_to_total, width, label="Improvement Ratio (to total improved)")

    plt.xlabel("Range")
    plt.ylabel("Ratio Value")
    plt.title("각 범위별 Improvement Ratio 비교")
    plt.xticks(x, range_labels, rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


# Tkinter GUI 설정
root = tk.Tk()
root.title("데이터 입력창")

instruction = tk.Label(root, text="데이터를 아래에 붙여넣고 [Plot Graph] 버튼을 클릭하세요.", font=("Arial", 12))
instruction.pack(pady=10)

text_area = scrolledtext.ScrolledText(root, width=100, height=25, font=("Courier", 10))
text_area.pack(padx=10, pady=10)

plot_button = tk.Button(root, text="Plot Graph", font=("Arial", 12), command=plot_graph)
plot_button.pack(pady=10)

root.mainloop()
