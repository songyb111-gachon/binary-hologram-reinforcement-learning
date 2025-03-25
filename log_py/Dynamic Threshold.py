import tkinter as tk
import re
from collections import defaultdict

def process_log():
    # 입력창의 전체 로그 문자열 가져오기
    log_text = input_text.get("1.0", tk.END)

    # 파일명 추출 (경로의 마지막 부분만 추출)
    file_pattern = re.compile(r"\[Episode Start\].*dataset file: \('(?:.*/)?(.*?)',\)")
    # T_PSNR_DIFF 값 추출
    threshold_pattern = re.compile(r"\[Dynamic Threshold\] T_PSNR_DIFF set to: ([0-9.]+)")
    # Initial PSNR 값 추출
    psnr_pattern = re.compile(r"Initial PSNR: ([0-9.]+)")

    files = file_pattern.findall(log_text)
    thresholds = threshold_pattern.findall(log_text)
    initial_psnrs = psnr_pattern.findall(log_text)

    # 기존 출력창 초기화
    output_text.delete("1.0", tk.END)
    avg_output_text.delete("1.0", tk.END)

    # 추출한 파일명, Initial PSNR, 그리고 T_PSNR_DIFF 값을 출력
    for f, psnr, t in zip(files, initial_psnrs, thresholds):
        output_text.insert(tk.END, f"{f}\n")
        output_text.insert(tk.END, f"Initial PSNR: {psnr}\n")
        output_text.insert(tk.END, f"T_PSNR_DIFF set to: {t}\n\n")

    # 파일명별 T_PSNR_DIFF 및 Initial PSNR 값을 모아서 평균 계산
    groups_threshold = defaultdict(list)
    groups_psnr = defaultdict(list)
    for f, t, psnr in zip(files, thresholds, initial_psnrs):
        try:
            groups_threshold[f].append(float(t))
            groups_psnr[f].append(float(psnr))
        except ValueError:
            continue

    # 파일명의 숫자 부분을 기준으로 정렬 (예: "0001.png" → 1)
    for filename in sorted(groups_threshold.keys(), key=lambda x: int(re.search(r'\d+', x).group())):
        avg_threshold = sum(groups_threshold[filename]) / len(groups_threshold[filename])
        avg_psnr = sum(groups_psnr[filename]) / len(groups_psnr[filename])
        avg_output_text.insert(tk.END, f"{filename} 평균 T_PSNR_DIFF: {avg_threshold:.6f}, 평균 Initial PSNR: {avg_psnr:.6f}\n")

# GUI 구성
root = tk.Tk()
root.title("로그 정보 및 평균 계산기")

# 로그 입력창
input_label = tk.Label(root, text="로그 입력:")
input_label.pack(anchor="w", padx=5, pady=2)

input_text = tk.Text(root, height=20, width=100)
input_text.pack(padx=5, pady=2)

# 처리 버튼
process_button = tk.Button(root, text="로그 처리", command=process_log)
process_button.pack(pady=5)

# 추출된 정보 출력창
output_label = tk.Label(root, text="추출 결과:")
output_label.pack(anchor="w", padx=5, pady=2)

output_text = tk.Text(root, height=15, width=100)
output_text.pack(padx=5, pady=2)

# 파일별 평균 출력창
avg_output_label = tk.Label(root, text="파일별 평균 값 (숫자 순 정렬):")
avg_output_label.pack(anchor="w", padx=5, pady=2)

avg_output_text = tk.Text(root, height=10, width=100)
avg_output_text.pack(padx=5, pady=2)

root.mainloop()