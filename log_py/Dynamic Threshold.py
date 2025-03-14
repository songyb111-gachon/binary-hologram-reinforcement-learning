import tkinter as tk
import re
from collections import defaultdict


def process_log():
    # 입력창의 전체 로그 문자열 가져오기
    log_text = input_text.get("1.0", tk.END)

    # 파일명 추출 (경로 '10data/' 이후의 파일명만)
    file_pattern = re.compile(r"\[Episode Start\].*dataset file: \('10data/(.*?)',\)")
    # T_PSNR_DIFF 값 추출
    threshold_pattern = re.compile(r"\[Dynamic Threshold\] T_PSNR_DIFF set to: ([0-9.]+)")

    files = file_pattern.findall(log_text)
    thresholds = threshold_pattern.findall(log_text)

    # 기존 출력창 초기화
    output_text.delete("1.0", tk.END)
    avg_output_text.delete("1.0", tk.END)

    # 추출한 파일명과 T_PSNR_DIFF 값을 출력
    for f, t in zip(files, thresholds):
        output_text.insert(tk.END, f"{f}\n")
        output_text.insert(tk.END, f"T_PSNR_DIFF set to: {t}\n\n")

    # 파일명별 T_PSNR_DIFF 값을 모아서 평균 계산
    groups = defaultdict(list)
    for f, t in zip(files, thresholds):
        try:
            groups[f].append(float(t))
        except ValueError:
            continue

    # 파일명의 숫자 부분을 기준으로 정렬 (예: "0001.png" → 1)
    for filename in sorted(groups.keys(), key=lambda x: int(x.split('.')[0])):
        avg_val = sum(groups[filename]) / len(groups[filename])
        avg_output_text.insert(tk.END, f"{filename} 평균 T_PSNR_DIFF: {avg_val:.6f}\n")


# GUI 구성
root = tk.Tk()
root.title("로그 정보 및 평균 T_PSNR_DIFF 계산기")

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
avg_output_label = tk.Label(root, text="파일별 평균 T_PSNR_DIFF (숫자 순 정렬):")
avg_output_label.pack(anchor="w", padx=5, pady=2)

avg_output_text = tk.Text(root, height=10, width=100)
avg_output_text.pack(padx=5, pady=2)

root.mainloop()