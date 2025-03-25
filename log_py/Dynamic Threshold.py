import tkinter as tk
from tkinter import ttk
import re
from collections import defaultdict

# 전역 변수에 추출된 결과를 저장합니다.
# 각 결과는 (파일명, 평균 T_PSNR_DIFF, 평균 Initial PSNR, 평균 PSNR increase probability) 튜플입니다.
extracted_results = []

def process_log():
    global extracted_results
    # 입력창의 전체 로그 문자열 가져오기
    log_text = input_text.get("1.0", tk.END)

    # 파일명 추출 (경로의 마지막 부분만 추출)
    file_pattern = re.compile(r"\[Episode Start\].*dataset file: \('(?:.*/)?(.*?)',\)")
    # T_PSNR_DIFF 값 추출
    threshold_pattern = re.compile(r"\[Dynamic Threshold\] T_PSNR_DIFF set to: ([0-9.]+)")
    # Initial PSNR 값 추출
    psnr_pattern = re.compile(r"Initial PSNR: ([0-9.]+)")
    # PSNR increase probability 값 추출 (ANSI 컬러 코드 무시)
    psnr_inc_pattern = re.compile(r"PSNR increase probability: ([0-9.]+)")

    files = file_pattern.findall(log_text)
    thresholds = threshold_pattern.findall(log_text)
    initial_psnrs = psnr_pattern.findall(log_text)
    psnr_increases = psnr_inc_pattern.findall(log_text)

    # 기존 출력창 초기화
    output_text.delete("1.0", tk.END)
    avg_output_text.delete("1.0", tk.END)

    # 추출한 항목들을 출력
    for f, psnr, t, inc in zip(files, initial_psnrs, thresholds, psnr_increases):
        output_text.insert(tk.END, f"{f}\n")
        output_text.insert(tk.END, f"Initial PSNR: {psnr}\n")
        output_text.insert(tk.END, f"T_PSNR_DIFF set to: {t}\n")
        output_text.insert(tk.END, f"PSNR increase probability: {inc}\n\n")

    # 파일명별로 값을 모아서 평균 계산
    groups_threshold = defaultdict(list)
    groups_psnr = defaultdict(list)
    groups_inc = defaultdict(list)
    for f, t, psnr, inc in zip(files, thresholds, initial_psnrs, psnr_increases):
        try:
            groups_threshold[f].append(float(t))
            groups_psnr[f].append(float(psnr))
            groups_inc[f].append(float(inc))
        except ValueError:
            continue

    results = []
    for filename in groups_threshold:
        avg_threshold = sum(groups_threshold[filename]) / len(groups_threshold[filename])
        avg_psnr = sum(groups_psnr[filename]) / len(groups_psnr[filename])
        avg_inc = sum(groups_inc[filename]) / len(groups_inc[filename])
        results.append((filename, avg_threshold, avg_psnr, avg_inc))

    # 전역 변수에 저장 후 정렬 결과 자동 갱신
    extracted_results = results
    sort_results()

def sort_results(*args):
    global extracted_results
    # 파일별 평균 결과창 초기화
    avg_output_text.delete("1.0", tk.END)
    if not extracted_results:
        return

    # 사용자가 선택한 정렬 기준에 따라 정렬
    sort_key = sort_option.get()
    if sort_key == "파일명":
        sorted_results = sorted(extracted_results, key=lambda x: int(re.search(r'\d+', x[0]).group()))
    elif sort_key == "T_PSNR_DIFF":
        sorted_results = sorted(extracted_results, key=lambda x: x[1])
    elif sort_key == "Initial PSNR":
        sorted_results = sorted(extracted_results, key=lambda x: x[2])
    elif sort_key == "PSNR increase probability":
        sorted_results = sorted(extracted_results, key=lambda x: x[3])
    else:
        sorted_results = extracted_results

    # 사용자가 선택한 정렬 순서에 따라 조정 (오름차순/내림차순)
    order = order_option.get()
    if order == "내림차순":
        sorted_results = sorted_results[::-1]

    # 헤더 출력: 현재 정렬 기준과 정렬 순서를 맨 위에 표시
    header_text = f"정렬 기준: {sort_option.get()} / 정렬 순서: {order_option.get()}\n\n"
    avg_output_text.insert(tk.END, header_text)

    # 정렬 기준에 따른 출력 순서 설정
    if sort_key == "PSNR increase probability":
        # 출력 순서: 파일명, 평균 PSNR increase probability, 평균 T_PSNR_DIFF, Initial PSNR
        for filename, avg_threshold, avg_psnr, avg_inc in sorted_results:
            avg_output_text.insert(tk.END,
                f"{filename} 평균 PSNR increase probability: {avg_inc:.6f}, "
                f"평균 T_PSNR_DIFF: {avg_threshold:.6f}, Initial PSNR: {avg_psnr:.6f}\n")
    elif sort_key == "T_PSNR_DIFF":
        # 출력 순서: 파일명, 평균 T_PSNR_DIFF, 평균 PSNR increase probability, Initial PSNR
        for filename, avg_threshold, avg_psnr, avg_inc in sorted_results:
            avg_output_text.insert(tk.END,
                f"{filename} 평균 T_PSNR_DIFF: {avg_threshold:.6f}, "
                f"평균 PSNR increase probability: {avg_inc:.6f}, Initial PSNR: {avg_psnr:.6f}\n")
    elif sort_key == "Initial PSNR":
        # 출력 순서: 파일명, Initial PSNR, 평균 T_PSNR_DIFF, 평균 PSNR increase probability
        for filename, avg_threshold, avg_psnr, avg_inc in sorted_results:
            avg_output_text.insert(tk.END,
                f"{filename} Initial PSNR: {avg_psnr:.6f}, "
                f"평균 T_PSNR_DIFF: {avg_threshold:.6f}, 평균 PSNR increase probability: {avg_inc:.6f}\n")
    else:
        # 기본 출력 순서: 파일명, 평균 T_PSNR_DIFF, Initial PSNR, 평균 PSNR increase probability
        for filename, avg_threshold, avg_psnr, avg_inc in sorted_results:
            avg_output_text.insert(tk.END,
                f"{filename} 평균 T_PSNR_DIFF: {avg_threshold:.6f}, "
                f"Initial PSNR: {avg_psnr:.6f}, 평균 PSNR increase probability: {avg_inc:.6f}\n")

# GUI 구성
root = tk.Tk()
root.title("로그 정보 및 평균 계산기")
root.geometry("900x1200")
root.configure(bg="#F0F0F0")

# ttk 스타일 설정
style = ttk.Style()
style.theme_use("clam")
style.configure("TLabel", font=("Helvetica", 11), background="#F0F0F0")
style.configure("TButton", font=("Helvetica", 11), padding=5)
style.configure("TFrame", background="#F0F0F0")

# 상단 헤더
header = ttk.Label(root, text="로그 정보 및 평균 계산기", font=("Helvetica", 16, "bold"))
header.pack(pady=10)

# 로그 입력 영역
input_frame = ttk.Frame(root)
input_frame.pack(fill="both", padx=10, pady=5)
input_label = ttk.Label(input_frame, text="로그 입력:")
input_label.pack(anchor="w", pady=(0, 5))
input_text = tk.Text(input_frame, height=10, width=100, font=("Helvetica", 10))
input_text.pack(fill="both", padx=5, pady=5)

# 정렬 옵션 영역
option_frame = ttk.Frame(root)
option_frame.pack(fill="x", padx=10, pady=5)
sort_option = tk.StringVar(value="파일명")
order_option = tk.StringVar(value="오름차순")
sort_label = ttk.Label(option_frame, text="정렬 기준:")
sort_label.pack(side="left", padx=(0, 5))
sort_menu = ttk.OptionMenu(option_frame, sort_option, "파일명", "파일명", "T_PSNR_DIFF", "Initial PSNR", "PSNR increase probability")
sort_menu.pack(side="left", padx=(0, 15))
order_label = ttk.Label(option_frame, text="정렬 순서:")
order_label.pack(side="left", padx=(0, 5))
order_menu = ttk.OptionMenu(option_frame, order_option, "오름차순", "오름차순", "내림차순")
order_menu.pack(side="left", padx=(0, 15))
# 옵션 변경 시 자동 재정렬
sort_option.trace("w", sort_results)
order_option.trace("w", sort_results)

# 처리 버튼 영역
button_frame = ttk.Frame(root)
button_frame.pack(fill="x", padx=10, pady=5)
process_button = ttk.Button(button_frame, text="로그 처리", command=process_log)
process_button.pack(pady=5)

# 추출 결과 영역
output_frame = ttk.Frame(root)
output_frame.pack(fill="both", padx=10, pady=5, expand=True)
output_label = ttk.Label(output_frame, text="추출 결과:")
output_label.pack(anchor="w", pady=(0, 5))
output_text = tk.Text(output_frame, height=10, width=100, font=("Helvetica", 10))
output_text.pack(fill="both", padx=5, pady=5, expand=True)

# 파일별 평균 결과 영역
avg_frame = ttk.Frame(root)
avg_frame.pack(fill="both", padx=10, pady=5, expand=True)
avg_output_label = ttk.Label(avg_frame, text="파일별 평균 값 (선택한 정렬 기준과 순서에 따름):")
avg_output_label.pack(anchor="w", pady=(0, 5))
avg_output_text = tk.Text(avg_frame, height=8, width=100, font=("Helvetica", 10))
avg_output_text.pack(fill="both", padx=5, pady=5, expand=True)

root.mainloop()