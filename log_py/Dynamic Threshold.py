import tkinter as tk
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

    # 정렬된 결과를 출력
    for filename, avg_threshold, avg_psnr, avg_inc in sorted_results:
        avg_output_text.insert(tk.END,
                               f"{filename} 평균 T_PSNR_DIFF: {avg_threshold:.6f}, 평균 Initial PSNR: {avg_psnr:.6f}, "
                               f"평균 PSNR increase probability: {avg_inc:.6f}\n"
                               )


# GUI 구성
root = tk.Tk()
root.title("로그 정보 및 평균 계산기")

# 로그 입력창
input_label = tk.Label(root, text="로그 입력:")
input_label.pack(anchor="w", padx=5, pady=2)

input_text = tk.Text(root, height=20, width=100)
input_text.pack(padx=5, pady=2)

# 정렬 기준 및 순서 선택 OptionMenu 추가
option_frame = tk.Frame(root)
option_frame.pack(anchor="w", padx=5, pady=2)

# 정렬 기준 선택
sort_option = tk.StringVar(value="파일명")
sort_label = tk.Label(option_frame, text="정렬 기준:")
sort_label.pack(side=tk.LEFT)
sort_menu = tk.OptionMenu(option_frame, sort_option, "파일명", "T_PSNR_DIFF", "Initial PSNR", "PSNR increase probability")
sort_menu.pack(side=tk.LEFT, padx=(0, 10))

# 정렬 순서 선택
order_option = tk.StringVar(value="오름차순")
order_label = tk.Label(option_frame, text="정렬 순서:")
order_label.pack(side=tk.LEFT)
order_menu = tk.OptionMenu(option_frame, order_option, "오름차순", "내림차순")
order_menu.pack(side=tk.LEFT)

# 정렬 옵션 변경 시 자동으로 재정렬
sort_option.trace("w", sort_results)
order_option.trace("w", sort_results)

# 처리 버튼
process_button = tk.Button(root, text="로그 처리", command=process_log)
process_button.pack(pady=5)

# 추출된 정보 출력창
output_label = tk.Label(root, text="추출 결과:")
output_label.pack(anchor="w", padx=5, pady=2)

output_text = tk.Text(root, height=15, width=100)
output_text.pack(padx=5, pady=2)

# 파일별 평균 출력창
avg_output_label = tk.Label(root, text="파일별 평균 값 (선택한 정렬 기준과 순서에 따름):")
avg_output_label.pack(anchor="w", padx=5, pady=2)

avg_output_text = tk.Text(root, height=10, width=100)
avg_output_text.pack(padx=5, pady=2)

root.mainloop()
