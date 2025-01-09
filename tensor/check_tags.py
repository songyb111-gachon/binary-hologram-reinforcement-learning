import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os

# 같은 폴더에 있는 TensorBoard 로그 파일의 이름을 자동으로 검색
log_dir = "."  # 현재 폴더
event_files = [f for f in os.listdir(log_dir) if f.startswith("events.out")]

if not event_files:
    print("No TensorBoard event files found in the directory.")
else:
    file_path = os.path.join(log_dir, event_files[0])  # 첫 번째 로그 파일 선택

# 고유 태그 저장
unique_tags = set()

# 로그 파일 읽기
for event in tf.compat.v1.train.summary_iterator(file_path):
    for value in event.summary.value:
        unique_tags.add(value.tag)

# 모든 태그 출력
print("Unique Tags in the TensorBoard file:")
for tag in unique_tags:
    print(tag)