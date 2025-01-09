import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os

# 같은 폴더에 있는 TensorBoard 로그 파일의 이름을 자동으로 검색
log_dir = ".."  # 현재 폴더
event_file = [f for f in os.listdir(log_dir) if f.startswith("events.out")]

if not event_file:
    print("No TensorBoard event files found in the directory.")
else:
    file_path = os.path.join(log_dir, event_file[0])  # 첫 번째 로그 파일 선택

    # 데이터를 저장할 리스트
    data = []

    # TensorBoard 로그 파일 읽기
    for event in tf.compat.v1.train.summary_iterator(file_path):
        for value in event.summary.value:
            if value.tag == "rollout/ep_rew_mean":  # Stable-Baselines의 에피소드 평균 리워드 태그
                data.append((event.step, value.simple_value))

    # 데이터를 DataFrame으로 변환
    df = pd.DataFrame(data, columns=["Step", "ep_rew_mean"])

    # 그래프 그리기
    if not df.empty:
        plt.figure(figsize=(10, 6))
        plt.plot(df["Step"], df["ep_rew_mean"], label="Episode Reward Mean")
        plt.xlabel("Step")
        plt.ylabel("Episode Reward Mean")
        plt.title("Episode Reward Mean over Training Steps")
        plt.legend()
        plt.grid()
        plt.show()
    else:
        print("No data found for the tag 'rollout/ep_rew_mean'.")