import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction

# 기준점 (Goal-Reaching 성공 비율과 보상 값)
success_ratios = np.array([1/8, 1/4, 1/2, 1])
rewards = np.array([-300, -100, 100, 300])

# 3차 다항식으로 보상 함수 추정
coefficients = np.polyfit(success_ratios, rewards, 3)
reward_function = np.poly1d(coefficients)

# Success ratio 범위 및 계산된 보상 값
x_vals = np.linspace(0, 1, 100)
reward_vals = reward_function(x_vals)

# Target points (with fractions for x-axis)
target_points = [(4000, -300), (2500, -100), (1000, 100), (1, 300)]

# 그래프 그리기
plt.figure(figsize=(8, 6))
plt.plot(x_vals, reward_vals, label="Cubic Goal-Reaching Reward Function", color='orange')
plt.scatter(*zip(*target_points), color='red', label="Target Points")
plt.axhline(0, color='black', linewidth=0.5, linestyle='--', label="Zero Reward or Penalty")
plt.axvline(1/8, color='purple', linestyle='--', label="1/8 Success Ratio")
plt.axvline(1/4, color='red', linestyle='--', label="1/4 Success Ratio")
plt.axvline(1/2, color='green', linestyle='--', label="1/2 Success Ratio")
plt.axvline(1, color='blue', linestyle='--', label="1 Success Ratio")

# x축 분수를 표시
plt.xticks([0, 1/8, 1/4, 1/2, 1], [str(Fraction(x).limit_denominator()) for x in [0, 1/8, 1/4, 1/2, 1]])

# 계산된 다항식 식 추가
equation_text = (
    f"y = {coefficients[0]:.2f}x³ + {coefficients[1]:.2f}x² + {coefficients[2]:.2f}x + {coefficients[3]:.2f}"
)
plt.text(0.05, 250, equation_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

# 그래프 꾸미기
plt.title("Goal-Reaching Reward or Penalty vs Success Ratio (Fractional x-axis)")
plt.xlabel("Success Ratio")
plt.ylabel("Goal-Reaching Reward or Penalty")
plt.legend()
plt.grid()
plt.show()