import numpy as np
import matplotlib.pyplot as plt

# Buckコンバータのパラメータ
Vin = 12.0
Vout_target = 5.0
L = 220e-6
C = 470e-6
R = 10.0
RL = 0.1
RC = 0.05
Ron = 0.2
Vf = 0.7
switching_frequency = 100e3

# シミュレーションパラメータ
switching_period = 1 / switching_frequency
dt = switching_period / 100  # PWM周期の1/100を刻み時間とする
duration = 0.1  # 10ms に拡張して安定性を見る
time_steps = int(duration / dt)

# 初期値
i_L = 0.0
Vout = 0.0
integral = 0.0
previous_error = 0.0

# PIDゲイン（調整後）
Kp = 0.5
Ki = 30.0
Kd = 0.002

# 結果の保存用配列
time = np.linspace(0, duration, time_steps)
vout_values = np.zeros(time_steps)
i_l_values = np.zeros(time_steps)
duty_values = np.zeros(time_steps)
switch_values = np.zeros(time_steps)

# PWM制御用変数
switching_time = 0.0

# シミュレーションループ
for n in range(time_steps):
    # PID制御
    error = Vout_target - Vout
    integral += error * dt
    integral = np.clip(integral, -5, 5)  # 積分の発散を防ぐ
    derivative = (error - previous_error) / dt
    duty = Kp * error + Ki * integral + Kd * derivative
    duty = np.clip(duty, 0, 1)

    # PWM制御
    switching_time += dt
    if switching_time >= switching_period:
        switching_time -= switching_period

    if switching_time < duty * switching_period:
        # MOSFETオン
        di_L = (Vin - i_L * (Ron + RL) - Vout) / L
        dVout = (i_L - Vout / R) / C
        switch_values[n] = 1
    else:
        # ダイオードオン
        di_L = (-Vout - Vf - i_L * RL) / L
        dVout = (i_L - Vout / R) / C
        switch_values[n] = 0

    # 状態変数の更新
    i_L += di_L * dt
    Vout += dVout * dt

    # 結果の保存
    vout_values[n] = Vout
    i_l_values[n] = i_L
    duty_values[n] = duty

    # 前回の誤差を更新
    previous_error = error

# ==== ここからプロット部分を修正 ====

# ✅【見たい区間】の設定
start_time = 0.0  # 0ms
end_time = 0.001   # 100ms

# 指定した区間のインデックスを取得
start_index = np.searchsorted(time, start_time)
end_index = np.searchsorted(time, end_time)

# 指定した範囲のデータをスライス
time_slice = time[start_index:end_index]
vout_slice = vout_values[start_index:end_index]
i_l_slice = i_l_values[start_index:end_index]
duty_slice = duty_values[start_index:end_index]
switch_slice = switch_values[start_index:end_index]

# 結果のプロット
plt.figure(figsize=(10, 6))
plt.plot(time_slice, vout_slice, label="Output Voltage (Vout)")
plt.plot(time_slice, i_l_slice, label="Inductor Current (i_L)")
plt.plot(time_slice, duty_slice, label="Duty Cycle")
plt.plot(time_slice, switch_slice, label="Switch State", linestyle="dotted")
plt.axhline(Vout_target, color="r", linestyle="--", label="Target Voltage")
plt.xlabel("Time (s)")
plt.ylabel("Value")
plt.title(f"Buck Converter Simulation (Zoomed: {start_time*1e3:.1f}ms - {end_time*1e3:.1f}ms)")
plt.legend()
plt.grid(True)
plt.show()
