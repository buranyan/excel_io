import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

# Buckコンバータのパラメータ（ESR, DCR, ON抵抗含む）
Vin = 12.0      # 入力電圧
Vout_target = 5.0  # 目標出力電圧
D = Vout_target / Vin  # DUTY比 (0 < D < 1)
L = 220e-6      # インダクタンス (H)
C = 470e-6      # キャパシタンス (F)
R = 10.0        # 負荷抵抗 (Ω)
R_L = 0.1       # インダクタのDCR (Ω)
R_C = 0.05      # コンデンサのESR (Ω)
R_on = 0.2      # FETのオン抵抗 (Ω)
Vf = 0.7        # ダイオードの順方向電圧 (V)

# スイッチング周波数とサンプリング時間
fs = 1000e3  # スイッチング周波数 (1 MHz)
Ts = 1 / fs  # サンプリング時間 (1 μs)

# Buckコンバータの連続時間伝達関数 G(s)
num = [D * Vin * (L * C)]  # 分子（デューティ比Dを考慮）
den = [
    L * C,                          # s^2 の係数
    L / R + R_C * C + R_L * C,      # s の係数
    1 + R_C / R                     # 定数項
]
G_continuous = ctrl.TransferFunction(num, den)  # 連続時間伝達関数

# Z変換（ゼロ次ホールドで離散化）
G_discrete = ctrl.sample_system(G_continuous, Ts, method='zoh')

# 周波数範囲の修正 (0.01 Hz から Nyquist 周波数の 95% 以下)
omega_min = 2 * np.pi * 0.01  # 0.01 Hz -> rad/s
omega_max = 0.95 * np.pi * fs  # Nyquist周波数の95% (rad/s)
omega = np.logspace(np.log10(omega_min), np.log10(omega_max), 1000)  # rad/s

# ボード線図のプロット (離散系では ctrl.bode_plot を推奨)
plt.figure(figsize=(8, 6))
ctrl.bode_plot(G_discrete, omega, dB=True, Hz=True, deg=True)
plt.show()

# ダイオードの順方向電圧補正
Vout_max = D * Vin - Vf
print(f"目標出力電圧: {Vout_target} V")
print(f"DUTY比 (D): {D:.2f}")
print(f"最大出力電圧（理論値）: {D * Vin} V -> 実際の最大電圧（ダイオード影響考慮）: {Vout_max} V")

# 離散時間伝達関数の表示
print("離散時間伝達関数 G(z):")
print(G_discrete)
