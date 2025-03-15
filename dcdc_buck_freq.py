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
fs = 100e3  # スイッチング周波数 (100 kHz)
Ts = 1 / fs # サンプリング時間 (10 μs)

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

# ボード線図のプロット
omega = np.logspace(-2, 6, 1000)  # 周波数範囲（rad/s）
mag, phase, w = ctrl.frequency_response(G_discrete, omega)

# rad/s -> Hz に変換
freq_Hz = w / (2 * np.pi)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

# ゲイン線図
ax1.semilogx(freq_Hz, 20 * np.log10(mag), label="Gain")
ax1.set_title("Bode Plot")
ax1.set_ylabel("Gain [dB]")
ax1.grid(True, which="both", ls="--")
ax1.set_xlim([10**-2, 10**6])

# 位相線図
ax2.semilogx(freq_Hz, phase * 180 / np.pi, label="Phase")
ax2.set_xlabel("Frequency [Hz]")
ax2.set_ylabel("Phase [deg]")
ax2.grid(True, which="both", ls="--")
ax2.set_xlim([10**-2, 10**6])
ax2.set_ylim(-180, 180)
plt.tight_layout()
plt.show()

# ダイオードの順方向電圧補正
Vout_max = D * Vin - Vf
print(f"目標出力電圧: {Vout_target} V")
print(f"DUTY比 (D): {D:.2f}")
print(f"最大出力電圧（理論値）: {D * Vin} V -> 実際の最大電圧（ダイオード影響考慮）: {Vout_max} V")

# 離散時間伝達関数の表示
print("離散時間伝達関数 G(z):")
print(G_discrete)
