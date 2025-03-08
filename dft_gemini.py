import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def trapezoidal_wave(t, rise_time, on_time, fall_time):
    """台形波を生成する関数（1周期に1回のみ）"""
    if t < rise_time:
        return t / rise_time
    elif t < rise_time + on_time:
        return 1.0
    elif t < rise_time + on_time + fall_time:
        return 1.0 - (t - rise_time - on_time) / fall_time
    else:
        return 0.0

# パラメータ設定
period = 400e-6  # 周期 (400us)
rise_time = 0.1e-6  # 立上がり時間 (0.1us)
on_time = 0.4e-6  # オン時間 (0.4us)
fall_time = 0.1e-6  # 立下がり時間 (0.1us)
sampling_rate = 100e6  # サンプリングレート (100MHz)
pulse_width = rise_time + on_time + fall_time  # パルス幅
duration = pulse_width * 2  # パルスの 2 倍の範囲で表示（視認性向上）

# 時間軸の生成
time = np.arange(0, period, 1 / sampling_rate)

# 台形波の生成（1周期に1回だけ）
power_signal = np.array([trapezoidal_wave(t, rise_time, on_time, fall_time) for t in time])

# 平方根を取って振幅信号を生成
amplitude_signal = np.sqrt(power_signal)

# 離散フーリエ変換
fft_result = np.fft.fft(amplitude_signal)
frequencies = np.fft.fftfreq(len(amplitude_signal), 1 / sampling_rate)

# 周波数軸の調整 (折り返し成分を負の周波数で表現)
frequencies = np.fft.fftshift(frequencies)
fft_result = np.fft.fftshift(fft_result)

# パワースペクトルの計算 (振幅の2乗)
power_spectrum = np.abs(fft_result)**2 / len(amplitude_signal)

# デシベル表示に変換（ゼロ回避のため小さい値を加える）
power_spectrum_db = 10 * np.log10(power_spectrum + 1e-12)

# 最大値をデシベルで取得
max_power_db = np.max(power_spectrum_db)

# グラフの表示
plt.figure(figsize=(5, 5))

# 時間領域の信号 (電力)
plt.subplot(2, 1, 1)
plt.plot(time * 1e6, power_signal)  # 時間をマイクロ秒で表示
plt.title("Trapezoidal Wave (Power)")
plt.xlabel("Time [µs]")
plt.ylabel("Power [mW]")
plt.xlim(0, duration * 1e6)  # 視認性のためパルス幅の2倍を表示
plt.grid(True)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(0.1))  # x軸を0.1 us刻みに設定
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.1))  # y軸を0.1 mW刻みに設定

# 周波数領域の信号 (パワースペクトル dBm)
plt.subplot(2, 1, 2)
plt.plot(frequencies * 1e-6, power_spectrum_db)  # 周波数をMHzで表示
plt.title(f"Power Spectrum (dBm), Max: {max_power_db:.2f} dBm")  # 最大値をタイトルに表示
plt.xlabel("Frequency [MHz]")
plt.ylabel("Power (dBm)")
plt.xlim(-sampling_rate / 2 * 1e-6, sampling_rate / 2 * 1e-6)  # 折り返し周波数範囲
plt.ylim(-120, 0)  # y軸の表示範囲
plt.grid(True)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(10))  # x軸を10 MHz刻みに設定
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(10))  # y軸を10 dB刻みに設定

plt.tight_layout()
plt.show()
