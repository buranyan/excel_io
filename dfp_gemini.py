import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def trapezoidal_wave(t, period, rise_time, on_time, fall_time):
    """台形波を生成する関数"""
    t = t % period  # 周期的な繰り返し
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
pulse_width = rise_time + on_time + fall_time # パルス幅
duration = pulse_width * 2  # パルス幅の2倍に変更

# 時間軸の生成(計算範囲はperiod)
time = np.arange(0, period, 1 / sampling_rate) # periodの2倍に変更

# 台形波の生成 (電力)
power_signal = np.array([trapezoidal_wave(t, period, rise_time, on_time, fall_time) for t in time])

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

# デシベル表示に変換
power_spectrum_db = 10 * np.log10(power_spectrum)

# 最大値をデシベルで取得
max_power_db = np.max(power_spectrum_db)

# グラフの表示
plt.figure(figsize=(6, 6))

# 時間領域の信号 (電力)
plt.subplot(2, 1, 1)
plt.plot(time, power_signal)
plt.title("Trapezoidal Wave (Power)")
plt.xlabel("Time [s]")
plt.ylabel("Power [mW]")
plt.xlim(0, duration) # 表示範囲をdurationに修正

# 周波数領域の信号 (パワースペクトル dBm)
plt.subplot(2, 1, 2)
plt.plot(frequencies, power_spectrum_db)
plt.title(f"Power Spectrum (dBm), Max: {max_power_db:.2f} dBm") # 最大値をタイトルに表示
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power (dBm)")
plt.xlim(-sampling_rate / 2, sampling_rate / 2)  # 表示範囲を調整
plt.ylim(-120, 0) # y軸の表示範囲を-120から0に設定
plt.grid(True)
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(10)) # y軸を10刻みに設定

plt.tight_layout()
plt.show()