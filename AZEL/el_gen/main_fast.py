import numpy as np
from numba import njit
import matplotlib.pyplot as plt

# =======================
# 定数定義（グローバル定数化）
# =======================
azv_d = 36
aza_d = 21.6
ta = azv_d / aza_d
tp = 20
tc = tp - ta
theta_pitch = 10
phi_roll = 20
elev_ang = -20

gretio_el = 480 #100だと負荷が見えてくる。
owntq = 2.51 / gretio_el
JM = 6.4e-7
DM = 1.0e-5
JL = 0.0613 / gretio_el**2
DL = 0.005 / gretio_el**2
kgb = 1000
dbdeg = 0.1
dbrad = dbdeg * np.pi / 180
kpp = 30
kvp = 90
kvi = 30
v_lim_val = 5000 / 60 * 2 * np.pi
v_min_val = -v_lim_val
v_max_val = v_lim_val
t_lim_val = 0.038  # 20Wモータ
t_min_val = -t_lim_val
t_max_val = t_lim_val
fc = 10e3 # LPFのカットオフ周波数

sim_time = 30 # 計算時間
dt = 10e-6 # 10e-6 刻み値
steps = int(sim_time / dt) # 全サンプリング数
sample_time = 10e-3 # サンプリング期間
sample_n = int(sample_time / dt) # サンプリング数/期間

# =======================
# 入力信号の定義
# =======================

# 連続回転時のAZ角速度
@njit
def az_v_rotate_deg_numba(t):
    if 1 <= t <= 1 + ta:
        return 0
    elif 1 + ta < t <= 1 + tc:
        return 0
    else:
        return 0
        
# セクタースキャン時のAZ角速度
@njit
def az_v_sector_deg_numba(t):
    if 0 <= t <= 1:
        return 0
    elif 1 < t <= 1 + ta:
        return aza_d / 2 * (t  - 1)
    elif 1 + ta < t <= 1 + 2 * ta:
        return -aza_d / 2 * (t - (1 + ta)) + azv_d / 2
    elif 1 + 2 * ta < t <= 1 + 3 * ta:
        return -aza_d * (t - (1 + 2 *ta))
    elif 1 + 3 * ta < t <= 1 + 5 * ta:
        return aza_d * (t - (1 + 3 * ta)) - azv_d
    elif 1 + 5 * ta < t <= 1 + 7 * ta:
        return -aza_d * (t - (1 + 5 *ta)) + azv_d
    elif 1 + 7 * ta < t <= 1 + 9 * ta:
        return aza_d* (t - (1 + 7 * ta)) - azv_d
    elif 1 + 9 * ta < t <= 1 + 11 * ta:
        return -aza_d * (t - (1 + 9 *ta)) + azv_d
    elif 1 + 11 * ta < t <= 1 + 12* ta:
        return aza_d * (t - (1 + 11 * ta)) - azv_d
    elif 1 + 12 * ta < t <= 1 + 13* ta:
        return aza_d / 2 * (t - (1 + 12 * ta))
    elif 1 + 13 * ta < t <= 1 + 14 * ta:
        return -aza_d / 2 *(t - (1 + 13 * ta)) + azv_d / 2
    else:
        return 0

# ピッチ角
@njit
def theta_pitch_func_numba(t):
    if 0 <= t <= 1:
        return 0
    elif 1 < t <= 28:
        return theta_pitch
    else:
        return 0

# ロール角
@njit
def phi_roll_func_numba(t):
    if 0 <= t <= 1:
        return 0
    elif 1 < t <= 28:
        return phi_roll
    else:
        return 0

# 仰角
@njit
def elev_ang_func_numba(t):
    if 0 <= t <= 1:
        return 0
    elif 1 < t <= 28:
        return elev_ang
    else:
        return 0

# =======================
# 機能維持する関数群（Numba対応）
# =======================

# 積分
@njit
def integ_numba(inp, prev_state, dt):
    new_state = prev_state + inp * dt
    return new_state

# 座標変換（オイラー角）
@njit
def elev_calc_numba(COS, SIN, theta, phi):
    xe = np.cos(theta * np.pi / 180) * COS + np.sin(phi * np.pi / 180) * np.sin(theta * np.pi / 180) * SIN
    ye = np.cos(phi * np.pi / 180) * SIN
    ze = -np.sin(theta * np.pi / 180) * COS + np.sin(phi * np.pi / 180) * np.cos(theta * np.pi / 180) * SIN
    r = np.sqrt(max(0,min(xe**2 + ye**2, 1))) # arccos の入力範囲を [-1, 1] に制限
    clipped_r = max(-1.0, min(r, 1.0))
    arccos_r = np.arccos(clipped_r) * 180 / np.pi
    sgn_ze = 1 if ze >= 0 else -1
    el = sgn_ze * arccos_r
    return el

# 角速度リミッター
@njit
def v_limiter_numba(inp, v_min=v_min_val, v_max=v_max_val):
    return max(v_min, min(inp, v_max))

# トルクリミッター
@njit
def t_limiter_numba(inp, t_min=t_min_val, t_max=t_max_val):
    return max(t_min, min(inp, t_max))

# LPF
@njit
def lpf_numba(x, y_prev, dt, fc):
    tau = 1 / (2 * np.pi * fc)
    alpha = dt / (tau + dt)
    y = alpha * x + (1 - alpha) * y_prev
    return y

@njit
def sh_numba(inp, hold_p, i_n, sample_n):
    if i_n % sample_n == 0:
        hold = inp
    else:
        hold = hold_p
    return hold

# =======================
# メインループ用関数
# =======================
@njit
def simulate_numba(
    steps, dt, azv_d, aza_d, ta, tc, theta_pitch, phi_roll, elev_ang,
    gretio_el, owntq, JM, DM, JL, DL, kgb, dbrad, kpp, kvp, kvi, v_min_val,
    v_max_val, t_min_val, t_max_val, fc, sample_n):
    
    # 初期化
    az_p_deg = 0.0 # AZ角度
    az_p_sh_deg = 0.0 # AZ角度（サンプルホールド後）
    v_int = 0.0 # 速度積分
    th1 = 0.0
    th1dt1 = 0.0
    th2 = 0.0
    th2dt1 = 0.0
    y_prev_lpf = 0.0 # LPF
    
    th1_hist = np.zeros(steps)
    th1dt1_hist = np.zeros(steps)
    th2_hist = np.zeros(steps)
    th2dt1_hist = np.zeros(steps)
    tq_hist = np.zeros(steps)
    az_v_s_d_hist = np.zeros(steps)
    el_cmd_d_hist = np.zeros(steps)

    # ループ計算
    for i in range(steps):
        ts = i * dt
        az_v_rotate_deg_val = az_v_rotate_deg_numba(ts)
        az_v_sector_deg_val = az_v_sector_deg_numba(ts)
        
        theta_pitch_val = theta_pitch_func_numba(ts)
        phi_roll_val = phi_roll_func_numba(ts)
        elev_ang_val = elev_ang_func_numba(ts)

        az_p_deg = integ_numba(az_v_sector_deg_val, az_p_deg, dt)
        az_p_sh_deg = sh_numba(az_p_deg, az_p_sh_deg, i, sample_n)
        
        cos_az_p = np.cos(az_p_sh_deg / 180 * np.pi)
        sin_az_p = np.sin(az_p_sh_deg / 180 * np.pi)
        
        el_calc_deg = elev_calc_numba(cos_az_p, sin_az_p, theta_pitch_val, phi_roll_val)
        el_cmd_deg = el_calc_deg + elev_ang_val
        el_cmd_rad = el_cmd_deg / 180 * np.pi
        mtr_rad = gretio_el * el_cmd_rad   

        v_limit = v_limiter_numba(kpp * (mtr_rad - th1))
        v_err = v_limit - th1dt1
        v_int = integ_numba(v_err, v_int, dt)
        tq = t_limiter_numba((kvi * v_int + v_err) * kvp* (JM + JL))
        tq = lpf_numba(tq, y_prev_lpf, dt, fc)    
        tha = np.abs(th2 - th1)
        fn = -np.sign(th2 - th1) * kgb * (tha - dbrad)
        
        if tha < dbrad:
            th1dt2 = (tq - DM * th1dt1) / (JM + 1e-9)
            th2dt2 = (- DL * th2dt1 - owntq) / (JL + 1e-9)
        else:
            th1dt2 = (tq - DM * th1dt1 - fn) / (JM + 1e-9)
            th2dt2 = (fn - DL * th2dt1 - owntq) / (JL + 1e-9)
        
        th1dt1 = integ_numba(th1dt2, th1dt1, dt)
        th1  = integ_numba(th1dt1, th1, dt)
        th2dt1 = integ_numba(th2dt2, th2dt1, dt)
        th2  = integ_numba(th2dt1, th2, dt)

        # 記録
        th1_hist[i] = th1
        th1dt1_hist[i] = th1dt1
        th2_hist[i] = th2
        th2dt1_hist[i] = th2dt1
        tq_hist[i] = tq
        el_cmd_d_hist[i] = el_cmd_deg
        az_v_s_d_hist[i] = az_v_sector_deg_val

    return az_v_s_d_hist, el_cmd_d_hist, tq_hist, th1_hist, th2_hist

# =======================
# 実行 & 可視化
# =======================
az_v_s_d_hist, el_cmd_d_hist, tq_hist, th1_hist, th2_hist = simulate_numba(
    steps, dt, azv_d, aza_d, ta, tc, theta_pitch, phi_roll, elev_ang,
    gretio_el, owntq, JM, DM, JL, DL, kgb, dbrad, kpp, kvp, kvi, v_min_val,
    v_max_val, t_min_val, t_max_val, fc, sample_n)

time = np.linspace(0, sim_time, steps)

fig, axs = plt.subplots(2, 2, figsize=(9, 6))  # 2行2列

# 左上：el_cmd_deg, th2_deg
axs[0, 0].plot(time, el_cmd_d_hist, label='el_cmd_deg')
axs[0, 0].plot(time, th2_hist/np.pi*180/gretio_el, label='th2_deg')
axs[0, 0].set_title('Elevation Angles')
axs[0, 0].set_xlabel('Time [s]')
axs[0, 0].set_ylabel('Angle [deg]')
axs[0, 0].legend()
axs[0, 0].grid(True)

# 右上：delta_deg（y軸設定あり）
axs[0, 1].plot(time, el_cmd_d_hist - th2_hist/np.pi*180/gretio_el, label='delta_deg')
axs[0, 1].set_title('delta_deg')
axs[0, 1].set_xlabel('Time [s]')
axs[0, 1].set_ylabel('Angle [deg]')
axs[0, 1].set_ylim(-1, 1)
axs[0, 1].set_yticks(np.arange(-1, 1.1, 0.2))  # 0.2度刻み
axs[0, 1].legend()
axs[0, 1].grid(True)

# 左下：az_p_deg
axs[1, 0].plot(time, az_v_s_d_hist, label='az_v_s_deg')
axs[1, 0].set_title('az_v_s_deg')
axs[1, 0].set_xlabel('Time [s]')
axs[1, 0].set_ylabel('Angle [deg]')
axs[1, 0].legend()
axs[1, 0].grid(True)

# 右下：tq
axs[1, 1].plot(time, tq_hist, label='tq')
axs[1, 1].set_title('tq (Torque)')
axs[1, 1].set_xlabel('Time [s]')
axs[1, 1].set_ylabel('Torque')
axs[1, 1].set_ylim(-0.04, 0.04)
axs[1, 1].set_yticks(np.arange(-0.04, 0.045, 0.01))  # 0.01度刻み
axs[1, 1].legend()
axs[1, 1].grid(True)

# 全体レイアウト調整と保存
plt.tight_layout()
plt.savefig("simulation_results.png", dpi=300)
plt.show()
