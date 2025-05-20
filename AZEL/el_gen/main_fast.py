import numpy as np
from numba import njit
import matplotlib.pyplot as plt

# =======================
# 定数定義（グローバル定数化）
# =======================
azv_d = 36 # AZ角速度[deg/s]
aza_d = 21.6 # AZ角加速度[deg/s2]
ta = azv_d / aza_d # AZ加速時間[s]
tp = 20 # 連続回転時、一定角速度に換算した場合のAZ回転時間[s]
tc = tp - ta # AZ加速時間を引き算して、一定角速度の期間を計算[s]
psi_yaw = 0 # ヨー角[deg]
az_center = 0 #セクタースキャン方位角の中央値[deg]
theta_pitch = 0 # ピッチ角[deg]
phi_roll = 20 # ロール角[deg]
elev_ang = 0 # 仰角設定値[deg]

# EL
gretio_el = 480 #　ELギヤ比。100だと負荷が見えてくる。
owntq = 2.51 / gretio_el # 自重トルク[Nm]
JM_el = 6.4e-7 # モータイナーシャ[kgm2]
DM_el = 1.0e-5 # モータ粘性係数[Nm/(rad/s)]
JL_el = 0.0613 / gretio_el**2 # モータ軸換算負荷イナーシャ[kgm2]
DL_el = 0.005 / gretio_el**2 # モータ軸換算負荷粘性係数[Nm/(rad/s)]
kgb_el = 1000 # バネ係数
dbdeg_el = 0.1 # バックラッシ[deg]
dbrad_el = dbdeg_el * np.pi / 180 # バックラッシ[rad]
kpp_el = 120 # 位置比例ゲイン
kvp_el = 90 # 速度比例ゲイン
kvi_el = 30 # 速度積分ゲイン
v_lim_val_el = 5000 / 60 * 2 * np.pi # 角速度リミット値[rad/s]
v_min_val_el = -v_lim_val_el
v_max_val_el = v_lim_val_el
t_lim_val_el = 0.038  # トルクリミット値[Nm] 20Wモータ 0.038[Nm]
t_min_val_el = -t_lim_val_el
t_max_val_el = t_lim_val_el
fc_el = 100 # LPFのカットオフ周波数[Hz]

# AZ
gretio_az = 100 #　AZギヤ比。100だと負荷が見えてくる。
JM_az = 6.4e-7 # モータイナーシャ[kgm2]
DM_az = 1.0e-5 # モータ粘性係数[Nm/(rad/s)]
JL_az = 0.1 / gretio_az**2 # モータ軸換算負荷イナーシャ[kgm2]
DL_az = 0.005 / gretio_az**2 # モータ軸換算負荷粘性係数[Nm/(rad/s)]
kgb_az = 1000 # バネ係数
dbdeg_az = 0.1 # バックラッシ[deg]
dbrad_az = dbdeg_az * np.pi / 180 # バックラッシ[rad]
kpp_az = 360 # 位置比例ゲイン
kvp_az = 30 # 速度比例ゲイン
kvi_az = 10 # 速度積分ゲイン
v_lim_val_az = 5000 / 60 * 2 * np.pi # 角速度リミット値[rad/s]
v_min_val_az = -v_lim_val_az
v_max_val_az = v_lim_val_az
t_lim_val_az = 0.038 # トルクリミット値[Nm] 20Wモータ 0.038[Nm]
t_min_val_az = -t_lim_val_az
t_max_val_az = t_lim_val_az
fc_az = 100 # LPFのカットオフ周波数[Hz]

# 計算設定値
sim_time = 30 # 計算時間[s]
dt = 10e-6 # 10e-6 刻み値[s]
steps = int(sim_time / dt) # 全計算数
sample_time = 0.1e-3 # 1e-3 サンプリング期間[s]
sample_n = int(sample_time / dt) # サンプリング数/期間

# =======================
# 入力信号の定義
# =======================

# 連続回転時のAZ角速度(6rpm)
@njit
def az_v_6rpm_deg_numba(t):
    if 1 <= t <= 1 + ta:
        return aza_d * (t - 1)
    elif 1 + ta < t <= 1 + ta + tc:
        return azv_d
    elif 1 + ta + tc < t <= 1 + tc + 2 * ta:
        return -aza_d * (t - (1 + ta + tc)) + azv_d
    else:
        return 0

# 連続回転時のAZ角速度(30rpm)
@njit
def az_v_30rpm_deg_numba(t):
    if 1 <= t <= 1 + 5 * ta:
        return aza_d * (t - 1)
    elif 1 + 5 * ta < t <= 5 + 5 * ta:
        return 5 * azv_d
    elif 5 + 5 * ta < t <= 5 + 10 * ta:
        return -aza_d * (t - (5 + 5 * ta)) + 5 * azv_d
    else:
        return 0
        
# セクタースキャン幅60deg時のAZ角速度
@njit
def az_v_s_60deg_numba(t):
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

# セクタースキャン幅180deg時のAZ角速度
@njit
def az_v_s_180deg_numba(t):
    if 0 <= t <= 1:
        return 0
    elif 1 < t <= 1 + ta:
        return aza_d / 2 * (t  - 1)
    elif 1 + ta < t <= 1 + 3 * ta:
        return azv_d / 2
    elif 1 + 3 * ta < t <= 1 + 4 * ta:
        return -aza_d / 2 * (t - (1 + 3 * ta)) + azv_d / 2
    elif 1 + 4 * ta < t <= 1 + 5 * ta:
        return -aza_d * (t - (1 + 4 *ta))
    elif 1 + 5 * ta < t <= 1 + 7 * ta:
        return -azv_d
    elif 1 + 7 * ta < t <= 1 + 8 * ta:
        return aza_d * (t - (1 + 7 * ta)) - azv_d
    elif 1 + 8 * ta < t <= 1 + 9 * ta:
        return aza_d / 2 * (t - (1 + 8 * ta))
    elif 1 + 9 * ta < t <= 1 + 11 * ta:
        return azv_d / 2
    elif 1 + 11 * ta < t <= 1 + 12* ta:
        return -aza_d / 2 * (t - (1 + 11 * ta)) + azv_d /2
    else:
        return 0

# ピッチ角設定
@njit
def theta_pitch_func_numba(t):
    if 0 <= t <= 1:
        return 0
    elif 1 < t <= 30:
        return theta_pitch
    else:
        return 0

# ロール角
@njit
def phi_roll_func_numba(t):
    if 0 <= t <= 1:
        return 0
    elif 1 < t <= 30:
        return phi_roll
    else:
        return 0

# ヨー角
@njit
def psi_yaw_func_numba(t):
    if 0 <= t <= 1:
        return 0
    elif 1 < t <= 30:
        return psi_yaw / 29 * (t - 1)   
    else:
        return 0
    
# 仰角設定
@njit
def elev_ang_func_numba(t):
    if 0 <= t <= 1:
        return 0
    elif 1 < t <= 30:
        return elev_ang
    else:
        return 0

# 方位角設定
@njit
def az_center_func_numba(t):
    if 0 <= t <= 1:
        return 0
    elif 1 < t <= 30:
        return az_center / 29 * (t - 1)   
    else:
        return 0
    
# 積分
@njit
def integ_numba(inp, prev_state, dt):
    new_state = prev_state + inp * dt
    return new_state

# 座標変換（機体と地面のxy面の角度差を計算）
@njit
def elev_calc_numba(COS, SIN, theta, phi):
    xe =  np.cos(theta * np.pi / 180) * COS + np.sin(phi * np.pi / 180) * np.sin(theta * np.pi / 180) * SIN
    ye =  np.cos(phi   * np.pi / 180) * SIN
    ze = -np.sin(theta * np.pi / 180) * COS + np.sin(phi * np.pi / 180) * np.cos(theta * np.pi / 180) * SIN
    r = np.sqrt(np.maximum(0, np.minimum(xe**2 + ye**2, 1))) # arccos の入力範囲を [-1, 1] に制限
    clipped_r = np.maximum(-1.0, np.minimum(r, 1.0))
    arccos_r = np.arccos(clipped_r) * 180 / np.pi
    sgn_ze = 1 if ze >= 0 else -1
    el = sgn_ze * arccos_r
    return el

# 角速度リミッター
@njit
def v_limiter_numba(inp, v_min, v_max):
    return np.minimum(np.maximum(inp, v_min),v_max)

# トルクリミッター
@njit
def t_limiter_numba(inp, t_min, t_max):
    return np.minimum(np.maximum(inp, t_min),t_max)

# LPF
@njit
def lpf_numba(x, y_prev, dt, fc):
    tau = 1 / (2 * np.pi * fc)
    alpha = dt / (tau + dt)
    y = alpha * x + (1 - alpha) * y_prev
    return y

# サンプルホールド
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
    steps, dt,
    # azv_d, aza_d, ta, tc, theta_pitch, phi_roll, elev_ang,
    gretio_el, owntq, JM_el, DM_el, JL_el, DL_el, kgb_el, dbrad_el, kpp_el, kvp_el, kvi_el,
    v_min_val_el, v_max_val_el, t_min_val_el, t_max_val_el, fc_el,
    gretio_az,        JM_az, DM_az, JL_az, DL_az, kgb_az, dbrad_az, kpp_az, kvp_az, kvi_az,
    v_min_val_az, v_max_val_az, t_min_val_az, t_max_val_az, fc_az,
    sample_n):
    
    # 初期化（共通）
    az_p_deg = 0.0 # AZ角度
    az_p_sh_deg = 0.0 # AZ角度（サンプルホールド後）

    # 初期化（EL）
    v_int_el = 0.0 # 速度積分
    th1_el = 0.0 # モータ角度
    th1dt1_el = 0.0 # モータ角速度
    th2_el = 0.0 # モータ軸換算負荷角度
    th2dt1_el = 0.0 # モータ軸換算負荷角速度
    y_prev_lpf_el = 0.0 # LPF

    th1_hist_el = np.zeros(steps)
    th1dt1_hist_el = np.zeros(steps)
    th2_hist_el = np.zeros(steps)
    th2dt1_hist_el = np.zeros(steps)
    tq_hist_el = np.zeros(steps)

    # 初期化（AZ）
    v_int_az = 0.0 # 速度積分
    th1_az = 0.0 # モータ角度
    th1dt1_az = 0.0 # モータ角速度
    th2_az = 0.0 # モータ軸換算負荷角度
    th2dt1_az = 0.0 # モータ軸換算負荷角速度
    y_prev_lpf_az = 0.0 # LPF

    th1_hist_az = np.zeros(steps)
    th1dt1_hist_az = np.zeros(steps)
    th2_hist_az = np.zeros(steps)
    th2dt1_hist_az = np.zeros(steps)
    tq_hist_az = np.zeros(steps)
    
    az_v_d_hist = np.zeros(steps)
    az_cmd_d_hist = np.zeros(steps)
    el_cmd_d_hist = np.zeros(steps)

    # ループ計算
    for i in range(steps):
        ts = i * dt
        # 連続回転/セクター幅60度/セクター幅180度を選択する。
        # az_v_deg_val = az_v_6rpm_deg_numba(ts) # 連続回転(6rpm)
        az_v_deg_val = az_v_30rpm_deg_numba(ts) # 連続回転(30rpm)
        # az_v_deg_val = az_v_s_60deg_numba(ts) # セクター幅60度
        # az_v_deg_val = az_v_s_180deg_numba(ts) # セクター幅180度

        theta_pitch_val = theta_pitch_func_numba(ts) # ピッチ角
        phi_roll_val = phi_roll_func_numba(ts) # ロール角
        elev_ang_val = elev_ang_func_numba(ts) # 仰角

        az_p_deg = integ_numba(az_v_deg_val, az_p_deg, dt) # AZ角度
        az_p_sh_deg = sh_numba(az_p_deg, az_p_sh_deg, i, sample_n) # AZ角度(サンプルホールド)
        az_cmd_deg = az_p_sh_deg + az_center_func_numba(ts) - psi_yaw_func_numba(ts)
        # AZ角度作成 = セクタースキャン角度幅 + 方位角中央値 - ヨー角
        az_cmd_rad = az_cmd_deg / 180 * np.pi # ragianに変換
        az_mtr_rad = gretio_az * az_cmd_rad # モータ軸に角度を換算  
        
        cos_az_p = np.cos(az_cmd_deg / 180 * np.pi) # 空中線正面方向ベクトルのX軸成分
        sin_az_p = np.sin(az_cmd_deg / 180 * np.pi) # 空中線正面方向ベクトルのY軸成分
        
        el_calc_deg = elev_calc_numba(cos_az_p, sin_az_p, theta_pitch_val, phi_roll_val)
        el_cmd_deg = el_calc_deg + elev_ang_val # 計算結果 + 仰角
        el_cmd_rad = el_cmd_deg / 180 * np.pi # ragianに変換
        el_mtr_rad = gretio_el * el_cmd_rad # モータ軸に角度を換算  

        # ELサーボ
        v_limit_el = v_limiter_numba(kpp_el * (el_mtr_rad - th1_el), v_min_val_el, v_max_val_el) # 角速度リミッター
        v_err_el = v_limit_el - th1dt1_el # 角速度誤差
        v_int_el = integ_numba(v_err_el, v_int_el, dt) # 角速度誤差の積分
        tqlim_el = t_limiter_numba((kvi_el * v_int_el + v_err_el) * kvp_el * (JM_el + JL_el), t_min_val_el, t_max_val_el)
        # トルク発生
        tq_el = lpf_numba(tqlim_el, y_prev_lpf_el, dt, fc_el) # LPF
        y_prev_lpf_el = tq_el
        
        tha_el = np.abs(th2_el - th1_el) # 負荷軸とモータ軸角度の差
        fn_el = -np.sign(th2_el - th1_el) * kgb_el * (tha_el - dbrad_el) # ギヤのバネモデルによる反力
        
        if tha_el < dbrad_el: # 角度がギヤのバックラッシより小さい時
            th1dt2_el = (tq_el - DM_el * th1dt1_el) / (JM_el + 1e-9) # モータ軸角加速度
            th2dt2_el = (- DL_el * th2dt1_el - owntq) / (JL_el + 1e-9) # 負荷軸モータ軸換算加速度
        else:
            th1dt2_el = (tq_el - DM_el * th1dt1_el - fn_el) / (JM_el + 1e-9)
            th2dt2_el = (fn_el - DL_el * th2dt1_el - owntq) / (JL_el + 1e-9)
        
        th1dt1_el = integ_numba(th1dt2_el, th1dt1_el, dt) # 角加速度を積分して角速度を計算
        th1_el  = integ_numba(th1dt1_el, th1_el, dt) # 角速度を積分して角度を計算
        th2dt1_el = integ_numba(th2dt2_el, th2dt1_el, dt) # 角加速度を積分して角速度を計算
        th2_el  = integ_numba(th2dt1_el, th2_el, dt) # 角速度を積分して角度を計算

        # AZサーボ
        v_limit_az = v_limiter_numba(kpp_az * (az_mtr_rad - th1_az), v_min_val_az, v_max_val_az) # 角速度リミッター
        v_err_az = v_limit_az - th1dt1_az # 角速度誤差
        v_int_az = integ_numba(v_err_az, v_int_az, dt) # 角速度誤差の積分
        tqlim_az = t_limiter_numba((kvi_az * v_int_az + v_err_az) * kvp_az * (JM_az + JL_az), t_min_val_az, t_max_val_az)
        # トルク発生
        tq_az = lpf_numba(tqlim_az, y_prev_lpf_az, dt, fc_az) # LPF
        y_prev_lpf_az = tq_az
        
        tha_az = np.abs(th2_az - th1_az) # 負荷軸とモータ軸角度の差
        fn_az = -np.sign(th2_az - th1_az) * kgb_az * (tha_az - dbrad_az) # ギヤのバネモデルによる反力
        
        if tha_az < dbrad_az: # 角度がギヤのバックラッシより小さい時
            th1dt2_az = (tq_az - DM_az * th1dt1_az) / (JM_az + 1e-9) # モータ軸角加速度
            th2dt2_az = (- DL_az * th2dt1_az) / (JL_az + 1e-9) # 負荷軸モータ軸換算加速度
        else:
            th1dt2_az = (tq_az - DM_az * th1dt1_az - fn_az) / (JM_az + 1e-9)
            th2dt2_az = (fn_az - DL_az * th2dt1_az) / (JL_az + 1e-9)
        
        th1dt1_az = integ_numba(th1dt2_az, th1dt1_az, dt) # 角加速度を積分して角速度を計算
        th1_az  = integ_numba(th1dt1_az, th1_az, dt) # 角速度を積分して角度を計算
        th2dt1_az = integ_numba(th2dt2_az, th2dt1_az, dt) # 角加速度を積分して角速度を計算
        th2_az  = integ_numba(th2dt1_az, th2_az, dt) # 角速度を積分して角度を計算

        # 記録
        # AZ/EL共通
        el_cmd_d_hist[i] = el_cmd_deg
        az_v_d_hist[i] = az_v_deg_val
        az_cmd_d_hist[i] = az_cmd_deg
        
        # EL
        th1_hist_el[i] = th1_el
        th1dt1_hist_el[i] = th1dt1_el
        th2_hist_el[i] = th2_el
        th2dt1_hist_el[i] = th2dt1_el
        tq_hist_el[i] = tq_el

        # AZ
        th1_hist_az[i] = th1_az
        th1dt1_hist_az[i] = th1dt1_az
        th2_hist_az[i] = th2_az
        th2dt1_hist_az[i] = th2dt1_az
        tq_hist_az[i] = tq_az

    return az_v_d_hist, az_cmd_d_hist, el_cmd_d_hist, tq_hist_el, th1_hist_el, th2_hist_el, tq_hist_az, th1_hist_az, th2_hist_az

# =======================
# 実行 & 可視化
# =======================
az_v_d_hist, az_cmd_d_hist, el_cmd_d_hist, tq_hist_el, th1_hist_el, th2_hist_el, tq_hist_az, th1_hist_az, th2_hist_az = simulate_numba(
    steps, dt,
    # azv_d, aza_d, ta, tc, theta_pitch, phi_roll, elev_ang,
    gretio_el, owntq, JM_el, DM_el, JL_el, DL_el, kgb_el, dbrad_el, kpp_el, kvp_el, kvi_el, v_min_val_el,
    v_max_val_el, t_min_val_el, t_max_val_el, fc_el,
    gretio_az,        JM_az, DM_az, JL_az, DL_az, kgb_az, dbrad_az, kpp_az, kvp_az, kvi_az, v_min_val_az,
    v_max_val_az, t_min_val_az, t_max_val_az, fc_az,
    sample_n)

time = np.linspace(0, sim_time, steps)

fig, axs = plt.subplots(2, 3, figsize=(10, 6)) # 2行3列

# 左上：el_cmd_deg, el_th2_deg
axs[0, 0].plot(time, el_cmd_d_hist, label='el_cmd_deg')
axs[0, 0].plot(time, th2_hist_el/np.pi*180/gretio_el, label='el_th2_deg')
axs[0, 0].set_title('EL_Elevation_Angles')
axs[0, 0].set_xlabel('Time [s]')
axs[0, 0].set_ylabel('Angle [deg]')
axs[0, 0].set_xticks(np.arange(0, 31, 5)) # 5秒刻み
axs[0, 0].legend()
axs[0, 0].grid(True)

# 中上：el_delta_deg（y軸設定あり）
axs[0, 1].plot(time, el_cmd_d_hist - th2_hist_el/np.pi*180/gretio_el, label='EL_delta_deg')
axs[0, 1].set_title('EL_Delta_Angles')
axs[0, 1].set_xlabel('Time [s]')
axs[0, 1].set_ylabel('Angle [deg]')
axs[0, 1].set_xticks(np.arange(0, 31, 5)) # 5秒刻み
axs[0, 1].set_yticks(np.arange(-2, 2.1, 0.4)) # 0.2度刻み
axs[0, 1].legend()
axs[0, 1].grid(True)

# 右上：az_delta_deg（y軸設定あり）
axs[0, 2].plot(time, az_cmd_d_hist - th2_hist_az/np.pi*180/gretio_az, label='AZ_delta_deg')
axs[0, 2].set_title('AZ_Delta_Angles')
axs[0, 2].set_xlabel('Time [s]')
axs[0, 2].set_ylabel('Angle [deg]')
axs[0, 2].set_xticks(np.arange(0, 31, 5)) # 5秒刻み
# axs[0, 2].set_yticks(np.arange(-2, 2.1, 0.4)) # 0.2度刻み
axs[0, 2].legend()
axs[0, 2].grid(True)

# 左下：az_v_deg az_p_deg
axs[1, 0].plot(time, az_v_d_hist, label='az_v_deg')
axs[1, 0].plot(time, az_cmd_d_hist, label='az_cmd_deg')
axs[1, 0].set_title('az_v_deg/az_cmd_deg')
axs[1, 0].set_xlabel('Time [s]')
axs[1, 0].set_ylabel('Angle v [deg/s] p [deg]')
axs[1, 0].set_xticks(np.arange(0, 31, 5)) # 5秒刻み
axs[1, 0].set_yticks(np.arange(-90, 2311, 180)) # 180度刻み
axs[1, 0].legend()
axs[1, 0].grid(True)

# 中下：el_tq
axs[1, 1].plot(time, tq_hist_el, label='EL_tq')
axs[1, 1].set_title('EL_tq (Torque)')
axs[1, 1].set_xlabel('Time [s]')
axs[1, 1].set_ylabel('Torque [Nm]')
axs[1, 1].set_xticks(np.arange(0, 31, 5)) # 5秒刻み
axs[1, 1].set_yticks(np.arange(-0.04, 0.045, 0.01)) # 0.01度刻み
axs[1, 1].legend()
axs[1, 1].grid(True)

# 右下：az_tq
axs[1, 2].plot(time, tq_hist_az, label='AZ_tq')
axs[1, 2].set_title('AZ_tq (Torque)')
axs[1, 2].set_xlabel('Time [s]')
axs[1, 2].set_ylabel('Torque [Nm]')
axs[1, 2].set_xticks(np.arange(0, 31, 5)) # 5秒刻み
axs[1, 2].set_yticks(np.arange(-0.04, 0.045, 0.01)) # 0.01度刻み
axs[1, 2].legend()
axs[1, 2].grid(True)

# 全体レイアウト調整と保存
plt.tight_layout()
plt.savefig("sim_out.png", dpi=100)
plt.show()
