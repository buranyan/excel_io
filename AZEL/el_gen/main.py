import numpy as np
import matplotlib.pyplot as plt
import subcircuits  # subcircuits.py から関数をインポート

# シミュレーション設定 (ascファイルから取得)
t_start = 0
t_end = 30
dt = 10e-6  # より小さなタイムステップを試す
t_eval = np.arange(t_start, t_end, dt)

# 初期条件 (ascファイルから取得)
th1 = 0
th1dt1 = 0
th2 = 0
th2_deg = 0
th2dt1 = 0
azv_d = 36
aza_d = 21.6
ta = azv_d / aza_d
tp = 20
tc = tp - ta
theta_pitch = 10
phi_roll = 20
elev_ang = -20

# 履歴リスト
th1dt1_history = [th1dt1]
th1_history = [th1]
th2dt1_history = [th2dt1]
th2_history = [th2]
th2_deg_history = [th2_deg]
el_calc_deg_history = [0]
el_cmd_deg_history = [0]
delta_deg_history = [0]
az_v_rotate_deg_history = [0]
az_v_sector_deg_history = [0]
az_p_deg_history = [0]
tq_history = [0]

# 積分器の状態
integ_az_p_deg_state = 0
integ_kvi_state = 0
integ_th1dt1_state = 0
integ_th2dt1_state = 0
integ_th1_state = 0
integ_th2_state = 0

# 入力信号定義
def az_v_rotate_deg(t):
    if 1 <= t <= 1 + ta:
        return 0
    elif 1 + ta < t <= 1 + tc:
        return 0
    else:
        return 0

def az_v_sector_deg(t):
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

def theta_pitch_func(t):
    if 0 <= t <= 1:
        return 0
    elif 1 < t <= 28:
        return theta_pitch
    else:
        return 0

def phi_roll_func(t):
    if 0 <= t <= 1:
        return 0
    elif 1 < t <= 28:
        return phi_roll
    else:
        return 0

def elev_ang_func(t):
    if 0 <= t <= 1:
        return 0
    elif 1 < t <= 28:
        return elev_ang
    else:
        return 0

# シミュレーションループ
for t_idx, t in enumerate(t_eval):
    az_v_rotate_deg_val = az_v_rotate_deg(t)
    az_v_sector_deg_val = az_v_sector_deg(t)
    theta_pitch_val = theta_pitch_func(t)
    phi_roll_val = phi_roll_func(t)
    elev_ang_val = elev_ang_func(t)

    az_p_deg, integ_az_p_deg_state = subcircuits.integ(az_v_sector_deg_val, integ_az_p_deg_state, dt)
    cos_az_p_deg = subcircuits.cos_gen(az_p_deg)
    sin_az_p_deg = subcircuits.sin_gen(az_p_deg)
    el_calc_deg = subcircuits.elev_calc(cos_az_p_deg, sin_az_p_deg, theta_pitch_val, phi_roll_val)
    el_cmd_deg = subcircuits.adder(el_calc_deg, elev_ang_val)
    el_cmd_rad = subcircuits.deg_to_rad(el_cmd_deg)
    mtr_rad = subcircuits.gear_el(el_cmd_rad)
    
    err_v = subcircuits.sub(subcircuits.v_limiter(subcircuits.kpp_gain( subcircuits.sub(mtr_rad, th1_history[-1]))),
        th1dt1_history[-1])
    kvi_n, integ_kvi_state = subcircuits.integ(err_v, integ_kvi_state, dt)
    tq_val = subcircuits.t_limiter(
        subcircuits.t_gen(subcircuits.kvp_gain(subcircuits.adder(err_v, subcircuits.kvi_gain(kvi_n)))),
        t_min=subcircuits.t_min_val,
        t_max=subcircuits.t_max_val
    )
    tha, fn = subcircuits.f_gen(th1_history[-1], th2_history[-1])
    th1dt2_val = subcircuits.th1dt2(tq_val, tha, fn, th1dt1_history[-1])
    th1dt1, integ_th1dt1_state = subcircuits.integ(th1dt2_val, th1dt1_history[-1], dt)
    th1, integ_th1_state = subcircuits.integ(th1dt1, th1_history[-1], dt)
    
    th2dt2_val = subcircuits.th2dt2_el(tha, fn, th2_history[-1], th2dt1_history[-1])
    th2dt1, integ_th2dt1_state = subcircuits.integ(th2dt2_val, th2dt1_history[-1], dt)
    th2, integ_th2_state = subcircuits.integ(th2dt1, th2_history[-1], dt)
    th2_deg = th2 * 180 / np.pi / 480
    delta_deg = el_cmd_deg - th2_deg

    # 履歴更新
    th1_history.append(th1)
    th1dt1_history.append(th1dt1)
    th2_history.append(th2)
    th2_deg_history.append(th2_deg)
    th2dt1_history.append(th2dt1)
    el_calc_deg_history.append(el_calc_deg)
    el_cmd_deg_history.append(el_cmd_deg)
    delta_deg_history.append(delta_deg)
    az_v_rotate_deg_history.append(az_v_rotate_deg_val)
    az_v_sector_deg_history.append(az_v_sector_deg_val)
    az_p_deg_history.append(az_p_deg)
    tq_history.append(tq_val)

# === 結果プロットと保存 ===

fig, axs = plt.subplots(2, 2, figsize=(9, 6))  # 2行2列

# 左上：el_calc_deg, el_cmd_deg, th2_deg
axs[0, 0].plot(t_eval, el_calc_deg_history[:len(t_eval)], label='el_calc_deg')
axs[0, 0].plot(t_eval, el_cmd_deg_history[:len(t_eval)], label='el_cmd_deg')
axs[0, 0].plot(t_eval, th2_deg_history[:len(t_eval)], label='th2_deg')
axs[0, 0].set_title('Elevation Angles')
axs[0, 0].set_xlabel('Time [s]')
axs[0, 0].set_ylabel('Angle [deg]')
axs[0, 0].legend()
axs[0, 0].grid(True)

# 右上：delta_deg（y軸設定あり）
axs[0, 1].plot(t_eval, delta_deg_history[:len(t_eval)], label='delta_deg')
axs[0, 1].set_title('delta_deg')
axs[0, 1].set_xlabel('Time [s]')
axs[0, 1].set_ylabel('Angle [deg]')
axs[0, 1].set_ylim(-1, 1)
axs[0, 1].set_yticks(np.arange(-1, 1.1, 0.2))  # 0.2度刻み
axs[0, 1].legend()
axs[0, 1].grid(True)

# 左下：az_p_deg
axs[1, 0].plot(t_eval, az_p_deg_history[:len(t_eval)], label='az_p_deg')
axs[1, 0].set_title('az_p_deg')
axs[1, 0].set_xlabel('Time [s]')
axs[1, 0].set_ylabel('Angle [deg]')
axs[1, 0].legend()
axs[1, 0].grid(True)

# 右下：tq
axs[1, 1].plot(t_eval, tq_history[:len(t_eval)], label='tq')
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
