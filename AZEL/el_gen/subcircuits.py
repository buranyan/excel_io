import numpy as np

# パラメータ定義 (ascファイルから取得)
gretio_el = 480
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
v_lim_val = 5000 / 60 * 2 * np.pi  # v_lim.txt のパラメータ名と合わせる
v_min_val = -v_lim_val
v_max_val = v_lim_val
t_lim_val = 0.038  # t_lim.txt のパラメータ名と合わせる
t_min_val = -t_lim_val
t_max_val = t_lim_val

# サブサーキットの関数定義
def th2dt2_el(tha, fn, th2, th2dt1):
    # 分母がゼロになるのを避けるための微小な値を加える (必要に応じて)
    return (fn - DL * th2dt1 - owntq * np.cos(th2 / gretio_el)) / (JL + 1e-9)

def elev_calc(COS, SIN, theta, phi):
    xe = np.cos(theta * np.pi / 180) * COS + np.sin(phi * np.pi / 180) * np.sin(theta * np.pi / 180) * SIN
    ye = np.cos(phi * np.pi / 180) * SIN
    ze = -np.sin(theta * np.pi / 180) * COS + np.sin(phi * np.pi / 180) * np.cos(theta * np.pi / 180) * SIN
    r = np.sqrt(np.clip(xe**2 + ye**2, 0, 1)) # arccos の入力範囲を [-1, 1] に制限
    arccos_r = np.arccos(np.clip(r, -1, 1)) * 180 / np.pi
    sgn_ze = 1 if ze >= 0 else -1
    el = sgn_ze * arccos_r
    return el

def gear_el(inp):
    return inp * gretio_el

def cos_gen(inp):
    return np.cos(inp / 180 * np.pi)

def sin_gen(inp):
    return np.sin(inp / 180 * np.pi)

def deg_to_rad(inp):
    return inp * np.pi / 180

def sub(in1, in2):
    return in1 - in2

def th1dt2(tq, tha, fn, th1dt1):
    # 分母がゼロになるのを避けるための微小な値を加える (必要に応じて)
    return (tq - DM * th1dt1 - fn) / (JM + 1e-9)

def integ(inp, prev_state, dt):
    new_state = prev_state + inp * dt
    return new_state, new_state

def f_gen(th1, th2):
    tha = abs(th2 - th1)
    fn = -np.sign(th2 - th1) * kgb * (tha - dbrad)
    return tha, fn

def v_limiter(inp, v_min=v_min_val, v_max=v_max_val):
    return np.clip(inp, v_min, v_max)

def t_limiter(inp, t_min=t_min_val, t_max=t_max_val):
    return np.clip(inp, t_min, t_max)

def t_gen(inp):
    return inp * (JM + JL)

def kvp_gain(inp):
    return inp * kvp

def kvi_gain(inp):
    return inp * kvi

def kpp_gain(inp):
    return inp * kpp

def adder(in1, in2):
    return in1 + in2
