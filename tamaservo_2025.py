import numpy as np
import matplotlib.pyplot as plt

# パラメータ
PKpp = 10.0                        # 位置比例ゲイン
PKvp = 30.0                        # 速度比例ゲイン
PKvi = 10.0                        # 速度積分ゲイン

PlimV = 2000                       # 速度リミット[rpm]
Limvr = PlimV / 60.0 * 2.0 * np.pi # 速度リミット[rad/s]
PlimT = 8.76                       # トルクリミット[Nm]
        
PDm = 0.0001                       # モータ粘性抵抗[Nm s/rad]
PJm = 0.00188                      # モータ慣性モーメント[kg m^2]
PDL = 0.0001                       # AZ粘性抵抗[Nm s/rad]
PJL = 0.0196                       # AZ慣性モーメント[kg m^2] C2=0.0189 C3=0.0196

Pkgb = 10.0                        # GBバネ定数[Nm/rad]
Pdb = 0.1                          # 不感帯[deg]
DB = Pdb * np.pi / 180.0           # 不感帯[rad]

Dt = 0.0001                        # 刻み値[s]
Tbase = -2.5                       # 計算開始時間[s]
# Tendで計算時間を調整する
# 計算終了時間[s]
Tend = 12.5                        
Ts = Tbase                         # グラフX軸始点[s]
Te = Tend + Dt                     # グラフX軸終点[s]　

# 指令角度を作成する
# 上位の指令角度を入力する(azposd)
# [deg] at az軸
azposd = 180.0                     
azposr = np.pi / 180.0 * azposd    # [rad] at az軸
gretio = 81.0 * 110.0 / 23.0       # 減速比
mtrpos = azposr * gretio           # [rad] at motor
# print(mtrpos)
# モータ回転速度を入力する(mtrvm)
# [rpm] at motor
mtrvm = 2000                       
mtrvs = mtrvm / 60.0               # [rps] at motor
mtrvr = mtrvs * 2 * np.pi          # [rad/s] at motor
mtrvr100ms = mtrvr * 0.1           # [rad/100ms] at motor

# 100ms毎の角度をvcmd1に割り当てる
strg = mtrpos                      #[rad]目標までの角度
Tend100ms = int(Tend * 10)
vcmd1 = []

for i in range(Tend100ms):
    if strg > mtrvr100ms:
        setaz = mtrvr100ms
        strg = strg - mtrvr100ms
    else:
        setaz = strg
        strg = 0
    vcmd1.append(setaz)    

# 速度移動平均の準備(0秒以前のゼロデータをvcmd1の先頭に追加する)
num1 = 25                          # 移動平均数
zero1a = np.zeros(num1)            # num1個のゼロデータ
vcmd1 = np.append(zero1a,vcmd1)    # ゼロデータとvcmd1を結合する

# 1ms毎にvcmd1データを加工する。グラフ作成に使用する。

a1 =[]
vn1 = len(vcmd1)
cp1 = int(0.1 / Dt)                     # 0.1秒に同じ値のデータを並べる
an1 = vn1 * cp1

for i in range(vn1):
    va = vcmd1[i]
    ba = np.full(cp1,va)
    a1 = np.append(a1,ba)

# 1ms毎の速度データ
org = a1

# 速度移動平均(100ms毎にnum1個で移動平均)
b1 = np.ones(num1)/num1
vcmd1ave = np.convolve(vcmd1, b1, mode='valid')
# データ配列調整(先頭にゼロデータを追加)
zero1b = np.zeros(num1-1)
value1 = np.append(zero1b,vcmd1ave)

# CPUカードからサーボアンプへの位置指令を作成する
pos =[]
sumpos = 0
for i in range(vn1):
    sumpos += value1[i]
    pos.append(sumpos)

# サーボアンプ内で100ms毎位置指令を1ms毎に加工する
a2 =[]
vn2 = len(pos)
cp2 = int(0.1 / Dt)
an2 = vn2 * cp2

for i in range(vn2):
    vb = pos[i]
    bb = np.full(cp2,vb)
    a2 = np.append(a2,bb)

# X軸
x = np.linspace(Ts,Tend,an2)
# Y軸
pcmd2 = a2

# 位置移動平均(1ms毎にnum2個で移動平均)
num2 = int(0.1 / Dt)
b2 = np.ones(num2)/num2
pcmd2ave = np.convolve(pcmd2, b2, mode='valid')
# データ配列調整(先頭にゼロデータを追加)
zero2b = np.zeros(num2-1)
value2 = np.append(zero2b, pcmd2ave)
# print(value2[an2-1])

# 軌道を見たいときに#を外す
# plt.plot(x, org  ,'g-' ,label='input')
# plt.plot(x, pcmd2,'r-' ,label='pos')
# plt.plot(x,value2,'b-' ,label='posave')
# plt.legend()

# 表示
# plt.show()

# リミッター(速度、トルク）
def Limit(D, l):            
    data = D
    if D > l:
        data = l
    elif D < -l:
        data = -l    
    return data

# 積分
def FI(NV, PV):            
    return PV + NV * Dt

# グラフ表示用リスト初期化
LT = []                               # BaseT[s]
LV = []                               # フィードバック速度[rad/s]
LQ = []                               # トルク[Nm]
LP = []                               # フィードバック角度[rad]
LA = []                               # モニタ

# 初期値
Th1    = 0.0
Th1dt1 = 0.0
Th2    = 0.0
Th2dt1 = 0.0
Tha    = 0.0
Evi    = 0.0
cnt    = 0

while True:
    PC = value2[cnt]                  # 指令角度のセット[rad]
# print(cnt)
    Ep = PC - Th1                     # 指令角度 － フィードバック角度[rad]
    Vp = PKpp * Ep                    # 位置比例ゲイン X 位置誤差[rad/s]
    Vpl = Limit(Vp, Limvr)            # 速度リミッタ[rad/s]
    Ev = Vpl - Th1dt1                 # 指令速度 － フィードバック速度[rad]
    Evi = FI(Ev, Evi)                 # 速度誤差積分
    Tq = PKvp * (PJm + PJL) * (Ev + PKvi * Evi)
                                      # トルク[Nm]
    
    Tql = Limit(Tq, PlimT)            # トルクリミッタ[Nm]
    
    Tha = np.abs(Th2 - Th1)           # アンテナ角度 － GB角度の絶対値[rad]
    Sign = np.sign(Th2 - Th1)         # 符号
    fn = -Sign * Pkgb * (Tha - DB)    # 反力トルク[Nm]

    # GB角度、角速度、角加速度[rad] 
    if Tha < DB:
        Th1dt2 = (Tql - PDm * Th1dt1) / PJm
    else:
        Th1dt2 = (Tql - PDm * Th1dt1 - fn) / PJm
    Th1dt1 = FI(Th1dt2, Th1dt1)
    Th1    = FI(Th1dt1, Th1)

    # アンテナ角度、角速度、角加速度[rad] 
    if Tha < DB:
        Th2dt2 = ( - PDL * Th2dt1) / PJL
    else:
        Th2dt2 = (fn - PDL * Th2dt1) / PJL
    Th2dt1 = FI(Th2dt2, Th2dt1)
    Th2    = FI(Th2dt1, Th2)


    LT.append(Tbase)               # BaseTのリスト化 
    LV.append(Th1dt1)              # DB速度のリスト化
    LQ.append(Tql)                 # モータトルクのリスト化
    LP.append(Th1)                 # 角度のリスト化
    LA.append(Th2 - Th1)           # モニタのリスト化
    
    Tbase = Tbase + Dt             # 次回の計算時間
    cnt = cnt + 1                  # カウンター
    if cnt == an2 - 1:             # 計算時間が終了時間以上で計算終了
        break

print('時間=',Tbase)
print('速度=',Th1dt1)
print('トルク=',Tql)
print('角度=',Th1)

# 計算結果の表示
x = LT                             # X軸
y1 = LV                            # Y軸(LV)
y2 = LQ                            # Y軸(LQ)
y3 = LP                            # Y軸(LP)
y4 = LA                            # Y軸(LA)

# グラフの配置
fig,ax = plt.subplots(4, 1, figsize=(7,7))

# プロット、線の色、凡例の内容
ax[0].plot(x,y1,'blue',label='Velocity' ,ls='solid',lw=1)
ax[1].plot(x,y2,'red',label='Tolque',ls='solid' ,lw=1)
ax[2].plot(x,y3,'green',label='Th1',ls='solid' ,lw=1)
ax[3].plot(x,y4,'green',label='Th2-Th1',ls='solid' ,lw=1)

# 凡例表示設定
ax[0].legend(bbox_to_anchor=(0,0),loc='lower left',borderaxespad=1,fontsize=10)
ax[1].legend(bbox_to_anchor=(0,0),loc='lower left',borderaxespad=1,fontsize=10)
ax[2].legend(bbox_to_anchor=(0,0),loc='lower left',borderaxespad=1,fontsize=10)
ax[3].legend(bbox_to_anchor=(0,0),loc='lower left',borderaxespad=1,fontsize=10)

# X,Y軸表示
# ax[0].set_xlabel('time[sec]'   ,c='g',fontsize=10)
# ax[1]のグラフと重なるのでコメントにした。
ax[0].set_ylabel('Velocity[rad/s]',c='m',fontsize=10)
# ax[1].set_xlabel('time[sec]'   ,c='g',fontsize=10)
# ax[2]のグラフと重なるのでコメントにした。
ax[1].set_ylabel('Tolque[Nm]',c='m',fontsize=10)
ax[2].set_ylabel('Th1[rad]',c='m',fontsize=10)
ax[3].set_ylabel('Th2-Th1[rad]',c='m',fontsize=10)
ax[3].set_xlabel('time[s]',c='g',fontsize=10)

# x 軸 (major) の目盛りを設定する。
ax[0].set_xticks(np.arange(Ts, Te, 1))
ax[1].set_xticks(np.arange(Ts, Te, 1))
ax[2].set_xticks(np.arange(Ts, Te, 1))
ax[3].set_xticks(np.arange(Ts, Te, 1))

# x 軸 (minor) の目盛りを設定する。
ax[0].set_xticks(np.arange(Ts, Te, 0.25), minor=True)
ax[1].set_xticks(np.arange(Ts, Te, 0.25), minor=True)
ax[2].set_xticks(np.arange(Ts, Te, 0.25), minor=True)
ax[3].set_xticks(np.arange(Ts, Te, 0.25), minor=True)

# y 軸 (major) の目盛りを設定する。
ax[0].set_yticks(np.linspace(-25, 225, 11))
# ax[0].set_yticks(np.linspace(-25, 25, 11))
ax[1].set_yticks(np.linspace(-10, 10, 11))
ax[2].set_yticks(np.linspace(-150, 1350, 11))
# ax[3].set_yticks(np.linspace(-0.01, 0.01, 11))

# y 軸 (minor) の目盛りを設定する。
ax[0].set_yticks(np.linspace(-25, 225, 21), minor=True)
# ax[0].set_yticks(np.linspace(-25, 25, 21), minor=True)
ax[1].set_yticks(np.linspace(-10, 10, 21), minor=True)
ax[2].set_yticks(np.linspace(-150, 1350, 21), minor=True)
# ax[3].set_yticks(np.linspace(-0.01, 0.01, 21), minor=True)

# タイトル表示
ax[0].set_title('Velocity',color='b',fontsize=10)
ax[1].set_title('Tolque',color='b',fontsize=10)
ax[2].set_title('Th1',color='b',fontsize=10)
ax[3].set_title('Th2-Th1',color='b',fontsize=10)

# テキストの記載の仕方（参考）
# ax.text(0.05,20,'TEST',fontsize=15)

# グリッド
ax[0].grid(which='both', axis='both')
ax[1].grid(which='both', axis='both')
ax[2].grid(which='both', axis='both')
ax[3].grid(which='both', axis='both')

# タイトレイアウト
plt.tight_layout()

# 表示
plt.show()

# 保存
fig.savefig('sample.pdf')
