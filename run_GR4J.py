import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  

def sh1_curve(t, x4):
    """计算HU1的时间延迟"""
    if t <= 0:
        return 0.0
    elif t < x4:
        return (t / x4)**2.5
    else:  # t >= x4
        return 1.0

def sh2_curve(t, x4):
    """计算HU2的时间延迟"""
    if t <= 0:
        return 0.0
    elif t <= x4:
        return 0.5 * (t / x4)**2.5
    elif t < 2 * x4:
        return 1.0 - 0.5 * (2 - t / x4)**2.5
    else:  # t >= 2 * x4
        return 1.0

def run_gr4j_model(x1, x2, x3, x4, P, E, Qobs, area, upperTankRatio, lowerTankRatio):
    """
    运行GR4J水文模型。
    
    参数:
    x1 (float): 产流水库容量 (mm)
    x2 (float): 地下水交换系数 (mm)
    x3 (float): 汇流水库容量 (mm)
    x4 (float): 单位线汇流时间 (天)
    P (np.array): 日降雨量 (mm)
    E (np.array): 日蒸散发量 (mm)
    Qobs (np.array): 流域出口观测流量 (m3/s)
    area (float): 流域面积 (km2)
    upperTankRatio (float): 产流水库初始填充率 S0/x1
    lowerTankRatio (float): 汇流水库初始填充率 R0/x3
    
    返回:
    Q (np.array): 模拟的逐日径流量 (mm/d)
    NSE (float): 纳什效率系数
    Qobs_mm (np.array): 观测的逐日径流量 (mm/d)
    """
    
    # 将观测径流量单位从 m3/s 转换为 mm/d
    # 转换系数: (m3/s * 3600 * 24) / (km2 * 10^6) * 1000 = s/d / m2/km2 * mm/m = 86.4
    Qobs_mm = Qobs * 86.4 / area
    
    nStep = len(P)  # 计算总天数

    # --- 初始化中间变量 ---
    Pn = np.zeros(nStep)  # 净雨
    En = np.zeros(nStep)  # 净蒸发
    Ps = np.zeros(nStep)  # 补充土壤含水量
    Es = np.zeros(nStep)  # 消耗土壤含水量
    Perc = np.zeros(nStep) # 产流水库壤中流产流量
    Pr = np.zeros(nStep)   # 产流总量
    
    # --- 计算单位线 (Unit Hydrographs) ---
    # 假设单位线最大延迟，x4不应大于此值
    maxDayDelay = 20 # 增加延迟以处理更长的x4
    
    # 确保 maxDayDelay 足够大
    if x4 > maxDayDelay:
        print(f"警告: x4 ({x4}) 大于 maxDayDelay ({maxDayDelay}). 可能导致计算错误。")

    SH1 = np.zeros(maxDayDelay)
    SH2 = np.zeros(2 * maxDayDelay)
    
    for i in range(maxDayDelay):
        SH1[i] = sh1_curve(i + 1, x4)
    for i in range(2 * maxDayDelay):
        SH2[i] = sh2_curve(i + 1, x4)
        
    UH1 = np.diff(SH1, prepend=0)
    UH2 = np.diff(SH2, prepend=0)
    
    # --- 计算净雨 (Pn) 和净蒸发 (En) ---
    Pn = np.maximum(0, P - E)
    En = np.maximum(0, E - P)
    
    # --- 初始化产汇流计算所需变量 ---
    S0 = upperTankRatio * x1  # 产流水库初始水量
    R0 = lowerTankRatio * x3  # 汇流水库初始水量
    S = np.zeros(nStep)       # 产流水库逐日水量
    R = np.zeros(nStep)       # 汇流水库逐日水量
    
    # 用于存储前一天汇流信息的矩阵 (convolution matrix)
    UH_Fast_Ordinates = np.zeros((nStep, len(UH1)))
    UH_Slow_Ordinates = np.zeros((nStep, len(UH2)))
    
    S_TEMP = S0  # 当前产流水库储量
    R_TEMP = R0  # 当前汇流水库储量
    Qr = np.zeros(nStep) # 汇流水库快速流出流量
    Qd = np.zeros(nStep) # 汇流水库慢速流出流量
    Q = np.zeros(nStep)  # 汇流总出流量
    
    # --- 主循环：逐日计算 ---
    for i in range(nStep):
        S[i] = S_TEMP
        R[i] = R_TEMP
        
        # --- 产流模块 (Production) ---
        # 计算Ps和Es
        if Pn[i] > 0:
            Ps[i] = x1 * (1 - (S[i] / x1)**2) * np.tanh(Pn[i] / x1) / (1 + S[i] / x1 * np.tanh(Pn[i] / x1))
            Es[i] = 0
        else: # En[i] > 0
            Ps[i] = 0
            Es[i] = (S[i] * (2 - S[i] / x1) * np.tanh(En[i] / x1)) / (1 + (1 - S[i] / x1) * np.tanh(En[i] / x1))

        # 更新产流水库蓄水量
        S_TEMP = S[i] - Es[i] + Ps[i]
        
        # 计算产流水库渗漏 (Percolation)
        Perc[i] = S_TEMP * (1 - (1 + (4.0 / 9.0 * (S_TEMP / x1))**4)**(-0.25))
        
        # 计算总产流量 (地表产流 + 壤中流)
        Pr[i] = Perc[i] + (Pn[i] - Ps[i])
        
        # 更新产流水库水量，为下一天做准备
        S_TEMP = S_TEMP - Perc[i]
        
        # --- 汇流模块 (Routing) ---
        # 水量交换
        F = x2 * (R[i] / x3)**3.5
        
        # 划分快速和慢速径流
        R_Fast = Pr[i] * 0.9
        R_Slow = Pr[i] * 0.1
        
        # 使用单位线进行汇流演算 (更高效的实现)
        for j in range(len(UH1)):
            if i - j >= 0:
                UH_Fast_Ordinates[i,j] = R_Fast * UH1[j]
        for j in range(len(UH2)):
            if i - j >= 0:
                UH_Slow_Ordinates[i,j] = R_Slow * UH2[j]

        # 快速流 Qr 计算
        fast_flow_input = np.sum([UH_Fast_Ordinates[k, i-k] for k in range(i+1)]) if i < len(UH1) else np.sum([UH_Fast_Ordinates[k, i-k] for k in range(i-len(UH1)+1, i+1)])
        
        R_TEMP = max(0, R_TEMP + fast_flow_input + F)
        Qr[i] = R_TEMP * (1 - (1 + (R_TEMP / x3)**4)**(-0.25))
        R_TEMP = R_TEMP - Qr[i]
        
        # 慢速流 Qd 计算
        slow_flow_input = np.sum([UH_Slow_Ordinates[k, i-k] for k in range(i+1)]) if i < len(UH2) else np.sum([UH_Slow_Ordinates[k, i-k] for k in range(i-len(UH2)+1, i+1)])

        Qd[i] = max(0, slow_flow_input + F)
        
        # 总流量
        Q[i] = Qr[i] + Qd[i]

    # --- 精度评估 ---
    # 使用一年 (365天) 作为预热期
    warmup_period = 365
    if nStep > warmup_period:
        sim_eval = Q[warmup_period:]
        obs_eval = Qobs_mm[warmup_period:]
        
        # 使用向量化计算NSE，更高效
        numerator = np.sum((obs_eval - sim_eval)**2)
        denominator = np.sum((obs_eval - np.mean(obs_eval))**2)
        
        if denominator == 0:
            NSE = -np.inf # 如果观测值是常数，无法计算NSE
        else:
            NSE = 1 - (numerator / denominator)
    else:
        NSE = np.nan # 数据不足以进行评估

    return Q, NSE, Qobs_mm

if __name__ == '__main__':
    # --- 第一步：加载数据 ---
    try:
        para = np.loadtxt('GR4J_Parameter.txt')
        x1, x2, x3, x4 = para[0], para[1], para[2], para[3]

        other_para = np.loadtxt('others.txt')
        area, upperTankRatio, lowerTankRatio = other_para[0], other_para[1], other_para[2]

        data = np.loadtxt('inputData.txt')
        P = data[:, 0]
        E = data[:, 1]
        Qobs = data[:, 2]

    except FileNotFoundError as e:
        print(f"文件加载错误: {e}")
        print("请确保 'GR4J_Parameter.txt', 'others.txt', 和 'inputData.txt' 文件在当前目录下。")
        exit()
    except IndexError as e:
        print(f"数据格式错误: {e}")
        print("请检查输入文件中的列数是否正确。")
        exit()

    # --- 第二步：运行模型 ---
    Q_sim, NSE, Qobs_mm = run_gr4j_model(x1, x2, x3, x4, P, E, Qobs, area, upperTankRatio, lowerTankRatio)
    
    print(f"模型运行完成。")
    print(f"纳什效率系数 (NSE): {NSE:.4f}")

    # --- 第三步：绘制结果图 ---
    nStep = len(P)
    axis = np.arange(1, nStep + 1)
    
    plt.figure(figsize=(12, 6))
    plt.plot(axis, Q_sim, 'r--', label='模拟径流量')
    plt.plot(axis, Qobs_mm, 'k-', alpha=0.7, label='观测径流量')
    
    # 突出显示预热期
    warmup_period = 365
    if nStep > warmup_period:
        plt.axvspan(0, warmup_period, color='gray', alpha=0.2, label='预热期')

    plt.title(f'GR4J模型模拟效果图, NSE = {NSE:.4f}')
    plt.xlabel('时间（天）')
    plt.ylabel('流量（mm/d）')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, nStep)
    plt.show()