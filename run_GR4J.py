import numpy as np
import matplotlib.pyplot as plt
import random

# 用于处理中文字符显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 'SimHei' 是黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

# =============================================================================
# Helper Functions (S-curves) - [此部分未改变]
# =============================================================================

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

# =============================================================================
# Main GR4J Model Function - [此部分未改变]
# =============================================================================

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
    Qobs_mm = Qobs * 86.4 / area
    
    nStep = len(P)

    # --- 初始化中间变量 ---
    Pn = np.maximum(0, P - E)
    En = np.maximum(0, E - P)
    Ps = np.zeros(nStep)
    Es = np.zeros(nStep)
    Perc = np.zeros(nStep)
    Pr = np.zeros(nStep)
    
    # --- 计算单位线 (Unit Hydrographs) ---
    maxDayDelay = 20
    if x4 > maxDayDelay:
        print(f"警告: x4 ({x4}) 大于 maxDayDelay ({maxDayDelay}).")

    SH1 = np.array([sh1_curve(i + 1, x4) for i in range(maxDayDelay)])
    SH2 = np.array([sh2_curve(i + 1, x4) for i in range(2 * maxDayDelay)])
    
    UH1 = np.diff(SH1, prepend=0)
    UH2 = np.diff(SH2, prepend=0)
    
    # --- 初始化产汇流计算所需变量 ---
    S0 = upperTankRatio * x1
    R0 = lowerTankRatio * x3
    S = np.zeros(nStep)
    R = np.zeros(nStep)
    
    # 此部分为了与原始MATLAB逻辑保持一致，使用了较为繁琐的汇流演算方式
    UH_Fast_Ordinates = np.zeros((nStep, len(UH1)))
    UH_Slow_Ordinates = np.zeros((nStep, len(UH2)))
    
    S_TEMP = S0
    R_TEMP = R0
    Qr = np.zeros(nStep)
    Qd = np.zeros(nStep)
    Q = np.zeros(nStep)
    
    # --- 主循环：逐日计算 ---
    for i in range(nStep):
        S[i] = S_TEMP
        R[i] = R_TEMP
        
        if Pn[i] > 0:
            Ps[i] = x1 * (1 - (S[i] / x1)**2) * np.tanh(Pn[i] / x1) / (1 + S[i] / x1 * np.tanh(Pn[i] / x1))
            Es[i] = 0
        else:
            Ps[i] = 0
            Es[i] = (S[i] * (2 - S[i] / x1) * np.tanh(En[i] / x1)) / (1 + (1 - S[i] / x1) * np.tanh(En[i] / x1))

        S_TEMP = S[i] - Es[i] + Ps[i]
        Perc[i] = S_TEMP * (1 - (1 + (4.0 / 9.0 * (S_TEMP / x1))**4)**(-0.25))
        Pr[i] = Perc[i] + (Pn[i] - Ps[i])
        S_TEMP = S_TEMP - Perc[i]
        
        F = x2 * (R[i] / x3)**3.5
        R_Fast = Pr[i] * 0.9
        R_Slow = Pr[i] * 0.1
        
        # 汇流演算
        for j in range(len(UH1)):
            if i - j >= 0:
                UH_Fast_Ordinates[i, j] = R_Fast * UH1[j]
        for j in range(len(UH2)):
            if i - j >= 0:
                UH_Slow_Ordinates[i, j] = R_Slow * UH2[j]

        fast_flow_input = np.sum([UH_Fast_Ordinates[k, i-k] for k in range(i+1)]) if i < len(UH1) else np.sum([UH_Fast_Ordinates[k, i-k] for k in range(i-len(UH1)+1, i+1)])
        R_TEMP = max(0, R_TEMP + fast_flow_input + F)
        Qr[i] = R_TEMP * (1 - (1 + (R_TEMP / x3)**4)**(-0.25))
        R_TEMP = R_TEMP - Qr[i]
        
        slow_flow_input = np.sum([UH_Slow_Ordinates[k, i-k] for k in range(i+1)]) if i < len(UH2) else np.sum([UH_Slow_Ordinates[k, i-k] for k in range(i-len(UH2)+1, i+1)])
        Qd[i] = max(0, slow_flow_input + F)
        
        Q[i] = Qr[i] + Qd[i]

    # --- 精度评估 ---
    warmup_period = 365
    if nStep > warmup_period:
        sim_eval = Q[warmup_period:]
        obs_eval = Qobs_mm[warmup_period:]
        numerator = np.sum((obs_eval - sim_eval)**2)
        denominator = np.sum((obs_eval - np.mean(obs_eval))**2)
        NSE = 1 - (numerator / denominator) if denominator != 0 else -np.inf
    else:
        NSE = np.nan

    return Q, NSE, Qobs_mm

if __name__ == '__main__':
    
    # --- 第一步：随机生成GR4J参数并保存 ---
    
    # 1. 定义参数范围
    param_bounds = {
        'x1': [100.0, 1200.0],   # 产流水库容量 (mm)
        'x2': [-5.0, 3.0],       # 地下水交换系数 (mm)
        'x3': [20.0, 300.0],     # 汇流水库容量 (mm)
        'x4': [0.1, 7.0]         # 单位线汇流时间 (天)
    }
    
    # 2. 在范围内随机生成参数
    x1 = random.uniform(param_bounds['x1'][0], param_bounds['x1'][1])
    x2 = random.uniform(param_bounds['x2'][0], param_bounds['x2'][1])
    x3 = random.uniform(param_bounds['x3'][0], param_bounds['x3'][1])
    x4 = random.uniform(param_bounds['x4'][0], param_bounds['x4'][1])
    
    new_params = np.array([x1, x2, x3, x4])
    
    # 3. 将新生成的参数保存(覆盖)到文件中，并打印出来
    param_filename = 'GR4J_Parameter.txt'
    np.savetxt(param_filename, new_params, fmt='%.4f')
    
    print("="*40)
    print(f"已随机生成新参数并保存至 '{param_filename}':")
    print(f"  x1 (产流水库容量) = {x1:.4f}")
    print(f"  x2 (地下水交换)   = {x2:.4f}")
    print(f"  x3 (汇流水库容量) = {x3:.4f}")
    print(f"  x4 (汇流时间)     = {x4:.4f}")
    print("="*40)

    # --- 第二步：加载其他数据文件 ---
    try:
        other_para = np.loadtxt('others.txt')
        area, upperTankRatio, lowerTankRatio = other_para[0], other_para[1], other_para[2]

        data = np.loadtxt('inputData.txt')
        P = data[:, 0]
        E = data[:, 1]
        Qobs = data[:, 2]

    except FileNotFoundError as e:
        print(f"文件加载错误: {e}")
        print("请确保 'others.txt' 和 'inputData.txt' 文件在当前目录下。")
        exit()
    except IndexError as e:
        print(f"数据格式错误: {e}")
        print("请检查输入文件中的列数是否正确。")
        exit()

    # --- 第三步：使用新参数运行模型 ---
    Q_sim, NSE, Qobs_mm = run_gr4j_model(x1, x2, x3, x4, P, E, Qobs, area, upperTankRatio, lowerTankRatio)
    
    print(f"模型运行完成。")
    print(f"纳什效率系数 (NSE): {NSE:.4f}")

    # --- 第四步：绘制结果图 ---
    nStep = len(P)
    axis = np.arange(1, nStep + 1)
    
    plt.figure(figsize=(12, 6))
    plt.plot(axis, Q_sim, 'r--', label='模拟径流量 (随机参数)')
    plt.plot(axis, Qobs_mm, 'k-', alpha=0.7, label='观测径流量')
    
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