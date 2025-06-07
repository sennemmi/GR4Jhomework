import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import differential_evolution
import os

# 用于处理中文字符显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 输出文件名
OUTPUT_PLOT_FILENAME = 'GR4J_DE_Optimization_Result.png'
OUTPUT_PARAMS_FILENAME = 'GR4J_Parameter_Optimized_DE.txt'
CONVERGENCE_PLOT_FILENAME = 'DE_Convergence_Curve.png'

# 优化参数
MAX_ITER = 50
POP_SIZE = 15
MUTATION = (0.5, 1.0)
RECOMBINATION = 0.7
POLISH = True
WORKERS = 1

# =============================================================================
# 辅助函数 (S-curves)
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
# GR4J模型函数
# =============================================================================

def run_gr4j_model(x1, x2, x3, x4, P, E, Qobs, area, upperTankRatio, lowerTankRatio):
    """
    运行GR4J水文模型。
    """
    # 将观测径流量单位从 m3/s 转换为 mm/d
    Qobs_mm = Qobs * 86.4 / area
    
    nStep = len(P)  # 计算总天数

    # --- 初始化中间变量 ---
    Pn = np.maximum(0, P - E)  # 净雨
    En = np.maximum(0, E - P)  # 净蒸发
    Ps = np.zeros(nStep)  # 补充土壤含水量
    Es = np.zeros(nStep)  # 消耗土壤含水量
    Perc = np.zeros(nStep) # 产流水库壤中流产流量
    Pr = np.zeros(nStep)   # 产流总量
    
    # --- 计算单位线 (Unit Hydrographs) ---
    maxDayDelay = 20 # 增加延迟以处理更长的x4
    
    # 确保 maxDayDelay 足够大
    if x4 > maxDayDelay:
        print(f"警告: x4 ({x4}) 大于 maxDayDelay ({maxDayDelay}). 可能导致计算错误。")

    SH1 = np.array([sh1_curve(i + 1, x4) for i in range(maxDayDelay)])
    SH2 = np.array([sh2_curve(i + 1, x4) for i in range(2 * maxDayDelay)])
    
    UH1 = np.diff(SH1, prepend=0)
    UH2 = np.diff(SH2, prepend=0)
    
    # --- 初始化产汇流计算所需变量 ---
    S0 = upperTankRatio * x1  # 产流水库初始水量
    R0 = lowerTankRatio * x3  # 汇流水库初始水量
    S = np.zeros(nStep)       # 产流水库逐日水量
    R = np.zeros(nStep)       # 汇流水库逐日水量
    
    # 用于存储前一天汇流信息的矩阵
    Pr_fast_hist = np.zeros(len(UH1))
    Pr_slow_hist = np.zeros(len(UH2))
    
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
            ps_num = x1 * (1 - (S[i] / x1)**2) * np.tanh(Pn[i] / x1)
            ps_den = 1 + (S[i] / x1) * np.tanh(Pn[i] / x1)
            Ps[i] = ps_num / ps_den if ps_den != 0 else 0
            Es[i] = 0
        else: # En[i] > 0
            Ps[i] = 0
            es_num = S[i] * (2 - S[i] / x1) * np.tanh(En[i] / x1)
            es_den = 1 + (1 - S[i] / x1) * np.tanh(En[i] / x1)
            Es[i] = es_num / es_den if es_den != 0 else 0

        # 更新产流水库蓄水量
        S_TEMP = max(0, S[i] - Es[i] + Ps[i])
        
        # 计算产流水库渗漏 (Percolation)
        perc_base = (4.0/9.0) * (S_TEMP/x1)
        Perc[i] = S_TEMP * (1 - (1 + perc_base**4)**(-0.25))
        
        # 计算总产流量 (地表产流 + 壤中流)
        Pr[i] = Perc[i] + (Pn[i] - Ps[i])
        
        # 更新产流水库水量，为下一天做准备
        S_TEMP = max(0, S_TEMP - Perc[i])
        
        # --- 汇流模块 (Routing) ---
        # 水量交换
        F = x2 * (R[i] / x3)**3.5
        
        # 划分快速和慢速径流
        R_Fast = Pr[i] * 0.9
        R_Slow = Pr[i] * 0.1
        
        # 更新历史数组
        Pr_fast_hist = np.roll(Pr_fast_hist, 1)
        Pr_fast_hist[0] = R_Fast
        Pr_slow_hist = np.roll(Pr_slow_hist, 1)
        Pr_slow_hist[0] = R_Slow
        
        # 使用点积计算单位线汇流
        fast_flow_input = np.dot(UH1, Pr_fast_hist)
        slow_flow_input = np.dot(UH2, Pr_slow_hist)

        # 快速流 Qr 计算
        R_TEMP = max(0, R[i] + fast_flow_input + F)
        qr_base = R_TEMP / x3
        Qr[i] = R_TEMP * (1 - (1 + qr_base**4)**(-0.25))
        R_TEMP = max(0, R_TEMP - Qr[i])
        
        # 慢速流 Qd 计算
        Qd[i] = max(0, slow_flow_input + F)
        
        # 总流量
        Q[i] = Qr[i] + Qd[i]

    # --- 精度评估 ---
    # 使用一年 (365天) 作为预热期
    warmup_period = 365
    if nStep > warmup_period:
        sim_eval = Q[warmup_period:]
        obs_eval = Qobs_mm[warmup_period:]
        
        # 计算NSE
        numerator = np.sum((obs_eval - sim_eval)**2)
        denominator = np.sum((obs_eval - np.mean(obs_eval))**2)
        
        if denominator == 0:
            NSE = -np.inf # 如果观测值是常数，无法计算NSE
        else:
            NSE = 1 - (numerator / denominator)
    else:
        NSE = np.nan # 数据不足以进行评估

    return Q, NSE, Qobs_mm


# 获取固定参数
def get_initial_params():
    """返回固定的参数值"""
    # 固定参数值
    fixed_params = np.array([320.11, 2.42, 69.93, 1.39])
    
    print(f"使用固定的参数值:")
    print(f"参数值: x1={fixed_params[0]:.4f}, x2={fixed_params[1]:.4f}, x3={fixed_params[2]:.4f}, x4={fixed_params[3]:.4f}")
    
    # 计算NSE值
    try:
        data = np.loadtxt('inputData.txt')
        P, E, Qobs = data[:, 0], data[:, 1], data[:, 2]
        
        other_para = np.loadtxt('others.txt')
        area, upperTankRatio, lowerTankRatio = other_para[0], other_para[1], other_para[2]
        
        _, nse, _ = run_gr4j_model(
            fixed_params[0], fixed_params[1], fixed_params[2], fixed_params[3],
            P, E, Qobs, area, upperTankRatio, lowerTankRatio
        )
        print(f"对应NSE: {nse:.6f}")
    except Exception as e:
        print(f"计算初始NSE时出错: {e}")
    
    return fixed_params


# =============================================================================
# 主执行块 (使用差分进化算法)
# =============================================================================
if __name__ == '__main__':
    # 显示优化设置
    print("\n" + "="*60)
    print(f"差分进化优化设置:")
    print(f"最大迭代次数: {MAX_ITER}")
    print(f"种群大小: {POP_SIZE}")
    print(f"变异系数范围: {MUTATION}")
    print(f"交叉概率: {RECOMBINATION}")
    print(f"使用局部优化: {POLISH}")
    print(f"线程数: {WORKERS}")
    print("="*60 + "\n")
    
    # --- 1. 加载数据 ---
    try:
        other_para = np.loadtxt('others.txt')
        area, upperTankRatio, lowerTankRatio = other_para[0], other_para[1], other_para[2]
        data = np.loadtxt('inputData.txt')
        P, E, Qobs = data[:, 0], data[:, 1], data[:, 2]
        print("数据加载成功！")
        print(f"数据长度: {len(P)} 天")
        print(f"流域面积: {area} km²")
    except FileNotFoundError as e:
        print(f"文件加载错误: {e}")
        exit()

    # --- 2. 获取初始参数 ---
    print("\n正在获取初始参数...")
    best_params = get_initial_params()

    # --- 3. 定义目标函数 ---
    def objective_function(X):
        """
        目标函数：最大化NSE (通过最小化-NSE)
        """
        x1, x2, x3, x4 = X
        _, nse, _ = run_gr4j_model(x1, x2, x3, x4, P, E, Qobs, area, upperTankRatio, lowerTankRatio)
        return -nse if not (np.isnan(nse) or np.isinf(nse)) else 1e10

    # --- 4. 定义回调函数和进度记录列表 ---
    nse_progress = []
    iteration_count = 0
    
    def de_callback(xk, convergence):
        """差分进化优化回调函数"""
        global iteration_count
        iteration_count += 1
        
        # 计算当前参数的NSE值
        current_nse = -objective_function(xk)
            
        # 添加到进度记录
        nse_progress.append(current_nse)
        
        # 打印进度信息
        print(f"迭代次数: {iteration_count:4d}/{MAX_ITER} | 当前最优NSE: {current_nse:.6f}")
        
        # 始终返回False，继续优化
        return False

    # --- 5. 定义参数边界 ---
    # 参数边界参考文献值和经验值设置
    bounds = [(100.0, 1200.0),  # x1: 产流水库容量 (mm)
              (-5.0, 3.0),      # x2: 地下水交换系数 (mm)
              (20.0, 300.0),    # x3: 汇流水库容量 (mm)
              (0.1, 7.0)]       # x4: 单位线汇流时间 (天)

    # --- 6. 设置初始种群 ---
    # 在最佳参数周围生成初始种群
    # 确保最佳参数在边界内
    for i in range(4):
        best_params[i] = max(bounds[i][0], min(bounds[i][1], best_params[i]))
    
    # 创建初始种群
    init_population = []
    # 添加最佳参数作为第一个个体
    init_population.append(best_params)
    
    # 在最佳参数周围生成其他个体
    for i in range(POP_SIZE - 1):
        individual = []
        for j in range(4):
            # 在参数周围加入一些随机扰动（范围的±5%）
            param_range = bounds[j][1] - bounds[j][0]
            perturbation = np.random.uniform(-0.05, 0.05) * param_range
            value = best_params[j] + perturbation
            # 确保在边界内
            value = max(bounds[j][0], min(bounds[j][1], value))
            individual.append(value)
        init_population.append(individual)
    
    init_population = np.array(init_population)
    print(f"已创建包含最佳参数的初始种群 (大小: {POP_SIZE})")

    # --- 7. 运行差分进化优化器 ---
    print("\n开始进行差分进化优化... (将实时报告每次迭代的最优NSE)")
    start_time = time.time()
    
    result = differential_evolution(
        func=objective_function,
        bounds=bounds,
        callback=de_callback,
        strategy='best1bin',  # 使用最佳个体作为基础的变异策略
        maxiter=MAX_ITER,     # 最大迭代次数
        popsize=POP_SIZE,     # 种群大小
        mutation=MUTATION,    # 变异系数范围
        recombination=RECOMBINATION,  # 交叉概率
        polish=POLISH,        # 使用局部优化
        workers=WORKERS,      # 并行计算的线程数
        init=init_population  # 使用自定义初始种群
    )
    
    end_time = time.time()
    print(f"\n优化完成！总耗时: {end_time - start_time:.2f} 秒")

    # --- 8. 显示最终结果并保存参数 ---
    best_params_new = result.x
    best_nse_new = -result.fun
    print("\n" + "="*45)
    print("差分进化找到的最终最优参数组合:")
    print(f"  x1 = {best_params_new[0]:.4f}, x2 = {best_params_new[1]:.4f}, x3 = {best_params_new[2]:.4f}, x4 = {best_params_new[3]:.4f}")
    print(f"对应的最佳纳什效率系数 (NSE): {best_nse_new:.6f}")
    print(f"总评估次数: {result.nfev}")
    print("="*45)
    
    # 比较与之前最佳参数的改进情况
    _, previous_nse, _ = run_gr4j_model(
        best_params[0], best_params[1], best_params[2], best_params[3],
        P, E, Qobs, area, upperTankRatio, lowerTankRatio
    )
    improvement = best_nse_new - previous_nse
    print(f"相比先前最佳参数，NSE提高了: {improvement:.6f} ({improvement/previous_nse*100:.2f}%)")
    
    # 保存参数和NSE值
    np.savetxt(OUTPUT_PARAMS_FILENAME, best_params_new, fmt='%.4f', header='x1, x2, x3, x4', comments='')
    with open(os.path.splitext(OUTPUT_PARAMS_FILENAME)[0] + "_NSE.txt", 'w') as f:
        f.write(f"{best_nse_new:.6f}")
    
    print(f"\n最优参数已保存至 '{OUTPUT_PARAMS_FILENAME}'")
    print(f"最优NSE值已保存至 '{os.path.splitext(OUTPUT_PARAMS_FILENAME)[0]}_NSE.txt'")

    # --- 9. 绘制并保存收敛曲线图 ---
    plt.figure(figsize=(10, 6))
    plt.plot(nse_progress)
    plt.title('差分进化算法收敛曲线')
    plt.xlabel('迭代次数 (Iteration)')
    plt.ylabel('该次迭代后最优NSE值')
    plt.grid(True)
    plt.savefig(CONVERGENCE_PLOT_FILENAME, dpi=300)
    plt.close()
    print(f"收敛曲线图已保存至 '{CONVERGENCE_PLOT_FILENAME}'")

    # --- 10. 使用最优参数生成并保存最终结果图 ---
    print(f"正在生成最终结果图并保存至 '{OUTPUT_PLOT_FILENAME}'...")
    Q_sim_best, _, Qobs_mm = run_gr4j_model(
        best_params_new[0], best_params_new[1], best_params_new[2], best_params_new[3],
        P, E, Qobs, area, upperTankRatio, lowerTankRatio
    )
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(1, len(P) + 1), Q_sim_best, 'r--', label=f'模拟径流量 (最优NSE={best_nse_new:.4f})')
    plt.plot(np.arange(1, len(P) + 1), Qobs_mm, 'k-', alpha=0.7, label='观测径流量')
    if len(P) > 365:
        plt.axvspan(0, 365, color='gray', alpha=0.2, label='预热期')
    plt.title('GR4J模型差分进化优化结果')
    plt.xlabel('时间（天）'); plt.ylabel('流量（mm/d）')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, len(P)); plt.savefig(OUTPUT_PLOT_FILENAME, dpi=300); plt.close()
    print("最终结果图已成功保存。") 