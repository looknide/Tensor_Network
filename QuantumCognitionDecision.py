import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

# =========================
# 配置
# =========================
T_EVOLVE = 0.45          # 时间演化时长
U_LIST = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]  # u的取值
GAMMA_STEPS = 21         # γ 取样点个数（含端点0与1）

# =========================
# 物理模型：HA + HB
# =========================

# ==== 构造HA：对于非机动车或行人两种结果（< 与 ≥）的偏向（"理性/效用主导"部分） ====
# 参数u越趋近1，表示估测值与目标值越接近
def get_HA(u=1.0):
    norm = 1 / np.sqrt(1 + u ** 2)
    HAL = norm * np.array([[u, 1], [1, -u]], dtype=np.complex128) # 作用在（LL、LH）
    HAH = norm * np.array([[u, 1], [1, -u]], dtype=np.complex128) # 作用在（HL、HH）
    return scipy.linalg.block_diag(HAL, HAH)

# 构造认知失调矩阵HB
# γ表示非理性程度，0是完全理性，1是完全非理性
def get_HB(gamma=0.0):
    M1 = np.array([[1, 0, 1, 0],
                   [0, 0, 0, 0],
                   [1, 0, -1, 0],
                   [0, 0, 0, 0]], dtype=np.complex128)
    M2 = np.array([[0,  0,  0,  0],
                   [0, -1,  0,  1],
                   [0,  0,  0,  0],
                   [0,  1,  0,  1]], dtype=np.complex128)
    return -gamma / np.sqrt(2) * (M1 + M2)

# ==== 初始纯态 ====
def init_pure_state(mode):
    amp = 1 / np.sqrt(2)
    if mode == 'bike':
        return np.array([amp, amp, 0, 0], dtype=np.complex128)
    elif mode == 'pedestrian':
        return np.array([0, 0, amp, amp], dtype=np.complex128)
    elif mode == 'group_20%bike':     # α=0.2
        a = np.sqrt(0.2)/np.sqrt(2); b = np.sqrt(0.8)/np.sqrt(2)
        return np.array([a, a, b, b], dtype=np.complex128)
    elif mode == 'group_50%bike':     # α=0.5
        a = np.sqrt(0.5)/np.sqrt(2); b = np.sqrt(0.5)/np.sqrt(2)
        return np.array([a, a, b, b], dtype=np.complex128)
    elif mode == 'group_70%bike':     # α=0.7
        a = np.sqrt(0.7)/np.sqrt(2); b = np.sqrt(0.3)/np.sqrt(2)
        return np.array([a, a, b, b], dtype=np.complex128)
    else:
        raise ValueError("Unknown mode")

# ==== 时间演化 ====
def evolve_pure_state(psi0, u=1.0, gamma=0.0, t=T_EVOLVE):
    H = get_HA(u) + get_HB(gamma) # 总哈密顿量H=H_A(u)+H_B(gamma)
    U = scipy.linalg.expm(-1j * H * t) # 酉演化矩阵U(t)=e^{-iHt}
    psi_t = U @ psi0
    psi_t = psi_t / np.linalg.norm(psi_t) # 数值归一化
    return psi_t

# ==== 决策与置信度 ====
# 返回：
#   decision: 0 表示 "<"（Estimated < Target），1 表示 "≥"
#   margin:决策置信度，即两个概率的绝对差值|p_< - p_≥|
def decision_and_margin(psi_t):
    # 基底顺序：(LL, LH, HL, HH)
    # 概率振幅是模平方的形式
    p_less = np.abs(psi_t[0])**2 + np.abs(psi_t[2])**2      # 估计值<目标值
    p_ge   = np.abs(psi_t[1])**2 + np.abs(psi_t[3])**2      # "≥"
    if p_less >= p_ge:
        decision = 0 # 减速等待
        margin = float(p_less - p_ge)
    else:
        decision = 1 # 加速通过
        margin = float(p_ge - p_less)
    return decision, margin

# =========================
# 可视化（两种决策结果+决策置信度）
# =========================
def generate_decision_figure(u_val=1.0, gamma_steps=GAMMA_STEPS,
                             encoding="color", group_side_by_side=True):
    gamma_values = np.linspace(0, 1, gamma_steps)

    pure_modes = ['bike', 'pedestrian']
    group_modes = [
        ('group_20%bike', '20% Bike / 80% Ped'),
        ('group_50%bike', '50% Bike / 50% Ped'),
        ('group_70%bike', '70% Bike / 30% Ped')
    ]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8.6, 9.6), dpi=120, sharex=False)

    # ===== 子图1,2: bike / pedestrian =====
    for i, mode in enumerate(pure_modes):
        decisions, margins = [], []
        for gamma in gamma_values:
            psi0 = init_pure_state(mode) # 初始态
            psi_t = evolve_pure_state(psi0, u=u_val, gamma=gamma) # 时间演化
            dec, m = decision_and_margin(psi_t) # 决策，置信度计算
            decisions.append(dec); margins.append(m)

        decisions = np.array(decisions)  # 0: "<", 1: "≥"
        margins = np.array(margins)
        y = decisions # y坐标为决策值
        ax = axes[i]
        # 创建散点图，颜色表示置信度
        sc = ax.scatter(gamma_values, y, c=margins, cmap='coolwarm',
                            s=100, edgecolor='k', linewidth=0.5, alpha=0.9)
        cbar = plt.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(r"Margin  $|p_{<} - p_{\geq}|$", fontsize=9)
        # 坐标轴设置
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Slow down', 'Accelerated'])# ['(Estimated < Target)减速让行', '(Estimated ≥ Target)加速通过']
        ax.set_ylim(-0.5, 1.5)
        ax.set_xlim(-0.02, 1.02)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_title(f"({mode.capitalize()})", loc='left', fontsize=11) # 设置子图标题
        ax.set_xlabel("γ", fontsize=10)

    # ===== 子图3: Group   =====
    ax = axes[2]
    if group_side_by_side:
        W, GAP = 1.0, 0.18
        x_offsets = [0.0, W + GAP, 2*(W + GAP)]
        block_centers = [off + W/2 for off in x_offsets]
        block_labels = [lbl for _, lbl in group_modes]
        for off in x_offsets:
            ax.axvspan(off, off + W, color='0.95', zorder=0)
        markers = ['o', 's', '^'] # 三种不同的点标记形状
        cbar_added = False

        # 刻度与标签
        all_ticks = []
        all_ticklabels = []

        for idx, ((mode, label), off) in enumerate(zip(group_modes, x_offsets)):
            xs, ys, ms, ds = [], [], [], []
            for gamma in gamma_values:
                psi0 = init_pure_state(mode)
                psi_t = evolve_pure_state(psi0, u=u_val, gamma=gamma)
                dec, m = decision_and_margin(psi_t)
                xs.append(gamma + off); ys.append(dec); ms.append(m); ds.append(dec)

            xs = np.array(xs); ys = np.array(ys); ms = np.array(ms); ds = np.array(ds)
            # 创建散点图，使用颜色编码置信度
            sc = ax.scatter(xs, ys, c=ms, cmap='coolwarm',
                                s=100, marker=markers[idx], edgecolor='k',
                                linewidth=0.5, alpha=0.9, label=label)
            if not cbar_added:
                cbar = plt.colorbar(sc, ax=ax, pad=0.02)
                cbar.set_label(r"Margin  $|p_{<} - p_{\geq}|$", fontsize=9)
                cbar_added = True

            all_ticks.extend([off + 0.0, off + 0.5, off + 1.0])
            all_ticklabels.extend(['0', '0.5', '1'])

        # 设置刻度与标签
        ax.set_xticks(all_ticks)
        ax.set_xticklabels(all_ticklabels)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Slow down', 'Accelerated'])
        ax.set_xlim(-0.05, x_offsets[-1] + W + 0.05)
        ax.set_ylim(-0.5, 1.5)
        ax.grid(True, linestyle='--', alpha=0.5)

        # 每个区块中心上方标标签
        ax_top = ax.secondary_xaxis('top')
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_xticks(block_centers)
        ax_top.set_xticklabels(block_labels) # x轴刻度标签"20% Bike / 80% Ped" 等
        ax_top.tick_params(axis='x', pad=6, length=0)
        ax.set_title("(Group)",
             loc='left', fontsize=11, pad=12)
        for lbl in ax_top.get_xticklabels():
          lbl.set_fontsize(9)
          lbl.set_horizontalalignment('center')
        fig.tight_layout(rect=[0, 0.08, 1, 0.95])

        ax.set_xlabel("γ per block (three blocks: 20/50/70)", fontsize=10)
        # ax.legend(fontsize=8, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.22))

    fig.suptitle(f"u = {u_val} — Decision Map with Confidence Encoding", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



# =========================
# 运行
# =========================
if __name__ == "__main__":
    for u in U_LIST:
        generate_decision_figure(u_val=u, gamma_steps=GAMMA_STEPS, encoding="color")
