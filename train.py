import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import time
import os
import json
from multiprocessing import Pool, cpu_count
from functools import partial

from environment import (GridWorld, collect_transitions, collect_transitions_with_strategy,
                         generate_random_map, generate_training_maps)
from world_model import WorldModel, USE_GPU, to_numpy


def evaluate_model(model, env, num_episodes=80, max_steps=30):
    """评估模型在环境中的预测准确度"""
    state_errors = []
    reward_errors = []
    gs = float(model.grid_size)

    for _ in range(num_episodes):
        state = env.reset()
        for _ in range(max_steps):
            action = np.random.randint(0, GridWorld.NUM_ACTIONS)
            real_next, real_reward, done = env.step(action)

            pred_next, pred_reward = model.predict(state, action)

            state_errors.append(np.linalg.norm(pred_next - real_next))
            reward_errors.append(abs(pred_reward - real_reward))

            state = real_next
            if done:
                break

    return np.mean(state_errors), np.mean(reward_errors)


def collect_data_for_map(args):
    """多进程数据收集的工作函数"""
    env_config, num_episodes, max_steps, use_strategy = args
    size = env_config['size']
    layout = env_config['layout']
    env = GridWorld(size=size, layout_type=layout)
    if use_strategy:
        return collect_transitions_with_strategy(env, num_episodes=num_episodes,
                                                  max_steps=max_steps)
    else:
        return collect_transitions(env, num_episodes=num_episodes, max_steps=max_steps)


def visualize_confidence(model, env, step_num, save_path=None):
    """可视化模型置信度（轨迹不确定性）"""
    state = env.reset()
    start = state.copy()
    actions = [np.random.randint(0, GridWorld.NUM_ACTIONS) for _ in range(12)]

    traj, rewards, total_r, confidences, state_stds, reward_stds = \
        model.imagine_trajectory_with_confidence(start, actions)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. 轨迹 + 置信度颜色
    ax1 = axes[0]
    int_traj = [(int(round(r)), int(round(c))) for r, c in traj]
    env.agent_pos = int_traj[-1]
    env.render_visual(trajectory=int_traj, title='想象轨迹', ax=ax1)

    # 用颜色编码置信度
    if len(confidences) > 0:
        for i, (conf, (r, c)) in enumerate(zip(confidences, int_traj[1:])):
            color = plt.cm.RdYlGn(conf)  # 红=低置信度, 绿=高置信度
            ax1.plot(c, r, 'o', color=color, markersize=10, zorder=7)

    # 2. 置信度曲线
    ax2 = axes[1]
    steps = list(range(1, len(confidences) + 1))
    ax2.plot(steps, confidences, 'o-', color='#4CAF50', linewidth=2, markersize=6)
    ax2.fill_between(steps, confidences, alpha=0.2, color='#4CAF50')
    ax2.set_title('逐步置信度', fontsize=13, fontweight='bold')
    ax2.set_xlabel('步骤')
    ax2.set_ylabel('置信度 (0-1)')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    # 3. 不确定性（标准差）
    ax3 = axes[2]
    state_std_means = [np.mean(s) for s in state_stds]
    ax3.bar(steps, state_std_means, color='#FF9800', edgecolor='white', label='状态不确定性')
    ax3_r = ax3.twinx()
    ax3_r.plot(steps, reward_stds, 's-', color='#F44336', linewidth=2,
               markersize=5, label='奖励不确定性')
    ax3.set_title('预测不确定性', fontsize=13, fontweight='bold')
    ax3.set_xlabel('步骤')
    ax3.set_ylabel('状态不确定性 (std)', color='#FF9800')
    ax3_r.set_ylabel('奖励不确定性 (std)', color='#F44336')
    ax3.legend(loc='upper left')
    ax3_r.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    fig.suptitle(f'模型置信度分析 (Step {step_num})', fontsize=15, fontweight='bold')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def visualize_network_architecture(model, save_path=None):
    """可视化网络架构"""
    arch = model.architecture
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    layer_names = ['输入层\n(6维)'] + \
                  [f'隐藏层 {i + 1}\n({d}维)' for i, d in enumerate(arch[1:-1])] + \
                  ['输出层\n(3维)']
    n_layers = len(arch)
    x_positions = np.linspace(1, 9, n_layers)
    colors = ['#2196F3'] + ['#FF9800'] * (n_layers - 2) + ['#4CAF50']

    max_neurons = max(arch)
    for i, (x, n, name, color) in enumerate(zip(x_positions, arch, layer_names, colors)):
        height = max(0.5, 8 * n / max_neurons)
        y_center = 5
        rect = mpatches.FancyBboxPatch(
            (x - 0.3, y_center - height / 2), 0.6, height,
            boxstyle="round,pad=0.05", facecolor=color, edgecolor='white',
            linewidth=2, alpha=0.8
        )
        ax.add_patch(rect)
        ax.text(x, y_center, str(n), ha='center', va='center',
                fontsize=max(8, min(16, 60 // n_layers)), fontweight='bold', color='white')
        ax.text(x, y_center - height / 2 - 0.3, name, ha='center', va='top', fontsize=9)

        if i < n_layers - 1:
            next_x = x_positions[i + 1]
            for y_start in [y_center - height / 4, y_center, y_center + height / 4]:
                ax.annotate('', xy=(next_x - 0.35, y_center),
                            xytext=(x + 0.35, y_start),
                            arrowprops=dict(arrowstyle='->', color='gray', alpha=0.3, lw=0.5))

    title = f'网络架构: {" → ".join(str(d) for d in arch)}\n'
    title += f'参数量: {model.get_param_count():,} | 后端: {model.backend}'
    ax.set_title(title, fontsize=14, fontweight='bold', y=1.02)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def visualize_training_summary(history, model, env, save_path=None):
    """训练总结可视化"""
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # 1. 训练损失
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history['epoch'], history['loss'], color='#F44336', linewidth=1.5, alpha=0.7)
    window = min(20, len(history['loss']) // 3)
    if window > 1:
        smooth = np.convolve(history['loss'], np.ones(window) / window, mode='valid')
        ax1.plot(history['epoch'][window - 1:], smooth, color='#D32F2F', linewidth=2.5,
                 label=f'平滑({window})')
    ax1.set_title('训练损失', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 状态误差
    ax2 = fig.add_subplot(gs[0, 1])
    eval_epochs = [e for e, _ in history['state_error']]
    eval_s_errs = [s for _, s in history['state_error']]
    ax2.plot(eval_epochs, eval_s_errs, 'o-', color='#2196F3', linewidth=2, markersize=4)
    ax2.fill_between(eval_epochs, eval_s_errs, alpha=0.15, color='#2196F3')
    ax2.set_title('状态预测误差', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('L2 误差')
    ax2.grid(True, alpha=0.3)

    # 3. 奖励误差
    ax3 = fig.add_subplot(gs[0, 2])
    eval_r_errs = [r for _, r in history['reward_error']]
    ax3.plot(eval_epochs, eval_r_errs, 's-', color='#4CAF50', linewidth=2, markersize=4)
    ax3.fill_between(eval_epochs, eval_r_errs, alpha=0.15, color='#4CAF50')
    ax3.set_title('奖励预测误差', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('绝对误差')
    ax3.grid(True, alpha=0.3)

    # 4. 学习率
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(history['epoch'], history['lr'], color='#9C27B0', linewidth=2)
    ax4.set_title('学习率 (余弦退火)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('学习率')
    ax4.grid(True, alpha=0.3)

    # 5. 热力图
    ax5 = fig.add_subplot(gs[1, 1])
    env.render_heatmap(title='训练后热力图', ax=ax5)

    # 6. 轨迹示例
    ax6 = fig.add_subplot(gs[1, 2])
    state = env.reset()
    start = state.copy()
    actions = [np.random.randint(0, GridWorld.NUM_ACTIONS) for _ in range(15)]
    traj, _, _ = model.imagine_trajectory(start, actions)
    int_traj = [(int(round(r)), int(round(c))) for r, c in traj]
    env.render_visual(trajectory=int_traj, title='想象轨迹示例', ax=ax6)

    # 7. 置信度
    ax7 = fig.add_subplot(gs[2, 0])
    _, _, _, confs, _, _ = model.imagine_trajectory_with_confidence(start, actions[:10])
    steps = list(range(1, len(confs) + 1))
    ax7.bar(steps, confs, color=[plt.cm.RdYlGn(c) for c in confs], edgecolor='white')
    ax7.set_title('逐步置信度', fontsize=12, fontweight='bold')
    ax7.set_xlabel('步骤')
    ax7.set_ylabel('置信度')
    ax7.set_ylim(0, 1)
    ax7.grid(True, alpha=0.3)

    # 8. 训练时间线
    ax8 = fig.add_subplot(gs[2, 1:])
    ax8.axis('off')
    info_lines = [
        f'训练完成摘要',
        f'─────────────────────────────────',
        f'后端: {model.backend}',
        f'网格大小: {model.grid_size}×{model.grid_size}',
        f'模型架构: {" → ".join(str(d) for d in model.architecture)}',
        f'参数量: {model.get_param_count():,}',
        f'训练轮次: {len(history["epoch"])}',
        f'最终损失: {history["loss"][-1]:.6f}',
        f'最终状态误差: {eval_s_errs[-1]:.4f}' if eval_s_errs else '',
        f'最终奖励误差: {eval_r_errs[-1]:.4f}' if eval_r_errs else '',
        f'训练数据量: {history.get("data_size", "N/A")}',
    ]
    for i, line in enumerate(info_lines):
        weight = 'bold' if i < 2 else 'normal'
        fontsize = 13 if i < 2 else 11
        ax8.text(0.05, 0.95 - i * 0.09, line, fontsize=fontsize, fontweight=weight,
                 va='top', fontfamily='monospace')

    fig.suptitle('训练总结', fontsize=16, fontweight='bold')
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def train_world_model(num_epochs=200, num_episodes=5000, size=15,
                      layout_type="random", num_maps=5, use_multiprocessing=True):
    """
    训练世界模型（全面增强版）

    增强:
      - GPU (CuPy) / CPU (NumPy) 自适应
      - 多进程数据收集
      - 多地图训练（泛化能力）
      - 智能数据收集策略
      - 置信度/不确定性可视化
      - 网络架构可视化
      - 热力图分析
      - 完整的训练过程监控
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    print("=" * 60)
    print("    世界模型 (World Model) - 全面增强训练")
    print("=" * 60)

    # ========== 环境设置 ==========
    print(f"\n[环境] 网格大小: {size}×{size}, 布局: {layout_type}")

    # 生成多个训练地图
    print(f"[环境] 生成 {num_maps} 个训练地图...")
    envs = []
    for i in range(num_maps):
        if layout_type == "mixed":
            env = generate_random_map(size=size)
        else:
            env = GridWorld(size=size, layout_type=layout_type)
        envs.append(env)
        diff = env.difficulty_metrics
        print(f"  地图 {i + 1}: 布局={env.layout_type}, "
              f"最短路径={diff['shortest_path']}, 困难度={diff['difficulty_score']:.2f}")

    # 主评估环境
    env = envs[0]

    # 可视化所有地图
    fig_maps, axes_maps = plt.subplots(1, min(num_maps, 5), figsize=(4 * min(num_maps, 5), 4))
    if num_maps == 1:
        axes_maps = [axes_maps]
    for i, e in enumerate(envs[:5]):
        e.render_visual(title=f'地图 {i + 1}', ax=axes_maps[i], show_difficulty=True)
    fig_maps.suptitle('训练地图布局', fontsize=14, fontweight='bold')
    fig_maps.tight_layout()
    fig_maps.savefig("01_training_maps.png", dpi=150, bbox_inches='tight')
    plt.close(fig_maps)
    print("[可视化] 地图布局已保存: 01_training_maps.png")

    # ========== 数据收集（多进程） ==========
    print(f"\n[数据收集] 每张地图 {num_episodes} 个回合, 共 {num_maps} 张地图...")
    all_transitions = []

    if use_multiprocessing and num_maps > 1:
        env_configs = [{'size': e.size, 'layout': e.layout_type} for e in envs]
        worker_args = [(cfg, num_episodes, 60, True) for cfg in env_configs]

        with Pool(processes=min(num_maps, cpu_count())) as pool:
            results = list(tqdm(
                pool.imap(collect_data_for_map, worker_args),
                total=num_maps, desc="多进程收集", unit="map", ncols=80
            ))
        for trans in results:
            all_transitions.extend(trans)
    else:
        for i, e in enumerate(tqdm(envs, desc="收集数据", unit="map", ncols=80)):
            trans = collect_transitions_with_strategy(e, num_episodes=num_episodes, max_steps=60)
            all_transitions.extend(trans)

    print(f"[数据收集] 共收集 {len(all_transitions)} 条转移记录")

    # 数据统计可视化
    rewards_list = [t[3] for t in all_transitions]
    actions_list = [t[1] for t in all_transitions]

    fig_data, axes_data = plt.subplots(1, 3, figsize=(16, 5))
    axes_data[0].hist(rewards_list, bins=30, color='#4CAF50', edgecolor='white', alpha=0.85)
    axes_data[0].set_title('奖励分布', fontsize=13, fontweight='bold')
    axes_data[0].axvline(np.mean(rewards_list), color='red', linestyle='--',
                         label=f'均值={np.mean(rewards_list):.3f}')
    axes_data[0].legend()

    action_counts = [actions_list.count(a) for a in range(4)]
    bars = axes_data[1].bar(['↑', '↓', '←', '→'], action_counts,
                            color=['#2196F3', '#FF9800', '#9C27B0', '#F44336'], edgecolor='white')
    axes_data[1].set_title('动作分布', fontsize=13, fontweight='bold')

    axes_data[2].hist(rewards_list, bins=50, color='#2196F3', edgecolor='white', alpha=0.85)
    axes_data[2].set_title('奖励分布 (对数)', fontsize=13, fontweight='bold')
    axes_data[2].set_yscale('log')

    fig_data.suptitle('训练数据统计', fontsize=15, fontweight='bold')
    fig_data.tight_layout()
    fig_data.savefig("02_data_stats.png", dpi=150, bbox_inches='tight')
    plt.close(fig_data)
    print("[可视化] 数据统计已保存: 02_data_stats.png")

    # ========== 初始化模型 ==========
    model = WorldModel(grid_size=size, hidden_sizes=(256, 256, 128), lr=0.003)

    # 可视化网络架构
    visualize_network_architecture(model, save_path="03_network_architecture.png")
    print("[可视化] 网络架构已保存: 03_network_architecture.png")

    # ========== 训练循环 ==========
    history = {
        'epoch': [], 'loss': [], 'state_error': [], 'reward_error': [],
        'lr': [], 'data_size': len(all_transitions)
    }

    eval_interval = 10
    total_start = time.time()
    print(f"\n开始训练 {num_epochs} 轮 (每 {eval_interval} 轮评估)...")

    pbar = tqdm(range(1, num_epochs + 1), desc="训练中", unit="epoch", ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    for epoch in pbar:
        epoch_start = time.time()
        model.update_learning_rate(epoch, num_epochs)
        loss = model.train_epoch(all_transitions, batch_size=512)
        epoch_time = time.time() - epoch_start

        history['epoch'].append(epoch)
        history['loss'].append(loss)
        history['lr'].append(model.lr)

        pbar.set_postfix(loss=f"{loss:.4f}", lr=f"{model.lr:.5f}", t=f"{epoch_time:.2f}s")

        if epoch % eval_interval == 0 or epoch == 1:
            s_err, r_err = evaluate_model(model, env)
            history['state_error'].append((epoch, s_err))
            history['reward_error'].append((epoch, r_err))
            pbar.set_postfix(loss=f"{loss:.4f}", lr=f"{model.lr:.5f}",
                             s_err=f"{s_err:.3f}", r_err=f"{r_err:.3f}", t=f"{epoch_time:.2f}s")

    pbar.close()
    total_time = time.time() - total_start
    print(f"\n训练完成! 总耗时: {total_time:.1f}s")

    # 最终评估
    final_s_err, final_r_err = evaluate_model(model, env)
    print(f"[最终] 状态误差: {final_s_err:.4f}, 奖励误差: {final_r_err:.4f}")

    # ========== 可视化输出 ==========
    # 训练曲线
    fig_train, axes_train = plt.subplots(2, 2, figsize=(14, 10))
    axes_train[0, 0].plot(history['epoch'], history['loss'], color='#F44336', linewidth=1.5, alpha=0.7)
    window = min(20, len(history['loss']) // 3)
    if window > 1:
        smooth = np.convolve(history['loss'], np.ones(window) / window, mode='valid')
        axes_train[0, 0].plot(history['epoch'][window - 1:], smooth, color='#D32F2F', linewidth=2.5,
                              label=f'平滑({window})')
    axes_train[0, 0].set_title('训练损失', fontsize=13, fontweight='bold')
    axes_train[0, 0].set_yscale('log')
    axes_train[0, 0].legend()
    axes_train[0, 0].grid(True, alpha=0.3)

    eval_epochs_s = [e for e, _ in history['state_error']]
    eval_s = [s for _, s in history['state_error']]
    axes_train[0, 1].plot(eval_epochs_s, eval_s, 'o-', color='#2196F3', linewidth=2)
    axes_train[0, 1].fill_between(eval_epochs_s, eval_s, alpha=0.15, color='#2196F3')
    axes_train[0, 1].set_title('状态预测误差', fontsize=13, fontweight='bold')
    axes_train[0, 1].grid(True, alpha=0.3)

    eval_r = [r for _, r in history['reward_error']]
    axes_train[1, 0].plot(eval_epochs_s, eval_r, 's-', color='#4CAF50', linewidth=2)
    axes_train[1, 0].fill_between(eval_epochs_s, eval_r, alpha=0.15, color='#4CAF50')
    axes_train[1, 0].set_title('奖励预测误差', fontsize=13, fontweight='bold')
    axes_train[1, 0].grid(True, alpha=0.3)

    axes_train[1, 1].plot(history['epoch'], history['lr'], color='#9C27B0', linewidth=2)
    axes_train[1, 1].set_title('学习率调度', fontsize=13, fontweight='bold')
    axes_train[1, 1].grid(True, alpha=0.3)

    fig_train.suptitle('训练过程监控', fontsize=16, fontweight='bold')
    fig_train.tight_layout()
    fig_train.savefig("04_training_curves.png", dpi=150, bbox_inches='tight')
    plt.close(fig_train)
    print("[可视化] 训练曲线已保存: 04_training_curves.png")

    # 置信度可视化
    visualize_confidence(model, env, num_epochs, save_path="05_confidence.png")
    print("[可视化] 置信度分析已保存: 05_confidence.png")

    # 训练总结
    visualize_training_summary(history, model, env, save_path="06_training_summary.png")
    print("[可视化] 训练总结已保存: 06_training_summary.png")

    # 保存训练历史
    history_serializable = {k: [float(x) if isinstance(x, (np.floating, float)) else x
                                 for x in v] if isinstance(v, list) else v
                            for k, v in history.items()}
    with open("training_history.json", "w") as f:
        json.dump(history_serializable, f, indent=2)
    print("[保存] 训练历史已保存: training_history.json")

    return model, env, envs


if __name__ == "__main__":
    train_world_model()