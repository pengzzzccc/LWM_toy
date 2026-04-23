"""
世界模型 (World Model) 统一入口
支持: 训练、评估、可视化、交互式演示

用法:
    python demo.py                    # 完整流程（训练+可视化）
    python demo.py --mode train       # 仅训练
    python demo.py --mode visualize   # 仅可视化（需要已有模型）
    python demo.py --mode interactive # 交互式逐步可视化
"""
import sys
import io
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from environment import (GridWorld, generate_random_map, generate_training_maps,
                         collect_transitions_with_strategy)
from world_model import WorldModel, USE_GPU
from train import (train_world_model, evaluate_model, visualize_confidence,
                   visualize_network_architecture, visualize_training_summary)


def run_trajectory_comparisons(model, env, num_trials=3, num_steps=12):
    """轨迹对比可视化"""
    print("\n" + "=" * 60)
    print("    轨迹对比可视化")
    print("=" * 60)

    pbar = tqdm(range(num_trials), desc="轨迹对比", unit="trial", ncols=80)
    for trial in pbar:
        state = env.reset()
        start = state.copy()
        actions = [np.random.randint(0, GridWorld.NUM_ACTIONS) for _ in range(num_steps)]

        # 真实轨迹
        real_traj = [(int(start[0]), int(start[1]))]
        real_rewards = []
        s = start.copy()
        env.agent_pos = (int(s[0]), int(s[1]))
        for a in actions:
            ns, r, done = env.step(a)
            real_traj.append((int(ns[0]), int(ns[1])))
            real_rewards.append(r)
            if done:
                break

        # 模型想象轨迹
        imagined_traj, imagined_rewards, _ = model.imagine_trajectory(
            start, actions[:len(real_traj) - 1])

        # 匹配率
        total = min(len(real_traj), len(imagined_traj))
        matches = sum(1 for i in range(total) if
                      real_traj[i][0] == int(imagined_traj[i][0]) and
                      real_traj[i][1] == int(imagined_traj[i][1]))
        match_rate = matches / max(total, 1) * 100

        # 绘图
        fig = plt.figure(figsize=(18, 8))
        gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

        # 真实轨迹
        ax_real = fig.add_subplot(gs[0, 0])
        env.agent_pos = real_traj[-1]
        env.render_visual(trajectory=real_traj, title=f'真实轨迹 (Trial {trial + 1})', ax=ax_real)

        # 想象轨迹
        ax_imag = fig.add_subplot(gs[0, 1])
        int_imag = [(int(round(r)), int(round(c))) for r, c in imagined_traj]
        env.agent_pos = int_imag[-1]
        env.render_visual(trajectory=int_imag, title=f'模型想象 (Trial {trial + 1})', ax=ax_imag)

        # 匹配率饼图
        ax_match = fig.add_subplot(gs[0, 2])
        colors_pie = ['#4CAF50', '#E0E0E0']
        ax_match.pie([match_rate, 100 - match_rate], colors=colors_pie, autopct='%1.1f%%',
                     startangle=90, textprops={'fontsize': 14})
        ax_match.set_title('位置匹配率', fontsize=13, fontweight='bold')
        centre = plt.Circle((0, 0), 0.55, fc='white')
        ax_match.add_artist(centre)
        ax_match.text(0, 0, f'{matches}/{total}', ha='center', va='center',
                      fontsize=18, fontweight='bold')

        # 奖励对比
        ax_reward = fig.add_subplot(gs[1, 0])
        steps_idx = list(range(1, min(len(real_rewards), len(imagined_rewards)) + 1))
        real_r = real_rewards[:len(steps_idx)]
        imag_r = imagined_rewards[:len(steps_idx)]
        x = np.arange(len(steps_idx))
        w = 0.35
        ax_reward.bar(x - w / 2, real_r, w, label='真实', color='#2196F3', edgecolor='white')
        ax_reward.bar(x + w / 2, imag_r, w, label='想象', color='#FF9800', edgecolor='white')
        ax_reward.set_title('逐步奖励对比', fontsize=13, fontweight='bold')
        ax_reward.legend()
        ax_reward.grid(True, alpha=0.3, axis='y')

        # 累积奖励
        ax_cum = fig.add_subplot(gs[1, 1])
        ax_cum.plot(steps_idx, np.cumsum(real_r), 'o-', color='#2196F3', linewidth=2.5, label='真实')
        ax_cum.plot(steps_idx, np.cumsum(imag_r), 's--', color='#FF9800', linewidth=2.5, label='想象')
        ax_cum.fill_between(steps_idx, np.cumsum(real_r), np.cumsum(imag_r), alpha=0.15, color='gray')
        ax_cum.set_title('累积奖励对比', fontsize=13, fontweight='bold')
        ax_cum.legend()
        ax_cum.grid(True, alpha=0.3)

        # 误差热力图
        ax_err = fig.add_subplot(gs[1, 2])
        err_matrix = np.zeros((env.size, env.size))
        for i in range(min(len(real_traj), len(imagined_traj))):
            rr, rc = real_traj[i]
            ir, ic = int(imagined_traj[i][0]), int(imagined_traj[i][1])
            err = np.sqrt((rr - ir) ** 2 + (rc - ic) ** 2)
            err_matrix[rr][rc] = max(err_matrix[rr][rc], err)
        im = ax_err.imshow(err_matrix, cmap='YlOrRd', interpolation='nearest')
        plt.colorbar(im, ax=ax_err, label='误差')
        ax_err.set_title('位置误差热力图', fontsize=13, fontweight='bold')

        action_str = " ".join(GridWorld.ACTION_NAMES[a] for a in actions)
        fig.suptitle(f'试验 {trial + 1}: 起点({int(start[0])},{int(start[1])}) | '
                     f'匹配率: {match_rate:.1f}%', fontsize=14, fontweight='bold', y=1.02)
        fig.savefig(f"07_trajectory_{trial + 1}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

        pbar.set_postfix(trial=trial + 1, match=f"{match_rate:.0f}%")

    pbar.close()
    print("[可视化] 轨迹对比图已保存: 07_trajectory_*.png")


def run_dream_visualization(model, env, num_steps=20):
    """模型做梦可视化"""
    print("\n" + "=" * 60)
    print('    模型"做梦" — 纯想象轨迹')
    print("=" * 60)

    state = env.reset()
    actions = [np.random.randint(0, GridWorld.NUM_ACTIONS) for _ in range(num_steps)]

    # 带置信度的想象
    traj, rewards, total_r, confs, s_stds, r_stds = \
        model.imagine_trajectory_with_confidence(state, actions)
    int_traj = [(int(round(r)), int(round(c))) for r, c in traj]

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # 想象轨迹
    ax1 = fig.add_subplot(gs[0, 0])
    env.agent_pos = int_traj[-1]
    env.render_visual(trajectory=int_traj, title='想象轨迹', ax=ax1)
    # 置信度颜色
    for i, (conf, (r, c)) in enumerate(zip(confs, int_traj[1:])):
        color = plt.cm.RdYlGn(conf)
        ax1.plot(c, r, 'o', color=color, markersize=10, zorder=7)

    # 逐步奖励
    ax2 = fig.add_subplot(gs[0, 1])
    steps = list(range(1, len(rewards) + 1))
    colors = ['#4CAF50' if r > 0 else '#F44336' if r < -0.5 else '#FFC107' for r in rewards]
    ax2.bar(steps, rewards, color=colors, edgecolor='white')
    ax2.axhline(0, color='gray', linewidth=0.8)
    ax2.set_title('逐步奖励', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # 累积奖励
    ax3 = fig.add_subplot(gs[0, 2])
    cum_r = np.cumsum(rewards)
    ax3.plot(steps, cum_r, 'o-', color='#9C27B0', linewidth=2.5, markersize=6)
    ax3.fill_between(steps, cum_r, alpha=0.15, color='#9C27B0')
    ax3.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax3.set_title(f'累积奖励 (总计: {total_r:.2f})', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 位置变化
    ax4 = fig.add_subplot(gs[1, 0])
    rows = [p[0] for p in int_traj]
    cols = [p[1] for p in int_traj]
    ax4.plot(range(len(int_traj)), rows, 'o-', color='#E91E63', linewidth=2, label='行')
    ax4.plot(range(len(int_traj)), cols, 's-', color='#3F51B5', linewidth=2, label='列')
    ax4.set_title('位置变化', fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 置信度
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.bar(steps, confs, color=[plt.cm.RdYlGn(c) for c in confs], edgecolor='white')
    ax5.set_title('逐步置信度', fontsize=13, fontweight='bold')
    ax5.set_ylim(0, 1)
    ax5.grid(True, alpha=0.3)

    # 不确定性
    ax6 = fig.add_subplot(gs[1, 2])
    s_std_means = [np.mean(s) for s in s_stds]
    ax6.bar(steps, s_std_means, color='#FF9800', edgecolor='white', label='状态不确定')
    ax6_r = ax6.twinx()
    ax6_r.plot(steps, r_stds, 's-', color='#F44336', linewidth=2, label='奖励不确定')
    ax6.set_title('预测不确定性', fontsize=13, fontweight='bold')
    ax6.legend(loc='upper left')
    ax6_r.legend(loc='upper right')
    ax6.grid(True, alpha=0.3)

    fig.suptitle(f'模型"做梦" — 纯想象轨迹\n'
                 f'起始: ({int(state[0])},{int(state[1])}) | 总奖励: {total_r:.2f}',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.savefig("08_dream_trajectory.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("[可视化] 想象轨迹已保存: 08_dream_trajectory.png")


def run_model_evaluation(model, env, num_trials=10):
    """模型综合评估"""
    print("\n" + "=" * 60)
    print("    模型综合评估")
    print("=" * 60)

    state_errors_all = []
    reward_errors_all = []
    match_rates = []

    pbar = tqdm(range(num_trials), desc="评估", unit="trial", ncols=80)
    for _ in pbar:
        state = env.reset()
        start = state.copy()
        actions = [np.random.randint(0, GridWorld.NUM_ACTIONS) for _ in range(15)]

        real_traj = [(int(start[0]), int(start[1]))]
        s = start.copy()
        env.agent_pos = (int(s[0]), int(s[1]))
        for a in actions:
            ns, r, done = env.step(a)
            real_traj.append((int(ns[0]), int(ns[1])))
            pred_next, pred_reward = model.predict(s, a)
            state_errors_all.append(np.linalg.norm(pred_next - ns))
            reward_errors_all.append(abs(pred_reward - r))
            s = ns
            if done:
                break

        imagined_traj, _, _ = model.imagine_trajectory(start, actions[:len(real_traj) - 1])
        total = min(len(real_traj), len(imagined_traj))
        matches = sum(1 for i in range(total) if
                      real_traj[i][0] == int(imagined_traj[i][0]) and
                      real_traj[i][1] == int(imagined_traj[i][1]))
        match_rates.append(matches / max(total, 1) * 100)
        pbar.set_postfix(match=f"{match_rates[-1]:.0f}%")

    pbar.close()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].hist(state_errors_all, bins=25, color='#2196F3', edgecolor='white', alpha=0.85)
    axes[0].axvline(np.mean(state_errors_all), color='red', linestyle='--',
                    label=f'均值={np.mean(state_errors_all):.3f}')
    axes[0].set_title('状态误差分布', fontsize=13, fontweight='bold')
    axes[0].legend()

    axes[1].hist(reward_errors_all, bins=25, color='#4CAF50', edgecolor='white', alpha=0.85)
    axes[1].axvline(np.mean(reward_errors_all), color='red', linestyle='--',
                    label=f'均值={np.mean(reward_errors_all):.3f}')
    axes[1].set_title('奖励误差分布', fontsize=13, fontweight='bold')
    axes[1].legend()

    bp = axes[2].boxplot(match_rates, patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor('#FF9800')
    bp['boxes'][0].set_alpha(0.7)
    axes[2].scatter([1] * len(match_rates), match_rates, color='#333', alpha=0.6, zorder=5)
    axes[2].axhline(np.mean(match_rates), color='red', linestyle='--',
                    label=f'均值={np.mean(match_rates):.1f}%')
    axes[2].set_title('匹配率分布', fontsize=13, fontweight='bold')
    axes[2].legend()

    fig.suptitle(f'模型综合评估 ({num_trials} 次试验)', fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig("09_model_evaluation.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("[可视化] 综合评估已保存: 09_model_evaluation.png")


def main():
    parser = argparse.ArgumentParser(description='世界模型训练与可视化')
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'train', 'visualize', 'interactive'],
                        help='运行模式')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮次')
    parser.add_argument('--episodes', type=int, default=5000, help='每地图回合数')
    parser.add_argument('--size', type=int, default=15, help='网格大小')
    parser.add_argument('--maps', type=int, default=5, help='训练地图数')
    parser.add_argument('--layout', type=str, default='random',
                        choices=['random', 'maze', 'rooms', 'corridor', 'spiral', 'mixed'],
                        help='地图布局类型')
    args = parser.parse_args()

    np.random.seed(42)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    print("=" * 60)
    print("    世界模型 (World Model) - 全面增强版")
    print(f"    模式: {args.mode} | 后端: {'GPU (CuPy)' if USE_GPU else 'CPU (NumPy)'}")
    print("=" * 60)

    model, env, envs = None, None, None

    # ========== 训练阶段 ==========
    if args.mode in ['full', 'train']:
        model, env, envs = train_world_model(
            num_epochs=args.epochs,
            num_episodes=args.episodes,
            size=args.size,
            layout_type=args.layout,
            num_maps=args.maps
        )

    # ========== 可视化阶段 ==========
    if args.mode in ['full', 'visualize']:
        if model is None:
            print("[警告] 没有已训练的模型，跳过可视化")
        else:
            run_trajectory_comparisons(model, env, num_trials=3, num_steps=12)
            run_dream_visualization(model, env, num_steps=20)
            run_model_evaluation(model, env, num_trials=10)

    # ========== 交互式模式 ==========
    if args.mode == 'interactive':
        if model is None:
            print("[阶段1] 先训练模型...")
            model, env, envs = train_world_model(
                num_epochs=args.epochs,
                num_episodes=args.episodes,
                size=args.size,
                layout_type=args.layout,
                num_maps=args.maps
            )

        from visualize_interactive import (step_by_step_real_vs_model,
                                           step_by_step_dream,
                                           step_by_step_multi_trajectory)
        while True:
            print("\n" + "-" * 40)
            print("  选择可视化模式:")
            print("  1. 真实环境 vs 模型预测 (逐步)")
            print("  2. 模型\"做梦\" (纯想象, 逐步)")
            print("  3. 多条轨迹连续展示")
            print("  4. 运行全部")
            print("  0. 退出")
            print("-" * 40)
            try:
                choice = input("请输入选项 (0-4): ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if choice == '1':
                step_by_step_real_vs_model(model, env, num_steps=15, pause_time=1.0)
            elif choice == '2':
                step_by_step_dream(model, env, num_steps=20, pause_time=0.8)
            elif choice == '3':
                step_by_step_multi_trajectory(model, env, num_trajectories=3,
                                              steps_per=12, pause_time=0.6)
            elif choice == '4':
                step_by_step_real_vs_model(model, env, num_steps=15, pause_time=1.0)
                step_by_step_dream(model, env, num_steps=20, pause_time=0.8)
                step_by_step_multi_trajectory(model, env, num_trajectories=3,
                                              steps_per=12, pause_time=0.6)
            elif choice == '0':
                break

    # ========== 完成 ==========
    print("\n" + "=" * 60)
    print("    演示完成!")
    print("=" * 60)
    print("  生成的可视化文件:")
    print("    01_training_maps.png       - 训练地图布局")
    print("    02_data_stats.png          - 数据统计")
    print("    03_network_architecture.png - 网络架构")
    print("    04_training_curves.png     - 训练曲线")
    print("    05_confidence.png          - 置信度分析")
    print("    06_training_summary.png    - 训练总结")
    print("    07_trajectory_*.png        - 轨迹对比")
    print("    08_dream_trajectory.png    - 想象轨迹")
    print("    09_model_evaluation.png    - 模型评估")
    print("    training_history.json      - 训练历史")
    print("=" * 60)


if __name__ == "__main__":
    main()