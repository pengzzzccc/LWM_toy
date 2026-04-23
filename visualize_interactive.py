"""
交互式逐步可视化脚本
在训练完成后，逐步展示模型实际运行的每一步预测过程。
包含 tqdm 进度条、GPU/CPU 自适应、置信度显示。
"""
import sys
import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from environment import GridWorld, collect_transitions, generate_random_map
from world_model import WorldModel, USE_GPU, to_numpy
from train import train_world_model


def draw_grid_step(env, real_pos, pred_pos, step_num, action_name,
                   real_reward, pred_reward, real_done, pred_done,
                   title="", fig=None, ax=None, confidence=None):
    """绘制单步网格状态（支持置信度显示）"""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    ax.clear()
    size = env.size

    # 构建网格
    grid = np.ones((size, size, 3))

    # 墙壁: 深灰
    for r, c in env.walls:
        grid[r][c] = [0.2, 0.2, 0.2]

    # 目标: 绿色
    gr, gc = env.goal
    grid[gr][gc] = [0.3, 0.75, 0.3]

    # 水: 蓝色
    from environment import Terrain
    for r in range(size):
        for c in range(size):
            if env.terrain[r][c] == Terrain.WATER:
                grid[r][c] = [0.3, 0.6, 0.9]
            elif env.terrain[r][c] == Terrain.MUD:
                grid[r][c] = [0.6, 0.4, 0.2]

    ax.imshow(grid, interpolation='nearest')

    # 绘制网格线
    for i in range(size + 1):
        ax.axhline(i - 0.5, color='gray', linewidth=1, alpha=0.4)
        ax.axvline(i - 0.5, color='gray', linewidth=1, alpha=0.4)

    # 墙壁标记
    for r, c in env.walls:
        ax.text(c, r, '■', ha='center', va='center',
                fontsize=max(6, 14 - size // 3), color='white')

    # 目标标记
    ax.text(gc, gr, 'G', ha='center', va='center',
            fontsize=max(6, 14 - size // 3), fontweight='bold', color='white')

    # 真实位置: 蓝色圆形
    if real_pos is not None:
        rr, rc = real_pos
        circle_real = plt.Circle((rc, rr), 0.3, color='#2196F3', zorder=10)
        ax.add_patch(circle_real)
        ax.text(rc, rr, 'R', ha='center', va='center', fontsize=9,
                fontweight='bold', color='white', zorder=11)

    # 预测位置: 橙色菱形
    if pred_pos is not None and pred_pos != real_pos:
        pr, pc = pred_pos
        diamond = plt.Polygon([
            [pc, pr - 0.35], [pc + 0.35, pr],
            [pc, pr + 0.35], [pc - 0.35, pr]
        ], closed=True, color='#FF9800', zorder=9)
        ax.add_patch(diamond)
        ax.text(pc, pr, 'P', ha='center', va='center', fontsize=9,
                fontweight='bold', color='white', zorder=11)

    # 信息
    info = f"步骤 {step_num} | {action_name}\n"
    info += f"真实奖励: {real_reward:+.1f}" + (" [完成!]" if real_done else "") + "\n"
    info += f"预测奖励: {pred_reward:+.1f}" + (" [预测完成]" if pred_done else "")
    if confidence is not None:
        info += f"\n置信度: {confidence:.2f}"
    pos_info = ""
    if real_pos is not None:
        pos_info += f"\n真实: ({real_pos[0]},{real_pos[1]})"
    if pred_pos is not None:
        pos_info += f"  预测: ({pred_pos[0]},{pred_pos[1]})"

    ax.set_title(f"{title}\n{info}{pos_info}", fontsize=10, fontweight='bold',
                 loc='left', linespacing=1.4)

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2196F3',
                   markersize=10, label='真实位置 (R)'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#FF9800',
                   markersize=10, label='预测位置 (P)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#4CAF50',
                   markersize=10, label='目标 (G)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)
    ax.set_xlabel('列')
    ax.set_ylabel('行')

    if fig is not None:
        fig.canvas.draw_idle()
        fig.canvas.flush_events()


def step_by_step_real_vs_model(model, env, num_steps=15, pause_time=1.0):
    """逐步可视化: 真实环境 vs 模型预测"""
    print("\n" + "=" * 60)
    print("  逐步可视化: 真实环境 vs 模型预测")
    print("  关闭窗口或按 Ctrl+C 退出")
    print("=" * 60)

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    state = env.reset()
    start_pos = (int(state[0]), int(state[1]))

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle('逐步可视化: 真实环境 (左) vs 模型预测 (右)',
                 fontsize=15, fontweight='bold')

    state_errors = []
    reward_errors = []
    match_count = 0

    print(f"\n起始位置: {start_pos}")
    plt.ion()
    plt.show()

    pbar = tqdm(range(num_steps), desc="逐步对比", unit="step", ncols=80)
    for step in pbar:
        action = np.random.randint(0, GridWorld.NUM_ACTIONS)
        action_name = GridWorld.ACTION_NAMES[action]

        real_next, real_reward, real_done = env.step(action)

        # 使用置信度预测
        pred_next, pred_reward, confidence, _, _ = model.predict_with_confidence(state, action)
        pred_pos = (int(np.clip(np.round(pred_next[0]), 0, env.size - 1)),
                    int(np.clip(np.round(pred_next[1]), 0, env.size - 1)))
        real_pos = (int(real_next[0]), int(real_next[1]))

        s_err = np.linalg.norm(pred_next - real_next)
        r_err = abs(pred_reward - real_reward)
        state_errors.append(s_err)
        reward_errors.append(r_err)
        if pred_pos == real_pos:
            match_count += 1

        pred_done = (pred_pos == env.goal)

        draw_grid_step(env, real_pos, None, step + 1, action_name,
                       real_reward, 0, real_done, False,
                       title="真实环境", ax=axes[0])
        draw_grid_step(env, real_pos, pred_pos, step + 1, action_name,
                       real_reward, pred_reward, real_done, pred_done,
                       title="模型预测 (蓝色=真实, 橙色=预测)", ax=axes[1],
                       confidence=confidence)

        for txt in fig.texts:
            if txt.get_position()[1] < 0.05:
                txt.remove()
        status = f"状态误差: {s_err:.3f}  |  奖励误差: {r_err:.3f}  |  "
        status += f"匹配率: {match_count}/{step + 1} ({100 * match_count / (step + 1):.0f}%)"
        fig.text(0.5, 0.01, status, ha='center', fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        fig.tight_layout(rect=[0, 0.04, 1, 0.95])
        plt.draw()
        plt.pause(pause_time)

        pbar.set_postfix(s_err=f"{s_err:.3f}", match=f"{match_count}/{step + 1}",
                         conf=f"{confidence:.2f}")

        state = real_next
        if real_done:
            print(f"\n  到达目标! 在第 {step + 1} 步完成")
            break

    pbar.close()

    print(f"\n--- 统计 ---")
    print(f"总步数: {len(state_errors)}")
    print(f"平均状态误差: {np.mean(state_errors):.4f}")
    print(f"平均奖励误差: {np.mean(reward_errors):.4f}")
    print(f"位置匹配率: {match_count}/{len(state_errors)} "
          f"({100 * match_count / max(len(state_errors), 1):.0f}%)")

    plt.ioff()
    input("\n按 Enter 继续...")


def step_by_step_dream(model, env, num_steps=20, pause_time=0.8):
    """逐步可视化: 模型"做梦"（带置信度）"""
    print("\n" + "=" * 60)
    print('  逐步可视化: 模型"做梦" (纯想象 + 置信度)')
    print("=" * 60)

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    state = env.reset()
    start_pos = (int(state[0]), int(state[1]))

    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.3)
    ax_grid = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0, 1])

    fig.suptitle('模型"做梦" — 纯想象轨迹 + 置信度',
                 fontsize=15, fontweight='bold')

    dream_state = state.copy()
    trajectory = [(int(dream_state[0]), int(dream_state[1]))]
    all_rewards = []
    all_confidences = []
    total_reward = 0.0

    print(f"\n起始位置: {start_pos}")
    plt.ion()
    plt.show()

    pbar = tqdm(range(num_steps), desc="做梦中", unit="step", ncols=80)
    for step in pbar:
        action = np.random.randint(0, GridWorld.NUM_ACTIONS)
        action_name = GridWorld.ACTION_NAMES[action]

        pred_next, pred_reward, confidence, _, _ = model.predict_with_confidence(
            dream_state, action)
        pred_pos = (int(np.clip(np.round(pred_next[0]), 0, env.size - 1)),
                    int(np.clip(np.round(pred_next[1]), 0, env.size - 1)))

        dream_state = np.array([float(pred_pos[0]), float(pred_pos[1])], dtype=np.float32)
        trajectory.append(pred_pos)
        all_rewards.append(pred_reward)
        all_confidences.append(confidence)
        total_reward += pred_reward

        # 绘制网格图
        ax_grid.clear()
        draw_dream_grid(env, trajectory, step + 1, action_name,
                        pred_reward, total_reward, pred_pos == env.goal,
                        confidence=confidence, ax=ax_grid)

        # 绘制信息面板
        ax_info.clear()
        draw_dream_info_panel(ax_info, trajectory, all_rewards,
                              total_reward, step + 1, action_name,
                              confidences=all_confidences)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        plt.draw()
        plt.pause(pause_time)

        pbar.set_postfix(pos=f"({pred_pos[0]},{pred_pos[1]})",
                         r=f"{total_reward:+.2f}", conf=f"{confidence:.2f}")

        if pred_pos == env.goal:
            print(f"\n  模型在第 {step + 1} 步\"想象\"到达了目标!")
            break

    pbar.close()
    print(f"\n--- 想象统计 ---")
    print(f"总步数: {len(all_rewards)}")
    print(f"总奖励: {total_reward:.2f}")
    print(f"平均奖励: {total_reward / max(len(all_rewards), 1):.3f}")
    print(f"平均置信度: {np.mean(all_confidences):.3f}")

    plt.ioff()
    input("\n按 Enter 继续...")


def draw_dream_grid(env, trajectory, step_num, action_name, reward, total_reward,
                    done, confidence=None, ax=None):
    """绘制做梦过程中的网格"""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    size = env.size
    grid = np.ones((size, size, 3))
    for r, c in env.walls:
        grid[r][c] = [0.2, 0.2, 0.2]
    gr, gc = env.goal
    grid[gr][gc] = [0.3, 0.75, 0.3]

    from environment import Terrain
    for r in range(size):
        for c in range(size):
            if env.terrain[r][c] == Terrain.WATER:
                grid[r][c] = [0.3, 0.6, 0.9]
            elif env.terrain[r][c] == Terrain.MUD:
                grid[r][c] = [0.6, 0.4, 0.2]

    ax.imshow(grid, interpolation='nearest')

    for i in range(size + 1):
        ax.axhline(i - 0.5, color='gray', linewidth=1, alpha=0.4)
        ax.axvline(i - 0.5, color='gray', linewidth=1, alpha=0.4)

    for r, c in env.walls:
        ax.text(c, r, '■', ha='center', va='center',
                fontsize=max(6, 14 - size // 3), color='white')
    ax.text(gc, gr, 'G', ha='center', va='center',
            fontsize=max(6, 14 - size // 3), fontweight='bold', color='white')

    # 绘制轨迹
    if len(trajectory) > 1:
        rows = [p[0] for p in trajectory]
        cols = [p[1] for p in trajectory]
        for i in range(1, len(trajectory)):
            alpha = 0.3 + 0.7 * (i / len(trajectory))
            ax.plot([cols[i - 1], cols[i]], [rows[i - 1], rows[i]],
                    '-', color='#FF5722', linewidth=2.5, alpha=alpha, zorder=5)

        for i, (r, c) in enumerate(trajectory[:-1]):
            alpha = 0.3 + 0.5 * (i / len(trajectory))
            ax.plot(c, r, 'o', color='#FF9800', markersize=6, alpha=alpha, zorder=6)

    # 当前位置
    cur_r, cur_c = trajectory[-1]
    circle = plt.Circle((cur_c, cur_r), 0.35, color='#E91E63', zorder=10)
    ax.add_patch(circle)
    ax.text(cur_c, cur_r, '★', ha='center', va='center', fontsize=12,
            color='white', zorder=11)

    title = f"做梦步骤 {step_num} | {action_name} | 奖励: {reward:+.2f} | 累积: {total_reward:+.2f}"
    if confidence is not None:
        title += f" | 置信度: {confidence:.2f}"
    if done:
        title += "  🎯"
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)
    ax.set_xlabel('列')
    ax.set_ylabel('行')


def draw_dream_info_panel(ax, trajectory, rewards, total_reward, step_num,
                          action_name, confidences=None):
    """绘制信息面板"""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    if len(rewards) > 0:
        # 累积奖励曲线
        ax_inset = ax.inset_axes([0.05, 0.40, 0.90, 0.35])
        cum_r = np.cumsum(rewards)
        steps = list(range(1, len(rewards) + 1))
        ax_inset.plot(steps, cum_r, 'o-', color='#9C27B0', linewidth=2, markersize=4)
        ax_inset.fill_between(steps, cum_r, alpha=0.15, color='#9C27B0')
        ax_inset.axhline(0, color='gray', linewidth=0.8, linestyle='--')
        ax_inset.set_title('累积奖励', fontsize=10, fontweight='bold')
        ax_inset.set_xlabel('步骤', fontsize=9)
        ax_inset.grid(True, alpha=0.3)

    # 文本信息
    ax.text(5, 9.5, f'步骤: {step_num}', fontsize=13, fontweight='bold',
            ha='center', va='top')
    ax.text(5, 9.0, f'动作: {action_name}', fontsize=11, ha='center', va='top')
    ax.text(5, 8.5, f'位置: ({trajectory[-1][0]}, {trajectory[-1][1]})',
            fontsize=11, ha='center', va='top')
    ax.text(5, 8.0, f'总奖励: {total_reward:.2f}', fontsize=11,
            ha='center', va='top', color='#9C27B0')

    if len(rewards) > 0:
        ax.text(5, 7.5, f'平均奖励: {np.mean(rewards):.3f}', fontsize=10,
                ha='center', va='top')

    if confidences and len(confidences) > 0:
        ax.text(5, 7.0, f'平均置信度: {np.mean(confidences):.3f}', fontsize=10,
                ha='center', va='top', color='#4CAF50')

    # 最近几步
    recent = min(5, len(rewards))
    if recent > 0:
        ax.text(5, 6.4, '--- 最近步骤 ---', fontsize=10,
                ha='center', va='top', color='gray')
        for i in range(recent):
            idx = len(rewards) - recent + i
            step_r, step_c = trajectory[idx + 1]
            conf_str = f" c={confidences[idx]:.2f}" if confidences else ""
            ax.text(5, 5.9 - i * 0.45,
                    f'步{idx + 1}: ({step_r},{step_c}) r={rewards[idx]:+.2f}{conf_str}',
                    fontsize=9, ha='center', va='top')


def step_by_step_multi_trajectory(model, env, num_trajectories=3, steps_per=12, pause_time=0.6):
    """逐步可视化: 多条轨迹连续展示"""
    print("\n" + "=" * 60)
    print(f"  逐步可视化: {num_trajectories} 条连续想象轨迹")
    print("=" * 60)

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plt.ion()
    plt.show()

    total_steps = num_trajectories * steps_per
    pbar = tqdm(total=total_steps, desc="多轨迹展示", unit="step", ncols=80)

    for traj_id in range(num_trajectories):
        print(f"\n--- 轨迹 {traj_id + 1}/{num_trajectories} ---")
        state = env.reset()
        dream_state = state.copy()
        trajectory = [(int(dream_state[0]), int(dream_state[1]))]
        rewards = []
        total_reward = 0.0

        for step in range(steps_per):
            action = np.random.randint(0, GridWorld.NUM_ACTIONS)
            action_name = GridWorld.ACTION_NAMES[action]

            pred_next, pred_reward = model.predict(dream_state, action)
            pred_pos = (int(np.clip(np.round(pred_next[0]), 0, env.size - 1)),
                        int(np.clip(np.round(pred_next[1]), 0, env.size - 1)))

            dream_state = np.array([float(pred_pos[0]), float(pred_pos[1])], dtype=np.float32)
            trajectory.append(pred_pos)
            rewards.append(pred_reward)
            total_reward += pred_reward

            ax.clear()
            draw_dream_grid(env, trajectory, step + 1, action_name,
                            pred_reward, total_reward, pred_pos == env.goal, ax=ax)
            ax.set_title(f'轨迹 {traj_id + 1}/{num_trajectories} | '
                         f'步 {step + 1} | {action_name} | '
                         f'奖励: {pred_reward:+.2f} | 累积: {total_reward:+.2f}',
                         fontsize=11, fontweight='bold')

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(pause_time)

            pbar.update(1)
            pbar.set_postfix(traj=traj_id + 1, pos=f"({pred_pos[0]},{pred_pos[1]})")

            if pred_pos == env.goal:
                print(f"  到达目标!")
                break

        print(f"  轨迹 {traj_id + 1} 结束, 总奖励: {total_reward:.2f}")

    pbar.close()
    plt.ioff()
    print("\n所有轨迹展示完毕!")


def main():
    np.random.seed(42)

    print("=" * 60)
    print("  世界模型 - 交互式逐步可视化")
    print("=" * 60)

    # 第一阶段: 训练
    print("\n[阶段1] 训练世界模型...")
    model, env, envs = train_world_model(num_epochs=200, num_episodes=3000,
                                          size=15, num_maps=3)

    print("\n" + "=" * 60)
    print("  训练完成! 进入交互式可视化模式")
    print("=" * 60)

    # 第二阶段: 逐步可视化
    while True:
        print("\n" + "-" * 40)
        print("  选择可视化模式:")
        print("  1. 真实环境 vs 模型预测 (逐步)")
        print("  2. 模型\"做梦\" (纯想象+置信度)")
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
        else:
            print("无效选项，请重试")

    print("\n演示结束!")


if __name__ == "__main__":
    main()