import numpy as np
import sys
import traceback

np.random.seed(42)
errors = []

print('=== 测试 environment ===')
try:
    from environment import GridWorld, generate_random_map, collect_transitions
    
    for layout in ['random', 'maze', 'rooms', 'corridor', 'spiral']:
        try:
            env = GridWorld(size=10, layout_type=layout)
            state = env.reset()
            total_r = 0
            for _ in range(20):
                action = np.random.randint(0, 4)
                ns, r, done = env.step(action)
                total_r += r
                if done:
                    break
            print(f'  {layout:10s} OK - walls={len(env.walls)}, diff={env.difficulty_metrics["difficulty_score"]:.2f}')
        except Exception as e:
            print(f'  {layout:10s} FAILED: {e}')
            errors.append(f'env.{layout}: {traceback.format_exc()}')
    
    try:
        env = generate_random_map(size=12)
        print(f'  generate_random_map OK')
    except Exception as e:
        print(f'  generate_random_map FAILED: {e}')
        errors.append(f'generate_random_map: {traceback.format_exc()}')
    
    try:
        env = GridWorld(size=8, layout_type='random')
        trans = collect_transitions(env, num_episodes=10, max_steps=20)
        print(f'  collect_transitions OK - {len(trans)} transitions')
    except Exception as e:
        print(f'  collect_transitions FAILED: {e}')
        errors.append(f'collect_transitions: {traceback.format_exc()}')

except Exception as e:
    print(f'  IMPORT FAILED: {e}')
    errors.append(f'env import: {traceback.format_exc()}')

print()
print('=== 测试 world_model ===')
try:
    from world_model import WorldModel, USE_GPU, to_numpy
    
    model = WorldModel(grid_size=10, hidden_sizes=(64, 32), lr=0.001)
    print(f'  USE_GPU={USE_GPU}')
    
    state = np.array([3.0, 5.0], dtype=np.float32)
    
    try:
        pred_s, pred_r = model.predict(state, 1)
        print(f'  predict OK - state={pred_s}, reward={pred_r:.3f}')
    except Exception as e:
        print(f'  predict FAILED: {e}')
        errors.append(f'predict: {traceback.format_exc()}')
    
    try:
        pred_s, pred_r, conf, s_std, r_std = model.predict_with_confidence(state, 1)
        print(f'  predict_with_confidence OK - conf={conf:.3f}')
    except Exception as e:
        print(f'  predict_with_confidence FAILED: {e}')
        errors.append(f'predict_conf: {traceback.format_exc()}')
    
    try:
        env = GridWorld(size=8, layout_type='random')
        trans = collect_transitions(env, num_episodes=50, max_steps=30)
        loss = model.train_epoch(trans, batch_size=32)
        print(f'  train_epoch OK - loss={loss:.4f}')
    except Exception as e:
        print(f'  train_epoch FAILED: {e}')
        errors.append(f'train_epoch: {traceback.format_exc()}')
    
    try:
        start = np.array([1.0, 2.0], dtype=np.float32)
        actions = [np.random.randint(0, 4) for _ in range(10)]
        traj, rewards, total = model.imagine_trajectory(start, actions)
        print(f'  imagine_trajectory OK - len={len(traj)}, total_r={total:.2f}')
    except Exception as e:
        print(f'  imagine_trajectory FAILED: {e}')
        errors.append(f'imagine: {traceback.format_exc()}')
    
    try:
        traj, rewards, total, confs, s_stds, r_stds = model.imagine_trajectory_with_confidence(start, actions)
        print(f'  imagine_trajectory_with_confidence OK - avg_conf={np.mean(confs):.3f}')
    except Exception as e:
        print(f'  imagine_trajectory_with_confidence FAILED: {e}')
        errors.append(f'imagine_conf: {traceback.format_exc()}')

except Exception as e:
    print(f'  IMPORT FAILED: {e}')
    errors.append(f'model import: {traceback.format_exc()}')

print()
print('=== 测试 train.py 函数 ===')
try:
    from train import evaluate_model, visualize_network_architecture, visualize_confidence
    
    env = GridWorld(size=8, layout_type='random')
    
    try:
        s_err, r_err = evaluate_model(model, env, num_episodes=5, max_steps=10)
        print(f'  evaluate_model OK - s_err={s_err:.3f}, r_err={r_err:.3f}')
    except Exception as e:
        print(f'  evaluate_model FAILED: {e}')
        errors.append(f'evaluate: {traceback.format_exc()}')
    
    try:
        visualize_network_architecture(model, save_path='test_arch.png')
        print(f'  visualize_network_architecture OK')
    except Exception as e:
        print(f'  visualize_network_architecture FAILED: {e}')
        errors.append(f'viz_arch: {traceback.format_exc()}')
    
    try:
        visualize_confidence(model, env, step_num=1, save_path='test_conf.png')
        print(f'  visualize_confidence OK')
    except Exception as e:
        print(f'  visualize_confidence FAILED: {e}')
        errors.append(f'viz_conf: {traceback.format_exc()}')

except Exception as e:
    print(f'  IMPORT FAILED: {e}')
    errors.append(f'train import: {traceback.format_exc()}')

print()
print(f'=== 测试完成: {len(errors)} 个错误 ===')
for i, err in enumerate(errors):
    print(f'\n--- 错误 {i+1} ---')
    print(err)