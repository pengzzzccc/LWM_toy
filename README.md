# 世界模型 (World Model) - 增强版

基于 NumPy/CuPy 的纯手写世界模型（World Model），在网格世界环境中学习状态转移函数和奖励预测，并利用学到的模型进行"想象"和规划。

## 核心特性

- **GPU/CPU 自适应**: 自动检测并使用 CuPy (CUDA) 加速，回退到 NumPy/OpenBLAS
- **多地图训练**: 程序化生成多种布局地图，提升模型泛化能力
- **置信度估计**: 基于 MC Dropout 的预测不确定性量化
- **热力图分析**: 可视化智能体移动模式和探索覆盖度
- **困难度分析**: 自动评估地图路径规划难度
- **交互式可视化**: 逐步动画展示模型预测过程
- **多进程数据收集**: 并行收集多张地图的训练数据
- **tqdm 进度条**: 实时显示训练和数据收集进度

## 环境配置

```bash
pip install -r requirements.txt
```

依赖项:
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- tqdm >= 4.65.0
- cupy-cuda12x >= 12.0.0 (可选，GPU 加速)

## 快速开始

```bash
# 完整流程（训练 + 可视化输出）
python demo.py

# 交互式逐步可视化
python demo.py --mode interactive

# 仅训练（不生成可视化）
python demo.py --mode train

# 自定义参数
python demo.py --epochs 300 --size 20 --maps 10 --layout mixed
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | full | 运行模式: full/train/visualize/interactive |
| `--epochs` | 200 | 训练轮次 |
| `--episodes` | 5000 | 每张地图收集的回合数 |
| `--size` | 15 | 网格大小 (NxN) |
| `--maps` | 5 | 训练地图数量 |
| `--layout` | random | 地图布局: random/maze/rooms/corridor/spiral/mixed |

## 项目结构

```
LWM_building/
├── world_model.py          # 世界模型 MLP (GPU/CPU自适应)
├── environment.py          # 网格世界环境 (程序化地图生成)
├── train.py                # 训练流程 + 可视化函数
├── demo.py                 # 统一入口 + 命令行参数
├── visualize_interactive.py # 交互式逐步可视化
├── requirements.txt        # 依赖列表
└── README.md               # 本文档
```

## 模型架构

```
输入层 (6维) → 隐藏层1 (256) → 隐藏层2 (256) → 隐藏层3 (128) → 输出层 (3维)
   ↑                                                                  ↓
[状态(2) + 动作one-hot(4)]                              [下一状态(2) + 奖励(1)]
```

- 激活函数: ReLU
- 权重初始化: He initialization
- 学习率调度: 余弦退火
- 梯度裁剪: [-1.0, 1.0]
- 置信度估计: MC Dropout (10次采样)

## 地图布局类型

| 类型 | 说明 |
|------|------|
| random | 随机放置墙壁、水、泥地 |
| maze | 递归回溯算法生成迷宫 |
| rooms | 多房间通过门连接 |
| corridor | 走廊式布局 |
| spiral | 螺旋墙布局 |
| mixed | 随机混合以上布局 |

## 可视化输出

运行 `python demo.py` 后生成以下文件:

| 文件 | 内容 |
|------|------|
| `01_training_maps.png` | 训练地图布局 (困难度分析) |
| `02_data_stats.png` | 数据统计 (奖励/动作分布) |
| `03_network_architecture.png` | 网络架构图 |
| `04_training_curves.png` | 训练过程曲线 (损失/误差/学习率) |
| `05_confidence.png` | 置信度分析 (不确定性可视化) |
| `06_training_summary.png` | 训练总结 (综合信息) |
| `07_trajectory_*.png` | 轨迹对比 (真实 vs 想象) |
| `08_dream_trajectory.png` | 模型"做梦"轨迹 |
| `09_model_evaluation.png` | 模型综合评估 |
| `training_history.json` | 训练历史数据 |

## 技术亮点

### GPU 加速
自动检测 NVIDIA GPU 并使用 CuPy 加速矩阵运算:
```python
# world_model.py 顶部
try:
    import cupy as cp
    xp = cp  # GPU
except:
    xp = np  # CPU
```

### 置信度估计
通过 MC Dropout 近似贝叶斯推断:
```python
# 多次前向传播（开启 Dropout）
pred_state, pred_reward, confidence, state_std, reward_std = \
    model.predict_with_confidence(state, action, n_samples=10)
```

### 多进程数据收集
使用 Python multiprocessing 并行收集多张地图的数据:
```python
with Pool(processes=min(num_maps, cpu_count())) as pool:
    results = pool.imap(collect_data_for_map, worker_args)
```

## 许可证

MIT License