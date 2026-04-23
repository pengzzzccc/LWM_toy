import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from enum import IntEnum
from typing import List, Tuple, Optional, Set
import random


class Terrain(IntEnum):
    """地形类型枚举"""
    EMPTY = 0
    WALL = 1
    WATER = 2
    MUD = 3
    GOAL = 4
    START = 5


class GridWorld:
    """
    增强的网格世界环境。
    
    特性:
      - 程序化地图生成（多种布局类型）
      - 多种地形类型（墙壁、水、泥地）
      - 可配置网格大小
      - 热力图可视化
      - 改进的奖励系统
      - 路径规划困难度分析
    """

    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    ACTION_NAMES = {0: "↑", 1: "↓", 2: "←", 3: "→"}
    NUM_ACTIONS = 4

    def __init__(self, size: int = 10, layout_type: str = "random", 
                 walls: Optional[Set] = None, goal: Optional[Tuple] = None,
                 num_water: int = 3, num_mud: int = 5, num_walls: int = 8):
        """
        初始化网格世界。
        
        Args:
            size: 网格大小 (NxN)
            layout_type: 布局类型 ("random", "maze", "rooms", "corridor", "spiral")
            walls: 自定义墙壁位置（如果提供则忽略layout_type）
            goal: 目标位置
            num_water: 水地形数量
            num_mud: 泥地数量
            num_walls: 墙壁数量
        """
        self.size = size
        self.num_states = size * size
        self.layout_type = layout_type
        
        # 地形网格
        self.terrain = np.zeros((size, size), dtype=np.int8)
        
        # 生成地图
        if walls is not None:
            self.walls = set(walls)
            for r, c in walls:
                self.terrain[r][c] = Terrain.WALL
        else:
            self.walls = self._generate_layout(layout_type, num_walls, num_water, num_mud)
        
        self.goal = goal or (size - 1, size - 1)
        self.terrain[self.goal[0]][self.goal[1]] = Terrain.GOAL
        
        self.agent_pos = None
        self.visit_counts = {}
        self.heatmap = np.zeros((size, size), dtype=np.float32)
        self.reset()
        
        # 分析路径规划困难度
        self.difficulty_metrics = self._analyze_difficulty()
    
    def _generate_layout(self, layout_type: str, num_walls: int, 
                         num_water: int, num_mud: int) -> Set[Tuple]:
        """程序化生成地图布局"""
        walls = set()
        
        if layout_type == "random":
            walls = self._generate_random_layout(num_walls, num_water, num_mud)
        elif layout_type == "maze":
            walls = self._generate_maze_layout()
        elif layout_type == "rooms":
            walls = self._generate_rooms_layout()
        elif layout_type == "corridor":
            walls = self._generate_corridor_layout()
        elif layout_type == "spiral":
            walls = self._generate_spiral_layout()
        else:
            walls = self._generate_random_layout(num_walls, num_water, num_mud)
        
        # 确保起点和终点可达
        self._ensure_path_exists(walls)
        return walls
    
    def _generate_random_layout(self, num_walls: int, num_water: int, 
                                 num_mud: int) -> Set[Tuple]:
        """生成随机布局"""
        walls = set()
        positions = [(r, c) for r in range(self.size) for c in range(self.size)]
        positions.remove((0, 0))
        positions.remove(self.goal)
        
        # 随机放置墙壁
        np.random.shuffle(positions)
        wall_positions = positions[:num_walls]
        for r, c in wall_positions:
            walls.add((r, c))
            self.terrain[r][c] = Terrain.WALL
        
        # 放置水地形（可通行但惩罚高）
        water_positions = positions[num_walls:num_walls + num_water]
        for r, c in water_positions:
            if (r, c) not in walls:
                self.terrain[r][c] = Terrain.WATER
        
        # 放置泥地（减速）
        mud_positions = positions[num_walls + num_water:num_walls + num_water + num_mud]
        for r, c in mud_positions:
            if (r, c) not in walls and self.terrain[r][c] == Terrain.EMPTY:
                self.terrain[r][c] = Terrain.MUD
        
        return walls
    
    def _generate_maze_layout(self) -> Set[Tuple]:
        """生成迷宫布局（使用递归回溯算法）"""
        walls = set()
        # 初始化全部为墙
        for r in range(self.size):
            for c in range(self.size):
                if (r, c) != (0, 0) and (r, c) != self.goal:
                    walls.add((r, c))
                    self.terrain[r][c] = Terrain.WALL
        
        # 递归回溯生成迷宫
        visited = set()
        stack = [(0, 0)]
        visited.add((0, 0))
        
        while stack:
            current = stack[-1]
            r, c = current
            
            # 获取未访问的邻居（跳过2格）
            neighbors = []
            for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size and (nr, nc) not in visited:
                    neighbors.append((nr, nc, r + dr//2, c + dc//2))
            
            if neighbors:
                # 随机选择一个邻居
                nr, nc, wr, wc = neighbors[np.random.randint(len(neighbors))]
                visited.add((nr, nc))
                # 移除中间的墙
                walls.discard((wr, wc))
                walls.discard((nr, nc))
                self.terrain[wr][wc] = Terrain.EMPTY
                self.terrain[nr][nc] = Terrain.EMPTY
                stack.append((nr, nc))
            else:
                stack.pop()
        
        # 确保起点和终点可达
        walls.discard((0, 0))
        walls.discard(self.goal)
        self.terrain[0][0] = Terrain.EMPTY
        self.terrain[self.goal[0]][self.goal[1]] = Terrain.GOAL
        
        return walls
    
    def _generate_rooms_layout(self) -> Set[Tuple]:
        """生成房间布局（多个房间通过门连接）"""
        walls = set()
        room_size = max(3, self.size // 3)
        
        # 创建房间边界
        for r in range(self.size):
            for c in range(self.size):
                is_wall = False
                # 水平墙
                if r % room_size == 0 and r != 0:
                    is_wall = True
                # 垂直墙
                if c % room_size == 0 and c != 0:
                    is_wall = True
                
                if is_wall:
                    walls.add((r, c))
                    self.terrain[r][c] = Terrain.WALL
        
        # 创建门（随机移除一些墙）
        doors_to_create = max(4, self.size // 2)
        wall_list = list(walls)
        np.random.shuffle(wall_list)
        for i in range(min(doors_to_create, len(wall_list))):
            r, c = wall_list[i]
            # 确保门不在边缘
            if 0 < r < self.size - 1 and 0 < c < self.size - 1:
                walls.discard((r, c))
                self.terrain[r][c] = Terrain.EMPTY
        
        # 添加一些水和泥地
        empty_positions = [(r, c) for r in range(self.size) for c in range(self.size)
                          if (r, c) not in walls and self.terrain[r][c] == Terrain.EMPTY
                          and (r, c) != (0, 0) and (r, c) != self.goal]
        
        if empty_positions:
            # 水
            np.random.shuffle(empty_positions)
            for r, c in empty_positions[:self.size // 3]:
                self.terrain[r][c] = Terrain.WATER
            
            # 泥地
            for r, c in empty_positions[self.size // 3:self.size // 2]:
                self.terrain[r][c] = Terrain.MUD
        
        return walls
    
    def _generate_corridor_layout(self) -> Set[Tuple]:
        """生成走廊布局"""
        walls = set()
        corridor_width = max(2, self.size // 4)
        
        # 创建垂直走廊
        for r in range(self.size):
            for c in range(self.size):
                if c % (corridor_width + 1) == 0:
                    walls.add((r, c))
                    self.terrain[r][c] = Terrain.WALL
        
        # 创建水平走廊（每隔几行）
        horizontal_interval = max(3, self.size // 3)
        for r in range(0, self.size, horizontal_interval):
            for c in range(self.size):
                if self.terrain[r][c] != Terrain.WALL:
                    walls.add((r, c))
                    self.terrain[r][c] = Terrain.WALL
        
        # 创建门
        for r in range(0, self.size, horizontal_interval):
            door_pos = np.random.randint(1, self.size - 1)
            walls.discard((r, door_pos))
            self.terrain[r][door_pos] = Terrain.EMPTY
        
        return walls
    
    def _generate_spiral_layout(self) -> Set[Tuple]:
        """生成螺旋布局"""
        walls = set()
        
        # 创建螺旋墙
        r, c = 0, 0
        direction = 0  # 0=右, 1=下, 2=左, 3=上
        steps = self.size
        step_count = 0
        direction_changes = 0
        
        while steps > 0:
            # 移动
            dr, dc = [(0, 1), (1, 0), (0, -1), (-1, 0)][direction]
            for _ in range(steps):
                r += dr
                c += dc
                if 0 <= r < self.size and 0 <= c < self.size:
                    # 不在中心区域放置墙
                    center_r, center_c = self.size // 2, self.size // 2
                    if abs(r - center_r) > 1 or abs(c - center_c) > 1:
                        walls.add((r, c))
                        self.terrain[r][c] = Terrain.WALL
            
            direction = (direction + 1) % 4
            direction_changes += 1
            if direction_changes % 2 == 0:
                steps -= 2
        
        # 清理起点和终点
        walls.discard((0, 0))
        walls.discard(self.goal)
        self.terrain[0][0] = Terrain.EMPTY
        self.terrain[self.goal[0]][self.goal[1]] = Terrain.GOAL
        
        return walls
    
    def _ensure_path_exists(self, walls: Set[Tuple]):
        """确保从起点到终点存在路径（BFS检查）"""
        from collections import deque
        
        start = (0, 0)
        goal = self.goal
        
        if start in walls or goal in walls:
            walls.discard(start)
            walls.discard(goal)
            self.terrain[start[0]][start[1]] = Terrain.EMPTY
            self.terrain[goal[0]][goal[1]] = Terrain.GOAL
        
        # BFS检查路径
        visited = set()
        queue = deque([start])
        visited.add(start)
        
        while queue:
            r, c = queue.popleft()
            if (r, c) == goal:
                return True
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.size and 0 <= nc < self.size 
                    and (nr, nc) not in visited and (nr, nc) not in walls):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        
        # 如果没有路径，清除一些墙
        if goal not in visited:
            # 清除从最近访问点到目标的路径
            for r in range(self.size):
                for c in range(self.size):
                    if (r, c) not in visited and (r, c) in walls:
                        # 检查是否连接了已访问区域
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if (nr, nc) in visited:
                                walls.discard((r, c))
                                self.terrain[r][c] = Terrain.EMPTY
                                visited.add((r, c))
                                break
        
        return True
    
    def _analyze_difficulty(self) -> dict:
        """分析地图的路径规划困难度"""
        from collections import deque
        
        # BFS找最短路径
        start = (0, 0)
        goal = self.goal
        
        visited = {start: 0}
        queue = deque([start])
        parent = {start: None}
        
        while queue:
            r, c = queue.popleft()
            dist = visited[(r, c)]
            
            if (r, c) == goal:
                break
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.size and 0 <= nc < self.size 
                    and (nr, nc) not in visited and (nr, nc) not in self.walls):
                    visited[(nr, nc)] = dist + 1
                    parent[(nr, nc)] = (r, c)
                    queue.append((nr, nc))
        
        # 计算指标
        shortest_path = visited.get(goal, float('inf'))
        
        # 计算墙密度
        wall_density = len(self.walls) / (self.size * self.size)
        
        # 计算连通性（从起点能到达的格子数）
        connectivity = len(visited) / (self.size * self.size - len(self.walls))
        
        # 计算平均路径宽度
        path_cells = set()
        current = goal
        while current and current in parent:
            path_cells.add(current)
            current = parent[current]
        path_cells.add(start)
        
        # 路径周围的墙数量
        path_walls = 0
        for r, c in path_cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in self.walls:
                    path_walls += 1
        
        avg_path_width = len(path_cells) / max(1, path_walls)
        
        return {
            'shortest_path': shortest_path,
            'wall_density': wall_density,
            'connectivity': connectivity,
            'avg_path_width': avg_path_width,
            'difficulty_score': (shortest_path / self.size + wall_density * 2 + 
                               (1 - connectivity)) / 3,
            'path_cells': path_cells,
        }
    
    def reset(self):
        """重置环境"""
        self.visit_counts = {}
        self.heatmap = np.zeros((self.size, self.size), dtype=np.float32)
        
        while True:
            r, c = np.random.randint(0, self.size, size=2)
            pos = (int(r), int(c))
            if pos not in self.walls and pos != self.goal:
                self.agent_pos = pos
                self.visit_counts[pos] = 1
                self.heatmap[r][c] += 1
                return self._get_state()
    
    def _get_state(self):
        return np.array(self.agent_pos, dtype=np.float32)
    
    def step(self, action):
        """执行动作"""
        dr, dc = self.ACTIONS[action]
        new_r = self.agent_pos[0] + dr
        new_c = self.agent_pos[1] + dc
        
        hit_wall = False
        if (0 <= new_r < self.size and 0 <= new_c < self.size
                and (new_r, new_c) not in self.walls):
            self.agent_pos = (new_r, new_c)
        else:
            hit_wall = True
        
        # 更新热力图
        self.heatmap[self.agent_pos[0]][self.agent_pos[1]] += 1
        
        # 计算奖励
        done = self.agent_pos == self.goal
        terrain_type = self.terrain[self.agent_pos[0]][self.agent_pos[1]]
        
        if done:
            reward = 20.0  # 到达目标的奖励
        elif hit_wall:
            reward = -2.0
        elif terrain_type == Terrain.WATER:
            reward = -0.5  # 水的惩罚
        elif terrain_type == Terrain.MUD:
            reward = -0.2  # 泥地的轻微惩罚
        else:
            reward = -0.1
        
        # 重复访问惩罚
        visits = self.visit_counts.get(self.agent_pos, 0)
        if visits > 1:
            reward -= 0.3 * min(visits, 5)  # 限制最大惩罚
        self.visit_counts[self.agent_pos] = self.visit_counts.get(self.agent_pos, 0) + 1
        
        # 向目标移动的奖励
        old_dist = abs(self.agent_pos[0] - self.goal[0]) + abs(self.agent_pos[1] - self.goal[1])
        new_dist = abs(new_r - self.goal[0]) + abs(new_c - self.goal[1]) if not hit_wall else old_dist
        if new_dist < old_dist:
            reward += 0.1  # 向目标移动的奖励
        
        return self._get_state(), reward, done
    
    def render(self, trajectory=None):
        """文本渲染"""
        lines = []
        traj_set = {}
        if trajectory:
            for i, (r, c) in enumerate(trajectory):
                if (r, c) not in traj_set:
                    traj_set[(r, c)] = i
        
        lines.append("+" + "---+" * self.size)
        for r in range(self.size):
            row = "|"
            for c in range(self.size):
                if (r, c) == self.goal:
                    cell = " G "
                elif (r, c) in self.walls:
                    cell = "███"
                elif (r, c) == self.agent_pos:
                    cell = " A "
                elif (r, c) in traj_set:
                    cell = f" {traj_set[(r, c)] % 10} "
                elif self.terrain[r][c] == Terrain.WATER:
                    cell = " ≈ "
                elif self.terrain[r][c] == Terrain.MUD:
                    cell = " ≈ "
                else:
                    cell = "   "
                row += cell + "|"
            lines.append(row)
            lines.append("+" + "---+" * self.size)
        return "\n".join(lines)
    
    def render_visual(self, trajectory=None, title="GridWorld", ax=None, 
                      show_heatmap=False, show_difficulty=False):
        """使用 matplotlib 渲染可视化"""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # 构建颜色网格
        grid = np.zeros((self.size, self.size, 3))
        
        # 默认白色
        grid[:, :] = [1.0, 1.0, 1.0]
        
        # 墙壁: 深灰
        for r, c in self.walls:
            grid[r][c] = [0.2, 0.2, 0.2]
        
        # 水: 蓝色
        for r in range(self.size):
            for c in range(self.size):
                if self.terrain[r][c] == Terrain.WATER:
                    grid[r][c] = [0.3, 0.6, 0.9]
                elif self.terrain[r][c] == Terrain.MUD:
                    grid[r][c] = [0.6, 0.4, 0.2]
        
        # 目标: 绿色
        gr, gc = self.goal
        grid[gr][gc] = [0.3, 0.8, 0.3]
        
        # 显示热力图
        if show_heatmap and np.max(self.heatmap) > 0:
            heatmap_normalized = self.heatmap / np.max(self.heatmap)
            for r in range(self.size):
                for c in range(self.size):
                    if self.terrain[r][c] == Terrain.EMPTY:
                        intensity = heatmap_normalized[r][c]
                        grid[r][c] = [1.0, 1.0 - intensity * 0.5, 1.0 - intensity]
        
        ax.imshow(grid, interpolation='nearest')
        
        # 绘制网格线
        for i in range(self.size + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.5)
            ax.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.5)
        
        # 墙壁标记
        for r, c in self.walls:
            ax.text(c, r, '■', ha='center', va='center', fontsize=max(8, 14 - self.size//3), 
                    color='white')
        
        # 目标标记
        ax.text(gc, gr, 'G', ha='center', va='center', 
                fontsize=max(8, 14 - self.size//3), fontweight='bold', color='white')
        
        # 绘制轨迹
        if trajectory and len(trajectory) > 1:
            rows = [p[0] for p in trajectory]
            cols = [p[1] for p in trajectory]
            ax.plot(cols, rows, 'o-', color='#FF5722', linewidth=2.5,
                    markersize=max(4, 8 - self.size//5), markerfacecolor='#FF9800', zorder=5)
            ax.plot(cols[0], rows[0], 's', color='#E91E63', markersize=max(6, 12 - self.size//3), 
                    zorder=6, label='起点')
            ax.plot(cols[-1], rows[-1], 'X', color='#9C27B0', markersize=max(8, 14 - self.size//3), 
                    zorder=6, label='终点')
        
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(self.size - 0.5, -0.5)
        
        # 标题信息
        diff = self.difficulty_metrics
        title_text = title
        if show_difficulty:
            title_text += f"\n最短路径: {diff['shortest_path']} | 墙密度: {diff['wall_density']:.2f}"
            title_text += f" | 困难度: {diff['difficulty_score']:.2f}"
        ax.set_title(title_text, fontsize=max(10, 14 - self.size//4), fontweight='bold')
        ax.set_xlabel('列')
        ax.set_ylabel('行')
        
        if trajectory:
            ax.legend(loc='upper left', fontsize=9)
        
        return ax
    
    def render_heatmap(self, title="移动热力图", ax=None):
        """专门渲染热力图"""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # 创建热力图数据
        heatmap_display = self.heatmap.copy()
        
        # 标记墙和目标
        for r, c in self.walls:
            heatmap_display[r][c] = -1
        heatmap_display[self.goal[0]][self.goal[1]] = -2
        
        # 使用颜色映射
        cmap = plt.cm.YlOrRd.copy()
        cmap.set_under('white')  # 墙
        cmap.set_over('green')   # 目标
        
        im = ax.imshow(heatmap_display, cmap=cmap, interpolation='nearest', 
                       vmin=0, vmax=max(np.max(self.heatmap), 1))
        plt.colorbar(im, ax=ax, label='访问次数')
        
        # 绘制网格线
        for i in range(self.size + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.5)
            ax.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.5)
        
        # 标记墙和目标
        for r, c in self.walls:
            ax.text(c, r, '■', ha='center', va='center', 
                    fontsize=max(8, 14 - self.size//3), color='black')
        ax.text(self.goal[1], self.goal[0], 'G', ha='center', va='center',
                fontsize=max(8, 14 - self.size//3), fontweight='bold', color='white')
        
        ax.set_title(title, fontsize=max(10, 14 - self.size//4), fontweight='bold')
        ax.set_xlabel('列')
        ax.set_ylabel('行')
        
        return ax
    
    def render_difficulty_analysis(self, title="路径规划困难度分析", ax=None):
        """渲染困难度分析图"""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # 创建颜色网格
        grid = np.ones((self.size, self.size, 3))
        
        # 墙壁: 深灰
        for r, c in self.walls:
            grid[r][c] = [0.2, 0.2, 0.2]
        
        # 最短路径: 黄色
        if 'path_cells' in self.difficulty_metrics:
            for r, c in self.difficulty_metrics['path_cells']:
                grid[r][c] = [1.0, 1.0, 0.0]
        
        # 目标: 绿色
        grid[self.goal[0]][self.goal[1]] = [0.3, 0.8, 0.3]
        
        # 起点: 红色
        grid[0][0] = [1.0, 0.3, 0.3]
        
        ax.imshow(grid, interpolation='nearest')
        
        # 绘制网格线
        for i in range(self.size + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.5)
            ax.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.5)
        
        # 标记墙
        for r, c in self.walls:
            ax.text(c, r, '■', ha='center', va='center', 
                    fontsize=max(8, 14 - self.size//3), color='white')
        
        # 添加困难度信息
        diff = self.difficulty_metrics
        info_text = (f"最短路径: {diff['shortest_path']}\n"
                    f"墙密度: {diff['wall_density']:.2%}\n"
                    f"连通性: {diff['connectivity']:.2%}\n"
                    f"困难度: {diff['difficulty_score']:.2f}")
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(title, fontsize=max(10, 14 - self.size//4), fontweight='bold')
        ax.set_xlabel('列')
        ax.set_ylabel('行')
        
        return ax


def collect_transitions(env, num_episodes=200, max_steps=50):
    """收集转移数据"""
    transitions = []
    for _ in range(num_episodes):
        state = env.reset()
        for _ in range(max_steps):
            action = np.random.randint(0, GridWorld.NUM_ACTIONS)
            next_state, reward, done = env.step(action)
            transitions.append((state.copy(), action, next_state.copy(), reward, done))
            state = next_state
            if done:
                break
    return transitions


def collect_transitions_with_strategy(env, num_episodes=200, max_steps=50, 
                                      explore_ratio=0.3):
    """
    使用策略收集转移数据（混合策略）
    
    Args:
        env: 环境
        num_episodes: 回合数
        max_steps: 最大步数
        explore_ratio: 探索策略的比例（其余使用贪心策略）
    """
    transitions = []
    explore_episodes = int(num_episodes * explore_ratio)
    
    for episode in range(num_episodes):
        state = env.reset()
        
        for step in range(max_steps):
            if episode < explore_episodes:
                # 随机探索
                action = np.random.randint(0, GridWorld.NUM_ACTIONS)
            else:
                # 贪心策略（朝目标方向移动）
                r, c = env.agent_pos
                gr, gc = env.goal
                
                # 计算到目标的方向
                dr = gr - r
                dc = gc - c
                
                # 选择主要移动方向
                if abs(dr) > abs(dc):
                    action = 1 if dr > 0 else 0  # 下或上
                elif abs(dc) > 0:
                    action = 3 if dc > 0 else 2  # 右或左
                else:
                    action = np.random.randint(0, GridWorld.NUM_ACTIONS)
                
                # 随机扰动（避免过于确定性）
                if np.random.random() < 0.2:
                    action = np.random.randint(0, GridWorld.NUM_ACTIONS)
            
            next_state, reward, done = env.step(action)
            transitions.append((state.copy(), action, next_state.copy(), reward, done))
            state = next_state
            if done:
                break
    
    return transitions


def generate_random_map(size: int = 10, seed: Optional[int] = None) -> GridWorld:
    """生成随机地图"""
    if seed is not None:
        np.random.seed(seed)
    
    # 随机选择布局类型
    layout_types = ["random", "maze", "rooms", "corridor", "spiral"]
    layout_type = np.random.choice(layout_types)
    
    # 根据大小调整参数
    num_walls = max(5, size * size // 15)
    num_water = max(2, size // 3)
    num_mud = max(3, size // 2)
    
    env = GridWorld(size=size, layout_type=layout_type, 
                    num_walls=num_walls, num_water=num_water, num_mud=num_mud)
    
    return env


def generate_training_maps(num_maps: int = 10, size_range: Tuple[int, int] = (8, 15),
                          seed: Optional[int] = None) -> List[GridWorld]:
    """生成多个训练地图"""
    if seed is not None:
        np.random.seed(seed)
    
    maps = []
    for i in range(num_maps):
        size = np.random.randint(size_range[0], size_range[1] + 1)
        env = generate_random_map(size=size)
        maps.append(env)
    
    return maps


if __name__ == "__main__":
    # 测试地图生成
    print("测试不同的地图布局...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    layouts = ["random", "maze", "rooms", "corridor", "spiral"]
    
    for i, layout in enumerate(layouts):
        env = GridWorld(size=10, layout_type=layout)
        ax = axes[i // 3, i % 3]
        env.render_visual(title=f"{layout} 布局", ax=ax, show_difficulty=True)
    
    # 生成随机地图
    env = generate_random_map(size=12)
    env.render_visual(title="随机生成地图", ax=axes[1, 2], show_difficulty=True)
    
    plt.tight_layout()
    plt.savefig("map_layouts.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n测试热力图...")
    # 运行一些回合生成热力图数据
    for _ in range(100):
        env.reset()
        for _ in range(50):
            action = np.random.randint(0, 4)
            _, _, done = env.step(action)
            if done:
                break
    
    fig, ax = plt.subplots(figsize=(8, 8))
    env.render_heatmap(title="移动热力图 (100回合)", ax=ax)
    plt.tight_layout()
    plt.savefig("heatmap.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n测试困难度分析...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, layout in enumerate(["maze", "rooms", "spiral"]):
        env = GridWorld(size=10, layout_type=layout)
        env.render_difficulty_analysis(title=f"{layout} 困难度分析", ax=axes[i])
    
    plt.tight_layout()
    plt.savefig("difficulty_analysis.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n所有测试完成！")