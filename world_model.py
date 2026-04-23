import numpy as np
import os

# GPU/CPU 自适应后端选择
USE_GPU = False
try:
    import cupy as cp
    cp.cuda.Device(0).compute_capability
    USE_GPU = True
    xp = cp
    BACKEND = "GPU (CuPy/CUDA)"
except Exception:
    xp = np
    BACKEND = "CPU (NumPy/OpenBLAS)"

# 设置 OpenBLAS 多线程
os.environ["OPENBLAS_NUM_THREADS"] = str(os.cpu_count() or 8)
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count() or 8)


def to_numpy(arr):
    """统一转为 numpy 数组"""
    if USE_GPU:
        if isinstance(arr, np.ndarray):
            return arr
        return cp.asnumpy(arr)
    return arr


def to_xp(arr):
    """统一转为当前后端数组"""
    if USE_GPU:
        if isinstance(arr, np.ndarray):
            return cp.asarray(arr)
        return arr
    return arr


def relu(x):
    return xp.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(xp.float32)


class WorldModel:
    """
    世界模型 MLP (GPU/CPU 自适应版)。
    输入: [state_row, state_col, action_onehot(4)] → 共 6 维
    输出: [predicted_next_row, predicted_next_col, predicted_reward] → 共 3 维

    增强:
      - GPU (CuPy) / CPU (NumPy) 自动切换
      - 更大隐藏层 (256, 256, 128)
      - 置信度/不确定性估计 (MC Dropout 近似)
      - 向量化批量训练
      - 预计算输入缓存
      - 余弦退火学习率
    """

    def __init__(self, grid_size=20, hidden_sizes=(256, 256, 128), lr=0.002, lr_min=1e-5):
        self.lr = lr
        self.lr_init = lr
        self.lr_min = lr_min
        self.grid_size = grid_size
        input_dim = 6   # 2 (state) + 4 (action one-hot)
        output_dim = 3   # 2 (next state) + 1 (reward)

        dims = [input_dim] + list(hidden_sizes) + [output_dim]
        self.weights = []
        self.biases = []
        for i in range(len(dims) - 1):
            scale = xp.sqrt(xp.float32(2.0 / dims[i]))
            w = (xp.random.randn(dims[i], dims[i + 1]).astype(xp.float32) * scale)
            b = xp.zeros(dims[i + 1], dtype=xp.float32)
            self.weights.append(w)
            self.biases.append(b)

        self.num_layers = len(self.weights)
        self.architecture = dims
        self.backend = BACKEND

        print(f"[后端] {self.backend}")
        print(f"[模型] 架构: {' → '.join(str(d) for d in dims)}")
        print(f"[模型] 参数量: {self.get_param_count()}")

    def _encode_input(self, state, action):
        action_oh = xp.zeros(4, dtype=xp.float32)
        action_oh[action] = 1.0
        gs = xp.float32(self.grid_size)
        return xp.concatenate([state / gs, action_oh])

    def _encode_input_batch(self, states, actions):
        """向量化批量编码"""
        bs = len(states)
        encoded = xp.zeros((bs, 6), dtype=xp.float32)
        gs = xp.float32(self.grid_size)
        encoded[:, :2] = to_xp(xp.array(states, dtype=xp.float32)) / gs
        acts = to_xp(np.array(actions, dtype=np.int32))
        for i in range(bs):
            encoded[i, 2 + int(acts[i])] = xp.float32(1.0)
        return encoded

    def _forward_batch(self, X):
        """批量前向传播"""
        activations = [X]
        pre_activations = []
        for i in range(self.num_layers):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            pre_activations.append(z)
            if i < self.num_layers - 1:
                activations.append(relu(z))
            else:
                activations.append(z)
        return activations[-1], activations, pre_activations

    def _forward_batch_with_dropout(self, X, dropout_rate=0.1):
        """带 Dropout 的前向传播 (用于置信度估计)"""
        activations = [X]
        pre_activations = []
        for i in range(self.num_layers):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            pre_activations.append(z)
            if i < self.num_layers - 1:
                a = relu(z)
                # Dropout mask
                mask = (xp.random.rand(*a.shape) > dropout_rate).astype(xp.float32)
                a = a * mask / (1.0 - dropout_rate)
                activations.append(a)
            else:
                activations.append(z)
        return activations[-1], activations, pre_activations

    def _backward_batch(self, activations, pre_activations, grad_output):
        """批量反向传播"""
        grad = grad_output
        bs = grad.shape[0]
        for i in range(self.num_layers - 1, -1, -1):
            grad_w = (activations[i].T @ grad) / bs
            grad_b = xp.mean(grad, axis=0)
            xp.clip(grad_w, -1.0, 1.0, out=grad_w)
            xp.clip(grad_b, -1.0, 1.0, out=grad_b)
            self.weights[i] -= self.lr * grad_w
            self.biases[i] -= self.lr * grad_b
            if i > 0:
                grad = (grad @ self.weights[i].T) * relu_grad(pre_activations[i - 1])

    def forward(self, x):
        """单样本前向"""
        activations = [x]
        pre_activations = []
        for i in range(self.num_layers):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            pre_activations.append(z)
            if i < self.num_layers - 1:
                activations.append(relu(z))
            else:
                activations.append(z)
        return activations[-1], activations, pre_activations

    def predict(self, state, action):
        x = self._encode_input(state, action)
        output, _, _ = self.forward(x)
        gs = float(self.grid_size)
        pred_next_state = to_numpy(output[:2]) * gs
        pred_reward = float(to_numpy(output[2]))
        return pred_next_state, pred_reward

    def predict_with_confidence(self, state, action, n_samples=10):
        """带置信度的预测 (MC Dropout 近似)"""
        gs = float(self.grid_size)
        x = self._encode_input(state, action)
        preds_state = []
        preds_reward = []
        for _ in range(n_samples):
            out, _, _ = self._forward_batch_with_dropout(x.reshape(1, -1), dropout_rate=0.1)
            out_np = to_numpy(out[0])
            preds_state.append(out_np[:2] * gs)
            preds_reward.append(float(out_np[2]))
        preds_state = np.array(preds_state)
        preds_reward = np.array(preds_reward)
        mean_state = np.mean(preds_state, axis=0)
        mean_reward = np.mean(preds_reward)
        std_state = np.std(preds_state, axis=0)
        std_reward = np.std(preds_reward)
        confidence = 1.0 / (1.0 + np.mean(std_state) + std_reward)
        return mean_state, mean_reward, confidence, std_state, std_reward

    def train_epoch(self, transitions, batch_size=256):
        """向量化 mini-batch 训练"""
        n = len(transitions)
        indices = np.random.permutation(n)
        total_loss = xp.float32(0.0)
        count = 0
        gs = xp.float32(self.grid_size)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]
            actual_bs = len(batch_idx)

            states = [transitions[idx][0] for idx in batch_idx]
            actions = [transitions[idx][1] for idx in batch_idx]
            next_states = [transitions[idx][2] for idx in batch_idx]
            rewards = [transitions[idx][3] for idx in batch_idx]

            X = self._encode_input_batch(states, actions)
            Y = xp.zeros((actual_bs, 3), dtype=xp.float32)
            Y[:, :2] = to_xp(np.array(next_states, dtype=np.float32)) / gs
            Y[:, 2] = to_xp(np.array(rewards, dtype=np.float32)) / 10.0

            output, activations, pre_activations = self._forward_batch(X)
            diff = output - Y
            total_loss += xp.sum(diff ** 2) * xp.float32(0.5)
            count += actual_bs
            self._backward_batch(activations, pre_activations, diff)

        return float(to_numpy(total_loss)) / max(count, 1)

    def update_learning_rate(self, epoch, total_epochs):
        self.lr = self.lr_min + 0.5 * (self.lr_init - self.lr_min) * (
            1 + np.cos(np.pi * epoch / total_epochs)
        )

    def imagine_trajectory(self, start_state, actions):
        state = start_state.copy()
        gs = float(self.grid_size)
        trajectory = [(state[0], state[1])]
        total_reward = 0.0
        rewards = []
        for action in actions:
            pred_state, pred_reward = self.predict(state, action)
            pred_state = np.clip(np.round(pred_state), 0, gs - 1).astype(np.float32)
            state = pred_state
            total_reward += pred_reward
            rewards.append(pred_reward)
            trajectory.append((state[0], state[1]))
        return trajectory, rewards, total_reward

    def imagine_trajectory_with_confidence(self, start_state, actions):
        """带置信度的轨迹想象"""
        state = start_state.copy()
        gs = float(self.grid_size)
        trajectory = [(state[0], state[1])]
        total_reward = 0.0
        rewards = []
        confidences = []
        state_stds = []
        reward_stds = []
        for action in actions:
            pred_state, pred_reward, conf, s_std, r_std = self.predict_with_confidence(state, action)
            pred_state = np.clip(np.round(pred_state), 0, gs - 1).astype(np.float32)
            state = pred_state
            total_reward += pred_reward
            rewards.append(pred_reward)
            confidences.append(conf)
            state_stds.append(s_std)
            reward_stds.append(r_std)
            trajectory.append((state[0], state[1]))
        return trajectory, rewards, total_reward, confidences, state_stds, reward_stds

    def get_param_count(self):
        return sum(int(to_numpy(w).size) + int(to_numpy(b).size)
                   for w, b in zip(self.weights, self.biases))

    def get_architecture_info(self):
        """返回架构信息用于可视化"""
        return {
            'layers': self.architecture,
            'param_count': self.get_param_count(),
            'backend': self.backend,
            'grid_size': self.grid_size,
        }