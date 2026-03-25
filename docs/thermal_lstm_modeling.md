# 宇树G1全关节热动力学LSTM建模规格书

> **基准文档**: `plan.md` V2.0  
> **建模策略**: 按电机类型分组（GearboxS / GearboxM / GearboxL）独立LSTM  
> **预测目标**: 双通道联合预测（主任务：线圈温度，辅助任务：外壳温度）  
> **邻接策略**: 数据驱动的热耦合发现  
> **全局特征**: 作为消融实验项，初始版本不纳入  

---

## 1. 问题定义

### 1.1 数学形式化

给定过去 $L$ 个时间步的特征序列 $\mathbf{X} = \{x_{t-L+1}, \ldots, x_t\}$，预测目标关节未来 $H$ 个时间节点的温度轨迹：

$$f_\theta(\mathbf{X}) \rightarrow (\hat{\mathbf{y}}^{coil},\ \hat{\mathbf{y}}^{shell}) \in \mathbb{R}^{H} \times \mathbb{R}^{H}$$

其中：
- $\hat{\mathbf{y}}^{coil} = [\hat{T}^{coil}_{t+h_1}, \ldots, \hat{T}^{coil}_{t+h_H}]$，主预测目标（线圈/绕组温度）
- $\hat{\mathbf{y}}^{shell} = [\hat{T}^{shell}_{t+h_1}, \ldots, \hat{T}^{shell}_{t+h_H}]$，辅助预测目标（外壳/减速器温度）
- $h_k \in \{10, 20, 40, 60, 100, 140, 200, 240, 300, 400\}$ 步（对应 @ 20Hz 的 $\{0.5, 1, 2, 3, 5, 7, 10, 12, 15, 20\}$ 秒）

### 1.2 约束条件

| 指标 | 要求 |
|:-----|:-----|
| 线圈温度 MAE（全工况） | $\le 1.5°C$ |
| 线圈温度 MAE（高负载 Phase IV） | $\le 2.0°C$ |
| 最远视距 (20s) MAE | $\le 2.5°C$ |
| 单步最大绝对误差 | $\le 5.0°C$ |
| 前向推理延迟（含预处理） | $\le 5\text{ms}$（FP16, Jetson Orin） |

### 1.3 因果性约束

模型必须严格因果——仅使用时间 $\le t$ 的观测数据预测 $t+h$ 处的温度。**禁用双向LSTM (BiLSTM)**，禁用未来帧泄漏。

---

## 2. 物理热力学建模基础

LSTM 是数据驱动模型，但特征工程须以物理热力学为先验指导。本节建立简化的集总参数热模型 (Lumped Parameter Thermal Model)，用于指导特征选取和解释模型行为。

### 2.1 单关节集总热方程

将单个电机关节建模为双节点热网络（线圈节点 + 外壳节点）：

$$C_{coil} \frac{dT_{coil}}{dt} = \underbrace{I^2 R}_{\text{焦耳热 } Q_{joule}} + \underbrace{\mu \cdot |\omega|}_{\text{摩擦热 } Q_{friction}} - \underbrace{G_{c \to s}(T_{coil} - T_{shell})}_{\text{线圈→外壳传导}}$$

$$C_{shell} \frac{dT_{shell}}{dt} = G_{c \to s}(T_{coil} - T_{shell}) - \underbrace{G_{s \to amb}(T_{shell} - T_{amb})}_{\text{外壳→环境对流}} - \underbrace{G_{adj} \sum_{j \in \mathcal{N}(i)} (T_{shell}^{(i)} - T_{shell}^{(j)})}_{\text{相邻关节传导}}$$

其中：
- $C_{coil}, C_{shell}$：线圈和外壳的热容量 $[\text{J/°C}]$
- $G_{c \to s}$：线圈到外壳的热导 $[\text{W/°C}]$
- $G_{s \to amb}$：外壳到环境的对流热导，受风扇状态影响
- $G_{adj}$：相邻关节间结构传导热导
- $\mathcal{N}(i)$：关节 $i$ 的热耦合邻居集合（**由数据驱动发现，见第 4 节**）

### 2.2 物理先验 → 特征设计的映射

| 热方程项 | 对应 SDK 字段 | 特征变换 |
|:---------|:-------------|:---------|
| $I^2 R \propto \tau^2$ | `tau_est` | $\tau_{sq} = \tau_{est}^2$ |
| $\mu \cdot \|\omega\|$ | `dq` | $dq_{abs} = \|dq\|$ |
| $\frac{dQ}{dt}$ 动态突变 | `ddq` | $ddq_{abs} = \|ddq\|$ |
| $T_{coil}$ | `temperature[0]` | EMA 平滑 |
| $T_{shell}$ | `temperature[1]` | EMA 平滑 |
| 电功率 $P = V \cdot I$ | `vol` | 直接使用或 $P_{est} = V \cdot \|\tau \cdot dq\| / \eta$ |
| $T_{amb}$ | `IMUState_.temperature` 或 `MainBoardState_.temperature` | 取可用源 |
| 风扇强制对流 | `MainBoardState_.fan_state` | 归一化均值 |
| 邻居温度 | 邻接关节 `temperature[0]` | EMA 平滑 |

### 2.3 各电机类型的热力学差异

三类电机 (GearboxS/M/L) 的热容量、散热面积、减速比差异显著，导致其温升/冷却动态特性不同，这是按电机类型分组建模的物理依据。

| 电机类型 | 热容量 $C$ | 散热面积 | 温升速率特征 | 冷却时间常数 $\tau_{cool}$ |
|:---------|:----------|:---------|:------------|:--------------------------|
| **GearboxL** (膝关节) | 大 | 大 | 升温慢，但极限温度高 | 长（散热慢） |
| **GearboxM** (髋/腰偏航) | 中 | 中 | 动静态交替，脉冲式温升 | 中 |
| **GearboxS** (肩/肘/腕/踝) | 小 | 小 | 升温快，温度响应灵敏 | 短（但散热面积限制极限） |

---

## 3. 特征工程

### 3.1 原始特征提取

对每个关节 $i \in \{0, \ldots, 28\}$，在每个采样时刻 $t$（20Hz）提取以下原始信号：

```python
raw_features_per_joint = {
    "tau_est":        motor_state[i].tau_est,        # float32, 力矩 [Nm]
    "dq":             motor_state[i].dq,              # float32, 角速度 [rad/s]
    "ddq":            motor_state[i].ddq,             # float32, 角加速度 [rad/s²]
    "temperature_0":  motor_state[i].temperature[0],  # uint8, 线圈温度 [°C]
    "temperature_1":  motor_state[i].temperature[1],  # uint8, 外壳温度 [°C]
    "vol":            motor_state[i].vol,              # float32, 电压 [V]
}
```

### 3.2 特征变换

| 特征名 | 变换公式 | 维度 | 说明 |
|:-------|:---------|:----:|:-----|
| $\tau_{sq}$ | $\tau_{est}^2$ | 1 | 焦耳热功率代理 |
| $dq_{abs}$ | $\|dq\|$ | 1 | 摩擦热功率代理 |
| $ddq_{abs}$ | $\|ddq\|$ | 1 | 动态载荷突变检测 |
| $T_{coil}$ | $\text{EMA}(\text{temperature}[0],\ \alpha=0.05)$ | 1 | 线圈温度（平滑后） |
| $T_{shell}$ | $\text{EMA}(\text{temperature}[1],\ \alpha=0.05)$ | 1 | 外壳温度（平滑后） |
| $V_{mot}$ | `vol` 直接使用 | 1 | 电压 |

EMA 平滑公式：

$$S_t = \alpha \cdot Y_t + (1 - \alpha) \cdot S_{t-1}, \quad \alpha = 0.05$$

`uint8` 阶梯跳变的温度数据在 20Hz 采样下大量相邻帧值相同，EMA 将其转化为平滑连续信号，改善梯度流。

### 3.3 数据驱动的热耦合邻接发现

与物理拓扑预定义不同，本项目采用数据驱动方式发现实际的热耦合关系。

#### 3.3.1 发现流程

**Phase 2（数据采集）中的测试方案 C** 提供了专用的热耦合实验数据。基于此数据执行以下分析：

```
输入: 测试方案 C 的时间序列数据（单关节堵转，其余空载）
      热源关节 i 达到 50°C 时的全身温度快照

对每个热源关节 i:
    对每个非热源关节 j ≠ i:
        计算温度增量 ΔT_adj[j] = T_j(t_50) - T_j(t_0)
        若 ΔT_adj[j] ≥ 2°C:
            标记 (i, j) 为热耦合对
```

#### 3.3.2 邻接矩阵构建

定义热耦合权重矩阵 $\mathbf{A} \in \mathbb{R}^{29 \times 29}$：

$$A_{ij} = \begin{cases} \frac{\Delta T_{adj}^{(j)}}{\max_k \Delta T_{adj}^{(k)}} & \text{if } \Delta T_{adj}^{(j)} \ge 2°C \\ 0 & \text{otherwise} \end{cases}$$

对每个关节 $i$，选取权重最大的 Top-$K$ 个邻居构成集合 $\mathcal{N}(i)$：

$$\mathcal{N}(i) = \text{TopK}(\{j : A_{ij} > 0\}, K)$$

$K$ 的取值建议：
- 初始设 $K=2$（与 plan.md 4.3.1 一致）
- 若某关节的显著耦合邻居 $> 2$，在消融实验中对比 $K \in \{1, 2, 3\}$

#### 3.3.3 备选方案：互信息分析

若测试方案 C 数据不充分，可用完整采集数据计算滑动窗口互信息：

$$\text{MI}(T_i, T_j) = \sum_{t_i, t_j} p(t_i, t_j) \log \frac{p(t_i, t_j)}{p(t_i) \cdot p(t_j)}$$

取 MI 显著高于 baseline（随机关节对的 95th 分位数）的关节对作为耦合候选。

### 3.4 单关节输入特征向量

对关节 $i$，其时刻 $t$ 的特征向量为：

$$x_t^{(i)} = [\underbrace{\tau_{sq},\ dq_{abs},\ ddq_{abs},\ T_{coil},\ T_{shell},\ V_{mot}}_{\text{本体特征 (6D)}},\ \underbrace{T_{coil}^{(n_1)},\ T_{coil}^{(n_2)}}_{\text{邻居线圈温度 (}K\text{D)}}]$$

- **基础特征维度** $D_{base} = 6$
- **邻居特征维度** $D_{adj} = K$（默认 $K=2$）
- **总输入维度** $D = D_{base} + D_{adj} = 8$

> **注**: 环境特征（$T_{amb}$, $fan_{avg}$, BMS 电流等）不纳入初始模型。Phase 3 消融实验 #3 将评估加入 $T_{amb}$ 和 $fan_{avg}$（$D=10$）的效果；消融实验 #5（新增）将评估加入完整全局特征（$D=17$）的效果。

### 3.5 归一化策略

采用 Z-score 标准化，统计量基于训练集计算：

$$\hat{x} = \frac{x - \mu_{train}}{\sigma_{train} + \epsilon}, \quad \epsilon = 10^{-8}$$

各特征的归一化分组与存储：

```python
normalization_stats = {
    "per_joint": {
        # 每个关节独立统计（力矩/速度分布因关节而异）
        "tau_sq": {"mean": ..., "std": ...},   # shape: [29]
        "dq_abs": {"mean": ..., "std": ...},
        "ddq_abs": {"mean": ..., "std": ...},
        "vol": {"mean": ..., "std": ...},
    },
    "per_motor_type": {
        # 同类电机共享温度统计量（训练集更大）
        "T_coil": {"mean": ..., "std": ...},   # shape: [3] (S/M/L)
        "T_shell": {"mean": ..., "std": ...},
    }
}
```

---

## 4. 模型架构

### 4.1 分组策略

按电机类型将 29 个关节划分为 3 组，每组训练独立的 LSTM 模型：

| 模型 ID | 电机类型 | 关节索引 | 关节数 |
|:--------|:---------|:---------|:------:|
| `Model_S` | GearboxS | 4, 5, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28 | 20 |
| `Model_M` | GearboxM | 0, 1, 2, 6, 7, 8, 12 | 7 |
| `Model_L` | GearboxL | 3, 9 | 2 |

分组建模的优势：
- 同类电机的热动态特性（热容量、散热系数、温升速率）相近，共享参数有物理意义
- 训练数据可跨同类关节聚合，`Model_S` 有 20 个关节的数据池
- 模型参数量和推理部署更可控

### 4.2 网络拓扑（每个模型共用此架构，超参数按类型调整）

```
Input (B, L, D)
      │
      ▼
┌─────────────────────────┐
│ InputProjection          │
│ Linear(D → d_proj)       │
│ LayerNorm(d_proj)        │
│ GELU()                   │
└────────────┬────────────┘
             │  (B, L, d_proj)
             ▼
┌─────────────────────────┐
│ CausalLSTM               │
│ LSTM(input=d_proj,       │
│      hidden=d_hidden,    │
│      num_layers=n_layers,│
│      batch_first=True,   │
│      dropout=p_drop)     │
└────────────┬────────────┘
             │  取最后时间步: (B, d_hidden)
             ▼
┌─────────────────────────┐
│ PredictionHead_Coil      │
│ Linear(d_hidden → d_mid) │
│ GELU()                   │
│ Linear(d_mid → H)        │  → ŷ_coil (B, H)
└─────────────────────────┘
             ┊
┌─────────────────────────┐
│ PredictionHead_Shell     │
│ Linear(d_hidden → d_mid) │
│ GELU()                   │
│ Linear(d_mid → H)        │  → ŷ_shell (B, H)
└─────────────────────────┘
```

双预测头共享同一个 LSTM 编码器，但各自有独立的输出映射层。

### 4.3 超参数配置表

| 超参数 | `Model_S` | `Model_M` | `Model_L` | 说明 |
|:-------|:---------:|:---------:|:---------:|:-----|
| $D$（输入维度） | 8 | 8 | 8 | $6 + K$，$K=2$ |
| $d_{proj}$（投影维度） | 48 | 48 | 32 | L 型关节仅 2 个，降低容量防过拟合 |
| $d_{hidden}$（LSTM 隐层） | 96 | 96 | 64 | 同上 |
| $n_{layers}$（LSTM 层数） | 2 | 2 | 2 | — |
| $p_{drop}$（Dropout） | 0.15 | 0.15 | 0.20 | L 型数据少，增强正则 |
| $d_{mid}$（预测头中间层） | 48 | 48 | 32 | — |
| $H$（预测视距） | 10 | 10 | 10 | — |
| $L$（序列长度） | 100 | 100 | 100 | 5s @ 20Hz |
| 参数量（估算） | ~85K | ~85K | ~40K | — |

### 4.4 PyTorch 模型定义

```python
import torch
import torch.nn as nn

class ThermalLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int = 8,
        proj_dim: int = 48,
        hidden_dim: int = 96,
        num_layers: int = 2,
        dropout: float = 0.15,
        mid_dim: int = 48,
        horizon: int = 10,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head_coil = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, horizon),
        )
        self.head_shell = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, horizon),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, L, D) 输入特征序列
        Returns:
            y_coil:  (B, H) 线圈温度预测
            y_shell: (B, H) 外壳温度预测
        """
        x = self.input_proj(x)                  # (B, L, d_proj)
        lstm_out, _ = self.lstm(x)              # (B, L, d_hidden)
        h_last = lstm_out[:, -1, :]             # (B, d_hidden)
        y_coil = self.head_coil(h_last)         # (B, H)
        y_shell = self.head_shell(h_last)       # (B, H)
        return y_coil, y_shell
```

---

## 5. 损失函数设计

### 5.1 双通道联合损失

$$\mathcal{L} = \mathcal{L}_{coil} + \lambda \cdot \mathcal{L}_{shell}$$

其中 $\lambda = 0.3$ 控制辅助任务权重。

#### 主任务损失（线圈温度）

采用 Huber Loss（Smooth L1），降低传感器尖峰噪声的梯度冲击：

$$\mathcal{L}_{coil} = \frac{1}{B \cdot H} \sum_{b=1}^{B} \sum_{k=1}^{H} w_k \cdot \text{Huber}_\delta (\hat{y}^{coil}_{b,k} - y^{coil}_{b,k})$$

$$\text{Huber}_\delta(e) = \begin{cases} \frac{1}{2} e^2 & |e| \le \delta \\ \delta(|e| - \frac{1}{2}\delta) & |e| > \delta \end{cases}, \quad \delta = 1.0$$

#### 视距权重 $w_k$

远期预测本身更难，为避免模型过度关注近期精度而忽略远期，对不同 Horizon 步施加权重：

$$w_k = 1 + \beta \cdot \frac{k - 1}{H - 1}, \quad \beta = 0.5$$

这使最近步权重 $w_1 = 1.0$，最远步权重 $w_H = 1.5$。

#### 辅助任务损失（外壳温度）

$$\mathcal{L}_{shell} = \frac{1}{B \cdot H} \sum_{b,k} \text{Huber}_\delta (\hat{y}^{shell}_{b,k} - y^{shell}_{b,k})$$

辅助任务不加视距权重，保持简单。

### 5.2 PyTorch 损失实现

```python
class DualChannelThermalLoss(nn.Module):
    def __init__(self, lambda_shell: float = 0.3, delta: float = 1.0, beta: float = 0.5):
        super().__init__()
        self.lambda_shell = lambda_shell
        self.huber = nn.HuberLoss(reduction='none', delta=delta)
        self.beta = beta

    def forward(
        self,
        pred_coil: torch.Tensor,   # (B, H)
        target_coil: torch.Tensor,  # (B, H)
        pred_shell: torch.Tensor,   # (B, H)
        target_shell: torch.Tensor, # (B, H)
    ) -> torch.Tensor:
        H = pred_coil.shape[1]
        weights = 1.0 + self.beta * torch.linspace(0, 1, H, device=pred_coil.device)
        loss_coil = (self.huber(pred_coil, target_coil) * weights).mean()
        loss_shell = self.huber(pred_shell, target_shell).mean()
        return loss_coil + self.lambda_shell * loss_shell
```

---

## 6. 数据流水线

### 6.1 整体数据流

```
原始 DDS 数据 (500Hz)
       │
       ▼
 CRC32 校验 → 丢弃坏帧
       │
       ▼
 motorstate_ 位域检查 → 标记异常帧（保留）
       │
       ▼
 降采样至 20Hz（每 25 帧取 1 帧）
       │
       ▼
 存储为 HDF5:  {date}_{round}_{phase}.h5
       │
       ▼
 ═══════════════════════════════════════
 离线预处理流水线
 ═══════════════════════════════════════
       │
       ▼
 EMA 平滑 temperature[0], temperature[1]
       │
       ▼
 特征变换（τ², |dq|, |ddq|）
       │
       ▼
 热耦合邻接发现（Phase 2 测试方案 C 结果）→ 构建邻接表
       │
       ▼
 拼接邻居温度特征
       │
       ▼
 按 session 划分 Train / Val / Test
       │
       ▼
 计算训练集 Z-score 统计量 (μ, σ)
       │
       ▼
 滑动窗口切片 → PyTorch Dataset
```

### 6.2 HDF5 数据格式

```
{date}_{round}_{phase}.h5
├── metadata/
│   ├── room_temperature    # float, 室温 [°C]
│   ├── sample_rate         # int, 20
│   ├── robot_mode          # str, mode_machine_ 值
│   └── collection_notes    # str, 人工备注
├── timestamps/             # float64, shape (N,), Unix 时间戳
├── joints/
│   ├── tau_est             # float32, shape (N, 29)
│   ├── dq                  # float32, shape (N, 29)
│   ├── ddq                 # float32, shape (N, 29)
│   ├── temperature_0       # uint8,   shape (N, 29)
│   ├── temperature_1       # uint8,   shape (N, 29)
│   ├── vol                 # float32, shape (N, 29)
│   ├── motorstate          # uint32,  shape (N, 29)
│   └── sensor              # float32, shape (N, 29, 4)  [如标定后有价值]
├── imu/
│   ├── rpy                 # float32, shape (N, 3)
│   ├── gyroscope           # float32, shape (N, 3)
│   ├── accelerometer       # float32, shape (N, 3)
│   └── temperature         # int16,   shape (N,)
├── mainboard/              # [如 Topic 可用]
│   ├── fan_state           # uint16,  shape (N, 6)
│   └── temperature         # int16,   shape (N, 6)
└── bms/                    # [如 Topic 可用]
    ├── current             # int32,   shape (N,)
    ├── soc                 # uint8,   shape (N,)
    └── temperature         # int16,   shape (N, 6)
```

### 6.3 Dataset 与 DataLoader

```python
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class ThermalDataset(Dataset):
    """
    按电机类型分组的滑动窗口数据集。
    """
    def __init__(
        self,
        h5_paths: list[str],
        joint_indices: list[int],
        adjacency: dict[int, list[int]],
        seq_len: int = 100,
        horizon_steps: list[int] = [10, 20, 40, 60, 100, 140, 200, 240, 300, 400],
        norm_stats: dict | None = None,
    ):
        self.seq_len = seq_len
        self.horizon_steps = horizon_steps
        self.max_horizon = max(horizon_steps)
        self.joint_indices = joint_indices
        self.adjacency = adjacency
        self.norm_stats = norm_stats

        self.samples = []  # list of (h5_path, joint_idx, start_t)
        for path in h5_paths:
            with h5py.File(path, 'r') as f:
                n_frames = f['timestamps'].shape[0]
                valid_len = n_frames - seq_len - self.max_horizon
                for ji in joint_indices:
                    for t in range(0, valid_len):
                        self.samples.append((path, ji, t))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, ji, t = self.samples[idx]
        with h5py.File(path, 'r') as f:
            sl = slice(t, t + self.seq_len)

            tau = f['joints/tau_est'][sl, ji]
            dq  = f['joints/dq'][sl, ji]
            ddq = f['joints/ddq'][sl, ji]
            t0  = f['joints/temperature_0'][sl, ji].astype(np.float32)
            t1  = f['joints/temperature_1'][sl, ji].astype(np.float32)
            vol = f['joints/vol'][sl, ji]

            features = np.stack([
                tau ** 2,
                np.abs(dq),
                np.abs(ddq),
                t0,  # 应为 EMA 平滑后的值（预处理阶段完成）
                t1,
                vol,
            ], axis=-1)  # (L, 6)

            neighbor_temps = []
            for nj in self.adjacency[ji]:
                nt = f['joints/temperature_0'][sl, nj].astype(np.float32)
                neighbor_temps.append(nt)
            if neighbor_temps:
                neighbor_feats = np.stack(neighbor_temps, axis=-1)  # (L, K)
                features = np.concatenate([features, neighbor_feats], axis=-1)

            targets_coil = np.array([
                f['joints/temperature_0'][t + self.seq_len + h - 1, ji]
                for h in self.horizon_steps
            ], dtype=np.float32)
            targets_shell = np.array([
                f['joints/temperature_1'][t + self.seq_len + h - 1, ji]
                for h in self.horizon_steps
            ], dtype=np.float32)

        features = torch.from_numpy(features).float()
        if self.norm_stats is not None:
            features = (features - self.norm_stats['mean']) / (self.norm_stats['std'] + 1e-8)

        return features, torch.from_numpy(targets_coil), torch.from_numpy(targets_shell)
```

### 6.4 数据集划分策略

按采集 session（单轮采集）为单位划分，**不打破时间序列连续性**：

| 划分 | 占比 | 约束 |
|:-----|:----:|:-----|
| Train | 70% | 包含 Phase I~V 各阶段的多个 session |
| Val | 15% | 必须包含至少 1 个完整 Phase IV（开门循环）session |
| Test | 15% | 完全独立 session，训练过程不可见 |

---

## 7. 训练协议

### 7.1 优化器与调度器

| 配置项 | 值 |
|:-------|:---|
| 优化器 | AdamW ($\beta_1=0.9, \beta_2=0.999$, weight_decay = $10^{-4}$) |
| 初始学习率 | $1 \times 10^{-3}$ |
| 调度器 | CosineAnnealingWarmRestarts ($T_0=20, T_{mult}=2$) |
| 梯度裁剪 | `max_norm = 1.0` |
| Batch Size | 128 |
| 最大 Epochs | 200 |
| Early Stopping | patience = 15（监控 Val $\mathcal{L}_{coil}$） |

### 7.2 训练伪代码

```python
for model_type in ["GearboxS", "GearboxM", "GearboxL"]:
    joint_ids = MOTOR_TYPE_TO_JOINTS[model_type]
    config = HYPERPARAMS[model_type]

    model = ThermalLSTM(**config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    criterion = DualChannelThermalLoss(lambda_shell=0.3)

    train_dataset = ThermalDataset(train_h5s, joint_ids, adjacency, norm_stats=stats)
    val_dataset   = ThermalDataset(val_h5s,   joint_ids, adjacency, norm_stats=stats)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(200):
        # --- Train ---
        model.train()
        for features, y_coil, y_shell in DataLoader(train_dataset, batch_size=128, shuffle=True):
            pred_coil, pred_shell = model(features)
            loss = criterion(pred_coil, y_coil, pred_shell, y_shell)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        # --- Validate ---
        model.eval()
        val_loss = evaluate(model, val_dataset, criterion)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, f"best_{model_type}.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 15:
                break
```

### 7.3 训练监控指标

每个 epoch 记录并可视化：

| 指标 | 用途 |
|:-----|:-----|
| Train / Val $\mathcal{L}_{total}$ | 总损失收敛曲线 |
| Train / Val $\mathcal{L}_{coil}$ | 主任务损失（用于 Early Stopping） |
| Train / Val $\mathcal{L}_{shell}$ | 辅助任务损失 |
| Val MAE (线圈) per horizon step | 各预测视距的精度趋势 |
| Learning Rate | 调度器行为验证 |
| Gradient Norm | 监控梯度健康度 |

---

## 8. 消融实验方案

消融实验在 Phase 3 执行，用于验证特征设计和架构选择的合理性。所有消融基于 `Model_S`（数据量最大、关节最多）进行。

### 8.1 实验矩阵

| 实验 ID | 变量 | 基线 | 消融配置 | 评估指标 |
|:--------|:-----|:-----|:---------|:---------|
| A1 | `ddq` 特征 | 包含 ($D=8$) | 移除 ($D=7$) | Val MAE 变化量 |
| A2 | `temperature[1]` (外壳温度) | 作为输入 + 辅助目标 | 仅从输入移除 ($D=7$) | Val MAE 变化量 |
| A3 | 辅助任务 $\mathcal{L}_{shell}$ | $\lambda=0.3$ | $\lambda=0$（仅预测线圈） | Val MAE 变化量 |
| A4 | 环境特征 | 无 ($D=8$) | 加入 $T_{amb}$, $fan_{avg}$ ($D=10$) | Val MAE 变化量 |
| A5 | 全局特征 | 无 ($D=8$) | 加入全部全局特征 ($D=15$) | Val MAE 变化量 |
| A6 | 序列长度 $L$ | 100 (5s) | $L \in \{50, 200\}$ | Val MAE + 推理延迟 |
| A7 | LSTM 隐层维度 | 96 | $d_{hidden} \in \{64, 128\}$ | Val MAE + 参数量 |
| A8 | LSTM 层数 | 2 | $n_{layers} \in \{1, 3\}$ | Val MAE + 推理延迟 |
| A9 | 邻居数量 $K$ | 2 | $K \in \{0, 1, 3\}$ | Val MAE 变化量 |
| A10 | $\lambda_{shell}$ 权重 | 0.3 | $\lambda \in \{0.1, 0.5, 1.0\}$ | 主/辅任务 MAE 平衡 |

### 8.2 消融实验决策规则

- 若移除某特征后 MAE 变化 < 0.05°C → 该特征可移除（简化模型）
- 若加入某特征后 MAE 下降 > 0.1°C → 该特征应纳入最终模型
- 若增大 $L$ 或 $d_{hidden}$ 带来的 MAE 收益 < 0.1°C，但推理延迟增加 > 1ms → 保持较小配置
- 最终模型选择：在验证集 MAE 最优且满足延迟约束的消融组合

---

## 9. 离线评估

### 9.1 评估指标体系

在**测试集**（Phase 3 中从未参与训练的独立 session）上计算：

| 指标 | 定义 | 维度 | 通过标准 |
|:-----|:-----|:-----|:--------:|
| MAE@h | $\frac{1}{N}\sum_i \|y_{i,h} - \hat{y}_{i,h}\|$ | per horizon step | — |
| MAE (全工况) | 所有关节、所有 Horizon 步的均值 | scalar | $\le 1.5°C$ |
| MAE (高负载) | 仅 Phase IV session | scalar | $\le 2.0°C$ |
| MAE (冷却) | 仅 Phase V session | scalar | $\le 1.5°C$ |
| MaxAE | 全测试集的单步最大绝对误差 | scalar | $\le 5.0°C$ |
| RMSE | $\sqrt{\frac{1}{N}\sum_i (y_i - \hat{y}_i)^2}$ | scalar | 参考 |
| $R^2$ | $1 - \frac{\sum(y - \hat{y})^2}{\sum(y - \bar{y})^2}$ | scalar | $\ge 0.95$ |

### 9.2 分关节误差热力图

生成 $29 \times H$ 的误差矩阵，行为关节索引，列为预测视距步，颜色编码 MAE：

```
              h=0.5s  h=1s   h=2s   h=3s   h=5s   h=7s   h=10s  h=12s  h=15s  h=20s
Joint 0  (LHP) [                                                                      ]
Joint 1  (LHR) [                                                                      ]
...
Joint 28 (RWY) [                                                                      ]
```

分析维度：
- 按电机类型分组的均值误差是否存在系统性差异
- 哪些关节在哪些视距步的误差异常偏高
- 误差是否在特定工况转换点集中

### 9.3 失败案例分析协议

1. 提取全测试集 Top-50 最大误差样本
2. 对每个样本记录：关节 ID、所处 Phase、时间戳、误差方向（高估/低估）
3. 统计是否集中于：
   - 特定关节 → 可能需要该关节的专用微调
   - 特定 Phase → 可能需要补充该工况数据
   - 动作转换点 → 可能需要增大 $L$ 或加入 $ddq$ 的高阶特征
   - 与 `motorstate_` 异常标志相关 → 异常帧的标签质量问题

---

## 10. 部署与在线推理

### 10.1 模型导出链路

```
PyTorch (.pt)
     │
     ▼
ONNX (opset 17) ── 验证: ONNX Runtime 输出与 PyTorch 误差 < 1e-4
     │
     ▼
TensorRT FP16 (.engine) ── 验证: 输出误差 < 0.1°C
```

```python
dummy = torch.randn(1, 100, 8)
torch.onnx.export(
    model, dummy,
    f"thermal_{motor_type}.onnx",
    input_names=["state_seq"],
    output_names=["temp_coil", "temp_shell"],
    dynamic_axes={"state_seq": {0: "batch"}},
    opset_version=17,
)
```

### 10.2 在线推理流水线

```
 DDS Subscribe                Ring Buffer              Inference              Control
 ─────────────               ───────────              ─────────              ───────
 rt/lowstate ──┐
               │    CRC32    ┌─────────────┐  每 50ms  ┌──────────┐         ┌────────┐
               ├──►  Check ──►  EMA + Norm  ├──trigger──► TensorRT ├──────►  │  CBF / │
               │    Filter   │  L × D Ring  │          │  FP16    │  ŷ_coil │  RL    │
               │             └─────────────┘          └──────────┘         └────────┘
 rt/mainboard* ┘                   ▲
 rt/bmsstate*  ┘               20Hz push
```

**延迟预算分解**:

| 阶段 | 预算 |
|:-----|:----:|
| 数据提取 + CRC 校验 | ≤ 0.5ms |
| EMA 平滑 + Z-score 归一化 | ≤ 0.5ms |
| Ring Buffer → Tensor 拷贝 | ≤ 0.5ms |
| TensorRT FP16 前向传播 | ≤ 2.0ms |
| 结果后处理（反归一化） | ≤ 0.5ms |
| **总计** | **≤ 4.0ms** (余量 1ms) |

### 10.3 热保护闭环逻辑

```python
T_SOFT = 50.0   # °C — 软约束阈值
T_HARD = 60.0   # °C — 硬约束阈值

def thermal_protection(pred_coil: np.ndarray, joint_id: int) -> str:
    """
    pred_coil: shape (H,), 未来 H 步的线圈温度预测
    """
    t_max = pred_coil.max()
    if t_max >= T_HARD:
        return "HARD_LIMIT"     # 强制切换低功率姿态
    elif t_max >= T_SOFT:
        return "SOFT_LIMIT"     # 降低该关节力矩限幅
    return "NORMAL"
```

力矩限幅衰减策略（线性插值）：

$$\tau_{max}^{eff}(i) = \tau_{max}^{rated}(i) \cdot \text{clip}\left(\frac{T_{hard} - \hat{T}_{max}^{(i)}}{T_{hard} - T_{soft}},\ 0,\ 1\right)$$

当 $\hat{T}_{max} \le T_{soft}$ 时满力矩；当 $\hat{T}_{max} \ge T_{hard}$ 时力矩降为 0。

---

## 11. 项目目录结构

```
ThermalManagement/
├── data/
│   ├── raw/                 # 原始 HDF5 采集文件
│   ├── interim/             # EMA 平滑 + 特征变换后的中间文件
│   └── processed/           # 滑动窗口切片 + 归一化后的训练数据
├── src/
│   ├── data/
│   │   ├── collector.py     # DDS 数据采集脚本
│   │   ├── preprocessing.py # EMA、特征变换、CRC 过滤
│   │   ├── adjacency.py     # 数据驱动邻接发现
│   │   └── dataset.py       # ThermalDataset 定义
│   ├── models/
│   │   ├── thermal_lstm.py  # ThermalLSTM 模型定义
│   │   └── loss.py          # DualChannelThermalLoss
│   ├── training/
│   │   ├── train.py         # 训练主脚本
│   │   ├── evaluate.py      # 离线评估与可视化
│   │   └── ablation.py      # 消融实验运行器
│   └── deployment/
│       ├── export_onnx.py   # ONNX 导出
│       ├── inference.py     # 在线推理 Pipeline
│       └── protection.py    # 热保护闭环逻辑
├── configs/
│   ├── model_s.yaml         # GearboxS 超参数
│   ├── model_m.yaml         # GearboxM 超参数
│   └── model_l.yaml         # GearboxL 超参数
├── docs/
│   └── thermal_lstm_modeling.md  # 本文档
├── plan.md
└── pyproject.toml
```

---

## 附录 A: 关节-电机类型映射表

| 电机类型 | 关节索引 | 关节名称 |
|:---------|:---------|:---------|
| **GearboxL** | 3 | LeftKnee |
| | 9 | RightKnee |
| **GearboxM** | 0 | LeftHipPitch |
| | 1 | LeftHipRoll |
| | 2 | LeftHipYaw |
| | 6 | RightHipPitch |
| | 7 | RightHipRoll |
| | 8 | RightHipYaw |
| | 12 | WaistYaw |
| **GearboxS** | 4, 5 | LeftAnklePitch, LeftAnkleRoll |
| | 10, 11 | RightAnklePitch, RightAnkleRoll |
| | 13, 14 | WaistRoll, WaistPitch |
| | 15~21 | LeftShoulderPitch ~ LeftWristYaw |
| | 22~28 | RightShoulderPitch ~ RightWristYaw |

## 附录 B: 预测视距时间步映射

| Horizon 索引 $k$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|:-----------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:--:|
| 未来时间 (秒) | 0.5 | 1.0 | 2.0 | 3.0 | 5.0 | 7.0 | 10.0 | 12.0 | 15.0 | 20.0 |
| 未来步数 (@ 20Hz) | 10 | 20 | 40 | 60 | 100 | 140 | 200 | 240 | 300 | 400 |

## 附录 C: EMA 平滑参数选择依据

在 20Hz 采样率下，$\alpha = 0.05$ 对应的等效截止频率为：

$$f_c = \frac{\alpha \cdot f_s}{2\pi(1 - \alpha)} \approx \frac{0.05 \times 20}{2\pi \times 0.95} \approx 0.167 \text{ Hz}$$

即 EMA 的 -3dB 点约在 0.167Hz（周期 ~6 秒），这与电机热时间常数（数十秒至分钟级）匹配，可有效滤除 `uint8` 量化跳变噪声而不损失热动态信息。
