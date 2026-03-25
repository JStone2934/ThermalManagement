# 宇树G1人形机器人全关节热动力学LSTM建模工程实施计划书 (V2.0)

## 0. 文档说明与项目基准

* **文档状态**: Release V2.0 — 基于 `unitree_sdk2` (C++/Python) 源码级逆向审计重构
* **核心目标**: 基于宇树 G1 底层 DDS 通信接口（`unitree_sdk2` / `unitree_sdk2_python`），构建完全数据驱动的因果 LSTM 热预测网络。
* **验收标准**: 在 G1 (EDU 29DOF) 挂载 Dex3-1 灵巧手执行"持续抗阻开门"等高负载任务下，对肩/肘/腰核心关节未来 15 秒的温度预测绝对平均误差 (MAE) $\le 1.5^\circ\text{C}$。
* **运行约束**: 预测模型单次前向推理延迟 $\le 5\text{ms}$（Jetson Orin / PC，FP16），满足实时强化学习 (RL) 控制障碍函数 (CBF) 的约束计算需求。
* **V2.0 变更摘要**:
  1. 修正 `temperature` 字段类型：从 V1.0 的 `int8` 修正为实际 IDL 定义的 `uint8_t[2]`（双通道温度）
  2. 修正 DDS Topic 名称：从 `rt/lf/lowstate` 修正为 `rt/lowstate`
  3. 新增 `MainBoardState_` 的风扇状态 + 主板温度、`BmsState_` 电池温度作为环境特征源
  4. 新增 `motor_state[i].vol`（电压）、`ddq`（角加速度）、`sensor[4]` 为辅助特征
  5. 重构特征工程：双通道温度+多源温度融合 → 特征维度提升，LSTM 隐层扩容
  6. 新增数据有效性校验（CRC32 + `motorstate_` 位域故障码解析）

---

## 1. 系统架构与接口规范 (System Architecture & Interfaces)

### 1.1 硬件与通信栈

* **机器人本体**: Unitree G1 EDU (29DOF)，搭配 Dex3-1 灵巧手（手部热管理不在本项目范围内）
* **通信协议**: CycloneDDS 0.10.2，IDL 命名空间 `unitree_hg::msg::dds_`
* **SDK**: `unitree_sdk2` (C++ 实时控制) / `unitree_sdk2_python` (数据采集)
* **主控频率**: G1 底层控制回路 500Hz (`control_dt_ = 0.002s`)，示例代码确认

### 1.2 DDS Topic 清单

> **V1.0 勘误**: 原文档中 `rt/lf/lowstate` 为错误 topic 名称。

| Topic                         | IDL 类型                   | 方向        | 用途                            |
| :---------------------------- | :------------------------- | :---------- | :------------------------------ |
| **`rt/lowstate`**             | `LowState_`                | Subscribe   | 全身 29 关节状态 + IMU          |
| `rt/lowcmd`                   | `LowCmd_`                  | Publish     | 底层电机控制指令                |
| `rt/mainboardstate`*          | `MainBoardState_`          | Subscribe   | 主板温度 + 风扇状态             |
| `rt/bmsstate`*                | `BmsState_`                | Subscribe   | 电池温度 + 电流                 |

> \* `mainboardstate` 与 `bmsstate` Topic 名需实机验证，SDK IDL 已定义但示例代码未直接展示。

### 1.3 采样频率规范

* **底层控制频率**: 500Hz（SDK 示例 `control_dt_ = 0.002s`）
* **数据采集频率 (Downsampling)**: **20Hz**
  * 热力学传导为慢动态过程，过高采样率会导致相邻时间步数据差异趋近于 0，引发 LSTM 梯度消失与内存溢出。
  * 20Hz 下每步间隔 50ms，足以捕获电机 $I^2R$ 发热的瞬态跃变。

---

## 2. 核心数据结构逆向解析 (IDL Struct Reverse Engineering)

> 以下均从 `unitree_sdk2/include/unitree/idl/hg/*.hpp` 源码直接提取。

### 2.1 MotorState_ — 单关节电机状态 (每个电机实例)

```cpp
class MotorState_ {
    uint8_t   mode_;                      // 电机使能状态: 0=Disable, 1=Enable
    float     q_;                         // 关节角度 [rad]
    float     dq_;                        // 关节角速度 [rad/s]
    float     ddq_;                       // 关节角加速度 [rad/s²]   ← V1.0 遗漏
    float     tau_est_;                   // 估计力矩 [Nm]
    std::array<uint8_t, 2> temperature_;  // ★ 双通道温度 [°C]      ← V1.0 错误：非 int8
    float     vol_;                       // 电机供电电压 [V]        ← V1.0 遗漏
    std::array<float, 4>   sensor_;       // 4 路辅助传感器          ← V1.0 遗漏
    uint32_t  motorstate_;                // 电机状态位域/故障码      ← V1.0 遗漏
    std::array<uint32_t, 5> reserve_;     // 保留字段
};
```

**关键发现**:
* `temperature_` 是 `uint8_t[2]`，**不是 V1.0 所述的 `int8`**。两个通道可能分别对应电机线圈温度（绕组 NTC）和外壳/减速器温度，精度 1°C，范围 0~255°C。
* `vol_` 提供电压信息，结合 `tau_est_` 与 `dq_` 可估算电功率 $P = V \cdot I \approx \frac{\tau \cdot \omega}{\eta}$。
* `ddq_` (角加速度) 可用于检测动态载荷突变，是温升速率前馈的有效先验。
* `motorstate_` 位域包含过温/过流/通信超时等故障标志，必须在数据清洗阶段检查。

### 2.2 LowState_ — G1 全身低层状态消息

```cpp
class LowState_ {
    std::array<uint8_t, 2>  version_;
    uint8_t                 mode_pr_;        // 0=PR(Pitch/Roll), 1=AB(A/B)
    uint8_t                 mode_machine_;   // 机型标识
    uint32_t                tick_;           // 时间戳 tick
    IMUState_               imu_state_;
    std::array<MotorState_, 35> motor_state_; // ★ 数组长度 35，G1 使用 [0..28]
    std::array<uint8_t, 40> wireless_remote_;
    std::array<uint32_t, 4> reserve_;
    uint32_t                crc_;            // CRC32 校验
};
```

**注意**: `motor_state_` 数组大小为 **35**（非 29），G1-29DOF 仅使用索引 0~28，其余为预留。数据采集时仅提取有效索引。

### 2.3 IMUState_ — 惯性测量单元

```cpp
class IMUState_ {
    std::array<float, 4> quaternion_;     // 四元数
    std::array<float, 3> gyroscope_;      // 角速度 [rad/s]
    std::array<float, 3> accelerometer_;  // 加速度 [m/s²]
    std::array<float, 3> rpy_;            // 欧拉角 [rad]
    int16_t              temperature_;    // IMU 芯片温度 [°C]
};
```

### 2.4 MainBoardState_ — 主板热环境

```cpp
class MainBoardState_ {
    std::array<uint16_t, 6> fan_state_;    // ★ 6 路风扇状态 (转速/占空比)
    std::array<int16_t, 6>  temperature_;  // ★ 6 路主板温度探头 [°C]
    std::array<float, 6>    value_;        // 扩展数值
    std::array<uint32_t, 6> state_;        // 状态字
};
```

### 2.5 BmsState_ — 电池管理系统

```cpp
class BmsState_ {
    // ...
    int32_t              current_;         // 系统总电流 [mA]
    uint8_t              soc_;             // 剩余电量 [%]
    std::array<int16_t, 6> temperature_;   // ★ 6 路电池温度探头 [°C]
    // ...
};
```

---

## 3. G1 关节索引表与电机类型映射 (Joint Index & Motor Type)

> 源自 `unitree_sdk2/include/unitree/dds_wrapper/robots/g1/defines.h` 及示例代码。

### 3.1 完整关节索引 (29DOF)

| 索引 | 关节名称            | 部位   | 电机类型   | 默认 Kp | 默认 Kd | 热建模优先级 |
| :--: | :------------------ | :----- | :--------- | :-----: | :-----: | :----------: |
|  0   | LeftHipPitch        | 左腿   | GearboxM   |   60    |    1    |     中       |
|  1   | LeftHipRoll         | 左腿   | GearboxM   |   60    |    1    |     中       |
|  2   | LeftHipYaw          | 左腿   | GearboxM   |   60    |    1    |     低       |
|  3   | LeftKnee            | 左腿   | GearboxL   |  100    |    2    |    **高**    |
|  4   | LeftAnklePitch      | 左腿   | GearboxS   |   40    |    1    |     低       |
|  5   | LeftAnkleRoll       | 左腿   | GearboxS   |   40    |    1    |     低       |
|  6   | RightHipPitch       | 右腿   | GearboxM   |   60    |    1    |     中       |
|  7   | RightHipRoll        | 右腿   | GearboxM   |   60    |    1    |     中       |
|  8   | RightHipYaw         | 右腿   | GearboxM   |   60    |    1    |     低       |
|  9   | RightKnee           | 右腿   | GearboxL   |  100    |    2    |    **高**    |
| 10   | RightAnklePitch     | 右腿   | GearboxS   |   40    |    1    |     低       |
| 11   | RightAnkleRoll      | 右腿   | GearboxS   |   40    |    1    |     低       |
| 12   | WaistYaw            | 腰部   | GearboxM   |   60    |    1    |    **高**    |
| 13   | WaistRoll           | 腰部   | GearboxS   |   40    |    1    |     中†      |
| 14   | WaistPitch          | 腰部   | GearboxS   |   40    |    1    |     中†      |
| 15   | LeftShoulderPitch   | 左臂   | GearboxS   |   40    |    1    |    **高**    |
| 16   | LeftShoulderRoll    | 左臂   | GearboxS   |   40    |    1    |    **高**    |
| 17   | LeftShoulderYaw     | 左臂   | GearboxS   |   40    |    1    |     中       |
| 18   | LeftElbow           | 左臂   | GearboxS   |   40    |    1    |    **高**    |
| 19   | LeftWristRoll       | 左臂   | GearboxS   |   40    |    1    |     中       |
| 20   | LeftWristPitch      | 左臂   | GearboxS   |   40    |    1    |     低‡      |
| 21   | LeftWristYaw        | 左臂   | GearboxS   |   40    |    1    |     低‡      |
| 22   | RightShoulderPitch  | 右臂   | GearboxS   |   40    |    1    |    **高**    |
| 23   | RightShoulderRoll   | 右臂   | GearboxS   |   40    |    1    |    **高**    |
| 24   | RightShoulderYaw    | 右臂   | GearboxS   |   40    |    1    |     中       |
| 25   | RightElbow          | 右臂   | GearboxS   |   40    |    1    |    **高**    |
| 26   | RightWristRoll      | 右臂   | GearboxS   |   40    |    1    |     中       |
| 27   | RightWristPitch     | 右臂   | GearboxS   |   40    |    1    |     低‡      |
| 28   | RightWristYaw       | 右臂   | GearboxS   |   40    |    1    |     低‡      |

> † WaistRoll (13) / WaistPitch (14) 在 G1 23DOF/29DOF waist-locked 版本中为无效关节，需运行时通过 `mode_machine_` 判断。
> ‡ WristPitch (20,27) / WristYaw (21,28) 在 G1 23DOF 版本中无效。

### 3.2 电机类型热力学特征

| 电机类型    | 减速比特征 | 峰值力矩 | 热惯性  | 热建模策略                                  |
| :---------- | :--------- | :-------- | :------ | :------------------------------------------ |
| **GearboxL** | 大减速比   | 最高      | 大      | 膝关节承重，焦耳热累积慢但极限温度最高      |
| **GearboxM** | 中减速比   | 中等      | 中      | 髋关节 + 腰偏航，动静态负载交替频繁        |
| **GearboxS** | 小减速比   | 较低      | 小      | 肩/肘/腕/踝，散热面积小，温升速率最快      |

---

## 4. 特征工程与状态空间定义 (Feature Engineering & State Space)

### 4.1 原始特征提取映射表（V2.0 修正）

> 下表修正了 V1.0 中的类型错误，并新增了 V1.0 遗漏的关键字段。

| 物理量            | SDK 字段                              | 实际数据类型          | 物理意义与转化逻辑                                                              |
| :---------------- | :------------------------------------ | :-------------------- | :------------------------------------------------------------------------------ |
| **估计力矩**      | `motor_state[i].tau_est`              | float32               | 焦耳发热源。预处理须转化为平方：$\tau_{sq} = \tau_{est}^2$                      |
| **关节角速度**    | `motor_state[i].dq`                   | float32               | 机械摩擦发热源。预处理取绝对值：$dq_{abs} = \|dq\|$                             |
| **角加速度**      | `motor_state[i].ddq`                  | float32               | 动态载荷突变检测。预处理取绝对值：$ddq_{abs} = \|ddq\|$。**V1.0 遗漏**          |
| **线圈温度**      | `motor_state[i].temperature[0]`       | **uint8_t** (0~255°C) | 绕组 NTC 温度，主预测目标。**V1.0 类型错误 (非 int8)**                          |
| **外壳温度**      | `motor_state[i].temperature[1]`       | **uint8_t** (0~255°C) | 减速器/外壳温度，辅助热传导建模。**V1.0 完全遗漏**                              |
| **电机电压**      | `motor_state[i].vol`                  | float32               | 电功率估算辅助：$P_{elec} \approx V \cdot I$。**V1.0 遗漏**                    |
| **辅助传感器**    | `motor_state[i].sensor[0..3]`         | float32 × 4           | 可能包含电流/编码器补充数据，需实机标定。**V1.0 遗漏**                          |
| **电机故障码**    | `motor_state[i].motorstate`           | uint32_t (位域)       | 过温/过流/通信异常标志，用于数据清洗。**V1.0 遗漏**                             |
| **IMU 温度**      | `imu_state.temperature`               | int16_t               | 机体内部环境温度参考                                                            |
| **主板温度**      | `MainBoardState_.temperature[0..5]`   | int16_t × 6           | 机体散热环境。**V1.0 遗漏**                                                     |
| **风扇状态**      | `MainBoardState_.fan_state[0..5]`     | uint16_t × 6          | 强制对流散热状态，影响散热系数。**V1.0 遗漏**                                   |
| **电池温度**      | `BmsState_.temperature[0..5]`         | int16_t × 6           | 电池散热环境基准。**V1.0 遗漏**                                                 |
| **系统总电流**    | `BmsState_.current`                   | int32_t               | 全身功耗宏观指标。**V1.0 遗漏**                                                 |

### 4.2 信号平滑与降噪算法 (Signal Smoothing)

由于 G1 的 `temperature` 为 `uint8_t` 阶梯状跳变数据（精度 1°C），直接输入神经网络会导致梯度震荡。

* **处理方案**: 对 `temperature[0]` 和 `temperature[1]` 分别采用指数移动平均 (EMA)：
  $$S_t = \alpha \cdot Y_t + (1 - \alpha) \cdot S_{t-1}$$
  *其中 $\alpha$ 根据 20Hz 采样率经验值设为 $0.05$。*
* **CRC32 校验**: 每帧数据入库前须执行 CRC 校验（SDK 源码确认 `Crc32Core` 算法），丢弃校验失败帧。
* **`motorstate_` 过滤**: 若位域包含过温 / 通信超时标志，该帧标记为异常但仍保留（异常事件本身是训练数据的一部分）。

### 4.3 网络输入输出张量设计（V2.0 重构）

#### 4.3.1 单关节局部耦合特征向量（以右肩俯仰 + 邻近关节为例）

$$X_t^{joint} = [\underbrace{\tau_{sq}, dq_{abs}, ddq_{abs}, T_{coil}, T_{shell}, V_{mot}}_{\text{目标关节 (6D)}}, \underbrace{T_{coil}^{adj1}, T_{coil}^{adj2}}_{\text{相邻关节温度 (2D)}}, \underbrace{T_{amb}, fan_{avg}}_{\text{环境 (2D)}}]$$

* **输入特征维度**: $D = 10$（V1.0 为 6，提升 67%）
* 其中 $T_{amb}$ = 主板温度均值 或 IMU 温度（取可用者）
* $fan_{avg}$ = 风扇状态归一化均值

#### 4.3.2 全局共享特征（可选：多任务学习架构）

若采用全关节联合预测架构，额外追加全局特征：

$$X_t^{global} = [I_{bms}, SOC, T_{bms\_avg}, \text{body\_rpy}[3], \|\omega_{gyro}\|]$$

维度 = 7，拼接在每个关节局部特征之后。

#### 4.3.3 张量形状

* **输入张量 Shape**: `[Batch_Size, Sequence_Length, D]`
  * 滑动窗口大小 $L = 100$（对应过去 5 秒 @ 20Hz）
* **输出张量 Shape**: `[Batch_Size, Prediction_Horizon]`
  * 预测视距 $H = 10$ 步（未来关键时间节点的温度轨迹，可配置为 [0.5s, 1s, 2s, 3s, 5s, 7s, 10s, 12s, 15s, 20s]）

---

## 5. 数据集采集协议 (Data Collection Protocol)

### 5.1 采集脚本核心框架 (Python SDK2)

```python
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC

TOPIC = "rt/lowstate"  # ★ V2.0 修正: 非 rt/lf/lowstate
G1_NUM_MOTOR = 29
SAMPLE_RATE = 20  # Hz

class ThermalDataCollector:
    def __init__(self):
        self.crc = CRC()
        self.subscriber = ChannelSubscriber(TOPIC, LowState_)
        self.subscriber.Init(self.callback, 10)

    def callback(self, msg: LowState_):
        # CRC 校验
        # 提取 motor_state[0..28] 的 tau_est, dq, ddq, temperature[2], vol
        # 降采样至 20Hz
        # EMA 平滑
        # 写入 HDF5 / Parquet
        pass
```

### 5.2 数据采集矩阵

必须覆盖边界工况以消除 OOD (Out-of-Distribution) 风险。

| 阶段 | 名称                                   | 占比 | 动作设计                                                                                           | 目的                                   |
| :--: | :------------------------------------- | :--: | :------------------------------------------------------------------------------------------------- | :------------------------------------- |
|  I   | 基础基准集 (Base)                      | 15%  | 空载原地踏步、手臂随机空间轨迹运动                                                                 | 拟合基础对流散热系数与轻载稳态热响应   |
|  II  | 单关节极限堵转集 (Stall)               | 20%  | 对右臂肩/肘关节施加 $10\text{Nm} \sim 25\text{Nm}$ 阶梯恒定力矩，保持至 $55^\circ\text{C}$        | 纯焦耳热条件下的最大温升斜率           |
|  III | 步态复合集 (Locomotion)                | 15%  | 平地行走 → 上下坡 → 转弯，持续 30 分钟                                                            | 下肢周期性负载的热累积曲线             |
|  IV  | **强耦合 Loco-manipulation (核心)**    | 35%  | Dex3-1 抗阻开门循环：末端抓取带阻尼反馈的门把手 → ZMP 偏移 → 腰部+支撑腿高扭矩 → 触发温度预警     | 上下肢耦合热传导、最高仿真价值场景     |
|  V   | 散热恢复集 (Cooldown)                  | 15%  | 高负载后立即进入静止/低负载状态，记录自然冷却曲线                                                   | 拟合散热时间常数，**V1.0 完全遗漏**    |

> **V2.0 新增 Phase V**: 冷却曲线对于 LSTM 学习"温度下降"动态至关重要。V1.0 仅关注升温，模型在冷却工况下将产生系统性高估偏差。

### 5.3 数据完整性要求

* **最短单次采集**: 不低于 30 分钟（从冷启动到热平衡再到冷却）
* **总数据量目标**: ≥ 10 小时有效数据（去除 CRC 失败帧和通信异常帧后）
* **温度标签对齐**: `temperature[0]`（线圈温度）作为主预测标签，`temperature[1]`（外壳温度）同步记录但用作辅助输入
* **环境温度记录**: 每次采集记录实验室室温（用于后期校准）

---

## 6. LSTM 网络架构与训练规范 (Model Architecture & Training)

### 6.1 网络拓扑设计 V2.0 (PyTorch)

严格因果架构，**禁用 BiLSTM**。

```
Input (D=10)
    │
    ▼
Linear(10 → 48) + LayerNorm + GELU          ← V2.0: 输入投影层升级
    │
    ▼
LSTM(input=48, hidden=96, layers=2,          ← V2.0: 隐层从 64 扩至 96
     batch_first=True, dropout=0.15)
    │
    ▼
取最后时间步 hidden state
    │
    ▼
Linear(96 → 48) + GELU
    │
    ▼
Linear(48 → H=10)                           ← 预测视距 H 步
```

**参数量估算**: ~85K（V1.0 ~35K），在 TensorRT FP16 下推理延迟仍 < 2ms。

### 6.2 双通道温度利用策略

$$\mathcal{L} = \mathcal{L}_{coil}(\hat{T}_{coil}, T_{coil}) + \lambda \cdot \mathcal{L}_{shell}(\hat{T}_{shell}, T_{shell})$$

* 主任务：预测线圈温度 `temperature[0]`
* 辅助任务（$\lambda = 0.3$）：联合预测外壳温度 `temperature[1]`，通过多任务学习隐式注入热传导物理先验

### 6.3 超参数与优化器

| 超参数                | 值                             | 说明                                                              |
| :-------------------- | :----------------------------- | :---------------------------------------------------------------- |
| Loss Function         | Huber Loss ($\delta = 1.0$)    | 降低传感器尖峰噪声的梯度惩罚                                     |
| Optimizer             | AdamW (wd = 1e-4)             | 权重衰减正则化                                                    |
| Learning Rate         | 初始 $1 \times 10^{-3}$       | 配合 CosineAnnealingWarmRestarts ($T_0=20, T_{mult}=2$)          |
| Batch Size            | 128                            |                                                                   |
| Max Epochs            | 200 (Early Stopping patience=15) |                                                                |
| Normalization         | Z-score (在 Dataset `__getitem__` 阶段) |                                                           |
| Gradient Clip         | max_norm = 1.0                 | 防止温度跳变导致梯度爆炸                                         |
| Sequence Length $L$   | 100 (5s @ 20Hz)                |                                                                   |
| Prediction Horizon $H$ | 10                            |                                                                   |

### 6.4 数据集划分

* **训练集**: 70%（随机打散但保持时间序列连续性——按"采集 session"划分）
* **验证集**: 15%（包含至少 1 个完整的 Phase IV 开门循环 session）
* **测试集**: 15%（包含至少 1 个未在训练中出现过的全新 session）

---

## 7. 工程部署与整合方案 (Deployment & Integration)

### 7.1 模型导出 (Model Export)

```python
torch.onnx.export(
    model, dummy_input,
    "thermal_predictor_g1.onnx",
    input_names=["state_seq"],
    output_names=["temp_pred"],
    dynamic_axes={"state_seq": {0: "batch"}},
    opset_version=17
)
```

### 7.2 C++ 实时推理流 (Online Inference Pipeline)

```
┌──────────────┐     ┌──────────────┐     ┌────────────────┐     ┌──────────────┐
│  DDS Sub     │     │  Ring Buffer  │     │  TensorRT      │     │  CBF /       │
│  rt/lowstate │────▶│  L × D       │────▶│  FP16 Infer    │────▶│  RL Policy   │
│              │     │  EMA+Norm    │     │  < 2ms         │     │              │
└──────────────┘     └──────────────┘     └────────────────┘     └──────────────┘
                           │
                           ▼
                     CRC32 Check +
                     motorstate_ 故障码过滤
```

1. **环形缓冲区 (Ring Buffer)**: 分配 $L \times D$ 的循环数组
2. **数据填充**: 每次 `rt/lowstate` 回调，提取特征 → CRC 校验 → EMA 平滑 → Z-score 归一化 → 压入缓冲区
3. **TensorRT 加速**: 调用 TensorRT C++ API 执行 ONNX 模型前向传播

### 7.3 闭环控制整合 (Closed-Loop with CBF)

将预测出的极限温度 $\hat{T}_{t+H}$ 实时推入 RL 策略的控制观测空间中：

$$\text{If } \hat{T}_{coil, t+H} \ge T_{threshold} \text{, Trigger Thermal-Aware Posture Adaptation.}$$

阈值推荐值：
* **软约束** ($T_{soft}$): 50°C — 开始降低该关节力矩限幅
* **硬约束** ($T_{hard}$): 60°C — 强制切换为低功率姿态

---

## 8. 风险与待确认事项 (Risks & Open Questions)

| # | 事项                                                              | 优先级 | 状态     |
|:-:|:-----------------------------------------------------------------|:------:|:--------:|
| 1 | `temperature[0]` vs `temperature[1]` 的物理含义需实机标定确认     | P0     | 待验证   |
| 2 | `motor_state[i].sensor[0..3]` 的具体物理量（可能含电流）需标定    | P1     | 待验证   |
| 3 | `MainBoardState_` / `BmsState_` 的实际 DDS Topic 名称需实机嗅探   | P1     | 待验证   |
| 4 | WaistRoll(13) / WaistPitch(14) 在 waist-locked 机型的数据有效性   | P1     | 通过 `mode_machine_` 判断 |
| 5 | `motorstate_` 位域的具体故障码定义需联系宇树技术支持获取           | P2     | 待确认   |

---

## 9. 实验计划流程 (Experiment Roadmap)

整体实验分为 **6 个阶段**，各阶段之间存在强依赖关系，必须严格按序推进。每阶段末尾设有**门控检查点 (Gate)**，未通过则不得进入下一阶段。

```
Phase 0        Phase 1          Phase 2         Phase 3          Phase 4          Phase 5
环境搭建  ──▶  传感器标定  ──▶  数据采集  ──▶  模型训练  ──▶  离线评估  ──▶  在线闭环验证
 (1 周)        (1~2 周)         (2~3 周)        (1~2 周)        (1 周)           (1~2 周)
   │              │                │               │               │                │
   ▼              ▼                ▼               ▼               ▼                ▼
 Gate 0         Gate 1           Gate 2          Gate 3          Gate 4           Gate 5
 通信连通       字段语义         数据量/质量      Loss 收敛       MAE ≤ 1.5°C     实时延迟
 数据可读       物理含义确认     覆盖率达标       无过拟合        全工况通过       ≤ 5ms
```

---

### Phase 0: 实验环境搭建与通信验证 (约 1 周)

**目标**: 确保从 G1 本体到采集主机的完整数据链路畅通。

| 步骤 | 具体操作 | 验收标准 |
|:----:|:---------|:---------|
| 0.1 | 在采集主机（Ubuntu 20.04+）上编译安装 `unitree_sdk2_python`，配置 CycloneDDS 0.10.2 | `import unitree_sdk2py` 无报错 |
| 0.2 | 以太网直连 G1，运行官方 `g1_low_level_example.py`，确认 `rt/lowstate` Topic 可接收 | 终端持续打印 IMU RPY 数据 |
| 0.3 | 编写最小化温度打印脚本：遍历 `motor_state[0..28]`，打印 `temperature[0]`、`temperature[1]` | 29 个关节的双通道温度均有非零输出 |
| 0.4 | 嗅探 `MainBoardState_` / `BmsState_` 的实际 DDS Topic 名称（使用 `cyclonedds` 命令行工具或自写 discovery 脚本） | 确认可用 Topic 名称并记录 |
| 0.5 | 测量采集主机端的回调频率，确认 500Hz 原始数据流无丢帧 | 连续 60s 采集，帧丢失率 < 0.1% |

**Gate 0**: 所有 Topic 数据可读，双通道温度非零，网络带宽无瓶颈。

---

### Phase 1: 传感器标定与字段语义确认 (约 1~2 周)

**目标**: 解决文档第 8 节中 P0/P1 级别的所有待确认事项，尤其是 `temperature[0]` vs `temperature[1]` 的物理含义。

#### 实验 1.1: 双通道温度物理含义标定

| 步骤 | 操作 | 预期观察 |
|:----:|:-----|:---------|
| a | G1 冷启动（室温平衡 ≥ 30 min），记录 `temperature[0]` 和 `temperature[1]` 初始值 | 两个通道初始值应接近室温但可能略有偏差 |
| b | 选取右肩俯仰关节 (index=22)，施加 15Nm 恒定堵转力矩，持续 5 分钟 | 通道 0 和通道 1 均应上升，但**上升速率应不同** |
| c | 停止施力后记录冷却过程 10 分钟 | 通道 0（若为线圈）应下降更快（热源停止），通道 1（若为外壳）下降更慢（热惯性大） |
| d | 用手持式红外测温枪同步测量电机**外壳表面温度**，与两个通道对比 | 与测温枪读数更接近的通道即为外壳温度 |

**判定规则**:
* 若通道 0 升温速率 > 通道 1，且通道 1 与红外测温枪吻合 → 通道 0 = 线圈，通道 1 = 外壳
* 若两个通道读数始终相同 → 可能为冗余设计，仅使用通道 0

#### 实验 1.2: `sensor[0..3]` 字段标定

| 步骤 | 操作 | 预期观察 |
|:----:|:-----|:---------|
| a | 空载状态记录 `sensor[0..3]` 基线 | 记录静态偏移量 |
| b | 施加已知力矩（5Nm, 10Nm, 15Nm, 20Nm）并记录 | 若某通道与力矩成线性关系 → 可能是电流测量 |
| c | 关节以不同速度运动（0.5, 1.0, 2.0 rad/s）并记录 | 若某通道与速度成线性关系 → 可能是反电动势/编码器信号 |

#### 实验 1.3: `motorstate_` 故障码触发测试

| 步骤 | 操作 | 预期观察 |
|:----:|:-----|:---------|
| a | 正常运行时持续记录 `motorstate_` 位域值 | 正常值（预期为 0 或固定掩码） |
| b | 逐步加大力矩至接近过温保护阈值 | 观察特定 bit 是否翻转，记录对应温度值 |
| c | 在堵转状态拔掉一个电机编码器线（如果安全允许） | 观察通信超时 bit 是否置位 |

**Gate 1**: `temperature[0/1]` 物理含义明确，`sensor[]` 中有价值的字段已识别，`motorstate_` 关键位含义已知。

---

### Phase 2: 系统化数据采集 (约 2~3 周)

**目标**: 按第 5 节数据采集矩阵的 5 个 Phase 完成全部数据集构建。

#### 2.1 采集总时间表

| 采集轮次 | Phase I~V 内容 | 持续时间 | 环境条件 | 重复次数 |
|:--------:|:--------------|:--------:|:--------:|:--------:|
| Round 1 | Phase I (空载) + Phase V (冷却) | 45 min | 室温 22±2°C | ×3 |
| Round 2 | Phase II (堵转) + Phase V (冷却) | 60 min | 室温 22±2°C | ×3 |
| Round 3 | Phase III (步态) + Phase V (冷却) | 50 min | 室温 22±2°C | ×3 |
| Round 4 | Phase IV (Loco-manipulation) + Phase V (冷却) | 60 min | 室温 22±2°C | ×5 |
| Round 5 | Phase I~IV 混合 + Phase V | 60 min | **室温 28±2°C**（高温环境） | ×2 |
| Round 6 | Phase I~IV 混合 + Phase V | 60 min | **室温 16±2°C**（低温环境） | ×2 |

* **总有效数据**: 约 18 轮 × 50 min ≈ **15 小时**（去除 CRC 坏帧后 ≥ 10 小时目标）
* Round 5/6 的不同环境温度用于验证模型对环境条件的泛化能力

#### 2.2 单轮采集标准操作流程 (SOP)

```
1. 冷启动检查
   ├── 确认 G1 已在当前室温下静置 ≥ 30 min
   ├── 记录室温（温湿度计读数）
   └── 启动采集脚本，确认所有 29 关节温度读数 ≈ 室温 (±3°C)

2. 执行目标 Phase 动作序列
   ├── 按预定脚本/遥控执行
   ├── 实时监控温度上限: 任意关节 temperature[0] ≥ 58°C 时手动终止
   └── 异常中断时记录断点时间戳

3. 进入冷却阶段 (Phase V)
   ├── G1 切换为阻尼站立（sport_mode 低功率姿态），不断电
   ├── 持续记录直至所有关节温度回落至 室温+5°C 以内
   └── 最短冷却记录时间: 10 min

4. 数据落盘与校验
   ├── 保存 HDF5 文件: {date}_{round}_{phase}.h5
   ├── 自动化脚本检查: CRC 坏帧率、温度曲线连续性、采样率偏差
   └── 生成该轮的温度曲线快照图 (PNG)，人工复核
```

#### 2.3 温度测试专项方案

以下列出针对性温度测试实验，用于覆盖边界工况和验证传感器可靠性。

##### 测试方案 A: 单关节阶梯力矩温升测试

**目的**: 获取力矩 → 温升速率的映射关系，作为 LSTM 学习的核心物理规律。

| 目标关节 | 力矩台阶 (Nm) | 每阶保持时间 | 终止条件 |
|:--------:|:--------------:|:------------:|:--------:|
| RightShoulderPitch (22) | 5 → 10 → 15 → 20 → 25 | 3 min/阶 | temperature[0] ≥ 55°C |
| RightElbow (25) | 5 → 10 → 15 → 20 | 3 min/阶 | temperature[0] ≥ 55°C |
| WaistYaw (12) | 5 → 10 → 15 → 20 → 25 | 3 min/阶 | temperature[0] ≥ 55°C |
| LeftKnee (3) | 10 → 20 → 30 → 40 | 3 min/阶 | temperature[0] ≥ 55°C |

**关键记录**: 每个力矩台阶的稳态温升速率 $\Delta T / \Delta t$ [°C/min]，绘制 $\tau^2$ vs $dT/dt$ 散点图验证焦耳热线性假设。

##### 测试方案 B: 周期往复运动摩擦热测试

**目的**: 分离摩擦热贡献（与角速度相关），独立于焦耳热（与力矩相关）。

| 目标关节 | 运动范围 | 角速度 (rad/s) | 外部负载 | 持续时间 |
|:--------:|:--------:|:--------------:|:--------:|:--------:|
| RightShoulderPitch (22) | ±60° | 0.5, 1.0, 2.0, 3.0 | 无 (空载) | 5 min/速度 |
| RightElbow (25) | ±45° | 0.5, 1.0, 2.0, 3.0 | 无 (空载) | 5 min/速度 |

**关键记录**: 各速度下的稳态温度增量 $\Delta T_{friction}$，绘制 $|dq|$ vs $\Delta T$ 曲线。

##### 测试方案 C: 热耦合传导测试

**目的**: 量化相邻关节之间的热传导效应，验证局部耦合特征设计是否合理。

| 操作 | 观测目标 |
|:-----|:---------|
| 仅对 RightShoulderPitch (22) 施加 20Nm 堵转，**其余关节空载** | 监测相邻关节 RightShoulderRoll (23)、RightShoulderYaw (24)、RightElbow (25) 的温度变化 |
| 仅对 WaistYaw (12) 施加 20Nm 堵转 | 监测 WaistRoll (13)、WaistPitch (14)、LeftHipPitch (0)、RightHipPitch (6) 的温度变化 |
| 仅对 LeftKnee (3) 施加 30Nm 堵转 | 监测 LeftHipPitch (0)、LeftHipRoll (1)、LeftAnklePitch (4) 的温度变化 |

**关键记录**: 热源关节温度达到 50°C 时，各相邻关节的温度增量 $\Delta T_{adj}$。若 $\Delta T_{adj} \ge 2°C$，则该关节对必须在特征向量中耦合。

##### 测试方案 D: 冷却时间常数标定

**目的**: 精确测量不同电机类型在不同初始温度下的自然冷却时间常数 $\tau_{cool}$。

| 初始条件 | 冷却方式 | 记录要点 |
|:---------|:---------|:---------|
| 电机 temperature[0] 加热至 45°C | 关节锁定零位，电机使能但无力矩输出 | 温度每下降 1°C 的时间间隔 |
| 电机 temperature[0] 加热至 55°C | 同上 | 同上 |
| 电机 temperature[0] 加热至 45°C | G1 站立状态，腿部关节持续低功耗维持平衡 | 带基础负载的冷却曲线 |

**分析方法**: 拟合 $T(t) = T_{amb} + (T_0 - T_{amb}) \cdot e^{-t/\tau_{cool}}$，提取 $\tau_{cool}$ 并按电机类型 (GearboxS/M/L) 分组统计。

##### 测试方案 E: 多关节协同高负载温度分布测试

**目的**: 模拟真实任务场景下的全身温度分布，验证数据采集矩阵 Phase IV 的有效性。

| 场景 | 动作描述 | 持续时间 | 预期热点 |
|:-----|:---------|:--------:|:---------|
| 单臂负重搬运 | 右臂持 2kg 哑铃，反复举起/放下 (0.5Hz) | 15 min | RightShoulderPitch, RightElbow |
| 抗阻开门循环 | 双手交替推拉弹簧门把手 | 20 min | 双侧 Shoulder, WaistYaw, 支撑腿 Knee |
| 快速行走 + 上坡 | 0.8 m/s 行走 10 min + 10° 斜坡行走 5 min | 15 min | 双侧 HipPitch, Knee, AnklePitch |
| 长时间站立平衡 | 单腿站立 (交替)，维持 5 min/腿 | 10 min | 支撑腿全部关节 |

**关键记录**: 全身 29 关节的温度热力图时间序列，识别各场景的 Top-5 高温关节。

##### 测试方案 F: 环境温度影响测试

**目的**: 验证环境温度对散热效率的影响，评估模型中 $T_{amb}$ 特征的必要性。

| 环境温度 | 测试内容 | 对比指标 |
|:--------:|:---------|:---------|
| 16°C (空调制冷) | 重复测试方案 A (RightShoulderPitch, 20Nm 堵转) | 达到 50°C 所需时间 |
| 22°C (标准室温) | 同上 | 同上（作为基准） |
| 28°C (加热环境) | 同上 | 同上 |
| 22°C + 外置风扇对吹 | 同上 | 同上（模拟强制对流） |

**预期**: 环境温度每升高 6°C，达到 50°C 的时间缩短约 15~25%。若差异 < 5%，则 $T_{amb}$ 特征可降级为常数。

**Gate 2**: 总有效数据 ≥ 10 小时，5 个 Phase 覆盖率达标，CRC 坏帧率 < 1%，温度曲线无明显异常跳变。

---

### Phase 3: 模型训练与迭代 (约 1~2 周)

| 步骤 | 操作 | 验收标准 |
|:----:|:-----|:---------|
| 3.1 | 数据预处理流水线: HDF5 → EMA 平滑 → Z-score → 滑动窗口切片 → Dataset/DataLoader | 单 epoch 数据加载时间 < 训练时间的 10% |
| 3.2 | 基线模型训练 (第 6 节架构) | 验证集 Loss 在 50 epoch 内开始收敛 |
| 3.3 | 消融实验 #1: 移除 `ddq` 特征 | 评估 MAE 变化量 |
| 3.4 | 消融实验 #2: 移除 `temperature[1]` (外壳温度) | 评估 MAE 变化量 |
| 3.5 | 消融实验 #3: 移除环境特征 ($T_{amb}$, $fan_{avg}$) | 评估 MAE 变化量 |
| 3.6 | 消融实验 #4: 滑动窗口长度对比 (L=50, 100, 200) | 选择 MAE/推理延迟最优的 L |
| 3.7 | 超参数搜索: hidden_size ∈ {64, 96, 128}, layers ∈ {1, 2, 3} | 在验证集上选择最优组合 |
| 3.8 | 最终模型: 用最优超参数在 Train+Val 上重训 | — |

**Gate 3**: 验证集 Huber Loss 稳定下降，Train/Val Loss 差距 < 20%（无严重过拟合），消融实验结论清晰。

---

### Phase 4: 离线评估与误差分析 (约 1 周)

#### 4.1 核心指标评估

在**测试集**（Phase 3 中从未参与训练的独立 session）上计算以下指标：

| 指标 | 定义 | 通过标准 |
|:-----|:-----|:--------:|
| MAE (全工况) | 全部关节、全部 Horizon 步的平均绝对误差 | ≤ 1.5°C |
| MAE (高负载) | 仅 Phase IV session 中的 MAE | ≤ 2.0°C |
| MAE@H=10 (15s) | 仅最远预测视距 (15s) 处的 MAE | ≤ 2.5°C |
| Max Error | 单步最大绝对误差 | ≤ 5.0°C |
| 冷却工况 MAE | 仅 Phase V session 中的 MAE | ≤ 1.5°C |

#### 4.2 分关节误差热力图

对每个关节单独计算 MAE，生成 29 关节 × 10 Horizon 步的误差矩阵热力图，识别：
* 哪些关节预测困难（候选原因：热耦合复杂、传感器噪声大）
* 哪些 Horizon 步误差增长最快（候选原因：窗口长度不足、非线性动态）

#### 4.3 失败案例分析

人工审查 Top-20 最大误差样本：
* 是否集中在某特定动作转换时刻（如从高负载突然停止）
* 是否集中在某特定关节
* 是否与 `motorstate_` 异常标志相关
* 误差方向是高估还是低估

**Gate 4**: MAE (全工况) ≤ 1.5°C，冷却工况 MAE ≤ 1.5°C，无系统性偏差。

---

### Phase 5: 在线闭环验证 (约 1~2 周)

#### 5.1 实时推理部署验证

| 步骤 | 操作 | 验收标准 |
|:----:|:-----|:---------|
| 5.1a | PyTorch → ONNX 导出 | ONNX Runtime 推理结果与 PyTorch 误差 < 1e-4 |
| 5.1b | ONNX → TensorRT FP16 转换 | 输出误差 < 0.1°C（量化损失可接受） |
| 5.1c | 单帧推理延迟测试 (Jetson Orin / PC) | 端到端 ≤ 5ms（含数据预处理） |
| 5.1d | 连续运行 1 小时稳定性测试 | 无内存泄漏，无推理超时 |

#### 5.2 在线预测精度验证

G1 在线运行，模型实时输出预测温度，同时录制 ground truth：

| 场景 | 持续时间 | 评估内容 |
|:-----|:--------:|:---------|
| 空载踏步 + 手臂随机运动 | 10 min | 低负载在线 MAE |
| 单臂 15Nm 堵转 | 5 min | 高负载稳态在线 MAE |
| Phase IV 开门循环 (新动作轨迹) | 15 min | 未见过轨迹的泛化 MAE |
| 高负载后自然冷却 | 10 min | 冷却工况在线 MAE |

**通过标准**: 在线 MAE ≤ 1.8°C（允许略高于离线，因存在实时数据延迟）。

#### 5.3 热保护闭环功能验证

| 测试项 | 操作 | 预期结果 |
|:-------|:-----|:---------|
| 软约束触发 | 逐步加大力矩至模型预测 $\hat{T}_{t+H} \ge 50°C$ | RL 策略自动降低该关节力矩限幅 |
| 硬约束触发 | 持续加载至模型预测 $\hat{T}_{t+H} \ge 60°C$ | 强制切换低功率姿态，温度不再上升 |
| 误报测试 | 正常低负载运动 30 min | 不应触发任何热保护（假阳性率 = 0） |
| 漏报测试 | 在堵转实验中，真实温度超过 55°C 前模型是否已预警 | 预警提前量 ≥ 10s |

**Gate 5**: 在线 MAE ≤ 1.8°C，热保护无误报无漏报，推理延迟 ≤ 5ms 稳定。

---

### 实验总时间表 (甘特图概览)

```
Week     1    2    3    4    5    6    7    8    9   10
       ├────┤
Phase 0  ██

       ├─────────┤
Phase 1       ████

            ├──────────────┤
Phase 2           █████████

                      ├─────────┤
Phase 3                    ████

                              ├────┤
Phase 4                           ██

                                 ├─────────┤
Phase 5                               ████

总计: 约 8~10 周
```

---

## 附录 A: V1.0 → V2.0 勘误总表

| 条目                  | V1.0 (错误)                    | V2.0 (修正)                              | 来源文件                               |
| :-------------------- | :----------------------------- | :--------------------------------------- | :------------------------------------- |
| DDS Topic             | `rt/lf/lowstate`               | **`rt/lowstate`**                        | `g1_low_level_example.py` L91         |
| `temperature` 类型    | `int8` (单值)                  | **`uint8_t[2]`** (双通道)                | `MotorState_.hpp` L29                  |
| `motor_state` 数组长度| 29                             | **35** (G1 使用 0~28)                    | `LowState_.hpp` L33                    |
| `ddq` 字段            | 未提及                         | **`float ddq_`** (角加速度)              | `MotorState_.hpp` L27                  |
| `vol` 字段            | 未提及                         | **`float vol_`** (电压)                  | `MotorState_.hpp` L30                  |
| 控制频率              | 1000Hz                         | **500Hz** (`control_dt_=0.002s`)         | `g1_low_level_example.py` L76         |
| 散热恢复集            | 无                             | **新增 Phase V (15%)**                   | V2.0 工程判断                          |
| 特征维度 D            | 6                              | **10** (含双温度+电压+加速度+环境)       | V2.0 特征工程重构                      |
| LSTM hidden_size      | 64                             | **96** (适配 D=10)                       | V2.0 架构调整                          |
| CRC 校验              | 未提及                         | **必须校验** (`Crc32Core`)               | `g1_dual_arm_example.cpp` L82-92      |

---
*文档结束 — V2.0 基于 SDK 源码逆向审计，所有接口字段均可追溯至 IDL 头文件。*
