# 任务记忆文档（ThermalManagement / G1 热 LSTM）

> **用途**：记录项目进展、已做决策与理由、未完成事项，便于后续会话或成员接续工作。  
> **最后更新**：2025-03-25

---

## 1. 项目目标（基准）

- **来源**：`docs/plan.md` V2.0（宇树 G1 全关节热动力学、因果 LSTM、DDS 数据驱动）。
- **验收方向**（摘录）：高负载任务下肩/肘/腰等核心关节未来约 15s 温度预测 MAE ≤ 1.5°C；单次前向推理 ≤ 5ms（FP16，Jetson Orin / PC）。
- **数据与接口**：`rt/lowstate`（`LowState_`）、双通道电机温度、20Hz 降采样、CRC32、`motorstate_` 等见 plan 与 `docs/thermal_lstm_modeling.md`。

---

## 2. 当前进行到哪一步

| 阶段（对照 plan §9） | 状态 | 说明 |
|:---------------------|:-----|:-----|
| **文档与规格** | **已做** | `docs/plan.md` 为总计划；已撰写 `docs/thermal_lstm_modeling.md`（建模规格）。 |
| **Phase 0 通信/Topic 验证** | **部分** | `scripts/print_body_temperatures.py` 读全身温度并 CRC；`scripts/sniff_dds_topics.py` 仍为 stub。 |
| **Phase 1 标定** | **未做** | 双通道物理含义、`sensor[]`、`motorstate_` 位域等需实机。 |
| **Phase 2 系统化采集** | **未做** | `scripts/collect_session.py` 未实现；`thermal_g1.collector` 为空壳。 |
| **Phase 3+ 训练/评估/部署** | **未做** | `src/thermal_g1` 下 models/datasets/training 等仅为包占位。 |

**一句话**：处于 **「规格与只读温度验证脚本已具备，实机标定与采集/训练流水线尚未落地」** 的阶段。

---

## 3. 已做决策与理由

| 决策 | 内容 | 理由（摘要） |
|:-----|:-----|:-------------|
| **模型范围** | 按 **电机类型** 分三组 LSTM：`GearboxS` / `GearboxM` / `GearboxL` | 热动态按减速器类型更接近；同类关节可合并训练数据，比 29 独立模型易维护，比单模型更可解释。 |
| **预测目标** | **双通道联合**：主任务线圈 + 辅助任务外壳（共享编码器、双头） | 与 plan 多任务一致，注入热传导先验。 |
| **邻接关节** | **数据驱动** 发现热耦合（测试方案 C + Top-K / 互信息备选） | 机械拓扑未必等于热路径。 |
| **全局特征** | **初始不加入**，作为 **消融** | 降低对未确认 Topic 的硬依赖；先保证 `lowstate` 闭环可训。 |
| **文档深度** | **理论与工程平衡** | 便于评审与实现对照。 |

**脚本**（`print_body_temperatures.py`）：默认 CRC；关节名来自 `configs/joints_g1_29dof.yaml`；Python IDL 温度类型可能与 plan 表述不完全一致，以实机为准。

---

## 4. 已完成交付物（仓库内）

- `docs/thermal_lstm_modeling.md`：LSTM 建模规格。
- `scripts/print_body_temperatures.py`：订阅 `rt/lowstate`，打印 29 关节 T0/T1 + IMU，统计 CRC 失败。
- `configs/*.yaml`：Topic、关节表、邻接示例等。

---

## 5. 尚未解决 / 待办

- **Phase 0**：确认主板/BMS 等 Topic 真名；完善 `sniff_dds_topics.py`。
- **Phase 1**：双通道与 `sensor`、`motorstate_` 标定。
- **采集**：`thermal_g1.collector` + `collect_session.py`，HDF5，≥10h 有效数据。
- **训练/评估/部署**：Dataset、三模型训练、消融、ONNX/TensorRT。
- **依赖**：`unitree_sdk2_python` 需按宇树文档单独安装（未写入 `pyproject.toml`）。
- **跟踪**：Python IDL 与 plan 中电机温度类型表述可能不一致，以 IDL/实机为准。

---

## 6. 建议下一步

1. 运行 `print_body_temperatures.py` 完成 Gate 0。  
2. Phase 1 最小标定。  
3. 实现 HDF5 采集与回放，再接 Dataset 与训练。

---

## 7. 如何更新本文档

每完成 Gate/Phase 更新 §2、§5；策略变更在 §3 追加并注明日期。
