# reBot Arm B601-RS 夹爪控制

目前Seeed官方的夹爪使用前馈控制，存在扭矩小-速度小不跟手/扭矩大-速度大-容易损坏夹爪的情况。此优化是基于 Seeed B601-RS LeRobot 插件修改的夹爪控制版本。在保持 leader 实时跟随的同时，使用 MIT 位置控制夹爪，并通过 RobStride `0x700B` 限制电机最大力矩。

## 当前控制逻辑

- 夹爪命令：`send_mit(target_pos_rad, 0, kp, kd, 0)`
- 默认增益：`kp=12`、`kd=2` 
- 默认 `0x700B` 力矩限制：`0.5 N.m`
- 夹爪软件角度范围：`0~245 deg`
- 启动遥操作前，RS 会使用 minimum-jerk 轨迹平滑移动到 leader 当前姿态
- 遥操作、录制和回放均可记录 RS 实际角度、速度、力矩和反馈时间戳

> `0x700B` 是电机端持续生效的参数，程序退出后不会自动恢复。若更换机械臂，应重新评估力矩限制。

## 安装

已安装 LeRobot 环境时：

```bash
conda activate lerobot
git clone https://github.com/bryce301/Rebot-Arm-B601-RS-.git
cd Rebot-Arm-B601-RS-
pip install -e .
pip install lerobot-teleoperator-rebot-arm-102
```

主要依赖为 Python 3.10+、LeRobot 0.4+、MotorBridge 0.4.9+ 和 python-can。

## 配置 CAN

本项目使用 SocketCAN，B601-RS 波特率为 1 Mbps：

```bash
sudo ip link set can0 down 2>/dev/null
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
ip -details link show can0
```

正常状态应包含 `UP`、`LOWER_UP`、`ERROR-ACTIVE` 和 `bitrate 1000000`。

## 遥操作

默认 leader 为 `/dev/ttyUSB0`，RS 为 `can0`，控制频率为 150 Hz：

```bash
python scripts/rebot_teleop_position_gripper.py \
  --leader-port /dev/ttyUSB0 \
  --robot-port can0 \
  --gripper-kp 12 \
  --gripper-kd 2 \
  --gripper-torque-limit 0.5
```

遥操作日志默认写入 `logs/`。按 `Ctrl+C` 停止。

## 录制关节轨迹

下面录制 60 秒，频率为 150 Hz：

```bash
mkdir -p recordings
python scripts/rebot_record_teleop.py \
  --leader-port /dev/ttyUSB0 \
  --robot-port can0 \
  --fps 150 \
  --duration 60 \
  --out recordings/demo.npz
```

录制文件同时保存发送给 RS 的目标角度 `rs_target_pos_rad` 和 RS 反馈角度 `rs_actual_pos_rad`。默认回放目标角度。

## 回放

回放前，机械臂会先平滑回到零位，再平滑移动到录制第一帧：

```bash
python scripts/rebot_replay.py \
  --robot-port can0 \
  --file recordings/demo.npz \
  --source rs_target_pos_rad \
  --fps 150 \
  --log-out logs/demo_replay.npz
```

## 分析反馈延迟

```bash
python scripts/analyze_feedback_freshness.py recordings/demo.npz
```

## 安全说明

- 首次运行应清空机械臂工作范围，并准备断电或急停。
- `0.5 N.m` 是当前测试参数，不代表所有夹爪、传动或物体都安全。
- 增大 `kp` 会提高位置响应，也可能增加碰撞冲击；增大 `kd` 会提高阻尼。
- 运行前确认 leader 和 RS 零位、方向映射及关节限位正确。

## 来源

机器人插件基于 [Seeed-Projects/lerobot-robot-seeed-b601](https://github.com/Seeed-Projects/lerobot-robot-seeed-b601)，许可证见 [LICENSE](LICENSE)。本仓库增加了 MIT 位置夹爪、`0x700B` 写入与回读验证、时间戳反馈及配套遥操作/录制/回放脚本。
