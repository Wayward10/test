# ACT 策略训练指南 - UR5e 26自由度

本指南说明如何使用 lerobot v3.0 训练 ACT 策略，数据集为 `uni_ur5e_26`（26自由度 UR5e 机器人）。

## 文件说明

- `train.py`: 训练脚本，位于 `/workspace/lerobot/src/lerobot/scripts/train.py`
- `train_act_ur5e.sh`: 训练启动脚本，位于 `/workspace/train_act_ur5e.sh`

## 数据集信息

- **数据集路径（容器内）**: `/dataset/uni_ur5e_26`
- **数据集路径（宿主机）**: `/home/ouhuang/lerobot_v3.0/lerobot_data/uni_ur5e_26`
- **动作空间**: 26 自由度
- **状态空间**: 26 维关节状态
- **摄像头**: 3 个（right_cam, left_cam, head_cam）
- **FPS**: 30
- **总帧数**: 7953
- **总episode数**: 3

## Docker 容器映射关系

根据您提供的 Docker 运行命令：

```bash
docker run -id \
  --name lerobot \
  --gpus all \
  --privileged \
  --network host \
  --ipc host \
  --pid host \
  --group-add video \
  --device /dev/bus/usb \
  -v /dev:/dev \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /home/ouhuang/lerobot_v3.0/workspace:/workspace \
  -v /home/ouhuang/lerobot_v3.0/lerobot_data:/dataset \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e ROS_DOMAIN_ID=99 \
  -e ROS_LOCALHOST_ONLY=0 \
  lerobot:latest \
  bash
```

映射关系：
- 宿主机 `/home/ouhuang/lerobot_v3.0/workspace` → 容器 `/workspace`
- 宿主机 `/home/ouhuang/lerobot_v3.0/lerobot_data` → 容器 `/dataset`
- **Lerobot 源码**: 容器内 `/lerobot`（lerobot 容器根目录）

**重要**: 训练脚本会从容器内的 `/lerobot/src/lerobot/scripts/train.py` 运行，确保 lerobot 源码在容器的 `/lerobot` 目录下。

## 使用方法

### 方法 1: 使用提供的训练脚本（推荐）

```bash
# 进入容器
docker exec -it lerobot bash

# 运行训练脚本
bash /workspace/train_act_ur5e.sh
```

### 方法 2: 直接使用 Python 命令

```bash
# 进入容器
docker exec -it lerobot bash

# 切换到 lerobot 根目录并设置 PYTHONPATH
cd /lerobot
export PYTHONPATH="/lerobot/src:${PYTHONPATH}"

# 运行训练命令
python /lerobot/src/lerobot/scripts/train.py \
  --dataset.repo_id=uni_ur5e_26 \
  --dataset.root=/dataset \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=/workspace/outputs/train/act_ur5e_26 \
  --job_name=act_ur5e_26 \
  --batch_size=8 \
  --steps=100000 \
  --log_freq=200 \
  --save_freq=20000 \
  --eval_freq=20000 \
  --num_workers=4 \
  --seed=42 \
  --wandb.enable=false
```

### 方法 3: 从容器外部运行

```bash
docker exec -it lerobot bash /workspace/train_act_ur5e.sh
```

## 训练配置说明

### 关键参数

- `--dataset.repo_id=uni_ur5e_26`: 数据集标识符
- `--dataset.root=/dataset`: 数据集根目录（容器内路径）
- `--policy.type=act`: 使用 ACT 策略
- `--policy.device=cuda`: 使用 GPU 训练
- `--policy.push_to_hub=false`: 禁用推送到 Hugging Face Hub（本地训练）
- `--batch_size=8`: 批次大小（可根据 GPU 内存调整）
- `--steps=100000`: 总训练步数
- `--log_freq=200`: 每 200 步记录一次日志
- `--save_freq=20000`: 每 20000 步保存一次检查点
- `--num_workers=4`: 数据加载器工作进程数

### ACT 策略自动配置

ACT 策略会自动从数据集元数据中读取：
- **输入形状**: 从 `meta/info.json` 中的 `features` 自动推断
- **输出形状**: 从 `meta/info.json` 中的 `action` 特征自动推断（26维）
- **图像输入**: 自动检测 3 个摄像头（right_cam, left_cam, head_cam）

无需手动配置 26 自由度的动作空间，策略会根据数据集自动适配。

## 输出目录结构

训练完成后，检查点将保存在：

```
/workspace/outputs/train/act_ur5e_26/
├── checkpoints/
│   ├── step_020000/
│   ├── step_040000/
│   ├── ...
│   └── step_100000/
├── eval/
│   └── videos_step_*/
└── train_config.json
```

## 恢复训练

如果需要从检查点恢复训练：

```bash
python -m lerobot.scripts.train \
  --config_path=/workspace/outputs/train/act_ur5e_26/train_config.json \
  --resume=true
```

## 调整训练参数

### 根据 GPU 内存调整批次大小

如果遇到 GPU 内存不足，可以减小批次大小：

```bash
--batch_size=4  # 或更小
```

### 调整学习率

```bash
--policy.optimizer_lr=1e-5  # 默认值
--policy.optimizer_lr_backbone=1e-5  # 骨干网络学习率
```

### 启用 WandB 日志

```bash
--wandb.enable=true \
--wandb.project=act_ur5e_26 \
--wandb.entity=your_username
```

## 常见问题

### 1. 数据集路径错误

确保数据集路径正确：
- 容器内路径：`/dataset/uni_ur5e_26`
- 检查数据集是否存在：`ls /dataset/uni_ur5e_26/meta/info.json`

### 2. Lerobot 路径错误

确保 lerobot 源码在容器内的正确位置：
- 容器内路径：`/lerobot/src/lerobot/scripts/train.py`
- 检查脚本是否存在：`ls /lerobot/src/lerobot/scripts/train.py`
- 如果路径不同，请修改 `train_act_ur5e.sh` 中的 `LEROBOT_ROOT` 变量

### 3. GPU 内存不足

- 减小 `batch_size`
- 减小 `num_workers`
- 使用梯度累积（需要修改代码）

### 4. 数据加载慢

- 增加 `num_workers`
- 确保使用 `pin_memory=True`（已默认启用）

## 验证训练

训练过程中会输出类似以下信息：

```
创建数据集
创建策略
输出目录: /workspace/outputs/train/act_ur5e_26
steps=100000 (100k)
dataset.num_frames=7953 (7.95k)
dataset.num_episodes=3
有效批次大小: 8 x 1 = 8
开始在固定数据集上进行离线训练
```

## 参考

- [LeRobot ACT 文档](https://github.com/huggingface/lerobot)
- [ACT 论文](https://huggingface.co/papers/2304.13705)

