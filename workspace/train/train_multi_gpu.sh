#!/bin/bash

# ==================== 环境配置 ====================
# 自动检测并激活 conda 环境（如果在本地环境运行）
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    # 检查是否存在 lerobot 环境
    if conda env list | grep -q "^lerobot "; then
        conda activate lerobot
        echo "✓ 已激活 conda 环境: lerobot"
    else
        echo "⚠ 警告: 未找到 lerobot conda 环境"
        echo "  如果使用 Docker 容器，可以忽略此警告"
    fi
fi

# ==================== 配置区域 ====================

# Lerobot 路径（容器内根目录）
LEROBOT_ROOT="/lerobot"
TRAIN_SCRIPT="${LEROBOT_ROOT}/src/lerobot/scripts/train.py"

# 设置 PYTHONPATH 以确保可以导入 lerobot 模块
export PYTHONPATH="${LEROBOT_ROOT}/src:${PYTHONPATH}"

# 设置 PyTorch CUDA 内存分配配置（解决内存碎片问题）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ==================== 多 GPU 配置 ====================
# 检测可用 GPU 数量
NUM_GPUS=0
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    echo "检测到 ${NUM_GPUS} 个 GPU"
else
    echo "警告: 未找到 nvidia-smi 命令"
    echo "尝试使用 PyTorch 检测 GPU..."
    NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
fi

# 检查 GPU 数量
if [ "${NUM_GPUS}" -lt 2 ]; then
    echo "错误: 需要至少 2 个 GPU 进行多 GPU 训练，当前检测到 ${NUM_GPUS} 个 GPU"
    echo "请使用 train_multi.sh 进行单 GPU 训练"
    exit 1
fi

# 可以通过环境变量 NUM_GPUS 指定使用的 GPU 数量（默认使用所有 GPU）
USE_GPUS="${NUM_GPUS:-${NUM_GPUS}}"
if [ "${USE_GPUS}" -gt "${NUM_GPUS}" ]; then
    echo "警告: 指定的 GPU 数量 (${USE_GPUS}) 超过可用 GPU 数量 (${NUM_GPUS})"
    USE_GPUS="${NUM_GPUS}"
fi

echo "将使用 ${USE_GPUS} 个 GPU 进行训练"
echo ""

# 检查 accelerate 是否已安装
if ! command -v accelerate &> /dev/null; then
    echo "错误: 未找到 accelerate 命令"
    echo "请安装: pip install accelerate"
    exit 1
fi

echo "✓ accelerate 已安装"
echo ""

# ==================== GPU 内存清理和检查 ====================
# 注意：在设置 CUDA_VISIBLE_DEVICES 之前清理 GPU，使用原始 GPU 索引
echo "检查 GPU 状态..."
# 检查 GPU 使用情况（如果 nvidia-smi 可用）
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits | while IFS=',' read -r idx name used total; do
        echo "GPU $idx ($name): ${used}MB / ${total}MB 已使用"
    done
fi

# 清理 GPU 缓存和 Python 对象
echo "清理 GPU 缓存..."
python3 << 'EOF'
import torch
import gc
import sys

if torch.cuda.is_available():
    # 清理所有 GPU 缓存
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
    # 强制垃圾回收
    gc.collect()
    # 再次清理缓存
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
    
    # 显示清理后的 GPU 内存状态
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"GPU {i}: {allocated:.2f}GB 已分配, {reserved:.2f}GB 已保留, {total:.2f}GB 总计")
    
    # 如果仍有大量内存被占用，给出警告
    if torch.cuda.memory_allocated(0) > 1024**3:  # 如果超过 1GB
        print("警告: GPU 内存仍被占用，可能有其他进程在使用 GPU")
        print("建议: 检查并结束其他占用 GPU 的进程，或重启容器")
else:
    print("CUDA 不可用")
    sys.exit(1)
EOF

echo "GPU 清理完成"
echo ""

# 设置 CUDA_VISIBLE_DEVICES 以确保 accelerate 能正确识别 GPU
# 使用前 USE_GPUS 个 GPU（0, 1, 2, ...）
CUDA_VISIBLE_DEVICES_LIST=""
for i in $(seq 0 $((USE_GPUS - 1))); do
    if [ -z "${CUDA_VISIBLE_DEVICES_LIST}" ]; then
        CUDA_VISIBLE_DEVICES_LIST="${i}"
    else
        CUDA_VISIBLE_DEVICES_LIST="${CUDA_VISIBLE_DEVICES_LIST},${i}"
    fi
done
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_LIST}"

echo "已设置 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo ""

# 切换到 lerobot 根目录
cd "${LEROBOT_ROOT}"

# 检查训练脚本是否存在
if [ ! -f "${TRAIN_SCRIPT}" ]; then
    echo "错误: 找不到训练脚本 ${TRAIN_SCRIPT}"
    echo "请确认 lerobot 源码在容器内的 /lerobot 目录下"
    exit 1
fi

# ==================== 策略类型配置 ====================
# 修改这里选择策略类型: "act", "diffusion", "pi0"
export POLICY_TYPE="act"

# ==================== 数据集配置 ====================
# 修改这里设置你的数据集
DATASET_ROOT="/workspace/dataset"
DATASET_REPO_ID="uni_ur5e_26"  # 修改为你的数据集名称
OUTPUT_DIR="/workspace/outputs/${POLICY_TYPE}_train/${DATASET_REPO_ID}"
JOB_NAME="${DATASET_REPO_ID}"


# ==================== 根据策略类型设置超参数 ====================
case "$POLICY_TYPE" in
    "act")
        policy="act"
        BATCH_SIZE=4  # 每个 GPU 的批次大小
        STEPS=10000
        SAVE_FREQ=2000
        LOG_FREQ=200
        N_OBS_STEPS=1
        CHUNK_SIZE=100
        N_ACTION_STEPS=1  # 使用时间集成时必须为 1
        # 注意：使用 accelerate 时，--policy.use_amp 会被忽略，由 accelerate 的 --mixed_precision 控制
        EXTRA_ARGS="--policy.temporal_ensemble_coeff=0.01"
        MIXED_PRECISION="fp16"  # accelerate 的混合精度设置: fp16, bf16, 或 no
        ;;
    "diffusion")
        policy="dp"
        BATCH_SIZE=1  # 每个 GPU 的批次大小
        STEPS=10000
        SAVE_FREQ=1000
        LOG_FREQ=100
        N_OBS_STEPS=2
        HORIZON=64
        N_ACTION_STEPS=32
        DROP_N_LAST_FRAMES=$((HORIZON - N_ACTION_STEPS - N_OBS_STEPS + 1))
        EXTRA_ARGS="--policy.horizon=${HORIZON} --policy.crop_shape=[480,640] --policy.drop_n_last_frames=${DROP_N_LAST_FRAMES}"
        MIXED_PRECISION="fp16"
        ;;
    "pi0")
        policy="pi0"
        BATCH_SIZE=1  # 每个 GPU 的批次大小
        STEPS=10000
        SAVE_FREQ=10000
        LOG_FREQ=2000
        N_OBS_STEPS=1
        CHUNK_SIZE=4
        N_ACTION_STEPS=4
        PRETRAINED_PATH="lerobot/pi0_base"
        EXTRA_ARGS="--policy.gradient_checkpointing=true --policy.dtype=bfloat16"
        MIXED_PRECISION="bf16"  # PI0 使用 bfloat16
        ;;
    *)
        echo "错误: 未知的策略类型: ${POLICY_TYPE}"
        echo "可用选项: act, diffusion, pi0"
        exit 1
        ;;
esac

# 计算有效批次大小（所有 GPU 的总和）
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * USE_GPUS))

# 多 GPU 训练时，使用适当的 num_workers（每个 GPU 2-4 个 worker）
# 注意：在多 GPU 训练中，num_workers 是每个进程的 worker 数量
# 如果设置为 0，数据加载会在主进程中进行，可能导致卡顿
NUM_WORKERS=2  # 每个 GPU 进程使用 2 个 worker（可以根据需要调整）

# ==================== 输出配置信息 ====================
echo "=========================================="
echo "多 GPU 训练配置"
echo "=========================================="
echo "数据集: ${DATASET_ROOT}/${DATASET_REPO_ID}"
echo "策略类型: ${POLICY_TYPE}"
echo "输出目录: ${OUTPUT_DIR}"
echo "作业名称: ${JOB_NAME}"
echo "GPU 数量: ${USE_GPUS}"
echo "批次大小（单 GPU）: ${BATCH_SIZE}"
echo "有效批次大小（总）: ${EFFECTIVE_BATCH_SIZE} (${BATCH_SIZE} × ${USE_GPUS})"
echo "训练步数: ${STEPS}"
echo "混合精度: ${MIXED_PRECISION}"
echo "数据加载 workers（每进程）: ${NUM_WORKERS}"
if [ "${POLICY_TYPE}" = "pi0" ]; then
    echo "Chunk size: ${CHUNK_SIZE}"
    echo "Action steps: ${N_ACTION_STEPS}"
fi
echo "=========================================="
echo ""

# ==================== 构建训练命令 ====================
BASE_ARGS="--dataset.repo_id=${DATASET_REPO_ID} \
--dataset.root=${DATASET_ROOT} \
--output_dir=${OUTPUT_DIR} \
--job_name=${JOB_NAME} \
--policy.type=${POLICY_TYPE} \
--policy.device=cuda \
--policy.push_to_hub=false \
--num_workers=${NUM_WORKERS} \
--batch_size=${BATCH_SIZE} \
--steps=${STEPS} \
--save_freq=${SAVE_FREQ} \
--log_freq=${LOG_FREQ} \
--eval_freq=20000 \
--policy.n_obs_steps=${N_OBS_STEPS} \
${CHUNK_SIZE:+--policy.chunk_size=${CHUNK_SIZE}} \
${N_ACTION_STEPS:+--policy.n_action_steps=${N_ACTION_STEPS}} \
--seed=42 \
--wandb.enable=false \
${EXTRA_ARGS}"

# 根据策略类型添加特定参数
case "$POLICY_TYPE" in
    "act")
        # ACT 策略：注意 --policy.use_amp 在使用 accelerate 时会被忽略
        ;;
    "pi0")
        # 如果设置了预训练路径，添加微调参数
        if [ -n "${PRETRAINED_PATH:-}" ]; then
            BASE_ARGS="${BASE_ARGS} --policy.pretrained_path=${PRETRAINED_PATH}"
            echo "使用微调模式，预训练模型: ${PRETRAINED_PATH}"
        else
            echo "使用全量训练模式（从头开始训练）"
        fi
        ;;
esac

# ==================== 执行训练 ====================
# 检查输出目录是否存在，如果存在则删除（避免 FileExistsError）
if [ -d "${OUTPUT_DIR}" ]; then
    echo "警告: 输出目录已存在: ${OUTPUT_DIR}"
    echo "删除旧目录以继续训练..."
    rm -rf "${OUTPUT_DIR}"
    echo "已删除旧目录"
fi

# 再次清理 GPU 缓存（在训练前）
echo "训练前最后一次清理 GPU 缓存..."
python3 -c "import torch; [torch.cuda.set_device(i) or torch.cuda.empty_cache() for i in range(torch.cuda.device_count())]; import gc; gc.collect(); print('GPU 缓存已清理')" 2>/dev/null || true

echo ""
echo "=========================================="
echo "多 GPU 训练说明"
echo "=========================================="
echo "1. 使用 accelerate 进行分布式训练"
echo "2. 批次会自动分配到各个 GPU"
echo "3. 梯度会在所有 GPU 间同步"
echo "4. 只有主进程会保存检查点和日志"
echo "5. 有效批次大小 = ${BATCH_SIZE} × ${USE_GPUS} = ${EFFECTIVE_BATCH_SIZE}"
echo ""
echo "注意:"
echo "- 使用 accelerate 时，--policy.use_amp 会被忽略"
echo "- 混合精度由 accelerate 的 --mixed_precision=${MIXED_PRECISION} 控制"
echo "- 如果遇到 OOM，可以减小 BATCH_SIZE（当前: ${BATCH_SIZE}）"
echo "- 数据集创建可能需要一些时间，特别是首次加载视频数据时"
echo "- 多 GPU 训练时，主进程会先创建数据集，然后其他进程再创建"
echo "- 如果卡在'创建数据集'超过 10 分钟，可以尝试："
echo "  1. 检查是否有其他进程占用 GPU"
echo "  2. 启用 NCCL_DEBUG=INFO 查看详细日志"
echo "  3. 或者使用单 GPU 训练（train_multi.sh）"
echo "=========================================="
echo ""

echo "开始多 GPU 训练..."
echo "命令: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --multi_gpu --num_processes=${USE_GPUS} --mixed_precision=${MIXED_PRECISION} ${TRAIN_SCRIPT} ${BASE_ARGS}"
echo ""

# ==================== 改进的多 GPU 训练启动 ====================
# 设置 NCCL 和分布式训练环境变量
export NCCL_TIMEOUT=7200  # 2 小时超时（应对数据集加载）
export TORCH_DISTRIBUTED_TIMEOUT=7200
export NCCL_ASYNC_ERROR_HANDLING=1  # 启用异步错误处理

# 可选：如果遇到通信问题，可以取消注释以下行
# export NCCL_IB_DISABLE=1  # 禁用 InfiniBand，使用 TCP
# export NCCL_DEBUG=INFO     # 启用详细调试（会产生大量输出）

# 创建临时的 accelerate 配置文件（更稳定的方式）
ACCELERATE_CONFIG="/tmp/accelerate_config_${USE_GPUS}gpu.yaml"
cat > "${ACCELERATE_CONFIG}" << EOF
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
mixed_precision: ${MIXED_PRECISION}
num_processes: ${USE_GPUS}
use_cpu: false
gpu_ids: all
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
num_machines: 1
rdzv_backend: static
same_network: true
EOF

echo "使用 accelerate 配置文件: ${ACCELERATE_CONFIG}"
echo ""
echo "提示: 如果训练卡在'创建数据集'超过 10 分钟，"
echo "      可以按 Ctrl+C 中断，然后检查日志或使用单 GPU 训练"
echo ""

# 使用 accelerate launch 启动多 GPU 训练
# 使用配置文件方式（更稳定，避免命令行参数解析问题）
# 注意：使用 timeout 命令设置最大运行时间（可选，默认不限制）
if command -v timeout &> /dev/null; then
    # 如果系统有 timeout 命令，可以设置最大运行时间（例如 24 小时）
    # TIMEOUT_HOURS=24
    # timeout ${TIMEOUT_HOURS}h \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" accelerate launch \
        --config_file="${ACCELERATE_CONFIG}" \
        "${TRAIN_SCRIPT}" ${BASE_ARGS}
else
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" accelerate launch \
        --config_file="${ACCELERATE_CONFIG}" \
        "${TRAIN_SCRIPT}" ${BASE_ARGS}
fi

TRAIN_EXIT_CODE=$?

# 清理临时配置文件
rm -f "${ACCELERATE_CONFIG}" 2>/dev/null || true

echo ""
echo "=========================================="
if [ "${TRAIN_EXIT_CODE}" -eq 0 ]; then
    echo "训练完成！检查点保存在: ${OUTPUT_DIR}"
else
    echo "训练失败，退出代码: ${TRAIN_EXIT_CODE}"
    echo "请检查错误信息"
fi
echo "=========================================="

