#!/bin/bash

# ==================== PI0 模型预下载脚本 ====================
# 用途：预先下载 PI0 预训练模型，避免训练时等待
# 使用方法：bash download_pi0_model.sh

echo "=========================================="
echo "PI0 模型预下载脚本"
echo "=========================================="
echo ""

# 设置代理（如果需要）
if [ -n "${http_proxy:-}" ] || [ -n "${https_proxy:-}" ]; then
    echo "检测到代理设置:"
    [ -n "${http_proxy:-}" ] && echo "  http_proxy=${http_proxy}"
    [ -n "${https_proxy:-}" ] && echo "  https_proxy=${https_proxy}"
    export HTTP_PROXY="${http_proxy:-${HTTP_PROXY:-}}"
    export HTTPS_PROXY="${https_proxy:-${HTTPS_PROXY:-}}"
else
    echo "⚠ 提示: 未检测到代理设置"
    echo "  如果下载速度慢，可以设置代理:"
    echo "    export http_proxy=http://your-proxy:port"
    echo "    export https_proxy=http://your-proxy:port"
fi

echo ""
echo "开始下载 PI0 预训练模型: lerobot/pi0_base"
echo "模型大小: 约 14GB"
echo "这可能需要一些时间，请耐心等待..."
echo ""

# 设置 Hugging Face 环境变量
export HF_ENDPOINT=https://huggingface.co
export HF_HUB_DOWNLOAD_TIMEOUT=7200  # 2小时超时
export HF_HUB_DOWNLOAD_RETRIES=5     # 重试5次

# 如果设置了 HF_TOKEN，使用它（可能获得更好的下载速度）
if [ -n "${HF_TOKEN:-}" ]; then
    echo "检测到 HF_TOKEN，将使用它进行下载（可能获得更好的速度）"
fi

# 设置 Python 路径
export PYTHONPATH="/lerobot/src:${PYTHONPATH}"

# 下载模型
python3 << 'EOF'
import os
import sys
from huggingface_hub import snapshot_download
from pathlib import Path

print("=" * 60)
print("开始下载 PI0 模型")
print("=" * 60)

# 检查代理设置
if os.environ.get('http_proxy') or os.environ.get('https_proxy'):
    print(f"使用代理:")
    if os.environ.get('http_proxy'):
        print(f"  http_proxy: {os.environ.get('http_proxy')}")
    if os.environ.get('https_proxy'):
        print(f"  https_proxy: {os.environ.get('https_proxy')}")
else:
    print("⚠ 未设置代理，下载可能较慢")

print()
print("模型信息:")
print("  - Repo ID: lerobot/pi0_base")
print("  - 大小: 约 14GB")
print("  - 缓存目录: ~/.cache/huggingface/hub")
print()

try:
    # 检查是否已登录 Hugging Face（登录后下载速度可能更快）
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        if user_info:
            print(f"✓ 已登录 Hugging Face (用户: {user_info.get('name', 'Unknown')})")
            print("  登录用户通常可以获得更好的下载速度")
    except Exception:
        print("⚠ 未登录 Hugging Face")
        print("  提示: 登录后可能获得更好的下载速度")
        print("  可以运行: hf auth login --token YOUR_TOKEN")
    
    print()
    print("开始下载...")
    print("（这可能需要较长时间，请耐心等待）")
    print("提示: 下载支持断点续传，可以安全中断（Ctrl+C）后重新运行")
    print()
    
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    
    # 使用 token 参数（如果设置了 HF_TOKEN）可能获得更好的速度
    token = os.environ.get("HF_TOKEN")
    
    snapshot_download(
        repo_id="lerobot/pi0_base",
        cache_dir=cache_dir,
        resume_download=True,  # 支持断点续传
        local_files_only=False,
        token=token  # 如果设置了 HF_TOKEN，使用它
    )
    
    print()
    print("=" * 60)
    print("✓ 模型下载完成！")
    print("=" * 60)
    print()
    print("模型已保存到:")
    print(f"  {cache_dir}/models--lerobot--pi0_base")
    print()
    print("现在可以运行训练脚本，训练时会自动使用已下载的模型")
    
except KeyboardInterrupt:
    print()
    print("⚠ 下载被用户中断")
    print("可以稍后重新运行此脚本继续下载（支持断点续传）")
    sys.exit(1)
except Exception as e:
    print()
    print(f"✗ 下载失败: {e}")
    print()
    print("可能的解决方案:")
    print("1. 检查网络连接")
    print("2. 设置代理加速下载")
    print("3. 检查 Hugging Face 登录状态")
    print("4. 重新运行此脚本（支持断点续传）")
    sys.exit(1)
EOF

DOWNLOAD_EXIT_CODE=$?

echo ""
echo "=========================================="
if [ "${DOWNLOAD_EXIT_CODE}" -eq 0 ]; then
    echo "✓ 模型下载完成！"
    echo "现在可以运行训练脚本: bash train_multi.sh"
else
    echo "下载失败或中断"
    echo "可以重新运行此脚本继续下载（支持断点续传）"
fi
echo "=========================================="

