#!/bin/bash
# Franka 触觉数据转换脚本

# 配置
INPUT_DIR="/home/dataset-local/data/megvii/wipe_plate"
OUTPUT_DIR="/home/dataset-local/data/megvii_post/tac"
REPO_ID="wipe_plate_tac"
PROMPT="wipe the pen marks off the plate."

# 转换数据
echo "=== 开始转换数据 ==="
uv run python /home/dataset-local/code/openpi/franka_to_ResTacVLA.py \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --repo-id "$REPO_ID" \
    --prompt "$PROMPT" \
    --gripper-threshold 0.07

# 检查转换是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "=== 生成可视化视频 ==="
    uv run python /home/dataset-local/code/openpi/visualize_lerobot.py \
        --repo-id "$REPO_ID" \
        --root "$OUTPUT_DIR"
    echo "=== 完成 ==="
else
    echo "转换失败，跳过可视化"
    exit 1
fi
