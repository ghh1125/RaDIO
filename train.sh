#!/bin/bash

echo "Current working directory: $(pwd)"
base_dir="./config/Llama2-7b-chat"
echo "Base directory: $base_dir"

# 确保当前工作目录是预期的目录
cd "$(dirname "$0")" || exit 1

# 遍历 base_dir 下的所有子目录
for subdir in "$base_dir"/*/; do
    if [ -d "$subdir" ]; then
        echo "Checking directory: $subdir"

        # 构造 DRAGIN.json 文件的完整路径
        config_path="$subdir/DRAGIN.json"

        # 检查 DRAGIN.json 文件是否存在
        if [ -f "$config_path" ]; then
            echo "Found config file: $config_path"

            # 尝试运行 Python 脚本，使用相对路径指向 src/ 目录下的 main.py
            python src/main.py -c "$config_path"

            # 检查 Python 脚本的退出状态
            if [ $? -ne 0 ]; then
                echo "Error running python script for $config_path"
            fi
        else
            echo "No DRAGIN.json found in $subdir"
        fi
    else
        echo "Skipping non-directory entry: $subdir"
    fi
done