#!/bin/bash

# 定义输入和输出文件夹
input_folder="assets/shape_variant/urdf_heavy/"
output_folder="assets/shape_variant/usd_heavy/"

# 确保输出文件夹存在
mkdir -p "$output_folder"

# 遍历输入文件夹下的所有 .urdf 文件
for urdf_file in "$input_folder"*/model.urdf; do
    # 获取文件夹名（不带路径）
    foldername=$(basename "$(dirname "$urdf_file")")

# "$urdf_file"：这是一个变量，包含了一个文件的完整路径，例如 /path/to/your/file.urdf。

# dirname "$urdf_file"：dirname 命令会返回路径中去掉文件名后的部分。例如，如果 urdf_file 是 /path/to/your/file.urdf，那么 dirname "$urdf_file" 会返回 /path/to/your。

# basename "$(dirname "$urdf_file")"：basename 命令会返回路径中最后一部分。例如，如果 dirname "$urdf_file" 返回 /path/to/your，那么 basename "$(dirname "$urdf_file")" 会返回 your。

    # 定义输出文件路径
    output_file="$output_folder$foldername/model.usd"
    output_instanceable_file="$output_folder$foldername/model_instanceable.usd"


    echo $urdf_file
    echo $output_file
    ehco $output_instanceable_file

    # 确保输出文件夹存在
    mkdir -p "$(dirname "$output_file")"

    # 调用 convert_urdf.py 脚本
    python source/standalone/tools/convert_urdf.py \
        "$urdf_file" \
        "$output_file" \
        --headless
    python source/standalone/tools/convert_urdf.py \
        "$urdf_file" \
        "$output_instanceable_file" \
        --make-instanceable \
        --headless
done