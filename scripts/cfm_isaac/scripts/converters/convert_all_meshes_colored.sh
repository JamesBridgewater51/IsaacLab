#!/bin/bash

# 基目录
base_dir="/home/minghao/src/robotflow/cfm_isaac/third_party/IsaacLab/assets/shape_variant/thingi10k/colored_obj_stl"

# 遍历每个子目录
for dir in "$base_dir"/*/; do
  # 检查 model.obj 是否存在
  if [ -f "${dir}model.obj" ]; then
    # 执行 convert_mesh.py 脚本
    mkdir -p "${dir}usd_color/"
    python source/standalone/tools/convert_mesh.py \
      "${dir}model_color.obj" \
      "${dir}usd_color/model.usd" \
      --collision-approximation convexDecomposition \
      --mass 0.2 \
      --headless 
  else
    echo "model.obj not found in ${dir}"
  fi
done