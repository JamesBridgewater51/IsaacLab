#!/bin/bash

# 获取脚本自身的进程号
self_pid=$$

# 获取所有被删除文件的进程号
pids=$(lsof | grep deleted | awk '{print $2}' | sort | uniq)

# 检查是否有进程号
if [ -z "$pids" ]; then
  echo "没有找到被删除文件的进程。"
  exit 0
fi

# 杀掉所有相关进程，排除脚本自身的进程号
for pid in $pids; do
  if [ "$pid" -ne "$self_pid" ]; then
    echo "Killing process ID $pid"
    kill -9 $pid
  fi
done

echo "所有相关进程已被杀掉。"