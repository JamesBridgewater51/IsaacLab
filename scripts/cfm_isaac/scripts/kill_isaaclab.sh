#!/bin/bash

# 定义要搜索的进程名中的关键字
KEYWORD="gen_expert"

# 使用pgrep命令查找所有包含关键字的进程ID
PIDS=$(pgrep -f "$KEYWORD")

# 检查是否有找到进程
if [ -z "$PIDS" ]; then
    echo "没有找到包含'$KEYWORD'的进程。"
    exit 1
fi

# 遍历所有找到的进程ID，使用kill命令终止它们
for PID in $PIDS; do
    echo "终止进程 $PID ..."
    kill -9 $PID
    if [ $? -eq 0 ]; then
        echo "进程 $PID 已被终止。"
    else
        echo "无法终止进程 $PID。"
    fi
done

echo "所有包含'$KEYWORD'的进程已被终止。"