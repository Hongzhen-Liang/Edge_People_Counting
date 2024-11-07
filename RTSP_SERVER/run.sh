#!/bin/bash

# 启动第一个命令 (mediamtx)
mediamtx/mediamtx mediamtx/mediamtx.yml &

# 获取第一个进程的 PID
PID1=$!

rtsp_stream=$(yq '.share_settings.rtsp_stream' ../config.yaml)
video_path=$(yq '.server_settings.video_path' ../config.yaml)
FPS=$(yq '.share_settings.FPS' ../config.yaml)

# 启动第二个命令 (ffmpeg 推流)
ffmpeg -re -stream_loop -1 -i $video_path -r $FPS -c copy \
-fifo_size 5000000 -rtsp_transport tcp -buffer_size 10000000 \
-max_delay 500000 -f rtsp $rtsp_stream &

# 获取第二个进程的 PID
PID2=$!

# 定义清理函数，在脚本退出时终止两个进程
cleanup() {
    echo "Terminating processes..."
    kill $PID1
    kill $PID2
}

# 捕捉脚本退出信号（如 Ctrl+C），并调用 cleanup 函数
trap cleanup EXIT

# 等待两个进程结束
wait $PID1 $PID2

