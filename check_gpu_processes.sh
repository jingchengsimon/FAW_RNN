#!/bin/bash
# GPU进程查找和管理脚本

echo "=== GPU使用情况 ==="
nvidia-smi

echo ""
echo "=== 占用GPU的进程详情 ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | while IFS=',' read -r pid name memory; do
    if [ ! -z "$pid" ]; then
        echo "PID: $pid"
        echo "  进程名: $name"
        echo "  GPU内存: $memory"
        echo "  完整命令:"
        ps -p "$pid" -o cmd --no-headers 2>/dev/null | sed 's/^/    /'
        echo "  工作目录:"
        pwdx "$pid" 2>/dev/null | sed 's/^/    /'
        echo ""
    fi
done

echo "=== 查找特定GPU设备的进程 ==="
echo "GPU 0 的进程:"
fuser -v /dev/nvidia0 2>/dev/null || echo "  无进程使用 /dev/nvidia0"
echo ""
echo "GPU 1 的进程:"
fuser -v /dev/nvidia1 2>/dev/null || echo "  无进程使用 /dev/nvidia1"

echo ""
echo "=== 如果要终止进程，可以使用: ==="
echo "  kill <PID>          # 正常终止"
echo "  kill -9 <PID>       # 强制终止"
echo ""
echo "例如终止PID 3622231: kill 3622231"
