#!/bin/bash
# 大数据分析系统 - 一键启动脚本
# 用法: bash start.sh [--pipeline]

cd "$(dirname "$0")"

echo "=========================================="
echo "  大数据分析系统"
echo "  E2E: Producer -> ML/LLM -> DuckDB/Parquet -> FastAPI -> ECharts"
echo "=========================================="

# 检查 Python
PYTHON=""
for py in python3 python; do
    if command -v "$py" &> /dev/null; then
        PYTHON="$py"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "错误: 未找到 Python，请安装 Python 3.9+"
    exit 1
fi

echo "Python: $($PYTHON --version)"

# 检查核心依赖
echo "检查依赖..."
$PYTHON -c "import fastapi, uvicorn, pandas" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "正在安装依赖..."
    $PYTHON -m pip install -r requirements.txt -q
    echo "依赖安装完成"
fi

# 启动
echo "启动服务..."
$PYTHON run_app.py "$@"
