#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大数据分析系统 - 一键启动入口 (E2E Integration)
架构: Producer -> ML/LLM Worker -> DuckDB/Parquet -> FastAPI -> ECharts
用法: python run_app.py [--pipeline] [--port 8000]
"""

import argparse
import os
import sys
import signal
import subprocess
import time
import threading
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_app")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 自动检测 venv Python，优先使用
VENV_PYTHON = os.path.join(BASE_DIR, "venv", "Scripts", "python.exe")
if os.path.exists(VENV_PYTHON):
    PYTHON_EXE = VENV_PYTHON
else:
    PYTHON_EXE = PYTHON_EXE


def load_env():
    """加载 .env 环境变量"""
    env_file = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_file):
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
            logger.info(".env 环境变量加载成功")
        except ImportError:
            logger.info("python-dotenv 未安装，跳过 .env 加载")
    else:
        logger.info("未找到 .env 文件，LLM 功能将不可用")


def check_api_key():
    """检查 API Key 是否存在"""
    api_key = os.getenv("SILICONFLOW_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        logger.warning("API_KEY 未设置，LLM 功能将优雅降级（/api/system-status 可查看状态）")
    return bool(api_key)


def start_fastapi(port=8000, reload=False):
    """启动 FastAPI 服务器（子进程）"""
    logger.info(f"启动 FastAPI 服务器 -> http://127.0.0.1:{port}")
    cmd = [
        PYTHON_EXE, "-m", "uvicorn",
        "api.server:app",
        "--host", "0.0.0.0",
        "--port", str(port),
    ]
    if reload:
        cmd.append("--reload")

    try:
        process = subprocess.Popen(cmd, cwd=BASE_DIR)
        return process
    except Exception as e:
        logger.error(f"FastAPI 启动失败: {e}")
        return None


def start_pipeline(qps=100, batch_size=10, duration=None):
    """启动数据管线 Producer + Consumer（子进程）"""
    logger.info(f"启动数据管线 QPS={qps} batch={batch_size}")
    script = os.path.join(BASE_DIR, "pipeline", "run_pipeline.py")
    if not os.path.exists(script):
        logger.error(f"管线脚本不存在: {script}")
        return None

    cmd = [PYTHON_EXE, script, "--qps", str(qps), "--batch_size", str(batch_size)]
    if duration:
        cmd.extend(["--duration", str(duration)])

    try:
        process = subprocess.Popen(cmd, cwd=BASE_DIR)
        return process
    except Exception as e:
        logger.error(f"管线启动失败: {e}")
        return None


def open_browser(port=8000):
    """自动打开浏览器"""
    url = f"http://127.0.0.1:{port}"
    try:
        import webbrowser
        time.sleep(1.0)
        webbrowser.open(url)
        logger.info(f"浏览器已打开: {url}")
    except Exception:
        logger.info(f"请手动访问: {url}")


def main():
    parser = argparse.ArgumentParser(description="大数据分析系统 - 一键启动")
    parser.add_argument("--pipeline", action="store_true", help="同时启动数据管线")
    parser.add_argument("--port", type=int, default=8000, help="Web 服务端口（默认8000）")
    parser.add_argument("--qps", type=int, default=100, help="生产者速率")
    parser.add_argument("--batch-size", type=int, default=10, help="批处理大小")
    parser.add_argument("--duration", type=int, default=None, help="管线运行时长（秒）")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--reload", action="store_true", help="开启 uvicorn 热重载")
    args = parser.parse_args()

    print("=" * 56)
    print("  大数据分析系统 - E2E 一键启动")
    print("  Producer -> ML/LLM -> DuckDB/Parquet -> FastAPI -> ECharts")
    print("=" * 56)

    load_env()
    check_api_key()

    processes = []

    def shutdown(signum=None, frame=None):
        print("\n")
        logger.info("收到停止信号，优雅关闭中...")
        for p in processes:
            if p and p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    p.kill()
        logger.info("所有服务已停止")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # 启动管线（可选）
    if args.pipeline:
        proc = start_pipeline(args.qps, args.batch_size, args.duration)
        if proc:
            processes.append(proc)
            time.sleep(1.0)

    # 启动 FastAPI
    api_proc = start_fastapi(args.port, args.reload)
    if api_proc:
        processes.append(api_proc)
        time.sleep(2.0)

    # 打开浏览器
    if not args.no_browser:
        threading.Thread(target=open_browser, args=(args.port,), daemon=True).start()

    logger.info(f"系统就绪 -> http://127.0.0.1:{args.port}")
    logger.info("按 Ctrl+C 停止所有服务")

    # 主循环
    try:
        while True:
            time.sleep(0.5)
            for p in processes:
                if p and p.poll() is not None:
                    logger.warning(f"子进程退出 (code={p.returncode})，正在关闭...")
                    shutdown()
    except KeyboardInterrupt:
        shutdown()


if __name__ == "__main__":
    main()
