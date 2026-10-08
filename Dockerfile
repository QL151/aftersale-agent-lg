# ============================================================
# 售后业务 Agent —— 应用镜像
#
# 构建：docker build -t aftersale-agent .
# 运行：docker compose up -d
# ============================================================
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先装依赖，利用镜像层缓存（改代码不用重装依赖）
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir redis "psycopg[binary]" langgraph langfuse || true
# 说明：上面这一行装了"生产可选依赖"。若你的网络装不了，去掉即可 ——
# 应用会自动走等价降级实现（见 app/runtime.py 的 doctor()）。

COPY app/ ./app/
COPY mcp_server/ ./mcp_server/
COPY eval/ ./eval/
COPY tests/ ./tests/
COPY run_cli.py run_api.py ./

# 数据目录（Trace / 检查点 / 会话落盘）
RUN mkdir -p data/traces data/state

# 非 root 运行
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8077

# 健康检查：能力体检接口（同时能看出哪些依赖真接了）
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8077/health',timeout=4).status==200 else 1)"

CMD ["python", "run_api.py", "--host", "0.0.0.0", "--port", "8077"]
