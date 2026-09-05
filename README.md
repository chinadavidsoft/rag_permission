# 权限感知的企业知识库问答系统

这是一个可运行的企业级 RAG 学习项目，覆盖三格式解析、四级 chunking、bge-m3 向量化、Qdrant/BM25 混合检索、RRF、可选精排、服务端权限过滤、引用校验、流式生成、SQLite 反馈闭环和轻量 tracing。

## 快速开始

```bash
uv sync
cp .env.example .env
uv run python scripts/create_fixtures.py
uv run pytest -q
uv run uvicorn rag_permission.api:app --host 0.0.0.0 --port 8000
```

模型默认从 `HF_ENDPOINT=https://hf-mirror.com` 下载。`.env` 里配置：

- `LLM_BASE_URL`：OpenAI 兼容 endpoint
- `LLM_API_KEY`：API key
- `LLM_MODEL`：模型名
- `QDRANT_URL`：留空使用内存模式，Docker 里为 `http://qdrant:6333`

打开 `http://localhost:8000/`。演示页面顶部可填 `X-User-Id` 和 `X-User-Groups`；生产环境必须由网关或 JWT 在服务端注入身份，绝不能让客户端自由选择权限组。

## Docker Compose

```bash
cp .env.example .env
make up
make logs
make down
```

Compose 会启动 Qdrant 和 API。API 等待 Qdrant healthcheck 后启动，启动时预载模型并摄入三份不同权限 fixture。

## 验收命令

```bash
curl -N -X POST http://localhost:8000/ask-stream \
  -H "Content-Type: application/json" \
  -H "X-User-Id: public-user" -H "X-User-Groups: public" \
  -d '{"query":"E-1002 是什么故障？"}'

curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-User-Id: eng-user" -H "X-User-Groups: eng" \
  -d '{"query":"劳动法规考试占比多少？"}'
```

第二个请求应返回“未找到权限范围内相关资料”，不泄露 hr 文档是否存在。反馈接口使用返回的 `trace_id` 调用 `POST /feedback`。
