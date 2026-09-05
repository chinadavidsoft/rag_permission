# 权限感知的企业知识库问答系统

这是一个可运行的企业级 RAG 学习项目，覆盖三格式解析、四级 chunking、bge-m3 向量化、Qdrant/BM25 混合检索、RRF、可选精排、服务端权限过滤、引用校验、流式生成、SQLite 反馈闭环和轻量 tracing。

## 快速开始

```bash
uv sync
cp .env.example .env
python -c 'import secrets; print("AUTH_SECRET=" + secrets.token_hex(32))' >> .env
uv run python scripts/create_fixtures.py
uv run pytest -q
uv run uvicorn rag_permission.api:app --host 0.0.0.0 --port 8000
```

模型默认从 `HF_ENDPOINT=https://hf-mirror.com` 下载。`.env` 里配置：

- `LLM_BASE_URL`：OpenAI 兼容 endpoint
- `LLM_API_KEY`：API key
- `LLM_MODEL`：模型名
- `QDRANT_URL`：留空使用内存模式，Docker 里为 `http://qdrant:6333`
- `AUTH_MODE`：默认 `jwt`；仅当 API 位于认证网关后时才可改用 `trusted_header`
- `AUTH_SECRET`：JWT 签名密钥，生产环境必须使用足够长的随机值
- `PROMPT_COST_PER_1K` / `COMPLETION_COST_PER_1K`：按千 token 配置模型单价

打开 `http://localhost:8000/`，在顶部粘贴访问令牌。生成令牌：

```bash
uv run python scripts/create_token.py --user-id public-user --groups public
```

服务端只信任 JWT 中的 `sub` 和 `groups`；客户端传入的旧权限 header 不会覆盖 JWT。若使用 `trusted_header` 模式，API 必须只能由完成身份认证的网关访问。

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
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -d '{"query":"E-1002 是什么故障？"}'

curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ENG_ACCESS_TOKEN" \
  -d '{"query":"劳动法规考试占比多少？"}'
```

第二个请求应返回“未找到权限范围内相关资料”，不泄露 hr 文档是否存在。反馈接口使用返回的 `trace_id` 调用 `POST /feedback`。

## 评估与观测

```bash
make evaluate
make profile
uv run python scripts/compare_judges.py --samples 5 --output reports/judge_comparison.json
```

`make evaluate` 会对比开启/关闭精排，检查 10 条 golden set 和 9 条越权用例。`make profile` 默认使用固定 LLM 输出做分段链路剖面；加 `--real-llm` 后才调用真实模型。`scripts/compare_judges.py` 会多采样两套 judge prompt，保存逐样本分数和 prompt 漂移。
