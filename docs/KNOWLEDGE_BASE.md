# rag-permission 项目知识库

权限感知的企业知识库 RAG 系统。核心目标:不同用户只能检索到自己权限范围内的文档,答案必须带可校验的引用,越权时优雅拒答而不是暴露文档是否存在。

## 项目概览

| 项目 | 说明 |
| --- | --- |
| Python | 3.12,src layout(`src/rag_permission/`) |
| 包管理 | uv |
| 向量库 | Qdrant(`:memory:` 或 server 模式),COSINE 距离 |
| Embedding | BAAI/bge-m3,1024 维,归一化 |
| Reranker | BAAI/bge-reranker-v2-m3(cross-encoder,可开关) |
| LLM | OpenAI 兼容接口(默认 DeepSeek),支持流式 |
| 存储 | SQLite(反馈闭环) |
| API | FastAPI + SSE 流式 |
| 测试 | pytest,52 个用例 |

## 架构与数据流

```
用户请求(JWT Bearer token)
    |
    v
API 层(api.py)──鉴权中间件→ User{id, groups}
    |                          (groups 只来自 JWT,请求体无法注入)
    v
RAGService(runtime.py)
    |
    ├── 检索 HybridRetriever.search()
    |       ├── embed(query)           bge-m3,1024 维
    |       ├── dense 路               Qdrant query_points + ACL pre-filter
    |       ├── bm25 路                进程内 BM25 + ACL post-filter
    |       ├── rrf_fuse(k=60)        按 chunk_id 聚合两路排名
    |       └── [rerank]              cross-encoder 精排(可配置关闭)
    |
    ├── 生成 generate_answer()
    |       ├── build_prompt           编号注入 [1][2]... 资料
    |       ├── llm.complete / stream  system 要求只用资料、断言挂 [n]
    |       └── display_citations      正则抽 [n],越界→拒答
    |
    └── 落库 FeedbackStore.record_trace()
            trace_id 主键,answer/citations/retrieved_ids/usage/cost
            (反馈通过 POST /feedback 按 trace_id 异步关联)
```

## 目录结构

```
src/rag_permission/
├── models.py              # User, ParsedDocument, DocumentChunk, SearchHit
├── config.py              # Pydantic-Settings,读 .env
├── auth.py                # JWT 签发/校验(HS256,secret >=32 bytes)
├── llm.py                 # OpenAILLMClient:complete() + stream()
├── embeddings.py          # BGEEmbedding 单例 + cache-aside
├── vector_store.py        # QdrantVectorStore(COSINE, uuid5, ACL filter)
├── bm25.py                # 进程内 BM25Index(k1=1.5, b=0.75)
├── retriever.py           # HybridRetriever + rrf_fuse
├── reranker.py            # BGEReranker(cross-encoder)
├── citations.py           # CitationChecker(引用校验/幻觉检测)
├── generation.py          # generate_answer / stream_answer / prompt
├── feedback.py            # SQLite trace + 反馈 + 周报 + 成本
├── observability.py       # Span/Tracer/percentile/token_cost
├── runtime.py             # RAGService(编排) + build_runtime(组装)
├── api.py                 # FastAPI 路由 + 鉴权中间件
├── ingest/
|   ├── parsers.py         # .md / .docx / .pdf 解析
|   └── chunking.py        # 四种切分策略
├── ingest_pipeline.py     # 解析→切分→embed→upsert→BM25 入索引
├── evaluation/
|   ├── metrics.py         # recall / precision / MRR 纯函数
|   ├── runner.py          # GoldenCase + EvaluationRunner + 汇总
|   ├── judges.py          # LLM 裁判(faithfulness / relevance)
|   └── judge_experiment.py
└── static/index.html      # 前端单页(fetch 流式读 SSE)
```

脚本(`scripts/`):

| 脚本 | 用途 |
| --- | --- |
| `create_fixtures.py` | 生成三份不同权限的样例文档(md/docx/pdf) |
| `create_token.py` | 生成本地测试 JWT |
| `smoke_test.py` | 冒烟测试:LLM 非空 + embedder 1024 维 |
| `demo_ingest.py` | 演示四种 chunking 策略输出 |
| `demo_retrieval.py` | 演示 hybrid 检索(dense/bm25/fused 三列对照) |
| `demo_evaluation.py` | 演示单条评估指标计算 |
| `demo_feedback.py` | 演示 trace 落库 → 差评 → 周报 |
| `demo_generation.py` | 演示生成 + 引用校验 |
| `demo_runtime.py` | 演示 RAGService 全流程 |
| `evaluate_retrieval.py` | 批量评估(rerank on/off 对比),输出 JSON 报告 |
| `profile_runtime.py` | 延迟/成本剖析(p50/p95) |
| `compare_judges.py` | 对比 LLM 裁判 vs 宽松裁判 |

## 核心数据模型

### User

```python
User(id="u1", groups=frozenset({"public", "eng"}))
user.can_access(("public",))  # bool,集合交集 OR 语义
```

`groups` 是权限的唯一来源。空 groups 的用户检索必不命中(返回空结果,不是 None)。

### ParsedDocument

解析产物。`doc_id = sha1(文件内容)[:12]` 前缀 `doc-`,由内容决定而非文件名,保证幂等。

```python
ParsedDocument(doc_id, title, source, acl_groups, elements)
ParsedElement(kind, text, section_path, locator)
  # kind: "heading" | "paragraph" | "table_row"
  # section_path: 元组,如 ("通用故障码",),由标题层级拼接
```

### DocumentChunk

```python
DocumentChunk(chunk_id, doc_id, text, chunk_type, section_path,
              source, title, acl_groups, locator,
              parent_id, parent_text, metadata)
```

- `chunk_id` 格式:`{doc_id}:{strategy}:{序号}`,parent_child 下为 `{parent_id}:child:{n}`
- `acl_groups` 是复制而非引用,防止修改
- `payload` 属性产出 12 字段 dict,直接作为 Qdrant payload(含 chunk_id/doc_id/text/acl_groups/parent_text 等)

### SearchHit

```python
SearchHit(chunk, score, dense_rank=None, bm25_rank=None, rerank_score=None)
```

RRF 融合后带两路排名;rerank 后带原始 rerank_score(不 normalize)和 prior_score。

## 权限模型(灵魂)

三层防线,任何一层都能挡住越权:

1. **入口**:`AskRequest` 只有 `query` 字段,无 `groups`。`User` 只能来自 JWT 解码(`auth_mode=jwt`,默认)或可信网关注入的 header(`auth_mode=trusted_header`)。
2. **检索**:
   - Dense 路:`acl_filter(user) → Filter(must=[FieldCondition(key="acl_groups", match=MatchAny(...))])` 作为 Qdrant query_filter,是 pre-filter,越权向量根本不参与排序。
   - BM25 路:先按 `limit * oversample` 检索,再按 `frozenset(chunk.acl_groups) & user.groups` post-filter 兜底。
   - 两路都要求 `user.groups` 非空,空 groups 直接返回 `[]`。
3. **语义**:越权时统一回答"未找到权限范围内相关资料",不说"无权限",不泄露文档是否存在。

## 摄入管线

`IngestionPipeline.ingest(path, acl_groups)` 一步完成:

1. **解析**(按扩展名分发):
   - `.md`:标题层级构建 section_path,表格按行抽成 `列 | 值` 短句
   - `.docx`:遍历 `body.iterchildren()` 按原始顺序处理段落和表格(不丢序)
   - `.pdf`:每页 `extract_tables()` 抽表格为行短句 + `extract_text()` 按空行分段
2. **切分**(四种策略):
   - `fixed`:滑动窗口,chunk_size + overlap
   - `recursive`:分隔符优先级回退(`\n\n` → `\n` → `。` → `；` → `，` → `.`),带 overlap 合并
   - `section`:连续同 section 分组,绝不跨节
   - `parent_child`(默认):parents 大块(1000 字)+ children 小块(260 字),child 带 parent_id 和 parent_text,`child.text in parent.text` 恒成立
3. **Embedding**:`BGEEmbedding.encode()`,cache-aside 模式(sha1(text) 做 key,misses 收集后一次批量 encode 再回填缓存,第二次全命中)
4. **入库**:
   - Qdrant:point_id = `uuid5(NAMESPACE_URL, "rag-permission:{chunk_id}")`,确定性,重跑不堆积
   - BM25:`BM25Index.add_documents()` 去重后入索引

`doc_id` 由内容 sha1 决定,同一文件重跑产出相同 doc_id 和 chunk_id,upsert 幂等覆盖,不会堆积。

## 检索管线

`HybridRetriever.search(query, user)` 一函数串完:

1. `embedding.encode([query])[0]` → 1024 维向量
2. Dense:`vector_store.search(vector, dense_top_k, user.groups)`,ACL pre-filter
3. BM25:`bm25.search(query, bm25_top_k, user.groups)`,ACL post-filter
4. `rrf_fuse(dense, bm25, k=60)`:按 chunk_id 聚合 `1/(60+rank)` 两路分数,记录 `dense_rank` / `bm25_rank`
5. `[reranker.rerank(query, candidates)]`:cross-encoder 批量 predict `[(q, h.text)]`,按 rerank_score 降序(可配置关闭)
6. 返回 top-k(final_top_k=8)

### BM25 实现细节

- k1=1.5, b=0.75;公式含 query 去重(Counter)、idf 用 `.get` 防缺词
- **tokenize 先正则分段**:`[A-Za-z][A-Za-z0-9_-]*|[A-Za-z0-9]+|[\u4e00-\u9fff]+`,CJK 段走 jieba,字母数字段整段小写。jieba 直切整段会把 `E-1002` 切碎,这是 tokenizer 先分段的原因。
- 检索量按 `limit * 4` 超采样再过滤,保证 post-filter 后仍有足够结果

## 生成与引用校验

### Prompt

```
SYSTEM: 你是一个企业知识库助手。只使用用户消息中编号提供的资料回答问题。
每个实质性事实都必须紧跟对应资料编号,例如 [1] 或 [2]。不要使用外部知识。
如果资料不足或没有相关资料,只回答:"未找到权限范围内相关资料。"
```

User prompt 结构:`[n] 标题:...\n章节:...\n内容:{parent_text or text}`,最后附问题。

### CitationChecker

三件事:
1. 正则 `\[(\d+)\]` 抽取引用编号,去重保序
2. 编号在 1..k 范围内合法建 `Citation`,越界进 `invalid_citations`(fabricated)
3. 实质断言零引用且非拒答 → `suspected_hallucination=True`

拒答边界:回答含"未找到/没有足够/无法回答/资料不足/无法确定"关键词时,零引用不算幻觉。

`display_citations()`:检测到幻觉时强制降级为拒答(丢弃原回答),只保留报告记录。流式场景在 `AnswerCompleteEvent` 中用 `final_answer` 替换。

## 反馈闭环

SQLite(`data/rag_permission.db`),`check_same_thread=False` + `threading.Lock` 保证跨线程安全。

两张表:

- `qa_traces`:trace_id(PK)、created_at、user_id、query、answer、retrieved_ids(JSON)、citations(JSON)、prompt/completion tokens、三级 cost
- `feedback`:trace_id(PK/FK)、rating(up/down)、comment、created_at

`record_trace` 与 `save_feedback` 分离:先有 ask 落 trace,用户事后点 👍👎 按 trace_id 关联。

`weekly_failure_report()`:按 ISO 周聚合差评,doc_id 取 `retrieved_ids[0].split(":", 1)[0]`,同条差评内同 doc 去重,输出 `FailureCluster(week_start, doc_id, title, count, queries)`。

## 可观测性

自实现轻量 tracing,不依赖外部 APM:

```python
InMemoryTracer()
with tracer.span("llm_generate", trace_id, {"stream": True}):
    ...
tracer.latency_percentiles()  # {"llm_generate": {"p50_ms": ..., "p95_ms": ...}}
```

- 埋点:`retrieve`(属性含 user_id)、`embed/dense/bm25/rerank`、`generate`、`llm_generate`(属性含 stream)
- `nearest_rank_percentile`:手写最近邻法,`rank = ceil(p/100 * n)`
- `token_cost`:prompt/completion 分别按千 token 单价换算
- `RAGService` 每次请求自动落 trace 并累计成本到 SQLite

## 评估体系

### 检索指标(纯函数)

| 函数 | 语义 |
| --- | --- |
| `recall_at_k` | 命中 relevant 比例;空 expected → 0.0 哨兵 |
| `precision_at_k` | 分母 `min(k, len(dedup retrieved))` |
| `reciprocal_rank` | 按原始顺序找第一个命中,返回 1/rank |
| `mean_reciprocal_rank` | 多 query 平均 |

### 评估运行器

```python
runner = EvaluationRunner(retriever, k=8)
results = runner.run(cases)           # 依赖注入,不绑死实现
summary = summarize_evaluations(results)
# 空-expected 条目计入 case_count,均值剔除
```

`RetrievalEvaluation` 带 `leakage` 字段:检索结果中出现 `forbidden_chunk_ids` 中的 ID 即为越权泄漏。`authorization_set.json` 专测权限:空 groups 用户问所有文档应零命中零泄漏。

### 生成层裁判

| 裁判 | 方法 |
| --- | --- |
| `LLMJudge.faithfulness` | 拆原子断言 + adversarial prompt(JSON 输出)+ temp=0;拒答且有零引用直接 1.0 |
| `LLMJudge.answer_relevance` | 三档:relevant=1.0 / partially_relevant=0.5 / irrelevant=0.0 |
| `NaiveLooseJudge` | 宽松对照:词表重叠打分,校验 LLM 裁判是否过严 |

`scripts/evaluate_retrieval.py` 跑 rerank on/off 两套检索器对比,输出逐条 + 均值 + target_moves(哪些 query 因 rerank 提升)到 `reports/retrieval_evaluation.json`。

## API 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 前端单页 |
| GET | `/health` | 健康检查 |
| POST | `/ask` | 同步问答,返回 answer/citations/report/usage/cost/trace_id |
| POST | `/ask-stream` | SSE 流式:token 事件先流,citations 事件带完整引用+trace_id |
| POST | `/feedback` | 提交 👍👎,按 trace_id 关联 |

鉴权:`Authorization: Bearer <jwt>`,中间件解码出 `User` 注入 `request.state`。JWT payload 含 `sub`(user_id)和 `groups`,HS256 签名,默认 60 分钟过期。

生成测试令牌:

```bash
uv run python scripts/create_token.py --user-id alice --groups public,eng
```

### SSE 事件格式

```
event: token
data: {"content": "E-1002 表示"}

event: token
data: {"content": "冷却风扇故障"}

event: usage
data: {"prompt_tokens": 180, "completion_tokens": 48}

event: citations
data: {"answer": "完整答案 [1]", "citations": [...], "citation_report": {...}, "usage": {...}, "trace_id": "..."}
```

前端用 `fetch` + `ReadableStream` 手动解析(不用 EventSource,因为 POST + 自定义 header),逐 token 渲染,末尾显示引用卡片和 👍👎 按钮。

## 配置(.env)

```bash
LLM_BASE_URL=https://api.deepseek.com   # OpenAI 兼容 endpoint
LLM_API_KEY=sk-...                       # 必填
LLM_MODEL=deepseek-v4-flash              # 模型名
QDRANT_URL=                              # 空=内存模式;http://qdrant:6333=server
SQLITE_PATH=./data/rag_permission.db
ENABLE_RERANK=true                       # 精排开关
HF_ENDPOINT=https://hf-mirror.com        # HuggingFace 镜像
AUTH_MODE=jwt                            # jwt | trusted_header
AUTH_SECRET=<随机 32+ 字节>               # JWT 密钥
AUTH_TOKEN_TTL_MINUTES=60
PROMPT_COST_PER_1K=0                     # 千 token 单价
COMPLETION_COST_PER_1K=0
```

其他调参(有默认值,一般不改):`chunk_size=260`、`chunk_overlap=48`、`parent_chunk_size=1000`、`ingestion_strategy=parent_child`、`dense_top_k=24`、`bm25_top_k=24`、`final_top_k=8`。

## 测试 Fixture

三份文档,三种权限,三个格式:

| 文件 | 格式 | 权限 | 内容 |
| --- | --- | --- | --- |
| `sample.md` | Markdown | public | 设备手册:E-1001/1002/1003 故障码 + 500h 保养 |
| `sample.docx` | Word | eng | 发动机手册:E001/E002 故障码 + 维护记录 |
| `sample.pdf` | PDF | hr | 人力考试大纲:三科目权重表 + 24 个月有效期 |

`golden_set.json`:10 条标注(含 query/user_groups/relevant/forbidden),最后一条是权限哨兵(空 groups,expected 空,应零命中)。

`authorization_set.json`:专项权限测试集。

## 部署(Docker Compose)

```bash
cp .env.example .env
make up          # 起 Qdrant + API(Qdrant healthcheck 通过后才启动)
make logs
make down
```

- API 容器基于 `python:3.12-slim` + uv,依赖层和源码层分离
- Qdrant 数据持久化到 named volume `qdrant_storage`,重启不丢
- HF 模型缓存挂载 `huggingface_cache`,避免重复下载
- SQLite 挂载 `app_data`
- compose 里保留注释掉的 TEI(Text Embeddings Inference)服务作为可选替代

## 关键设计决策

1. **doc_id 由内容哈希决定** → 幂等重跑,不堆积垃圾 point
2. **ACL pre-filter(dense)+ post-filter(BM25)双保险** → 越权数据不进排序,不靠事后拦截
3. **空 groups 必不命中** → Qdrant filter `must=[]` 不等价于无 filter;BM25 显式检查
4. **拒答统一文案"未找到权限范围内相关资料"** → 不区分"没文档"和"没权限",防止侧信道探测
5. **parent_text 用于生成** → child 命中后送 LLM 的是完整 parent 上下文,信息不截断
6. **引用幻觉强制降级拒答** → 宁可拒答不给,不出无依据断言
7. **tokenize 先正则分段再 jieba** → 保护 `E-1002` 这类混合 code 不被切碎
8. **cache-aside embedding** → 同文本二次 encode 零计算开销
9. **trace 与 feedback 分离** → 交互异步自然解耦,trace_id 桥接
10. **评估指标全部纯函数** → 边界条件可独立测试,runner 依赖注入可换实现

## 常见问题

**Q: 换 embedding 模型怎么办?**
改 `.env` 的 `embedding_model`,删掉 Qdrant collection 重新摄入。向量维度不匹配会导致 collection 创建报错。

**Q: 为什么 public 用户查不到发动机文档?**
`sample.docx` 标了 `("eng",)` 权限,public 不在 ACL 集合里,两条检索路都过滤掉了。这是预期行为。

**Q: rerank 关了会怎样?**
`ENABLE_RERANK=false` 后走 RRF 排序。RRF 分数在 0~2/61 之间(两路都命中时最高),可作为对比基线。`evaluate_retrieval.py` 的 rerank_target_moves 列出哪些 query 因 rerank 提升。

**Q: SQLite 会不会有并发问题?**
`check_same_thread=False` 允许跨线程使用连接,配合 `threading.Lock` 串行化写操作。当前单进程部署下足够;多进程需换 PostgreSQL 或文件锁。

**Q: Qdrant `:memory:` 模式数据会丢吗?**
会,进程退出即清空。开发测试用 `:memory:` 方便;生产必须配置 `QDRANT_URL` 指向 server 并使用 compose volume。
