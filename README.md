##🎓 UESTC Forum Hybrid Search (校园论坛混合搜索引擎)
告别“搜不到”，拒绝“搜不准”。一个懂语义、更懂校园生活的 AI 搜索引擎。
##🚩 为什么做这个项目？
作为一名学生，在使用学校论坛（清水河畔）检索信息时，我经常遇到以下困扰，这促使我开发了这个工具：
##痛点 1：关键词的局限性
现象：我想找“容易过的课”，搜“水课”能搜到，搜“轻松的课”却搜不到。
问题：原生搜索只能机械匹配字面，无法理解“轻松”和“水”在校园语境下的语义关联。
##痛点 2：长尾问题难以回答
现象：输入“大二下学期挂科了会不会影响保研”，原生搜索往往返回 0 结果。
问题：复杂的自然语言描述无法被传统搜索引擎解析。
##痛点 3：数据获取门槛
现象：需要繁琐的登录验证，且翻页查找效率极低。
##💡 我的解决方案
本项目实现了一个基于混合检索（Hybrid Search）的垂直搜索引擎：
##🧠 语义检索 (Vector Search)：
使用 text2vec 模型将帖子向量化，让搜索引擎“听得懂”人话。
效果：搜“好吃的”，自动关联“火锅”、“烧烤”、“食堂”。
##🎯 关键词检索 (BM25)：
保留传统倒排索引，确保“CS101”、“张三老师”等专有名词不被遗漏。
##⚖️ RRF 融合排序：
使用倒数排名融合算法（Reciprocal Rank Fusion），结合两路检索的优势，显著提升 Top-K 结果的准确率。

## 技术栈

- **语言**: Python 3.9+
- **数据源处理**: requests, BeautifulSoup4
- **向量数据库**: ChromaDB (持久化存储)
- **关键词检索**: rank_bm25 (基于内存的倒排索引)
- **Embedding模型**: SentenceTransformers (shibing624/text2vec-base-chinese)
- **后端API**: FastAPI (提供RESTful接口)
- **前端UI**: Streamlit (极简Web交互)

## 项目结构

```
project_root/
├── data/                    # 数据目录
│   ├── posts_data.json      # 爬虫下来的原始数据
│   └── bm25_index.pkl       # BM25索引文件
├── chroma_db/               # 向量数据库自动生成的文件夹
├── etl_crawler.py           # 1. 爬虫脚本
├── build_index.py           # 2. 索引构建脚本
├── server.py                # 3. 后端服务 (FastAPI)
├── ui.py                    # 4. 前端界面 (Streamlit)
├── requirements.txt         # 依赖库
└── README.md                # 项目说明文档
```

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境（可选）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据爬取 (ETL)

首先需要配置爬虫参数，编辑 `etl_crawler.py` 中的配置部分：

```python
# 配置信息
TARGET_URL = "https://bbs.uestc.edu.cn/forum.php?mod=forumdisplay&fid=xxx"  # 目标版块URL
COOKIE = "your_cookie_string_here"  # 从浏览器复制的Cookie
MAX_PAGES = 50  # 爬取页数
```

然后运行爬虫：

```bash
python etl_crawler.py
```

爬取的数据将保存到 `data/posts_data.json`。

### 3. 构建索引

```bash
python build_index.py
```

该脚本将：
1. 读取 `data/posts_data.json`
2. 构建向量索引（保存到 `chroma_db/`）
3. 构建关键词索引（保存到 `data/bm25_index.pkl`）

### 4. 启动后端服务

```bash
python server.py
```

后端服务将在 `http://localhost:8000` 启动，提供以下API：
- `POST /search` - 执行混合搜索
- `GET /health` - 健康检查
- `GET /stats` - 统计信息

### 5. 启动前端界面

```bash
streamlit run ui.py
```

前端界面将在 `http://localhost:8501` 启动，提供完整的Web交互界面。

## 核心功能详解

### 混合检索流程

1. **向量检索**：使用SentenceTransformers将查询和文档转换为向量，在ChromaDB中进行相似度搜索
2. **关键词检索**：使用jieba分词和BM25算法进行关键词匹配
3. **RRF融合**：使用倒数排名融合算法合并两种检索结果

### RRF算法实现

```python
def _rrf_fusion(self, vector_results, keyword_results, k=60):
    """
    倒数排名融合 (Reciprocal Rank Fusion)
    Score(d) = Σ 1/(k + rank(d))
    """
    scores = {}
    
    # 处理向量检索结果
    for rank, doc in enumerate(vector_results, 1):
        doc_id = doc['id']
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank)
    
    # 处理关键词检索结果
    for rank, doc in enumerate(keyword_results, 1):
        doc_id = doc['id']
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank)
    
    # 按分数排序
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs
```

### 前端功能

- **实时搜索**：输入自然语言查询，获取相关帖子
- **参数调节**：可调节返回结果数量、融合方法等
- **历史记录**：自动保存搜索历史
- **结果展示**：美观的卡片式展示，包含标题、摘要、作者、发布时间
- **直接跳转**：点击标题可直接跳转到论坛原帖

## API接口说明

### POST /search

**请求体**:
```json
{
  "query": "研一新生如何选课",
  "top_k": 20,
  "fusion_method": "rrf"  # 可选: "rrf", "simple", "weighted"
}
```

**响应**:
```json
{
  "status": "success",
  "query": "研一新生如何选课",
  "results": [
    {
      "id": "12345",
      "title": "研一新生选课指南",
      "content": "研一新生选课需要注意...",
      "author": "张三",
      "url": "https://bbs.uestc.edu.cn/forum.php?mod=viewthread&tid=12345",
      "timestamp": "2023-09-01 10:30:00",
      "score": 0.85
    }
  ],
  "stats": {
    "vector_search_time": 0.12,
    "keyword_search_time": 0.08,
    "fusion_time": 0.01,
    "total_time": 0.21
  }
}
```

## 配置说明

### 模型配置

在 `build_index.py` 和 `server.py` 中，可以修改Embedding模型：

```python
# 使用CPU模式，适合无GPU环境
model = SentenceTransformer('shibing624/text2vec-base-chinese', device='cpu')
```

### 搜索参数

在 `ui.py` 中可调节的搜索参数：
- `top_k`：返回结果数量（默认20）
- `fusion_method`：融合方法（rrf/simple/weighted）
- `k_value`：RRF算法的k参数（默认60）

## 性能优化

1. **CPU优化**：所有模型默认使用CPU模式
2. **缓存机制**：Embedding模型和索引加载使用缓存
3. **并发处理**：数据爬取支持并发请求
4. **持久化存储**：索引数据持久化，避免重复构建

## 故障排除

### 常见问题

1. **爬虫失败**：检查Cookie是否过期，网络连接是否正常
2. **模型加载失败**：检查网络连接，确保能访问HuggingFace
3. **内存不足**：减少爬取页数或使用更小的模型
4. **端口冲突**：修改server.py或ui.py中的端口号

### 日志查看

各模块都有详细的日志输出，可通过控制台查看运行状态。

## 扩展开发

### 添加新的融合算法

在 `server.py` 的 `HybridSearchEngine` 类中添加新的融合方法：

```python
def _custom_fusion(self, vector_results, keyword_results):
    # 实现自定义融合算法
    pass
```

### 支持新的数据源

修改 `etl_crawler.py` 中的解析逻辑，适配不同的论坛结构。

### 界面定制

修改 `ui.py` 中的Streamlit组件，定制前端界面。

## 许可证

本项目仅供学习和研究使用。

## 贡献

欢迎提交Issue和Pull Request来改进本项目。

---

**注意**：使用爬虫功能时请遵守目标网站的robots.txt和相关使用条款，尊重网站版权和用户隐私。
