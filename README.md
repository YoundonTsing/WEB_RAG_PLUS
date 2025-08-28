# LightRAG

> **轻量级、可扩展的 RAG 框架**
>
> 通过渐进式、非侵入式优化，实现 *检索准确性*、*推理效率*、*长文档处理* 与 *持续学习* 的全面升级。

---

## ✨ 项目亮点

| 技术 | 作用 | 代码实现 |
| --- | --- | --- |
| **Multi-Query Parallelization（多查询并行检索）** | 生成多种查询变体并行检索，提升召回率 | `enhancers/multi_query.py` |
| **Adaptive Context Compression（自适应上下文压缩）** | 根据查询复杂度动态压缩上下文，降低推理延时 | `enhancers/compression.py` |
| **Hierarchical LongRAG / RAPTOR（分层长文档 RAG）** | 分层索引 + 分层检索，有效处理长文档 | `enhancers/long_context.py` |
| **End-to-End Retrieval Learning（检索器端到端学习）** | 反馈驱动、离线微调，实现检索器持续进化 | `feedback_store/` & `finetuning/` |

---

## 🖼️ 系统架构

```mermaid
%% 样式定义
classDef newComponent fill:#1a4d2e,stroke:#2b8a3e,stroke-width:2px,color:#fff;
classDef feedbackLoop fill:#5a189a,stroke:#9d4edd,stroke-width:2px,color:#fff;

graph TD
    subgraph "Ingestion Pipeline (数据注入流程)"
        A[输入文档] --> B(Chunking / 文档分块)
        B --> C(Entity & Relation Extraction / 实体关系提取)
        C --> D(Embedding / 向量化)
        D --> E[Vector & Graph Storage]
        A_LONG["长文档"] --> B_LONG("Hierarchical Indexer (分层索引器)<br>LongRAG / RAPTOR")
        B_LONG --> C_LONG["Multi-level Vector Storage (多层向量存储)"]
        class A_LONG,B_LONG,C_LONG newComponent
    end
    
    subgraph "Query Pipeline (查询处理流程)"
        H[用户查询] --> H_MULTI("Multi-Query Processor (多查询处理器)<br>RAG-R1")
        H_MULTI --> I{查询模式选择}
        I -- KG/Naive --> J("Parallel Retrieval (并行检索)")
        J --> J_FUSE("Result Fusion (结果融合)<br>Reciprocal Rank Fusion")
        I -- Hierarchical --> J_HIERARCHICAL("Hierarchical Retrieval (分层检索)")
        J_FUSE --> L(Reranking / 重排序)
        J_HIERARCHICAL --> L
        L --> L_COMPRESS("Adaptive Context Compressor (自适应上下文压缩)<br>ACC-RAG")
        L_COMPRESS --> M(Context Assembly / 上下文组装)
        M --> N(Prompt Engineering / 提示工程)
        N --> O(LLM Generator / LLM 生成器)
        O --> P[最终答案]
        class H_MULTI,J,J_FUSE,J_HIERARCHICAL,L_COMPRESS newComponent
    end

    subgraph "Continuous Learning Loop (持续学习回路)"
        direction LR
        P --> FB_COLLECT("Feedback Collector (反馈收集器)")
        FB_COLLECT --> FINETUNE("Offline Model Finetuning (离线模型微调)")
        FINETUNE -- "更新模型" --> EMBEDDING_MODEL["Embedding & Reranker Models"]
        class FB_COLLECT,FINETUNE,EMBEDDING_MODEL feedbackLoop
    end

    %% 连接
    E --> J
    C_LONG --> J_HIERARCHICAL
    EMBEDDING_MODEL --> J
    EMBEDDING_MODEL --> L
```

---

## 🚀 快速开始

```bash
# 克隆仓库
$ git clone https://github.com/yourname/LightRAG.git && cd LightRAG

# 安装依赖（以 Poetry 为例）
$ poetry install

# 运行示例
$ python examples/basic_usage.py
```

*更多配置与使用说明，请参考 `docs/` 目录。*

---

## 🗂️ 目录结构

```text
LightRAG/
├─ lightrag/              # 核心库
│  ├─ enhancers/          # 增强模块
│  │  ├─ multi_query.py
│  │  ├─ compression.py
│  │  └─ long_context.py
│  ├─ kg/                 # 知识图谱/存储
│  ├─ operate.py          # 检索与生成逻辑
│  └─ lightrag.py         # 主入口类
├─ finetuning/            # 离线微调脚本
├─ feedback_store/        # 反馈存储
├─ examples/              # 使用示例
└─ README_zh.md           # 中文说明
```

---

## 🛣️ Roadmap

1. **Phase 1**：多查询 + 自适应压缩 —— 提升召回率、降低成本 ✅
2. **Phase 2**：分层长文档 RAG —— 彻底解决长文档场景 🚧
3. **Phase 3**：持续学习 —— 反馈驱动，检索器自进化 🔜

欢迎提出 Issue / PR，共同完善！

---

## 🤝 贡献

1. Fork & Clone 本仓库
2. 创建功能分支
3. 提交 PR 并描述变更
4. 通过 CI 检查后合并

---

## 📄 许可证

项目遵循 [MIT License](LICENSE)。
