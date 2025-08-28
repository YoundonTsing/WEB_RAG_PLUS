# LightRAG

> **Simple, Fast & Extensible Retrieval-Augmented Generation Framework**
>
> 通过渐进式、非侵入式优化，让 RAG 在 *准确性*、*效率*、*长文档处理* 与 *持续学习* 上全面升级。

---

## ✨ 项目亮点

| 技术 | 作用 | 本仓库实现 |
| --- | --- | --- |
| **Multi-Query Parallelization (RAG-R1)** | 生成多种查询变体并行检索，提高召回率 | `enhancers/multi_query.py` |
| **Adaptive Context Compression (ACC-RAG)** | 根据查询复杂度动态压缩上下文，降低推理延迟 | `enhancers/compression.py` |
| **Hierarchical LongRAG / RAPTOR** | 分层索引 + 分层检索，专治长文档 | `enhancers/long_context.py` |
| **End-to-End Retrieval Learning (OpenRAG)** | 反馈驱动、离线微调，检索器持续进化 | `feedback_store/` & `finetuning/` |

---

## 🖼️ 系统架构

```mermaid
%% Styles
classDef newComponent fill:#1a4d2e,stroke:#2b8a3e,stroke-width:2px,color:#fff;
classDef feedbackLoop fill:#5a189a,stroke:#9d4edd,stroke-width:2px,color:#fff;

graph TD
    subgraph "Ingestion Pipeline (数据注入流程)"
        A[Input Documents] --> B(Chunking / 文档分块)
        B --> C(Entity & Relation Extraction / 实体关系提取)
        C --> D(Embedding / 向量化)
        D --> E[Vector & Graph Storage]
        A_LONG["Long Documents (长文档)"] --> B_LONG("Hierarchical Indexer (分层索引器)<br>LongRAG / RAPTOR")
        B_LONG --> C_LONG["Multi-level Vector Storage (多层向量存储)"]
        class A_LONG,B_LONG,C_LONG newComponent
    end
    
    subgraph "Query Pipeline (查询处理流程)"
        H[User Query] --> H_MULTI("Multi-Query Processor (多查询处理器)<br>RAG-R1")
        H_MULTI --> I{Query Mode Selection}
        I -- KG/Naive Modes --> J("Parallel Retrieval (并行检索)")
        J --> J_FUSE("Result Fusion (结果融合)<br>Reciprocal Rank Fusion")
        I -- Hierarchical Mode --> J_HIERARCHICAL("Hierarchical Retrieval (分层检索)")
        J_FUSE --> L(Reranking / 重排序)
        J_HIERARCHICAL --> L
        L --> L_COMPRESS("Adaptive Context Compressor (自适应上下文压缩)<br>ACC-RAG")
        L_COMPRESS --> M(Context Assembly / 上下文组装)
        M --> N(Prompt Engineering / 提示工程)
        N --> O(LLM Generator / LLM生成器)
        O --> P[Final Answer / 最终答案]
        class H_MULTI,J,J_FUSE,J_HIERARCHICAL,L_COMPRESS newComponent
    end

    subgraph "Continuous Learning Loop (持续学习回路)"
        direction LR
        P --> FB_COLLECT("Feedback Collector (反馈收集器)")
        FB_COLLECT --> FINETUNE("Offline Model Finetuning (离线模型微调)")
        FINETUNE -- "Updates models" --> EMBEDDING_MODEL["Embedding & Reranker Models"]
        class FB_COLLECT,FINETUNE,EMBEDDING_MODEL feedbackLoop
    end

    %% Connections
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

# 安装依赖（示例，以 poetry 为例）
$ poetry install

# 运行示例
$ python examples/basic_usage.py
```

*详细的运行指导和配置说明，请参考 `docs/` 目录。*

---

## 🛠️ 目录结构

```text
LightRAG/
├─ lightrag/              # 核心库
│  ├─ enhancers/          # 新增增强模块
│  │  ├─ multi_query.py
│  │  ├─ compression.py
│  │  └─ long_context.py
│  ├─ kg/                 # 知识图谱 & 存储实现
│  ├─ operate.py          # 检索 & 生成逻辑
│  └─ lightrag.py         # 主入口类
├─ finetuning/            # 离线微调脚本
├─ feedback_store/        # 反馈存储示例
├─ examples/              # 使用示例
└─ README.md              # 当前文件
```

---

## 🛣️ Roadmap

1. **Phase 1**  
   *Multi-Query + Adaptive Compression* — 提升复杂查询召回率与推理效率 ✅
2. **Phase 2**  
   *Hierarchical LongRAG* — 分层索引、分层检索，彻底解决长文档场景 🚧
3. **Phase 3**  
   *Continuous Learning* — 反馈驱动，模型自进化 🔜

欢迎提交 PR 或 issue，共建更强大的 LightRAG！

---

## 🤝 贡献指南

1. Fork & Clone
2. 创建特性分支
3. 提交 PR 并描述变更
4. 通过 CI 检查后合并

---

## 📄 许可证

[MIT](LICENSE)