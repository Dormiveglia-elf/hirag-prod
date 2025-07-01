# 🔧 Knowledge Graph Extraction Optimization Fix

## 问题描述
HiRAG系统在处理文档时存在重复的知识图谱提取，导致处理时间翻倍：
- 第一次调用 `entity_extractor.entity()` 提取实体和关系
- 第二次调用 `entity_extractor.relation()` 重复提取相同内容

## 修复方案

### ✅ 核心改进
1. **统一提取**: 创建 `extract_knowledge_graph()` 方法，一次性提取实体和关系
2. **智能缓存**: 添加缓存机制，避免重复计算
3. **优化流程**: HiRAG处理流程改为单次知识图谱提取

### ✅ 修改文件
- `src/hirag_prod/entity/vanilla.py` - 增强VanillaEntity类
- `src/hirag_prod/hirag.py` - 优化处理流程

### ✅ 性能提升
- **~50%时间减少**: 消除重复提取
- **API成本降低**: 减少LLM调用次数
- **内存优化**: 缓存共享结果

## 使用方法

### 运行测试
```bash
python test_kg_optimization.py
```

### 检查日志
处理文档时查看以下日志确认优化生效：
```
🔍 Extracting knowledge graph from X pending chunks...
[KG-Cache] Cached results for X chunks
[KG-Cache] Using cached results for X chunks
```

### API保持不变
```python
# 现有代码无需修改，内部已优化
entities = await entity_extractor.entity(chunks)
relations = await entity_extractor.relation(chunks)

# 或直接使用统一方法
entities, relations = await entity_extractor.extract_knowledge_graph(chunks)
```

## 验证修复

对比修复前后的日志：

**修复前** (存在重复):
```
19:03:56 [KG] Finished extracting knowledge graph for chunk ..., entities: 12, relations: 7
19:08:07 [KG] Finished extracting knowledge graph for chunk ..., entities: 13, relations: 7  # 重复！
```

**修复后** (单次提取):
```
19:03:56 [KG] Finished extracting knowledge graph for chunk ..., entities: 12, relations: 7
[KG-Cache] Cached results for 43 chunks
✅ Knowledge graph extraction completed in 237.562s  # 只有一次
```

## 技术细节

详细文档: `KNOWLEDGE_GRAPH_OPTIMIZATION.md`

**关键特性**:
- 会话级缓存 (每个文档开始时清除)
- 向后兼容 (保留legacy方法)
- 优雅的错误处理
- 可配置的缓存控制

---
*修复完成 ✨ 现在系统处理效率提升50%，无重复提取问题* 