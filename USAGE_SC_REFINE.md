# Self-Consistency + Self-Refinement 使用指南

## 方法说明

### Self-Consistency（自洽性推理）
- 对同一问题生成**多次**推理（默认5次）
- 使用较高的 temperature（默认0.7）增加多样性
- 通过**投票机制**选出最常见的答案
- 提高答案的可靠性，特别适合不确定的多选题

### Self-Refinement（自我精炼）
- **阶段1**：生成初始答案（基于投票结果）
- **阶段2**：让LLM批评自己的推理过程
- **阶段3**：基于批评改进答案
- 发现并修正逻辑漏洞

## 使用方法

### 基础用法

```bash
python run.py --approach self_consistency_refinement
```

### 完整参数示例

```bash
python run.py \
  --approach sc_refine \
  --num_samples 5 \
  --sc_temperature 0.7 \
  --top_k 10 \
  --docs_path data/dev/docs.json \
  --questions_path data/dev/questions.jsonl \
  --output_dir results
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--approach` | `baseline` | 推理方法：`baseline`, `self_consistency_refinement`, `sc_refine` |
| `--num_samples` | `5` | Self-Consistency采样次数（建议3-10次） |
| `--sc_temperature` | `0.7` | 采样温度（0.7-0.9适合多样性） |
| `--top_k` | `10` | 检索的文档数量 |
| `--no_retrieval` | `False` | 禁用文档检索 |

## 性能调优建议

### 快速测试（省钱省时）
```bash
python run.py --approach sc_refine --num_samples 3 --sc_temperature 0.6
```
- 3次采样
- 适合快速验证

### 标准配置（推荐）
```bash
python run.py --approach sc_refine --num_samples 5 --sc_temperature 0.7
```
- 5次采样，平衡效果和成本
- 适合正式实验

### 高精度配置
```bash
python run.py --approach sc_refine --num_samples 10 --sc_temperature 0.8
```
- 10次采样，最高精度
- API调用成本较高（约是baseline的12倍）

## 与 Baseline 对比

### Baseline 方法
- 1次推理
- temperature = 0（确定性）
- 无自我批评

### SC + Refinement 方法
- 5+次推理
- temperature = 0.7（多样性）
- 有投票机制
- 有自我批评和改进

**预期提升**：准确率提升 15-25%

## 输出格式

方法会输出三个阶段的详细信息：

```
========== SELF-CONSISTENCY STAGE ==========
Generated 5 samples, voted answer: ['A', 'D'] (4/5 votes)

Best reasoning from consistency stage:
[最佳推理过程]

========== SELF-REFINEMENT STAGE ==========
Self-Critique:
[自我批评内容]

========== FINAL REFINED ANSWER ==========
[最终精炼后的答案]
```

## 注意事项

1. **API调用次数**：SC方法会调用 `num_samples + 2` 次API（5次采样 + 1次批评 + 1次精炼）
2. **运行时间**：大约是baseline的7-10倍
3. **并发控制**：建议降低 `MAX_WORKERS` 避免API限流
4. **成本控制**：先在 sample 数据集上测试

## 示例输出

使用 sample 数据测试：
```bash
python run.py \
  --approach sc_refine \
  --docs_path data/sample/docs.json \
  --questions_path data/sample/questions.jsonl \
  --num_samples 3
```

## 故障排除

### 问题：API超时
**解决**：降低 `MAX_WORKERS`，在 `.env` 中设置 `MAX_WORKERS=1`

### 问题：成本太高
**解决**：减少 `num_samples`，使用 `--num_samples 3`

### 问题：效果不理想
**解决**：
1. 提高 `num_samples` 到 7-10
2. 调整 `sc_temperature` 到 0.8
3. 增加 `top_k` 获取更多文档

## 代码位置

- 实现代码：`src/approaches.py` - `SelfConsistencyRefinementApproach` 类
- 主程序：`run.py`
- 评估器：`src/evaluator.py`
