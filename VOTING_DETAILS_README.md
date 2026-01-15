# 投票详情保存功能说明

## 修改概述

为了分析LLM对每个选项的投票结果,我对代码做了以下修改:

## 修改的文件

### 1. src/approaches.py - SelfConsistencyRefinementApproach类

**修改内容:**
- 在`solve()`方法中新增了`voting_details`列表,记录每次采样的详细信息
- 每次采样都会保存:
  - `sample_id`: 采样编号 (1-5)
  - `selected_options`: 该次采样选择的选项 (如 ["A", "C"])
  - `response`: LLM的完整响应文本
- 将所有投票详情保存到实例属性`self.last_voting_details`,包含:
  - `question_id`: 问题ID
  - `num_samples`: 采样次数
  - `vote_threshold`: 投票阈值
  - `d_option_threshold`: D选项的特殊阈值
  - `voting_details`: 每次采样的详细信息列表
  - `option_votes`: 各选项的投票统计 (如 {"A": 3, "B": 1, "C": 4, "D": 0})
  - `voted_answers`: 经过阈值筛选后的答案
  - `final_answers`: 经过后处理的最终答案

### 2. src/evaluator.py - Evaluator类

**修改内容:**
- 在`__init__()`中新增`self.voting_details`列表用于收集所有问题的投票详情
- 修改`update()`方法:
  - 新增`voting_details`参数
  - 当有投票详情时,将其添加到`self.voting_details`列表
- 修改`save_results()`方法:
  - 在输出的JSON中新增`"voting_details"`字段

### 3. run.py - 主执行脚本

**修改内容:**
- 在处理每个问题后,检查solver是否有`last_voting_details`属性
- 如果有,将其传递给`evaluator.update()`方法

## 使用方法

运行程序后,结果文件(如`results/results_20260115_133742.json`)会包含新的`voting_details`字段:

```json
{
  "approach": "SelfConsistencyRefinementApproach",
  "summary": { ... },
  "error_cases": [ ... ],
  "partial_cases": [ ... ],
  "voting_details": [
    {
      "question_id": "train_0001",
      "num_samples": 5,
      "vote_threshold": 3,
      "d_option_threshold": 4,
      "voting_details": [
        {
          "sample_id": 1,
          "selected_options": ["A", "C"],
          "response": "根据文档分析...Final Answer I Reasoned: A,C"
        },
        {
          "sample_id": 2,
          "selected_options": ["A", "B", "C"],
          "response": "..."
        },
        ...
      ],
      "option_votes": {
        "A": 5,
        "B": 2,
        "C": 4,
        "D": 0
      },
      "voted_answers": ["A", "C"],
      "final_answers": ["A", "C"]
    },
    ...
  ]
}
```

## 数据分析示例

可以使用以下Python代码分析投票详情:

```python
import json

# 读取结果文件
with open('results/results_20260115_133742.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

# 分析投票详情
for detail in results['voting_details']:
    question_id = detail['question_id']
    option_votes = detail['option_votes']
    
    print(f"\n问题 {question_id}:")
    print(f"  投票统计: {option_votes}")
    print(f"  最终答案: {detail['final_answers']}")
    
    # 查看每次采样的选择
    for sample in detail['voting_details']:
        print(f"  采样 {sample['sample_id']}: {sample['selected_options']}")
```

## 分析价值

这些数据可以帮助你:
1. **理解模型不确定性**: 看哪些选项得票分散
2. **发现模型偏好**: 某些选项是否总是被高估或低估
3. **调试阈值设置**: 根据投票分布优化`vote_threshold`
4. **分析错误模式**: 对比错误案例的投票详情,找出共性问题
5. **评估一致性**: 看模型在多次采样中的稳定性

## 注意事项

- 投票详情功能目前只对`SelfConsistencyRefinementApproach`有效
- 如果需要为其他approach添加此功能,可以参考相同的模式进行修改
- 保存完整的response会显著增加结果文件大小,如果文件过大可以考虑只保存`selected_options`
