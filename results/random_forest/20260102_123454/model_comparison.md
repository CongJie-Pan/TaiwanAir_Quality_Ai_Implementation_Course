# Model Comparison: Decision Tree vs Random Forest

## Test Set Performance

| Metric | Decision Tree | Random Forest | Improvement |
|--------|---------------|---------------|-------------|
| **Accuracy** | 0.8211 | 0.8470 | +2.59% |
| **Precision (Macro)** | 0.6355 | 0.6671 | +3.17% |
| **Recall (Macro)** | 0.7374 | 0.7462 | +0.88% |
| **F1 (Macro)** | 0.6663 | 0.6922 | +2.59% |

## Model Configuration

| Parameter | Decision Tree | Random Forest |
|-----------|---------------|---------------|
| **max_depth** | 5 | 10 |
| **n_estimators** | 1 | 100 |
| **class_weight** | balanced | balanced |

## OOB Score (Random Forest Only)

Random Forest OOB Score: **0.7936**

> OOB (Out-of-Bag) Score 是隨機森林特有的交叉驗證指標，利用未被抽樣的資料進行驗證，
> 提供模型泛化能力的無偏估計。

## Conclusion

隨機森林模型 (Accuracy=0.8470) 優於決策樹模型 (Accuracy=0.8211)，提升了 2.59%。
