# 模型介绍

## 数据预处理

```python
# dataset.py
text_fields = ["主诉", "现病史", "既往史"...]  # 11个医疗文本字段
tokenized = tokenizer(txt, padding="max_length", truncation=True...)  # 统一截断/填充到256长度
main_label = main2id.get(...)  # 主要诊断编码转换为ID
other_vec = np.zeros(...)      # 其他诊断编码转换为多标签向量
```

## 模型架构

### 编码层

```python
# model.py
self.encoder = AutoModel.from_pretrained("hfl/chinese-roberta-wwm-ext")  # 中文RoBERTa基础模型
```

- 使用预训练RoBERTa分别编码11个医疗文本字段
- 每个字段独立编码后取[CLS]表征

### 特征融合层

```python
self.fusion = nn.Sequential(
    nn.Linear(11*768, 768),  # 将11x768=8448维特征压缩
    nn.GELU(),               # 高斯误差线性单元激活
    nn.Dropout(0.1)          # 防止过拟合
)
```

- 拼接所有字段的[CLS]特征
- 通过线性层+激活函数实现特征交互

### 多任务输出层

```python
self.main_classifier = nn.Linear(768, 5)   # 主要诊断分类(5类)
self.other_classifier = nn.Linear(768, 49) # 其他诊断多标签分类(49类)
```

- 共享底层特征，分支出两个分类器
- 主任务使用交叉熵损失，副任务使用二元交叉熵

## 训练策略

```python
# train.py
loss = (loss_main + loss_other) / GRAD_ACCUM_STEPS  # 梯度累积
scaler.scale(loss).backward()  # 混合精度训练
```

- 动态早停机制：连续10个epoch验证集综合指标无提升则停止
- 混合精度训练：使用torch.cuda.amp加速计算
- 梯度累积：每4个step更新一次参数，等效batch_size=8

## 评估指标

- 主诊断：准确率(Accuracy)
- 其他诊断：宏平均F1分数
- 综合指标：两者加权平均

## 推理流程

```python
# inference.py
main_preds = torch.argmax(...)  # 取最大概率
other_codes = [OTHER_CODES[idx] for idx... if prob > 0.5]  # 阈值0.5
```
