# Alibaba Tianchi: CCL2025 - Chinese Electronic Medical Record ICD Diagnosis Coding Evaluation

[中文](https://github.com/arnozeng98/icd-prediction/blob/main/README_zh.md)

## Background

In recent years, with the intensification of population aging and the rise in health awareness, the healthcare system faces increasing service pressure. The widespread application of electronic medical records (EMRs) in the process of medical informatization provides new possibilities for addressing this challenge. To achieve standardized management and sharing of medical data, the World Health Organization (WHO) has developed the International Classification of Diseases (ICD). This standard converts tens of thousands of diseases and their combinations into a standardized alphanumeric coding system, laying the foundation for cross-regional and cross-institutional medical data exchange and analysis.

However, manually converting EMR text into ICD codes is not only time-consuming and labor-intensive but also prone to human errors. Developing an automated ICD coding system can improve coding efficiency and consistency while providing more reliable data support for disease research and healthcare management. Against this backdrop, this evaluation constructs a dataset specifically designed to assess automatic ICD coding for Chinese EMRs. The dataset is based on de-identified medical records and includes five primary diagnoses and 53 additional diagnosis ICD (ICD-10) codes, totaling 1,485 data samples.

## Task Description

### 1. Task Overview

- **Objective:** Utilize natural language processing (NLP) techniques to analyze patients' clinical symptoms from EMR text, extract key symptom information, determine the primary diagnosis code, and identify additional diagnosis codes to assist doctors or coders in achieving more accurate ICD coding.
- **Definition:** Given a detailed patient description written in natural language (including chief complaint, current medical history, past medical history, discharge condition, etc.), the model needs to predict the patient's primary diagnosis code and additional diagnosis codes.

### 2. Task Details

This task provides clinical information text as input, and the model must output the corresponding primary and additional diagnosis codes.

- **Input:** A string-type field composed of various clinical information from medical records.
- **Output:** The corresponding primary diagnosis code and additional diagnosis codes.

The possible primary and additional diagnosis codes in this dataset are as follows:

```py
# Primary Diagnosis Codes:
['I10.x00x032', 'I20.000', 'I20.800x007', 'I21.401', 'I50.900x018']

# Additional Diagnosis Codes:
['E04.101', 'E04.102', 'E11.900', 'E14.900x001', 'E72.101', 'E78.500',
 'E87.600', 'I10.x00x023', 'I10.x00x024', 'I10.x00x027', 'I10.x00x028',
 'I10.x00x031', 'I10.x00x032', 'I20.000', 'I25.102', 'I25.103', 'I25.200',
 'I31.800x004', 'I38.x01', 'I48.x01', 'I48.x02', 'I49.100x001', 'I49.100x002',
 'I49.300x001', 'I49.300x002', 'I49.400x002', 'I49.400x003', 'I49.900',
 'I50.900x007', 'I50.900x008', 'I50.900x010', 'I50.900x014', 'I50.900x015',
 'I50.900x016', 'I50.900x018', 'I50.907', 'I63.900', 'I67.200x011',
 'I69.300x002', 'I70.203', 'I70.806', 'J18.900', 'J98.414', 'K76.000',
 'K76.807', 'N19.x00x002', 'N28.101', 'Q24.501', 'R42.x00x004',
 'R91.x00x003', 'Z54.000x033', 'Z95.501', 'Z98.800x612']
```

**Note:** The output format is a list where the primary diagnosis code and additional diagnosis codes are separated by `|`, and additional diagnosis codes are separated by `;`. The primary diagnosis code must be exactly one, while additional diagnosis codes must be one or more.

## Submission Format

The evaluation consists of two leaderboards: A leaderboard (validation set) and B leaderboard (test set).

- A leaderboard evaluation results will be displayed on the Alibaba Cloud Tianchi platform.
- Participants can submit up to **five times per day** on the A leaderboard. Failed submissions do not count toward this limit.

**Example submission format:**

```py
[
    {
        "病案标识":"ZYxxxxxxx", 
        "预测结果":"[Primary Diagnosis Code|Additional Diagnosis Code1;Additional Diagnosis Code2;...]"
    }, 
    {
        "..."
    },
    ...
]
```

## Dataset Description

This evaluation dataset is based on de-identified hospital medical records, consisting of 1,485 samples. The dataset is split into:

- **Training set**: 800 samples
- **Validation set**: 200 samples
- **Test set**: 485 samples (not publicly available)

The dataset is provided in JSON format with the following files:

- `ICD-Coding-train.json`: Training set with labeled data.
- `ICD-Coding-test-A.json`: A leaderboard test set (validation set).
- `ICD-Coding-A.json`: A leaderboard submission example.
- `ICD-Coding-test-B.json`: B leaderboard test set (test set, not publicly available).

### Data Fields

Each record in the dataset is stored in JSON format and contains the following fields:

- Case ID (`病案标识`): The unique patient case identifier in the hospital.
- Chief Complaint (`主诉`): The primary symptom described by the patient during consultation, typically summarized in a short sentence.
- Present Illness History (`现病史`): A detailed description of the patient's current illness, including onset, symptom characteristics, disease progression, past treatments, and response to therapy.
- Past Medical History (`既往史`): The patient’s previous health conditions, major diseases, surgeries, injuries, and allergies.
- Personal History (`个人史`): The patient’s lifestyle habits, occupational exposure, and epidemiological history.
- Marital History (`婚姻史`): Information on marital status, age at marriage, spouse’s health status, and number of children.
- Family History (`家族史`): Family history of hereditary or specific diseases among direct relatives.
- Admission Condition (`入院情况`): A summary of the patient's symptoms, signs, and overall condition at the time of hospital admission.
- Admission Diagnosis (`入院诊断`): The initial diagnosis made by the physician upon hospital admission based on the medical history and tests.
- Treatment Course (`诊疗经过`): Detailed records of the patient’s examinations, treatments, and disease progression during hospitalization.
- Discharge Condition (`出院情况`): A brief description of the patient’s health status at discharge.
- Discharge Instructions (`出院医嘱`): Guidelines provided by the physician regarding medication, follow-up, and lifestyle adjustments after discharge.
- Primary Diagnosis Code (`主诊断编码`): The primary ICD-10 code corresponding to the main diagnosis during hospitalization.
- Additional Diagnosis Codes (`其他诊断编码`): One or more ICD-10 codes corresponding to other diagnoses.

Sample Data (JSON Format):

```json
{
    "病案标识": "ZY020000982397",
    "主诉": "胸闷、喘7天。",
    "现病史": "患者于7天前无明显诱因出现胸闷、喘，呈阵发性，活动及情绪激动后明显加重，不能从事日常活动...",
    "既往史": "有“冠状动脉粥样硬化性心脏病”10余年，2021年于****行“冠状动脉移植术”（具体不详）",
    "个人史": "生长于原籍，否认疫区及地方病流行区长期居住史，生活规律...",
    "婚姻史": "适龄结婚，育有1子，配偶及孩子身体健康。",
    "家族史": "父母已逝，具体不详。否认家族性遗传病及传染病史。",
    "入院情况": "患者老年男性，76岁，因“胸闷、喘7天”入院...",
    "入院诊断": "1.急性失代偿性心力衰竭心功能II级（Killip分级）2.肺炎3.急性呼吸衰竭（I型）",
    "诊疗经过": "入院后完善相关辅助检查，凝血常规：凝血酶原时间：13.1秒...",
    "出院情况": "双侧瞳孔等大等圆，对光反射及调节反射存在...",
    "出院医嘱": "1、低盐低脂饮食，注意休息，避免劳累，按时服药...",
    "主要诊断编码": "J81.x00x002",
    "其他诊断编码": "I50.907; I50.903; I25.103; I20.000; I49.900; I48.x01;E11.900"
}
```

## Evaluation Metric

The competition evaluates ICD coding accuracy using the accuracy (Acc) metric, calculated as:

$$
\text{Acc} = \frac{1}{N} \sum_{i=1}^{N} \lbrace 0.5 \cdot I(\hat{y}_{\text{main}} == y_{\text{main}}) + 0.5 \cdot \frac{\text{NUM}(y_{\text{other}} \cap \hat{y}_{\text{other}})}{\text{NUM}(y_{\text{other}})} \rbrace_{i}
$$

Where $I(\cdot)$ is an indicator function that returns `1` if the condition is met and `0`otherwise. $\hat{y}_{\text{main}}$ and $y_{\text{main}}$ represent the predicted label and true label of the main diagnosis code, respectively. $\text{NUM}(x)$ represents a quantity function that is used to calculate the number of $x$ . $\hat{y}_{\text{other}}$ and $y_{\text{other}}$ represent the predicted label set and true label set of other diagnosis codes, respectively. $N$ is the number of test samples. $\lbrace \cdot \rbrace_i$ is the prediction accuracy of the `i-th` Chinese electronic medical record.

## Baseline Performance

| Accuracy |
|----------|
|  41.34%  |
