# SongDetect / 歌曲识别项目

> Bilingual README (中文 / English)

## 1) Project Overview / 项目概述

**中文**  
本项目基于 `MLEnd Hums and Whistles II` 数据集（800条音频），完成一个8分类音频识别任务：输入一段哼唱（hum）或口哨（whistle）音频，输出对应歌曲类别。项目包含完整流程：数据分析、特征工程、模型训练、集成学习、测试评估，以及最终推理函数封装。

**English**  
This project uses the `MLEnd Hums and Whistles II` dataset (800 audio clips) to solve an 8-class audio classification task: given a humming (`hum`) or whistling (`whistle`) clip, predict the corresponding song class. The notebook covers a full pipeline: EDA, feature engineering, model training, ensemble learning, test evaluation, and final inference function packaging.

---

## 2) Dataset / 数据集

**中文**  
数据来源：MLEnd Hums and Whistles II（sample_800）  
每首歌100条样本，共8首歌，共800条WAV音频。

下载链接：
- https://github.com/thekmannn/MLEndHW_QHM5703_Sample/raw/main/MLEndHWII_sample_400.zip
- https://github.com/thekmannn/MLEndHW_QHM5703_Sample/raw/main/MLEndHWII_sample_800.zip

**English**  
Data source: MLEnd Hums and Whistles II (`sample_800`).  
Each song has 100 clips, with 8 songs in total (800 WAV files).

Download links:
- https://github.com/thekmannn/MLEndHW_QHM5703_Sample/raw/main/MLEndHWII_sample_400.zip
- https://github.com/thekmannn/MLEndHW_QHM5703_Sample/raw/main/MLEndHWII_sample_800.zip

---

## 3) Task Definition / 任务定义

**中文**
- 输入：音频（统一裁剪到10秒）
- 采样率：统一为22050 Hz
- 输出：8个歌曲类别之一
- 问题类型：监督学习，多分类
- 随机基线：12.5%

**English**
- Input: audio clip (uniformly cropped to 10 seconds)
- Sampling rate: 22050 Hz
- Output: one of 8 song classes
- Task type: supervised multi-class classification
- Random baseline: 12.5%

---

## 4) Methodology / 方法流程

### 4.1 Preprocessing / 预处理

**中文**
- 音频长度标准化为10秒，避免输入长度不一致造成偏差。
- 降采样至22050 Hz，降低计算成本并保留主要旋律信息。
- 按歌曲分层划分：每首歌 90 训练 + 10 测试（总计 720/80）。

**English**
- Standardize audio length to 10 seconds to avoid length-induced bias.
- Downsample to 22050 Hz for efficiency while preserving melody cues.
- Song-stratified split: 90 train + 10 test per song (total 720/80).

### 4.2 Feature Engineering (26-dim) / 特征工程（26维）

**中文**
- 基础声学特征（4维）：功率、音高均值、音高标准差、有声帧比例
- MFCC（13维）
- 音高轮廓特征（6维）：max/min/median、变化率均值/标准差、趋势斜率
- 节奏与谱粗糙度特征（3维）：过零率均值/标准差、节奏强度

**English**
- Basic audio features (4): power, pitch mean, pitch std, voiced fraction
- MFCC features (13)
- Pitch contour features (6): max/min/median, diff mean/std, trend slope
- Rhythm & roughness features (3): ZCR mean/std, rhythm strength

### 4.3 Models / 模型

**中文**
基模型：
- Logistic Regression
- SVM
- KNN
- Random Forest
- Naive Bayes

调参与验证：
- 5折分层交叉验证（StratifiedKFold）
- 以 Macro-F1 为核心指标
- 参数搜索采用整数/对数空间二分策略

**English**
Base models:
- Logistic Regression
- SVM
- KNN
- Random Forest
- Naive Bayes

Tuning & validation:
- 5-fold stratified cross-validation (StratifiedKFold)
- Macro-F1 as the primary metric
- Binary-style search in integer/log parameter spaces

### 4.4 Ensemble Learning / 集成学习

**中文**
- Hard Voting（等权、F1加权）
- Soft Voting（等权、F1加权）
- Stacking（Logistic Regression 作为元学习器）

**English**
- Hard Voting (equal-weight, F1-weighted)
- Soft Voting (equal-weight, F1-weighted)
- Stacking (Logistic Regression as meta-learner)

### 4.5 Deep Learning Branch / 深度学习分支

**中文**
- 输入构建：将音频统一重采样并截断后，转换为时频图表示（以 CQT/频谱图为核心），再缩放到统一尺寸（如 256×256），并扩展为 3 通道张量，便于 CNN 训练。
- 数据组织：按歌曲标签构建监督学习数据集，并保持与主实验一致的分层划分思想（训练/验证分布尽量一致）。
- 基线模型：先建立标准 CNN 基线（多层卷积块 + BatchNorm + ReLU + MaxPooling + 全连接分类头），使用 Adam 优化器训练。
- 结构优化：在基线之上引入模拟退火（Simulated Annealing）进行离散超参数搜索，重点搜索卷积层数、基础通道数、全连接层宽度等结构参数，提升模型容量与泛化平衡。
- 训练策略：结合分层交叉验证与 EarlyStopping，减少过拟合并提高评估稳定性。
- 最终结果：深度学习分支最终达到 **52.5% 准确率（Accuracy = 0.525）**，显著高于传统 ML-only 基线。

**English**
- Input construction: audio is consistently resampled/cropped, converted into time-frequency representations (mainly CQT/spectrogram), resized to a fixed image size (e.g., 256×256), and expanded to 3-channel tensors for CNN training.
- Dataset setup: supervised labels are aligned with song classes, while data splitting follows the same stratified idea used in the ML branch to keep class distribution stable.
- Baseline network: a standard CNN is first built (stacked convolution blocks + BatchNorm + ReLU + MaxPooling + dense classification head) and trained with Adam.
- Architecture optimization: simulated annealing is then used to search discrete structural hyperparameters, including number of convolution blocks, base filter width, and dense-layer size.
- Training control: stratified validation and EarlyStopping are used to improve generalization and reduce overfitting risk.
- Final performance: the deep-learning branch reaches **52.5% test accuracy (Accuracy = 0.525)**, outperforming traditional ML-only baselines.

---

## 5) Evaluation / 评估方式

**中文**
- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Confusion Matrix
- Classification Report

**English**
- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Confusion Matrix
- Classification Report

---

## 6) Key Findings / 关键结论

**中文**
- 集成学习整体优于多数单模型。
- 从测试集混淆矩阵可见，部分类别之间仍有较强混淆。
- Notebook结论显示：纯ML方法指标约在0.3量级；DL方法约在0.5量级；进一步融合ML+DL后有额外提升（约10%量级，具体以实际运行输出为准）。

**English**
- Ensemble methods generally outperform most single models.
- The test confusion matrix shows noticeable confusion among several classes.
- According to notebook conclusions: ML-only methods are around ~0.3 level, DL methods around ~0.5 level, and ML+DL fusion yields additional gains (~10% level, depending on actual runs).

---

## 7) Environment & Dependencies / 环境与依赖

**中文**
建议 Python 3.9+，主要依赖：
- numpy
- pandas
- matplotlib
- seaborn
- librosa
- tqdm
- scikit-learn
- scipy
- spkit
- （深度学习部分）tensorflow / keras

**English**
Recommended Python 3.9+, main packages:
- numpy
- pandas
- matplotlib
- seaborn
- librosa
- tqdm
- scikit-learn
- scipy
- spkit
- (for DL part) tensorflow / keras

Install example:

```bash
pip install numpy pandas matplotlib seaborn librosa tqdm scikit-learn scipy spkit
```

---

## 8) How to Run / 运行说明

**中文**
1. 下载并解压 `MLEndHWII_sample_800` 到本地。  
2. 打开 `Songdetect.ipynb`，检查数据路径是否正确。  
3. 按顺序运行：数据读取 → 特征提取 → 数据划分 → 模型训练 → 集成评估 → 最终测试。  
4. 使用推理函数（如 `predict_song_final`）对新音频路径进行预测。

**English**
1. Download and extract `MLEndHWII_sample_800`.  
2. Open `Songdetect.ipynb` and verify dataset paths.  
3. Run cells in order: loading → feature extraction → split → training → ensemble evaluation → final testing.  
4. Use inference utilities (e.g., `predict_song_final`) to classify new audio files.

---

## 9) Limitations & Future Work / 局限与改进

**中文**
- 样本规模有限，类别相似旋律导致混淆。  
- 参与者个体差异（音域、节奏、表现方式）较大。  
- 可继续尝试数据增强、更强时频特征、端到端音频模型与更稳健融合策略。

**English**
- Limited dataset size and melody similarity cause class confusion.  
- Large inter-speaker variability (pitch range, rhythm, style) remains challenging.  
- Future directions: augmentation, stronger time-frequency representations, end-to-end audio models, and more robust fusion strategies.

