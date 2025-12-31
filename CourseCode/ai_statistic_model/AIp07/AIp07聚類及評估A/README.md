# AIp07 聚類及評估 (Clustering and Evaluation)

**課程教材** - AI Python 實作第七單元
**作者**: Jia-Sheng Heh
**日期**: 10/23/2024

## 📚 課程概覽

本單元為人工智慧實務課程的第七部分，主要涵蓋**非監督式學習(Unsupervised Learning)**中的**聚類分析(Clustering Analysis)**及其評估方法。課程從基礎理論到實務應用，逐步深入聚類演算法及其品質評估。

## 🎯 學習目標

1. 理解AI與機器學習的基本概念，特別是監督式與非監督式學習的區別
2. 掌握常見的聚類演算法：k-means, 層次聚類, DBSCAN
3. 能夠將聚類技術應用於實務問題（客戶分群分析）
4. 學會使用多種評估指標衡量聚類品質

## 📁 目錄結構

```
AIp07聚類及評估A/
├── README.md                              # 本文件 - 課程導覽
├── A_AI基礎與數據準備/
│   └── A_AI_ML_Basics_IrisData.py         # 第一部分：AI基礎概念與Iris數據
├── B_聚類演算法/
│   └── B_Clustering_Algorithms.py         # 第二部分：常見聚類演算法
├── C_實務應用/
│   └── C_Customer_Clustering_Practice.py  # 第三部分：客戶聚類實務
├── D_聚類評估/
│   └── D_Clustering_Evaluation.py         # 第四部分：聚類評估方法
├── data/
│   └── XXX.csv                            # 客戶交易數據（需自行準備）
└── AIp07聚類及評估A.py                     # 原始完整教材（保留參考）
```

## 📖 課程內容

### A. AI基礎與數據準備 (`A_AI_ML_Basics_IrisData.py`)
**學習重點**:
- AI、機器學習、深度學習的關係
- 監督式學習 vs. 非監督式學習
- KDD (Knowledge Discovery in Databases) 流程
- Iris（鳶尾花）數據集介紹
- 數據視覺化基礎函數

**執行時間**: ~5-10分鐘
**前置需求**: 無

---

### B. 聚類演算法 (`B_Clustering_Algorithms.py`)
**學習重點**:
- 聚類的基本概念與類型
- **k-means 聚類**: 劃分型聚類的代表
- **層次式聚類 (Hierarchical Clustering)**: 凝聚法與分裂法
- **DBSCAN**: 基於密度的聚類
- **Scipy層次聚類**: linkage與fcluster應用
- 各演算法在Iris數據上的實作與比較

**執行時間**: ~15-20分鐘
**前置需求**: 完成 A 部分

---

### C. 實務應用 (`C_Customer_Clustering_Practice.py`)
**學習重點**:
- 零售業客戶價值分析 (RFM模型)
- 交易數據 → 客戶數據框轉換
- 客戶-品類矩陣 (CP_matrix) 建構
- 階層式聚類在客戶分群的應用
- 聚類特徵分析與商業解讀
- 二次聚類細分策略

**執行時間**: ~20-30分鐘
**前置需求**:
- 完成 A, B 部分
- 需要 `XXX.csv` 數據文件

**數據需求**:
- 交易數據檔案: `XXX.csv` (84,008筆交易記錄)
- 欄位: invoiceNo, datetime, channel, customer, product, category, price, cost, quantity, amount

---

### D. 聚類評估 (`D_Clustering_Evaluation.py`)
**學習重點**:
- **內部評估方法**:
  - Silhouette Score (輪廓係數)
  - Davies-Bouldin Index (戴維斯-鮑丁指數)
  - Dunn Index (鄧恩指數)
  - 變異數分析 (Within/Between Cluster Variance)
- **外部評估方法**:
  - Adjusted Rand Index (ARI)
  - Normalized Mutual Information (NMI)
  - Homogeneity & Completeness
  - V-measure
- 混淆矩陣 (Confusion Matrix)
- F-ratio與Elbow Method

**執行時間**: ~15-20分鐘
**前置需求**: 完成 A, B, C 部分

---

## 🚀 使用說明

### 執行順序（推薦）

1. **第一步**: 閱讀本 README.md，了解課程架構
2. **第二步**: 執行 `A_AI基礎與數據準備/A_AI_ML_Basics_IrisData.py`
   ```bash
   python3 AIp07/AIp07聚類及評估A/A_AI基礎與數據準備/A_AI_ML_Basics_IrisData.py
   ```
3. **第三步**: 執行 `B_聚類演算法/B_Clustering_Algorithms.py`
   ```bash
   python3 AIp07/AIp07聚類及評估A/B_聚類演算法/B_Clustering_Algorithms.py
   ```
4. **第四步**: 準備數據後，執行 `C_實務應用/C_Customer_Clustering_Practice.py`
   ```bash
   python3 AIp07/AIp07聚類及評估A/C_實務應用/C_Customer_Clustering_Practice.py
   ```
5. **第五步**: 執行 `D_聚類評估/D_Clustering_Evaluation.py`
   ```bash
   python3 AIp07/AIp07聚類及評估A/D_聚類評估/D_Clustering_Evaluation.py
   ```

### 使用 Spyder IDE

所有檔案都支援 Spyder 的 `#%%` 區塊執行功能：
- 使用 `Ctrl+Enter` 或 `Shift+Enter` 逐區塊執行
- 方便觀察每個步驟的輸出結果

## 🔧 環境需求

### Python 版本
- Python 3.12.3 或以上

### 必要套件
```bash
pip install numpy pandas matplotlib scikit-learn scipy
```

### 完整套件清單
- `numpy` - 數值計算
- `pandas` - 數據處理
- `matplotlib` - 視覺化
- `scikit-learn` - 機器學習演算法
- `scipy` - 科學計算（層次聚類）

## 📊 數據說明

### Iris 數據集
- **來源**: scikit-learn內建數據集
- **規模**: 150筆資料，4個特徵
- **類別**: 3種鳶尾花品種
- **用途**: A, B, D 部分的演算法示範

### XXX.csv 客戶交易數據
- **來源**: 需自行準備或使用課程提供數據
- **規模**: 84,008筆交易記錄
- **欄位**: invoiceNo, datetime, channel, customer, product, category, price, cost, quantity, amount
- **用途**: C, D 部分的實務應用

## 💡 學習建議

1. **循序漸進**: 按照 A → B → C → D 的順序學習
2. **動手實作**: 執行每個區塊，觀察輸出結果
3. **理解原理**: 不只看結果，要理解為何選擇該演算法
4. **比較分析**: 比較不同聚類方法的優缺點
5. **實務思考**: 思考如何將技術應用到實際商業問題

## 📝 注意事項

1. **工作目錄**: 執行前請確認工作目錄設定正確
   ```python
   import os
   wkDir = "AIp07\\AIp07聚類及評估A"
   os.chdir(wkDir)
   ```

2. **路徑設定**:
   - Windows系統使用反斜線 `\\` 或正斜線 `/`
   - 根據實際路徑調整 `wkDir` 變數

3. **中文字型**: C部分的視覺化需要中文字型
   ```python
   font_path = 'C:/Users/jsheh/Desktop/newWorking/RDsys/RDSgpt/微軟正黑體-1.ttf'
   ```
   請根據系統調整字型路徑

4. **數據檔案**: C部分需要 `XXX.csv`，請確保數據檔案存在

## 🔗 相關課程

- **AIp06**: 數據框轉換（前置知識）
- **AIp08**: 關聯規則分析（延伸學習）
- **AIp09**: 迴歸分析（監督式學習）
- **AIp10**: 分類分析（監督式學習）

## 📖 參考資料

- [殷] 殷建平等，《數據挖掘：概念與技術》
- [Chollet] François Chollet, "Deep Learning with Python"
- UCI Machine Learning Repository: Iris Dataset

---

**祝學習順利！有問題歡迎討論。**
