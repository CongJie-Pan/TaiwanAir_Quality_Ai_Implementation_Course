# 期末報告工作清單

**課程**: 人工智慧實務
**組員**: 黎彥德、張家睿、潘驄杰
**繳交日期**: 2026/01/07
**最後更新**: 2026/01/01

> ⚠️ **補充要求**：不一定要有模型評估，但**至少要有 2-4 個模型解讀**
> 重點在於解釋模型結果的意義，而非僅呈現準確度數字

---

## Phase 1: 數據準備與特徵工程

### [x] **Task ID**: FR-001
- **Task Name**: 數據前處理與特徵工程
- **Work Description**:
    - Why: 為三個 AI 模型準備乾淨且標準化的輸入數據
    - How: 載入多年數據，產生季節、AQI 等級等衍生特徵，處理缺失值，分割訓練/測試集
- **Resources Required**:
    - Materials: `pandas`, `numpy`, `sklearn.model_selection`
    - Personnel: 潘驄杰
    - Reference Codes/docs: `docs/finalReport/spec/01-problem-and-data.md`
- **Deliverables**:
    - [x] 數據載入與清洗腳本完成
    - [x] 特徵工程函式 (季節、AQI 等級、風速等級)
    - [x] 訓練集/驗證集/測試集分割完成 (多年訓練方案)
    - [x] 多年訓練資料預處理與存檔
- **Testing Plan**:
    - Unit Test: 驗證特徵衍生邏輯正確 ✅ (24/24 passed)
    - 資料驗證: 確認無缺失值、資料型別正確 ✅
- **Dependencies**: 現有數據後端 (`src/main/python/utils/data_loader.py`)
- **Constraints**: 使用多年資料避免時間序列洩漏
- **Completion Status**: ✅ 已完成 (2026/01/01)
- **Notes**: 特徵工程需與期中報告的分類標準一致（風速等級、AQI 等級）
- **Complete Summary**: 
    - 建立 `models/` 模組，包含 `__init__.py` 與 `data_preprocessing.py`
    - 實作特徵工程函式：季節標籤 (12個月→4季)、AQI 等級 (≤50良好/≤100普通/>100不健康)、風速等級 (≤1.5無風/≤3.4輕風/>3.4微風以上)
    - 資料清洗：移除 NaN 與 AQI=-1 哨兵值，訓練集清理 7.10%，測試年清理 8.12%
    - **多年訓練方案**（防止時間序列洩漏）：
        - 訓練集：2017-2022 年（1,494,466 筆，6年×12月完整季節學習）
        - 驗證集：2023 年上半年（123,359 筆）
        - 測試集：2023 年下半年（123,359 筆）
    - 縣市篩選：延續期中報告使用 New Taipei City、Changhua County、Kaohsiung City
    - 特徵欄位：windspeed, month, hour, season_encoded, county_encoded
    - 資料存檔：`data/processed/model_used/cleaned/`
        - `train_2017_2022.parquet` (5.5 MB)
        - `val_2023_h1.parquet` (0.5 MB)
        - `test_2023_h2.parquet` (0.4 MB)
    - 新增函式：`save_multiyear_splits()`, `load_training_splits()`, `splits_exist()`
    - 單元測試：24 個測試案例全數通過 (pytest) 

### [x] **Task ID**: FR-001-B
- **Task Name**: 修正特徵洩漏問題（Lag Features 方案）
- **Work Description**:
    - Why: 用 $t$ 時刻的污染物預測 $t$ 時刻的 AQI 是 Data Leakage（循環論證）
    - Why Not Remove: 完全移除污染物會丟失最強預測因子（污染物具有自相關性/慣性）
    - How: 使用「滯後特徵 (Lag Features)」—— 用 $t-1$ 的污染物預測 $t$ 時刻的 AQI
- **Resources Required**:
    - Materials: `data_preprocessing.py`, `pandas.DataFrame.shift()`
    - Personnel: 潘驄杰
- **Deliverables**:
    - [x] 新增 `create_lag_features()` 函式
    - [x] 產生滯後特徵：`pm2.5_lag1`, `pm10_lag1`, `o3_lag1`
    - [x] 處理 lag 產生的 NaN（移除 shift 產生的空值，約 8.96%）
    - [x] 更新 `prepare_*_multiyear()` 使用新特徵列表
    - [x] 最終特徵：`pm2.5_lag1`, `pm10_lag1`, `o3_lag1`, `windspeed`, `month`, `hour`, `season_encoded`, `county_encoded`
- **Testing Plan**:
    - 驗證 lag 特徵正確偏移（$t$ 列的 `pm2.5_lag1` 等於 $t-1$ 列的 `pm2.5`）✅
    - 驗證無 NaN 殘留 ✅
    - 單元測試：31 個測試案例全數通過 (含 8 個新增 Lag Features 測試)
- **Dependencies**: FR-001
- **Constraints**: 需確保時間排序正確後才做 shift
- **Completion Status**: ✅ 已完成 (2026/01/01)
- **Priority**: P0
- **Complete Summary**:
    - 新增 `create_lag_features()` 函式，使用 `groupby().shift()` 產生滯後特徵
    - 新增 `get_lag_feature_names()` 輔助函式
    - 更新 `MODEL_COLS` 常數，加入 lag 特徵欄位
    - 更新 `prepare_regression_data_multiyear()` 與 `prepare_classification_data_multiyear()` 預設使用 lag 特徵
    - 更新 `save_multiyear_splits()` 加入 lag 特徵產生步驟
    - 新增 8 個單元測試 (`TestLagFeatures` 類別)
    - 重新產生 parquet 檔案：
        - `train_2017_2022.parquet` (1,365,839 rows, 14.0 MB)
        - `val_2023_h1.parquet` (118,963 rows, 1.2 MB)
        - `test_2023_h2.parquet` (118,963 rows, 1.2 MB)

### [x] **Task ID**: FR-001-C
- **Task Name**: 季節編碼改進
- **Work Description**:
    - Why: LabelEncoder 會讓模型誤認為季節有順序關係（如冬季=0 < 夏季=1）
    - How: 改用 One-Hot Encoding，產生 4 個二元欄位 (season_春/夏/秋/冬)
- **Resources Required**:
    - Materials: `pandas.get_dummies()` 或 `sklearn.preprocessing.OneHotEncoder`
    - Personnel: 潘驄杰
- **Deliverables**:
    - [x] 新增 `encode_onehot()` 函式
    - [x] 更新特徵欄位文件
- **Testing Plan**:
    - 驗證產出 4 個 season_* 欄位 ✅
- **Dependencies**: FR-001
- **Constraints**: 需確保編碼與既有資料相容
- **Completion Status**: ✅ 已完成 (2026/01/01)
- **Priority**: P1
- **Complete Summary**:
    - 實作 `encode_season_onehot()` 函式，手動建立 `season_spring/summer/autumn/winter` 4 個二元欄位 (使用英文避免編碼問題)
    - 移除原本的 `season_encoded` (Label Encoding)
    - 更新 `MODEL_COLS` 加入新特徵
    - 更新單元測試 `TestEncodeSeasonOneHot` 驗證編碼正確性
    - 重新產生訓練資料：`train_2017_2022.parquet`, `val_2023_h1.parquet`, `test_2023_h2.parquet`
    - 驗證資料：確認包含 `season_spring` 等欄位且數值正確 (0/1)

### [x] **Task ID**: FR-001-D
- **Task Name**: 類別不平衡檢查（觀察與記錄）
- **Work Description**:
    - Why: aqi_level 三類別可能分佈不均（如良好佔60%，不健康佔5%）
    - **Phase 1 職責**：觀察並記錄類別分佈（不刪除資料，這是真實現象）
    - **Phase 3/4 職責**：訓練時使用 `class_weight='balanced'` 或 SMOTE 處理不平衡
- **Resources Required**:
    - Materials: `pandas.value_counts()`
    - Personnel: 潘驄杰
- **Deliverables**:
    - [x] 新增 `check_class_distribution()` 函式（僅觀察用）
    - [x] 輸出類別分佈報告：數量、百分比、是否不平衡
- **Testing Plan**:
    - 確認訓練/驗證/測試集類別分佈一致 ✅
    - 單元測試驗證函式輸出格式 ✅ (9 passed)
- **Dependencies**: FR-001
- **Constraints**: 此 Task 只做觀察，不做處理（處理在 FR-003/FR-004）
- **Completion Status**: ✅ 已完成 (2026/01/01)
- **Priority**: P2
- **Complete Summary**:
    - 新增 `check_class_distribution()` 函式，輸出視覺化分佈報告
    - **訓練集** (2017-2022): 良好 47.9% / 普通 40.1% / 不健康 12.0% → ✅ 平衡
    - **測試集** (2023 H2): 良好 69.7% / 普通 27.7% / 不健康 **2.6%** → ⚠️ 嚴重不平衡
    - 不平衡比例：26.9:1 (良好 vs 對敏感族群不健康)
    - **結論**: Phase 3/4 必須使用 `class_weight='balanced'` 處理測試集不平衡問題
    - 新增 9 個單元測試 (`TestCheckClassDistribution` 類別)

### [x] **Task ID**: FR-001-E
- **Task Name**: 特徵標準化選項
- **Work Description**:
    - Why: 線性迴歸係數比較需要特徵有相同尺度（windspeed: 0-15, month: 1-12）
    - How: 加入 StandardScaler 選項，讓係數可直接比較重要性
- **Resources Required**:
    - Materials: `sklearn.preprocessing.StandardScaler`
    - Personnel: 潘驄杰
- **Deliverables**:
    - [x] 在 `prepare_regression_data_multiyear()` 加入 `standardize` 參數
    - [x] 新增 `standardize_features()` 函式
    - [ ] 標準化後係數與原始係數對照表（待 FR-002 實作時產出）
- **Testing Plan**:
    - 驗證標準化後特徵均值≈0、標準差≈1 ✅ (6 tests passed)
- **Dependencies**: FR-001
- **Constraints**: 僅對線性迴歸有影響，決策樹類模型不需要
- **Completion Status**: ✅ 已完成 (2026/01/01)
- **Priority**: P3
- **Complete Summary**:
    - 新增 `standardize_features()` 函式於 `data_preprocessing.py`
    - 使用 `sklearn.preprocessing.StandardScaler` 標準化數值特徵
    - Scaler 只在訓練集上 fit，驗證/測試集只做 transform（避免資料洩漏）
    - 修改 `prepare_regression_data_multiyear()` 新增 `standardize: bool = False` 參數
    - 回傳 8 元素 tuple，新增 `scaler` 物件供後續使用
    - 新增 6 個單元測試 (`TestStandardizeFeatures` 類別)

---

## Phase 2: 線性回歸模型實作

### [ ] **Task ID**: FR-002
- **Task Name**: 線性回歸模型開發
- **Work Description**:
    - Why: 量化風速、季節等因素對 AQI 的影響，驗證期中「AQI 隨風速增加而下降」的發現
    - How: 使用 scikit-learn 建立多元線性回歸模型，輸出係數表與評估指標
- **Resources Required**:
    - Materials: `sklearn.linear_model.LinearRegression`, `sklearn.metrics`
    - Personnel: 潘驄杰
    - Reference Codes/docs: 
        - `docs/finalReport/spec/02-models.md` (2.2 節)
        - `CourseCode/ai_statistic_model/AIp09/AIp09時序與回歸A.py`
- **Deliverables**:
    - [ ] `src/main/python/models/linear_regression.py` 模型類別
    - [ ] 訓練與預測函式
    - [ ] 係數表輸出（含 β 值解釋）
    - [ ] 評估指標：R², RMSE, MAE
- **Testing Plan**:
    - Unit Test: 驗證模型可正常訓練與預測
    - 評估驗證: R² > 0.5 表示模型有效
- **Dependencies**: FR-001 (數據準備)
- **Constraints**: 需確保係數可解釋（β₁ 對應風速影響量）
- **Class Imbalance**: 不適用（回歸任務預測連續數值 AQI，無類別概念）
- **Completion Status**: 未開始
- **Notes**: 課程來源 AIp09，重點產出為係數表與預測 vs 實際散點圖

---

## Phase 3: 決策樹模型實作

### [ ] **Task ID**: FR-003
- **Task Name**: 決策樹分類模型開發
- **Work Description**:
    - Why: 自動學習 AQI 等級分類規則，產生可視化決策樹圖對應期中的交叉表發現
    - How: 使用 scikit-learn 建立決策樹分類器，限制深度以確保可視化效果
- **Resources Required**:
    - Materials: `sklearn.tree.DecisionTreeClassifier`, `sklearn.tree.plot_tree`
    - Personnel: 潘驄杰
    - Reference Codes/docs:
        - `docs/finalReport/spec/02-models.md` (2.3 節)
        - `CourseCode/ai_statistic_model/AIp10/AIp10決策與分類A.py`
- **Deliverables**:
    - [ ] `src/main/python/models/decision_tree.py` 模型類別
    - [ ] 決策樹可視化圖（可放入 PPT）
    - [ ] 混淆矩陣 (Confusion Matrix)
    - [ ] 特徵重要性排名
- **Testing Plan**:
    - Unit Test: 驗證分類預測輸出正確類別
    - 評估驗證: Accuracy > 75%
- **Dependencies**: FR-001 (數據準備)
- **Constraints**: max_depth=5 限制深度，確保規則樹圖清晰可讀
- **Class Imbalance**: 使用 `class_weight='balanced'` 處理類別不平衡（依 FR-001-D 觀察結果決定）
- **Completion Status**: 未開始
- **Notes**: 課程來源 AIp10，決策樹圖為期末報告亮點

---

## Phase 4: 隨機森林模型實作

### [ ] **Task ID**: FR-004
- **Task Name**: 隨機森林分類模型開發
- **Work Description**:
    - Why: 提高分類準確度，產生更可靠的特徵重要性排名
    - How: 使用 scikit-learn 建立隨機森林模型，與決策樹進行準確度比較
- **Resources Required**:
    - Materials: `sklearn.ensemble.RandomForestClassifier`
    - Personnel: 潘驄杰
    - Reference Codes/docs:
        - `docs/finalReport/spec/02-models.md` (2.4 節)
        - `CourseCode/ai_statistic_model/AIp10/AIp10決策與分類A.py`
- **Deliverables**:
    - [ ] `src/main/python/models/random_forest.py` 模型類別
    - [ ] 特徵重要性條形圖
    - [ ] 與決策樹的準確度比較表
    - [ ] OOB 分數輸出
- **Testing Plan**:
    - Unit Test: 驗證模型訓練與預測正常
    - 評估驗證: Accuracy > 80% (預期高於決策樹)
- **Dependencies**: FR-001 (數據準備), FR-003 (用於比較)
- **Constraints**: n_estimators=100, max_depth=10
- **Class Imbalance**: 使用 `class_weight='balanced'` 處理類別不平衡（依 FR-001-D 觀察結果決定）
- **Completion Status**: 未開始
- **Notes**: 課程來源 AIp10，特徵重要性用於驗證期中觀察

---

## Phase 5: Streamlit 系統整合

### [ ] **Task ID**: FR-005
- **Task Name**: 模型整合至 Streamlit 系統
- **Work Description**:
    - Why: 提供互動式預測介面，展示模型應用價值
    - How: 修改現有 page5_prediction_model.py，整合三個模型後端，新增預測互動功能
- **Resources Required**:
    - Materials: `streamlit`, `plotly`, `matplotlib`
    - Personnel: 潘驄杰
    - Reference Codes/docs:
        - `docs/finalReport/spec/03-streamlit-ui.md`
        - `src/main/python/pages/page5_prediction_model.py`
- **Deliverables**:
    - [ ] 模型載入機制 (@st.cache_resource)
    - [ ] 預測輸入 UI (風速、季節、月份滑桿)
    - [ ] 預測結果展示 (AQI 值/等級、健康建議)
    - [ ] 模型解釋區 (係數表、決策樹圖、重要性圖)
    - [ ] 取消隱藏 page5，啟用預測功能
- **Testing Plan**:
    - Smoke Test: 確認 Streamlit 可正常啟動
    - Integration Test: 驗證預測按鈕可正確呼叫模型
- **Dependencies**: FR-002, FR-003, FR-004 (三個模型)
- **Constraints**: 需在現有 DIKW 架構下整合
- **Completion Status**: 未開始
- **Notes**: 需修改 app.py 中的 HIDE_ADVANCED_PAGES 設定

---

## Phase 6: 報告撰寫與交付

### [ ] **Task ID**: FR-006
- **Task Name**: 期末報告 PDF 撰寫
- **Work Description**:
    - Why: 符合繳交要求，完整呈現專案成果
    - How: 撰寫報告內容，包含研究動機、模型說明、結果分析、結論
- **Resources Required**:
    - Materials: Word
    - Personnel: 黎彥德、張家睿
    - Reference Codes/docs: 期中報告、spec 文件
- **Deliverables**:
    - [ ] 報告大綱確認
    - [ ] 模型結果截圖與圖表
    - [ ] PDF 報告完成
    - [ ] GitHub Repo 整理與連結
- **Testing Plan**:
    - 文件審閱: 組員互相校對
- **Dependencies**: FR-002, FR-003, FR-004, FR-005 (所有模型與系統)
- **Constraints**: 繳交日期 2026/01/07
- **Completion Status**: 未開始
- **Notes**: PDF 需附 GitHub Repo 連結

---

## 時間規劃總覽

| 日期 | Phase | Task ID | 工作內容 |
|------|-------|---------|----------|
| 12/31-1/1 | Phase 1 | FR-001 | 數據準備與特徵工程 |
| 1/2-1/3 | Phase 2 | FR-002 | 線性回歸模型實作 |
| 1/4 | Phase 3, 4 | FR-003, FR-004 | 決策樹與隨機森林實作 |
| 1/5 | Phase 5 | FR-005 | Streamlit 系統整合 |
| 1/6 | Phase 6 | FR-006 | 報告撰寫 |
| 1/7 | - | - | 繳交 |

---

## 工作進度追蹤

| Task ID | 任務名稱 | 負責人 | 狀態 |
|---------|----------|--------|------|
| FR-001 | 數據準備 | - | [ ] 未開始 |
| FR-002 | 線性回歸 | - | [ ] 未開始 |
| FR-003 | 決策樹 | - | [ ] 未開始 |
| FR-004 | 隨機森林 | - | [ ] 未開始 |
| FR-005 | Streamlit 整合 | - | [ ] 未開始 |
| FR-006 | 報告撰寫 | - | [ ] 未開始 |
