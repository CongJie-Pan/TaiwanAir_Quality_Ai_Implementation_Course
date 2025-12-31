# 期末報告工作清單

**課程**: 人工智慧實務
**組員**: 黎彥德、張家睿、潘驄杰
**繳交日期**: 2026/01/07
**最後更新**: 2024/12/31

---

## Phase 1: 數據準備與特徵工程

### [ ] **Task ID**: FR-001
- **Task Name**: 數據前處理與特徵工程
- **Work Description**:
    - Why: 為三個 AI 模型準備乾淨且標準化的輸入數據
    - How: 載入 2023 年數據，產生季節、AQI 等級等衍生特徵，處理缺失值，分割訓練/測試集
- **Resources Required**:
    - Materials: `pandas`, `numpy`, `sklearn.model_selection`
    - Personnel: 開發者
    - Reference Codes/docs: `docs/finalReport/spec/01-problem-and-data.md`
- **Deliverables**:
    - [ ] 數據載入與清洗腳本完成
    - [ ] 特徵工程函式 (季節、AQI 等級、風速等級)
    - [ ] 訓練集/驗證集/測試集分割完成 (70/15/15)
- **Testing Plan**:
    - Unit Test: 驗證特徵衍生邏輯正確
    - 資料驗證: 確認無缺失值、資料型別正確
- **Dependencies**: 現有數據後端 (`src/main/python/utils/data_loader.py`)
- **Constraints**: 優先使用 2023 年數據以延續期中報告
- **Completion Status**: 未開始
- **Notes**: 特徵工程需與期中報告的分類標準一致（風速等級、AQI 等級）

---

## Phase 2: 線性回歸模型實作

### [ ] **Task ID**: FR-002
- **Task Name**: 線性回歸模型開發
- **Work Description**:
    - Why: 量化風速、季節等因素對 AQI 的影響，驗證期中「AQI 隨風速增加而下降」的發現
    - How: 使用 scikit-learn 建立多元線性回歸模型，輸出係數表與評估指標
- **Resources Required**:
    - Materials: `sklearn.linear_model.LinearRegression`, `sklearn.metrics`
    - Personnel: 開發者
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
    - Personnel: 開發者
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
    - Personnel: 開發者
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
    - Personnel: 開發者
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
    - Materials: Word / LaTeX / Markdown
    - Personnel: 全體組員
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
