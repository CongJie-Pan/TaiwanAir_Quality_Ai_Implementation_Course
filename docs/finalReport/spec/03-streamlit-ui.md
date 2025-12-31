# 03 - Streamlit 系統整合規格

---

## 3.1 現有系統架構

當前 Streamlit 系統已有 5 個頁面（DIKW 架構）：

| 頁面 | 名稱 | DIKW 層級 | 狀態 |
|------|------|-----------|------|
| 1 | 數據總覽 | Data | ✅ 已完成 |
| 2 | 統計分析 | Information | ✅ 已完成 |
| 3 | 規律發現 | Knowledge | ✅ 已完成 |
| 4 | 智慧決策 | Wisdom | ⚠️ 隱藏中 |
| 5 | 預測模型 | Wisdom+ | ⚠️ UI 架構完成，後端待實作 |

---

## 3.2 期末整合目標

在 **Page 5（預測模型）** 整合三個 AI 模型。

### 3.2.1 頁面結構規劃

```
Page 5: 預測模型
├── 📊 模型選擇器（Tab 或 Radio）
│   ├── Tab 1: 線性回歸
│   ├── Tab 2: 決策樹
│   └── Tab 3: 隨機森林
│
├── 🎛️ 預測輸入區
│   ├── 風速滑桿 (0-15 m/s)
│   ├── 季節選擇 (春/夏/秋/冬)
│   ├── 月份選擇 (1-12)
│   └── 地區選擇 (縣市)
│
├── 📈 預測結果區
│   ├── AQI 預測值 / 等級
│   ├── 信心度（分類模型）
│   └── 健康建議
│
└── 📊 模型解釋區
    ├── 線性回歸：係數表
    ├── 決策樹：規則樹圖
    └── 隨機森林：特徵重要性圖
```

---

## 3.3 UI 元件規格

### 3.3.1 模型選擇器

```python
model_tab = st.tabs(["📈 線性回歸", "🌳 決策樹", "🌲 隨機森林"])
```

### 3.3.2 預測輸入區

```python
st.subheader("🎛️ 輸入預測參數")

col1, col2 = st.columns(2)

with col1:
    windspeed = st.slider("風速 (m/s)", 0.0, 15.0, 2.0, 0.1)
    season = st.selectbox("季節", ["春季", "夏季", "秋季", "冬季"])

with col2:
    month = st.slider("月份", 1, 12, 6)
    county = st.selectbox("縣市", counties_list)
```

### 3.3.3 預測結果展示

```python
st.subheader("📊 預測結果")

# 大型指標卡片
col1, col2, col3 = st.columns(3)
col1.metric("預測 AQI", f"{predicted_aqi:.0f}", delta=None)
col2.metric("AQI 等級", aqi_level, delta=None)
col3.metric("信心度", f"{confidence:.1%}", delta=None)

# 顏色指示
if aqi_level == "良好":
    st.success("🟢 空氣品質良好")
elif aqi_level == "普通":
    st.warning("🟡 空氣品質普通")
else:
    st.error("🔴 對敏感族群不健康")
```

### 3.3.4 模型解釋區

```python
# 線性回歸：係數表
st.dataframe(coefficients_df)

# 決策樹：規則圖
from sklearn.tree import plot_tree
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(dt_model, feature_names=features, class_names=classes, ax=ax)
st.pyplot(fig)

# 隨機森林：特徵重要性
st.bar_chart(importance_df)
```

---

## 3.4 後端整合

### 3.4.1 模型載入

```python
# models/__init__.py
from .linear_regression import LinearRegressionModel
from .decision_tree import DecisionTreeModel
from .random_forest import RandomForestModel

# 在 app 啟動時載入
@st.cache_resource
def load_models():
    return {
        'lr': LinearRegressionModel().load(),
        'dt': DecisionTreeModel().load(),
        'rf': RandomForestModel().load()
    }
```

### 3.4.2 預測 API

```python
def predict(model_type: str, features: dict) -> dict:
    """
    統一預測介面

    Args:
        model_type: 'lr' | 'dt' | 'rf'
        features: {'windspeed': 2.0, 'season': '冬季', ...}

    Returns:
        {
            'predicted_aqi': 85.0,      # 回歸用
            'predicted_level': '普通',   # 分類用
            'confidence': 0.82,          # 分類用
            'explanation': {...}         # 模型解釋
        }
    """
```

---

## 3.5 修改現有檔案

| 檔案 | 修改內容 |
|------|----------|
| `app.py` | 取消隱藏 page5，啟用預測功能 |
| `pages/__init__.py` | 重新 export page5 |
| `pages/page5_prediction_model.py` | 整合實際模型後端 |

---

## 3.6 展示流程

1. 使用者選擇模型（Tab）
2. 輸入預測參數（風速、季節等）
3. 點擊「預測」按鈕
4. 顯示預測結果 + 健康建議
5. 展示模型解釋（係數/規則/重要性）
