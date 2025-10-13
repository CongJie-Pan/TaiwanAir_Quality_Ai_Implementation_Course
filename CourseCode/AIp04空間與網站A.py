# AIp04空間與網站A.py: AI python 實作 - 04: 數據空間與網站設計
# Jia-Sheng Heh, 10/03/2024, revised from AIp1127數據空間與網站設計.py
# Usage: streamlit run AIp04空間與網站.py
    
import os   
wkDir = "c:/Users/jsheh/Desktop/postWorking/DIKW/AIp/";   os.chdir(wkDir)
print(os.getcwd())
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
st.set_page_config( page_title="S01_初步運營分析儀表板", page_icon="✅", layout="wide",)   #--這行只能是第一行

#%%####### (A) 數據準備: 視覺化的設計標籤 ##########

#%%##===== (A1) (KDD1) 讀取數據框 (file-->XXX) =====#####
XXX = pd.read_csv(wkDir+"XXX.csv")
print(XXX.shape)
print(XXX.head(2))  # -- (84008, 12)
#   invoiceNo channel customer product  ... amount  category2    cost
# 0        N1      s1       c1      p1  ...   1692       sub1  931.39
# 1        N2      s1       c2      p2  ...   1197       sub2  793.36

#%%##===== (A2) (KDD2) 數據空間 =====#####
##== (1).數據的維度(dimension)/軸向(axis): 收攏相關屬性，並具有索引鍵(key)
##== (2).可以進行數據空間分解 -- SPC模型的維度: 在 S通路銷售 P商品 給 C客戶
#        -- S (銷售): 通路—店舖—銷售者 ====================---> 通路銷售(動詞) 
#        -- P (商品): 品牌—品類—品項 ======================---> 商品 (受詞)
#        -- C (客戶): 會員/非會員—客層(客戶價值)—客群—客戶 =---> 客戶 (主詞)
#        -- T (時間): 年—季—月—週—日—時
print(XXX.iloc[0])
# channel                       s1 ===--> S維度 (通路銷售) 
# customer                      c1 ===--> C維度 (客戶)
# product                       p1 ===--> P維度 (商品)
# category                   kind1 ===------P的上階層維度
# category2                   sub1 ===------P的次上階層屬性
# price                       1980 ===------P的價格屬性
# cost                      931.39 ===------P的成本屬性
# datetime     2015-01-07 20:07:11 ===--> T維度
# invoiceNo                     N1 ===------T維度相關的順序
# date                  2015-01-07 ===------T維度的計算
# quantity                       1 ....... (量值)
# amount                      1692 ....... (量值)

#%%##===== (A3) (KDD3) 數據標籤的產生方式 =====#####
##== (1).算術計算的標籤
XXX["discount"] = XXX["amount"] / (XXX["quantity"] * XXX["price"])
XXX["profit"]   = XXX["amount"] - XXX["cost"]
##== (2).日期相關標籤
XXX["year"]  = pd.to_datetime(XXX["datetime"]).dt.year    #-- 還有很多其他產生此標籤的方法, 這裡只是取其中較方便的一種      
XXX["month"] = pd.to_datetime(XXX["datetime"]).dt.month
##== (3).連續數值的離散化
XXX["quarter"] = pd.cut( XXX["month"], bins=[0,3,6,9,12], labels=["1","2","3","4"])
##== (4).文字處理的標籤
XXX["yq"] = [ XXX["datetime"][k][2:4]+"Q"+XXX["quarter"][k] for k in np.arange(XXX.shape[0]) ]
print(XXX.iloc[0])
###--- channel, customer, product 等重複欄位, 在此不顯示...
# discount     0.854545 <-- (1).算術計算: XXX["discount"] = XXX["amount"] / (XXX["quantity"] * XXX["price"]) .... (衍生的量值)
# profit         760.61 <--               XXX["profit"]   = XXX["amount"] - XXX["cost"]                      .... (衍生的量值)
##                                 ===------------------------------------------------------------------以下均為T維度的衍生計算
# year             2015 <-- (2).日期相關: XXX["year"]  = pd.to_datetime(XXX["datetime"]).dt.year   (T維度)
# month               1 <--               XXX["month"] = pd.to_datetime(XXX["datetime"]).dt.month  
# quarter             1 <-- (3).數值離散化: XXX["quarter"] = pd.cut( XXX["month"], bins=[0,3,6,9,12], labels=["1","2","3","4"])
# year1            15Q1 <-- (4).文字處理: XXX["year1"] = [ XXX["datetime"][k][2:4]+"Q"+XXX["quarter"][k] for k in np.arange(XXX.shape[0]) ]

#%%##===== (A4) 量值 =====####
##== (1).一階統計量, 集中量值 (中間值,圓心)
#        -- 眾數 (mode): 用於nominal 變數(e.g.籍貫)中唯一可用之中間值
#           - (PRO).為最常出現之值, (CON).可能有兩個以上, 可能受極端值影響
#        -- 中位數 (median): 用於ordinal 變數(e.g.高矮), 唯一, 排序中間之值
#        -- (算術)平均數 (mean): (PRO) 可進行四則運算，誤差總和為0，最小誤差平方和
#        -- 幾何平均數 (geometric mean): 適用於 平均改變率, 平均成長率,平均比率
#        -- 調和平均數: 如：平均速度
##== (2).二階統計量, 變異量值 (半徑)
#        -- 全距/範圍 (range)：最大值 – 最小值
#           - (PRO)表示資料的分散程度,計算容易,容易理解 (CON)易受極端值影響, 無法精確反應所有資料的分布情形, 計算時只使用部份資料
#        -- 四分位距：Q=Q3-Q1
#           - (PRO)避免了受全距的影響, (CON)需先排序, 計算不便, 計算時只使用部份資料
#        -- 平均絕對離差(MAD,mean absolute deviation) MAD = |Xi - Xmed| / N
#           - 計算時使用全部資料, 但計算時比變異數複雜
#        -- 變異數(variance)–標準差(standard deviation),S^2 = sum(Xi – E(X))2 / (N-1)
#           - 計算時使用全部資料, 較MAD易受到極端值影響
#        -- 變異係數(coefficient of variation)= 標準差 / 平均數
#           - 可以消去平均數的影響, 可比較不同平均數的資料分布
##== (3).三階統計量, 偏態: 正偏態 (右偏態,往右邊延伸) / 負偏態 (左偏態,往左邊延伸)
##== (4).四階統計量, 峰值: 峰值>0 (資料分布較高聳且狹窄) / 峰值<0 (資料分布平坦且寬闊)
##== 實用零售量值
def 成交結構(Z):
    營業額 = sum(Z["amount"]);             銷售量 = sum(Z["quantity"])
    來客數 = len(Z["customer"].unique());  交易數 = len(Z["invoiceNo"].unique());
    客單價 = 營業額/交易數;                 客單件 = 銷售量/交易數
    return {'營業額':營業額, '銷售量':銷售量, '來客數':來客數, '交易數':交易數, '客單價':客單價, "客單件":客單件}
print(成交結構(XXX))
# {'營業額': 87910390, '銷售量': 83488, '來客數': 7774, '交易數': 40632, 
#  '客單價': 2163.5752608781254, '客單件': 2.0547351840913564}


#%%####### (B) 大數據中的多維度分析(MDA,Multi-Dimensional Analysis)的標準二維表格 ##########

#%%##===== (B1) 大數據分析(Big Data Analysis, BDA) =====#####
##== (1).數據分析的標準步驟：知識發現(KDD, Knowledge Discovery in Databases) [Fayyad et al., 1996]
##== (2).KDD步驟: 
#        -- (KDD1) 數據擷取(Data Acquisition)
#                  --> 當熟練後, 這部份最花時間，約需一半以上的時間
#        -- (KDD2) 數據處理(Data Processing): 一般 包括 數據清理、數據整合、數據轉換、數據化約
#                  --> 前兩者 (數據清理與數據整合) -- 融入 (KDD1)數據擷取, 為數據前處理(Pre-processing)
#                  --> 後兩者 (數據轉換與化約)     -- 通常在(KDD3)數據轉換 進行, 一般儘量不做化約(reduction)
#                  --> 實務上，將 KDD2 改為 數據探索(Data Exploration), 進行數據空間分析
#        -- (KDD3) 數據轉換(Data Transformation)
#                  --> 數據分析的核心，用以產生標籤(tag)
#        -- (KDD4) 數據模型(Data Modeling)，是數據分析的牛肉, 包括
#                  --> <1> 一維圖形呈現: 折線圖, 長條圖, 橫條圖, 圓餅圖, 散布圖, ...
#                  --> <2> OLAP (OnLine Analytic Processing, 在線分析處理, Power BI) 的 多維度分析(MDA, Multi-Dimensional Analysis) 
#                  --> <3> 數據探勘(Data Mining): 數據聚類(Clustering), 數據關連(Association), 數據序列(Sequencing)
#                  --> <4> 人工智能(AI切入點)/機器學習(Machine Learning): 預測(Prediction),決策(Decision),分類(Classification)
#        -- (KDD5) 數據解讀(Data Interpretation): 數據分析的亮點，要對模型進行洞察(insight)
#                  --> 結合數據視覺化(Visualization)，才能表現數據的價值，現在已以數位儀表板(dashboard)方式呈現

#%%##===== (B2) (KDD4) 多維度分析(MDA)中的二維表格: pd.crosstab() =====####
##== (1).頻次分布表
TYC = pd.crosstab(XXX["year"],XXX["channel"], margins=True);   print(TYC)
# year\channel    s1     s2    s3     s4    s5    All
# 2015          4784  12492  3157   2089  1451  23973
# 2016          5939  11225  1673   2916  1348  23101
# 2017         10118  16165   457   5439  4755  36934
# All          20841  39882  5287  10444  7554  84008
#%%== (2).數值累計表
TYCM = pd.crosstab(index=XXX["year"], columns=XXX["channel"], 
                   values=XXX["amount"], aggfunc="sum", margins=True);   print(TYCM)
# year\channel    s1        s2       s3       s4       s5       All
# 2015       4822236  12474878  3873440  1954859  1621785  24747198
# 2016       6670402  12542735  1818659  2758091  1649517  25439404
# 2017      11142368  16802196   515536  4820457  4443231  37723788
# All       22635006  41819809  6207635  9533407  7714533  87910390
#%%== (3).比例分布表
TYCMP = round(100*pd.crosstab(index=XXX["year"], columns=XXX["channel"], 
                              values=XXX["amount"], aggfunc="sum", normalize="index"),1);   print(TYCMP)
# year\channel  s1    s2    s3    s4    s5
# 2015        19.5  50.4  15.7   7.9   6.6
# 2016        26.2  49.3   7.1  10.8   6.5
# 2017        29.5  44.5   1.4  12.8  11.8

#%%##===== (B3) (KDD4) 數據框轉換/樞紐(pivot)轉換 (XXX-->Sv) =====#####
##== SPC轉換: 在數據空間中，從交易數據(X), 投射到某特定軸 (S軸/P軸/C軸) 的轉換，可以產生此軸上的新數據集
#    -- 如: 投射到 S軸 (+年份year) 的 通路導向樞紐轉換,
Sv = XXX.groupby(["year","channel"]).agg({"amount":"sum"}).reset_index();
print(Sv.shape);   print(Sv.head(4))   #-- (15, 3)
#   channel  year    amount
# 0      s1  2015   4822236
# 1      s1  2016   6670402
# 2      s1  2017  11142368
# 3      s2  2015  12474878

#%%##===== (B4) (KDD3-->KDD4) 初步的運營分析 =====#####
##== (4A).營業額趨勢(year) -- Sy: (KDD2)
Sy = XXX.groupby(["year"]).agg({"amount":"sum"}).reset_index();  print(Sy)
#    year    amount
# 0  2015  24747198
# 1  2016  25439404
# 2  2017  37723788
#%%== (4B).營業額趨勢(yq) -- Syq: (KDD2)
Syq = XXX.groupby(["yq"]).agg({"amount":"sum"}).reset_index();   print(Syq.shape);   print(Syq.head(2))   #-- (12, 2)
#      yq   amount
# 0  15Q1  7049821
# 1  15Q2  6195112
#%%== (4C).客戶vs交易數 -- Sc: (KDD2)
Sc = XXX.groupby(["customer"]).agg({"amount":"sum"}).reset_index();   
print(Sc.shape);   print(Sc.sort_values("amount",ascending=False)[0:7])   #-- (7774, 2)
#      customer    amount
# 5413    c5871  18251044
# 2157    c2940   5941165
# 1111       c2   3756426
# 7663       c9   3620920
# 2182    c2963    983970
# 2641    c3376    819695
# 5555       c6    665309
#%%== (4D).定義 X身分 -- XXX["X身分"]: (KDD3)
XXX["X身分"] = [ "非會員" if cc in ["c5871","c2940","c2","c9"] else "會員" for cc in XXX["customer"]]
#%%== (4E).不同的成交結構 -- TS: (KDD4)
TS = pd.DataFrame([成交結構(XXX), 
                   成交結構(XXX.loc[XXX["X身分"]=="會員"]),
                   成交結構(XXX.loc[XXX["X身分"]=="非會員"])])
TS.index = ["全客戶","會員","非會員"];   print(TS)
#           營業額  銷售量   來客數 交易數     客單價       客單件
# 全客戶  87910390  83488     7774  40632  2163.575261  2.054735
# 會員   56340835  53816      7770  24168  2331.216278  2.226746
# 非會員  31569555  29672        4  16464  1917.489978  1.802235
#%%== (4F).會員/非會員的趨勢 -- SMyq: (KDD4)
SMyq = XXX.groupby(["X身分","yq"]).agg({"amount":"sum"}).reset_index();   print(SMyq.head(2))
TSMyq = pd.crosstab(index=SMyq["yq"], columns=SMyq["X身分"], values=SMyq["amount"], aggfunc="sum");   print(TSMyq[0:2])
# yq\X身分 會員      非會員
# 15Q1     5233713  1816108
# 15Q2     4767800  1427312
#%%---> 以上的實驗結果, 應表現為一張進行數據分析的儀表板 (E3)


#%%####### (C) 常用streamlit指令 ##########
##-- [REF] 钱魏Way (2020). Streamlit：快速数据可视化界面工具, https://www.biaodianfu.com/streamlit.html

#%%##===== (C0) streamlit啟動 [Wiki] =====#####

##== (1).streamlit: 以 python 開發機器學習與數據科學之 open-source 平台的軟體公司
##== (2).streamlit基本程式撰寫 =====#####
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

#####===== (C1) 輸出文本/圖形 =====#####

##== (1)文字呈現組件(component)
st.title("網站標題")
st.header("主標題")
st.subheader("子標題")
st.write("字串")        #-- 經驗: 可以 st.write("字串1","子串2",...)
st.text("文字", help="浮動文字tooltip")
st.caption("圖表標題")
st.markdown(":red[紅色文字] 與 **:blue[藍色粗體文字]**")  #-- 其實,幾乎所有HTML均可嵌入

#%%== (2)數據(表格)呈現組件(component)
st.dataframe(XXX.head(3))
st.table( pd.crosstab(XXX["channel"],XXX["year"]))   
st.metric(label="Temperature", value="70 °F", delta="1.2 °F")
## st.data_editor()   ##==> 具有很高的應用潛力
##==> 可加上 style.format() 設定數值格式,文字格式,背景格式,橫條格式  [[AIp1129數據框轉換與客戶分析.py]]
##    [Trabelsi, (2019). Style Pandas Dataframe Like a Master, https://towardsdatascience.com/style-pandas-dataframe-like-a-master-6b02bf6468b0]
##-->另有 AgGrid 可顯示互動表格 (含輸入), 後述

#%%== (3) 統計圖形
df = pd.DataFrame(np.random.randn(20, 3), columns=['a', 'b', 'c'])
st.line_chart(df)
st.area_chart(df)
st.bar_chart(df)
arr = np.random.normal(1, 1, size=100)
plt.hist(arr, bins=20);   st.pyplot()

#%%== (4) 圖片呈現組件(component)
# st.plotly_chart(fig)   #-- 儀表板中的主要繪圖, 但還有很多其他的--> Chart elements [https://docs.streamlit.io/library/api-reference/charts]
## write(mpl_fig) : 显示Matplotlib图片 ---> 常用
## write(altair) : 显示Altair图表
## write(keras) : 显示Keras模型
## write(graphviz) : 显示Graphviz图片
## write(plotly_fig) : 显示Plotly图片 ---> 常用
## write(bokeh_fig) : 显示Bokeh图片

#%%== (5) 地圖
df_location = pd.DataFrame(np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4], columns=['lat', 'lon'])
st.map(df_location)

#%%== (6) 媒體
# st.image()
# st.audio()
# st.video()

#%%##===== (C2) 輸入機制 =====#####

##== (1) 控制輸入元件(widget)
from datetime import datetime
A1 = st.checkbox("提示文字")
A2 = st.button("提示文字")
A3 = st.selectbox("提示文字", ("選項1","選項2","選項3") )
A4 = st.slider("提示文字", min_value=0, max_value=100, step=5 )
A5 = st.slider("提示文字",value=datetime(2023,10,10,12,30),format="MM/DD/YY hh:mm")

##== (2) 檔案輸入 (上傳檔案)
Afile = st.file_uploader("在本地端選擇一個CSV檔", type="csv");  # A6 = pd.read_csv(Afile)

##== (3)其他進階組件...
#-- Forms: st.form()
#-- Chat: st.chat_message(), st.chat_input()
#-- Streamlit Community Cloud-->
#-- database: st.experimental_connection()
#-- LLM: openAI, LangChain... 

#%%##===== (C3) 互動儀表板網站架構 =====#####

##== (1).後台: 數據處理與計算 (KDD 1-3) 

##== (2).前台: 通常包括 -- 
#    -- (2A).sidebar(邊欄,控制盤): st.sidebar. title/head/控制項/顯示項
#    -- (2B).canvas(主畫布): st. title/head/顯示項(包括圖形)
#    -- (2C).navbar(導航列): 目前不具備此功能(未來可能會加入), 後述以sidebar兼具

#%%##===== (C4) Streamlit的網頁佈局法 (Layouts) =====#####
##== [REF: Abhi Saini. (2021). How to make a great Streamlit app: Part II, https://blog.streamlit.io/designing-streamlit-apps-for-the-user-part-ii/]
#--    (A).Sidebar (邊欄):   st.sidebar.write("...")
#--    (B).Columns (直行):   col1,col2=st.columns(2);  col1.write("...");  col2.write("...")
#--    (C).Tabs (標籤,標格): tab1,tab2=st.tabs(2);     tab1.write("...");  tab2.write("...") ===> 只能用在canvas,不能用到sidebar,故不宜做為切換流程,所以我們沒用到...
#--    (D).Expander (展開):  with st.expander("展開處");   st.write("展開字串")
#--    (E).Container (多組件容器): c=st.container();   st.write("最後一組件");   c.write("第一組件");   c.write("第二組件")
#--    (F).Empty (單組件容器): c=st.container();   st.write("最後顯示組件");   c.write("第二顯示組件,");   c.write("最早顯示組件,被換了)

##== (1).水平流程: 用 st.columns
##== (2).垂直流程: 用 st.sidebar
##== (3).流程中的文字說明
##== (4).主題,顏色與對比: 編輯 .streamlit/config.toml
##== (5).加上頁註,頁標題等: page_icon: 是一種 Favicon
##== (6).其他附加功能: 如 展開頁, 浮動標籤, 收藏標籤, 點評標籤等等


#%%####### (W).網站系統基本架構-->參見 AIp04空間與網站W.py ##########

