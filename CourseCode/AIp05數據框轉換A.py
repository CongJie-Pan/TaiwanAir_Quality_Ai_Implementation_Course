# AIp05數據框轉換.py: AI python 實作 - 05: 數據框轉換與客戶模型
# Jia-Sheng Heh, 09/08/2024, revised from AIp1129數據框轉換與客戶分析.py
# Usage: streamlit run AIp05數據框轉換.py

import os
wkDir = "c:/Users/jsheh/Desktop/postWorking/DIKW/AIp/";    os.chdir(wkDir);   print(os.getcwd())
import pandas as pd
import numpy as np
from st_aggrid import AgGrid #, GridUpdateMode, JsCode, ColumnsAutoSizeMode
import plotly_express as px

#%%##===== (*).啟動streamlit並設定 =====#####
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title="AIp05數據框轉換", page_icon="✅", layout="wide",)


#%%####### (A) 從數據空間到客戶模型 ##########

#%%##===== (A1) (KDD1)數據框與(KDD2)數據標籤 (file-->XXX-->X["date"]) =====#####
XXX = pd.read_csv(wkDir+"XXX.csv")
XXX["date"] = pd.to_datetime(XXX["datetime"]).dt.date   #-- 還有很多其他產生此標籤的方法, 這裡只是取其中較方便的一種
print(XXX.shape);   print(XXX.head(2))  # -- (84008, 12)
#   invoiceNo channel customer product  ... amount  category2    cost        date
# 0        N1      s1       c1      p1  ...   1692       sub1  931.39  2015-01-07
# 1        N2      s1       c2      p2  ...   1197       sub2  793.36  2015-01-18

#%%##===== (A2) SPC數據空間: (KDD2)數據探索 =====#####
##== (1).維度與量值
#        == 數據維度(dimension)/軸向(axis): 收攏相關屬性，並具有索引鍵(key)
#           -- SPC模型的維度: 在 S通路銷售 P商品 給 C客戶
#              > S (通路銷售,動詞): 通路—店舖(channel)—銷售者
#              > P (商品,受詞):     品牌—品類(category,category2)—品項(product-price,cost)
#              > C (客戶,主詞):     會員/非會員—客層(客戶價值)—客群—客戶(customer)
#              > T (時間):         年—季—月—週—日—時(datetime-->date, invoiceNo)
#        == 量值(measure):         金額(amount), 數量(quantity)
##== (2).客戶旅程(customer journey)
#        -- 通過品牌提供的產品或服務，從客戶的角度收集有關過程和體驗的信息 [Folstad & Kvale, 2018]
#        -- 是使用者與品牌互動的 “引人入勝的故事 (engaging story)” ]Stickdorn & Schneider, 2010]
#        -- 從交易行為來說，是當客戶對商品或服務滿意後，基於再購意願形成的再購行為 [Francken, 1993]
##== (3).接觸點(touchpoint): 
#        -- 顧客與品牌之間的互動或溝通的模式與過程 [Zomerdijk & Voss, 2010, 2011; Patr´ıcio, et al., 2011]
print(XXX.loc[XXX["customer"]=="c995"])
#       invoiceNo channel customer product  ... amount  category2     cost        date
# 6507      N2683      s3     c995   p1605  ...   1264       sub1   696.63  2015-03-25 --> 接觸點 1
# 6620      N2683      s3     c995   p1443  ...   2110       sub1  1402.62  2015-03-25
# 12041     N5692      s3     c995   p1554  ...    969       sub1   579.00  2015-06-03 --> 接觸點 2
# 12077     N5692      s3     c995    p654  ...    285       sub3   184.30  2015-06-03
# 20038     N9106      s3     c995   p3268  ...    922       sub1   507.64  2015-10-17 --> 接觸點 3

#%%##===== (A3) (KDD3) 數據轉換: 數據框轉換/樞紐(pivot)轉換 (XXX-->Cv) =====#####
##== SPC轉換: 在數據空間中，從交易數據(X), 投射到某特定軸 (S軸/P軸/C軸) 的轉換，可以產生此軸上的新數據集
#    -- 如: (2A4)是投射到 S軸 (+年份year) 的 通路導向樞紐轉換,
#       而此處是 投射到 C軸上的 客戶導向樞紐轉換
Cv = XXX.groupby("customer").agg({"invoiceNo":"nunique", "amount":"sum", "quantity":"sum",
                                  "date":["min","max"],
                             })                         ##== (2B1-2).(KDD3) 產生客戶價值(Cv)標籤
Cv.columns = ["FF","MM","TT", "D0","Df"];   print(Cv.shape);   print(Cv.head(2))   #-- (7774, 5)
#           FF      MM     TT           D0          Df
# customer                                      
# c1         1    3335      3   2015-01-07  2015-01-07
# c10        1    2770      1   2015-01-28  2015-01-28
#     造訪頻次 購買金額 購買件數  首次造訪日  最近造訪日
##== 客戶圖像(customer profiles) [Brownon, 2019]
#    -- 企業客戶的具體描述
#    -- 包括其人口分布、背景、習性、興趣，與行為等等
##== 社會科學中的變量 [林清山, 社會科學研究法]
#    -- (1) 基本/本質變量(property): 關於年齡,性別,排行,身高,體重,學歷,經歷,職業,住址,email等資料, 常與個資有關
#    -- (2) 傾向變量(inclination): 關於顏色,尺寸,風格,商品性能等喜好或傾向, 常以問卷進行收集
#    -- (3) 行為變量(behavior): 關於動作,行經路徑,造訪,點選網頁,購買,使用,選購商品,發表言論等行為的紀錄, 常為擷取數據的內容

#%%##===== (A4) 客戶價值模型: RFM模型 =====#####
##== 客戶分層: 區隔不同價值的顧客 [Arthur Hughes, 1994] [Stone, 1989]
##== RFM模型 [Miglautsch, 2000]
#    -- (1).顧客最近的消費/購買時間 (Recency,R): --> D0 (首次造訪日), Df (最後造訪日)
#           -- 即顧客最近一次購買的時間與現在時間的距離天數
#           -- 用來衡量顧客再次購買的可能性。時間距離愈近則再次購買程度愈高
#    -- (2).造訪頻次(Frequency,F)---量: 客戶"腳"的行為 --> FF (常客)
#           -- 在某段期間內購買該企業產品的總次數，此期間可定義為一個月、一季或任何時間長度
#           -- 用來衡量顧客在購買行為中與企業的互動程度，頻率愈高表示顧客的熱衷程度愈高。
#    -- (3).購買金額(Monetary,M)---質: 客戶"手"的行為 --> MM (貴客)
#           -- 在某段期間內購買該企業產品的總金額，
#           -- 用來評價顧客對該企業的貢獻度及顧客價值。金額愈高表示價值較高

#%%##===== (A5) (KDD3) 數據離散化 (FF,MM-->FF0,MM0) 到 客戶圖像(Customer Profile) =====#####
##== pd.cut: 切分(segment)數據 
Cv["FF0"] = pd.cut(Cv["FF"], bins=[0,1,9,99,999,19999])  #.astype(str)  
Cv["MM0"] = pd.cut(Cv["MM"], bins=[-5000,0,999,9999,99999,999999,19999999]) #.astype(str)
##== pandas中的時間切分
Cv["yq0"] = pd.PeriodIndex(Cv.D0, freq='Q')                                     #--> 用於客戶佇留模型
Cv["yqf"] = pd.PeriodIndex(Cv.Df, freq='Q')
print(Cv.shape);   print(Cv.head(2))   #-- (7774, 7)
#           FF    MM  TT          D0          Df     FF0          MM0     yq0     yqf
# customer                                                           
# c1         1  3335   3  2015-01-07  2015-01-07  (0, 1]  (999, 9999]  2015Q1  2015Q1
# c10        1  2770   1  2015-01-28  2015-01-28  (0, 1]  (999, 9999]  2015Q1  2015Q1
##== 這是一種 與行為有關的 客戶圖像(customer profile) --> FF: 腳的行為(造訪), MM: 手的行為(購買)
### Cv.to_csv("Cv.csv")

#%%##===== (A6) (KDD4) 客戶價值模型 (Cv['FF0','MM0']) =====#####
##== (1).表格(pd.crosstab())是最常見,也最有用的模型
##== (2).客戶價值(customer value)
#        -- 客戶對於給予和獲取的主觀感知態度, 影響消費者對產品的整體評價 [Zang, 2022]
#        -- 客戶價值模型: 表現 每位客戶的 質(quality,MM0) 與 量(quantity,FF) 的模型
TFM = pd.crosstab(Cv["FF0"], Cv["MM0"], margins=True);   print(TFM)
# MM0           (-5000, 0]  (0, 999]  ...  (999999, 19999999]   All
# FF0                                 ...                          
# (0, 1]                14       901  ...                   0  4169
# (1, 9]                 6        36  ...                   0  3225
# (9, 99]                0         0  ...                   0   368
# (99, 999]              0         0  ...                   0     8
# (999, 19999]           0         0  ...                   4     4
# All                   20       937  ...                   4  7774

#%%##===== (A7) (KDD5) 客戶價值模型的解讀 =====#####
# 一次客4169位，宜加強，尤其是35位的萬元客。
# 數次客3225位，可深耕，尤其是816位的萬元客。
# 超過十次客的342+25位，應做進一步分析，並可深入做VIP方案。
# 超過百次客的3+1位，可能為非會員，並可深入並做非會員分析。


#%%####### (B) 客戶的動態 ##########

#%%##===== (B1).回購週期 =====#####
##== 回購意向(repurchase intention)或重複購買(repeat patronage) [Tsiros, Mittal, & Regret, 2000]
#    -- 顧客再次購買同一品牌的產品或服務的可能性 
##== 客戶的回購意向 [Gronholdt, Martensen, & Kristensen, 2000]
#    -- (1) 欲回購的意向(intents to repurchase): 客戶未來對公司產品或服務的回購意願
#    -- (2) 初級行為(primary behaviors)：包括顧客的購買次數、頻率、金額和數量
#    -- (3) 次要行為(secondary behaviors)：客戶幫助公司介紹、推薦、建立聲譽的行為
##== 平均回購週期 (BB) = BB = (Df-D0)/(FF-1)
Cv["BB"] = [(Cv["Df"][k]-Cv["D0"][k]).days/(Cv["FF"][k]-1)   
             if (Cv["FF"][k] != 1) else np.nan for k in np.arange(Cv.shape[0])] #--> 用於流失客模型
Cv["BB0"] = pd.cut(Cv["BB"], bins=[0, 1, 7, 30, 99, 300, 1999])  # .astype(str)

#%%##===== (B2).Tnow 與 R0/Rf =====#####
##== Zee. (2020). 什麼是 NES？經營 CRM 不可不知的顧客分群模型. https://ezorderly.com/blog/2020/08/30/NES/
##== 分析計算
Tnow = pd.to_datetime("2017/12/31", format="%Y/%m/%d")
print(Tnow)  # -- 2023-07-01 00:00:00
##== R (Recency) in RFM模型: 分為
#    -- 客戶年資(seniority): R0 = Tfinal – D0
#    -- 客戶未消費時期:       Rf = Tfinal – Df
Cv["R0"] = [(Tnow - pd.to_datetime(d)).days for d in Cv.D0]
Cv["Rf"] = [(Tnow - pd.to_datetime(d)).days for d in Cv.Df]
print(Cv.loc[Cv.index=="c995"].transpose() )   #-- 加上動態的客戶圖像
# customer         c995
# FF                  3
# MM               5550
# FF0            (1, 9]
# MM0       (999, 9999]
# TT                  5
# D0         2015-03-25
# Df         2015-10-17
# DD                  3
# BB              103.0 = (Df-D0) / (FF-1) = (2015/10/17-2015/03/25) / (3-1)
# BB0         (99, 300]
# R0               1012 = Tnow - D0 = 2017/12/31 - 2015/03/25
# Rf                806 = Tnow - Df = 2017/12/31 - 2015/10/17
# yq0            2015Q1
# yqf            2015Q4
# status        S3沈睡客 ====>參見下方 (B3)

#%%##===== (B3).(KDD3) NES3模型/客戶動態標籤(status) =====#####
##== NES模型/客戶動態模型
#    -- 將客戶視為變動的個體，在乎的是顧客與品牌互動實際的消費行為，以及其現狀。
#    -- 客戶五動態: N (新客)--> 主力客戶 (Active,Existing) --> 瞌睡客(S1) --> 半睡客(S2) --> 沈睡客/流失客(S3)
KK = np.mean(Cv["BB"])      #-- 91.41137022993303 ===>全體平均回購週期 K = 非一次客BB 的平均值 (時尚業約為一季)
MM = 1.2*np.mean(Cv["MM"])  #-- 13569.908412657576 ===> 較高消費M = 1.2*新客平均消費
def NES3(Ck, K, M):
    if Ck["R0"] < 0: status = "U尚未消費"
    elif Ck["R0"] < 2*K:
        if Ck["MM"] > M: status = "N1新貴客"
        else:            status = "N2新客"
    else:
        if Ck["Rf"] < 2*K:
            if Ck["R0"]/Ck["FF"] < 0.75*K: status = "A1較活躍客"
            else:                          status = "A2活躍客"
        elif Ck["Rf"] < 3*K: status = "S1瞌睡客"
        elif Ck["Rf"] < 4*K: status = "S2半睡客"
        elif Ck["FF"] < 10:  status = "S3沈睡客"
        else:                status = "S4沈睡忠誠客"    
    return status
Cv.loc[Cv.index=="c995"].apply(NES3, K=KK, M=MM, axis=1)   #-- c995    S3沈睡客

#%%##===== (B4).(KDD4) 私域流量池 =====#####
##== 公域流量 (Public Traffic) [MBA智庫百科]
#    -- 如Google/Facebook/淘寶等網路平台，具相當大的公共流量
#    -- 利用付費廣告/網站SEO優化，來爭取流量，吸引客戶消費
##== 私域流量 (Private Traffic): 自主掌控、封閉管道 [MBA智庫百科]
#    -- 企業可在任何時間，直接接觸到的客戶
#    -- 如: 自媒體、用戶群、微信號等社交平台，及企業會員
##== 客戶佇留模型(customer retention model)/客戶流量模型 -- 也就是私域流量池
TQQ = pd.crosstab(Cv["yq0"], Cv["yqf"], margins=True);   print(TQQ)  # +--> 流入(新進客)數目
# yqf     2015Q1  2015Q2  2015Q3  2015Q4  ...  2017Q2  2017Q3  2017Q4   All
# yq0                                     ...                              
# 2015Q1     389      65      53      61  ...     118       0       0  1052
# 2015Q2       0     342      53      37  ...      66       0       0   715
# 2015Q3       0       0     271      32  ...      57       0       0   537
# 2015Q4       0       0       0     331  ...      50       0       0   566 活躍忠誠客
# 2016Q1       0       0       0       0  ...      42       0       0   553 -+-- 活躍舊客
# 2016Q2       0       0       0       0  ...      81       0       0   621  !
# 2016Q3       0       0       0       0  ...      66       0       0   506  !
# 2016Q4       0       0       0       0  ...      73       0       0   475  !
# 2017Q1       0       0       0       0  ...      83       0       0   475  !
# 2017Q2       0       0       0       0  ...     369       1       0   370 - 
# 2017Q3       0       0       0       0  ...       0    1007     109  1116 -+-- 新進客
# 2017Q4       0       0       0       0  ...       0       0     788   788 -+
# All        389     407     377     461  ...    1005    1008     897  7774
#                +--> 流出(流失客)數目   S3  S2 瞌睡客S1  +-新進/活躍客客-+
#%%##===== (B5).私域流量池中的客戶動態 =====#####
##== NES3
# N (New) 新客:               R0/Rf < 2*K 的右下角2*2四格人數
# E (Existing/Active) 活躍客: Rf<2*K, R0>2*K 的 右方兩行的 上/中方人數
# S1 瞌睡客:                  2*K<Rf<3*K 的倒數第三行(2017Q2)人數
# S2 半睡客:                  3*K<Rf<4*K 的倒數第四行(2017Q1)人數
# S3 沈睡客/流失客(L):         4*K<Rf     的其他行(2015Q1-2016Q4)人數
##== 行銷分眾的策略方向
# 新進客(N)可分為 新客/新貴客
# 活躍客(E)可分為 一般活躍客/較活躍客
# 瞌睡客(S1)一般是 挽回客戶的重點之一
# 沈睡客(S3/L)中亦有 需挽回的客戶,如沈睡忠誠客

#%%##===== (B6).(KDD5) 私域流量池的解讀 =====#####
# 2015Q1-2016Q3的新進客每季均超過500人，而2016Q4-2017Q2的新進客大都少於500人，值得注意。
# 2016Q2開始，流失客大於新進客，要特別注意，而2017Q2-Q4的流失客均超過800人，應做回客動作。
# 2017Q3-Q4的每季新進客有1116+788位，但回頭客只有109位，而2017Q2前的舊客只有1位回頭，需強烈檢


#%%####### (C) 客戶模型的系統化工程 ##########

#%%##===== (C1).定義相關分析參數 =====#####
##== 各家企業的以下參數不儘相同
FFbreaks = [0, 1, 9, 99, 999, 19999]
MMbreaks = [-5000, 0, 999, 9999, 99999, 999999, 19999999]
BBbreaks = [0, 1, 7, 30, 99, 300, 1999]
RRbreaks = [0, 7, 30, 60, 99, 180, 360, 499, 700, 1999]               #--> 用於客戶漏斗 (下述)
Tnow = pd.to_datetime("2017/12/31", format="%Y/%m/%d");  print(Tnow)  # -- 數據分析點: 2023-07-01 00:00:00

#%%##===== (C2).(KDD2-->KDD4) 客戶圖像 包成 函式庫 (X-->Cv) =====#####
def buildCv(XX,FFbreaks,MMbreaks,BBbreaks):  #==> 建構客戶價值數據框
    Cv1 = XX.groupby("customer").agg({"invoiceNo": "nunique", "amount": "sum", "quantity": "sum",
                                      "date": ["min", "max", "nunique"],
                                      })
    Cv1.columns = ["FF", "MM", "TT", "D0", "Df", "DD"]
    Cv1["FF0"] = pd.cut(Cv1["FF"], bins=FFbreaks) #.astype(str)
    Cv1["MM0"] = pd.cut(Cv1["MM"], bins=MMbreaks) #.astype(str)
    Cv1["BB"] = [(Cv1["Df"][k]-Cv1["D0"][k]).days/(Cv1["FF"][k]-1)
                 if (Cv1["FF"][k] != 1) else np.nan for k in np.arange(Cv1.shape[0])]
    Cv1["BB0"] = pd.cut(Cv1["BB"], bins=BBbreaks) #.astype(str)
    Cv1["yq0"] = pd.PeriodIndex(Cv1.D0, freq='Q')
    Cv1["yqf"] = pd.PeriodIndex(Cv1.Df, freq='Q')
    #              FF     MM  TT         D0  ...            MM0     BB   yq0   yqf
    # customer                               ...
    # A0000000036   3  10488   6 2021-01-31  ...  (9999, 49999]  188.0  21Q1  22Q1
    # A0000000038   5  20735   8 2020-08-04  ...  (9999, 49999]   93.5  20Q3  21Q3
    return Cv1
def addCvNES3(Cv, Tnow, KK, MM, RRbreaks):   #==> 將 NES3 加入 Cv
    Cv["R0"] = [(Tnow - pd.to_datetime(d)).days for d in Cv.D0]
    Cv["Rf"] = [(Tnow - pd.to_datetime(d)).days for d in Cv.Df]
    Cv["R00"] = pd.cut(Cv["R0"], bins=RRbreaks) #.astype(str)
    Cv["Rf0"] = pd.cut(Cv["Rf"], bins=RRbreaks) #.astype(str)
    Cv["status"] = Cv.apply(NES3, K=KK, M=MM, axis=1)
    return Cv

#%%##===== (C3).(KDD2-->KDD4) 客戶圖像 包成 函式庫 (X-->Cv) =====#####
##== streamlit基本運作程序
#    -- (1).操作每個元件: 都會重新運行 整個應用程式
#    -- (2).全部填寫好,再一次觸發(提交submit): 使用 st.form()表單元件
#    -- (3).快取(cache)先前運算過的結果:      使用 @st.cache_..: 下次遇到一樣的輸入值，會直接使用先前的結果
##== 兩種快取(cache)機制: 使用裝飾器(Decorator)
#    -- @st.cache_data：    快取計算後的結果、如: 載入DataFrame、查詢 API 等等
#    -- @st.cache_resource：全域資源，如 ML 模型或資料庫連接等等
@st.cache_data
def getData():     #==> 自 X.csv 讀取 X (KDD1), 並設定標籤 (KDD3)
    X = pd.read_csv("XXX.csv")
    # -- 還有很多其他產生此標籤的方法, 這裡只是取其中較方便的一種
    X["date"] = pd.to_datetime(X["datetime"]).dt.date
    X["year"] = pd.to_datetime(X["datetime"]).dt.year
    X["yq"] = pd.PeriodIndex(X.date, freq='Q')
    X["ym"] = pd.PeriodIndex(X.date, freq='M')
    return(X)
@st.cache_data
def buildCvRDS(X,FFbreaks,MMbreaks,BBbreaks,RRbreaks,Tnow):  #==> 由交易數據 X 求取客戶數據框 Cv (KDD3)
    Cv = buildCv(X,FFbreaks,MMbreaks,BBbreaks)
    print(Cv.shape);   print(Cv[2:4])  # -- (52217, 17)
    KK = np.nanmean(Cv["BB"]);    print(KK)  # -- 43.070694784611675
    MM = np.nanmean(Cv["MM"]);    print(MM)  # -- 46998.990443725226
    Cv = addCvNES3(Cv, Tnow, KK, MM, RRbreaks)
    print(Cv.shape)
    print(Cv[2:4])  # -- (52217, 22)
    # Cv = pd.read_csv("cvv.csv")
    return(Cv)

#%%##===== (C4).試呼叫函式 =====#####

Cv = buildCvRDS(XXX,FFbreaks,MMbreaks,BBbreaks,RRbreaks,Tnow);   print(Cv.shape);   print(Cv.head(2))
#%%==
FFdata = pd.DataFrame(Cv.FF0.value_counts()).reset_index()              #-- (C1).客戶FF漏斗   
figFF = px.funnel(x=list(FFdata.FF0), y=list(FFdata.index.astype(str)), title="造訪頻次客戶漏斗") #==> px有漏斗的元件
figFF.show()
#%%==
MMdata = pd.DataFrame(Cv.MM0.value_counts()).reset_index().sort_values(by='index');   print(MMdata) #-- (C2).客戶MM漏斗
figMM = px.funnel(x=list(MMdata.MM0), y=list(MMdata["index"].astype(str)), title="消費金額客戶漏斗")
figMM.show()


#%%####### (W).網站系統基本架構-->AIp數據框轉換W.py ##########
