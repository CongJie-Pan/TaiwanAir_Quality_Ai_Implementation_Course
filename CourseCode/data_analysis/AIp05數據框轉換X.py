# AIp05數據框轉換X.py: AI python 實作 - 05: 數據框轉換 with 客戶系統
# Jia-Sheng Heh, 10/03/2024, revised from AIp05數據框轉換.py
# Usage: streamlit run AIp05數據框轉換X.py --> http:/localhost:8501

import numpy as np
import pandas as pd
from st_aggrid import AgGrid #, GridUpdateMode, JsCode, ColumnsAutoSizeMode
import plotly_express as px
import plotly.graph_objs as go
import os
wkDir = "c:/Users/jsheh/Desktop/postWorking/DIKW/AIp/";   os.chdir(wkDir);   print(os.getcwd())

#%%####### (W).網站系統基本架構 ##########
import streamlit as st
from streamlit_navigation_bar import st_navbar

#%%##===== (W1).自定分析參數 + 公用函式庫 + streamlit快取機制 =====#####

##== (1).定義相關分析參數: 各家企業的以下參數不盡相同 ==##
FFbreaks = [0, 1, 9, 99, 999, 19999]
MMbreaks = [-5000, 0, 999, 9999, 99999, 999999, 19999999]
BBbreaks = [0, 1, 7, 30, 99, 300, 1999]
RRbreaks = [0, 7, 30, 60, 99, 180, 360, 499, 700, 1999]               #--> 用於客戶漏斗 (下述)
Tnow = pd.to_datetime("2017/12/31", format="%Y/%m/%d");  print(Tnow)  # -- 數據分析點: 2023-07-01 00:00:00

#%%== (2).客戶圖像公用函式庫 (Cv) ==##
def buildCv(XX,FFbreaks,MMbreaks,BBbreaks):  ##== 建構客戶價值數據框: Cv = buildCv(X,FFbreaks,MMbreaks,BBbreaks)
    Cv1 = XX.groupby("customer").agg({"invoiceNo": "nunique", "amount": "sum", "quantity": "sum",
                                      "date": ["min", "max", "nunique"],
                                      })
    Cv1.columns = ["FF", "MM", "TT", "D0", "Df", "DD"]
    Cv1["FF0"] = pd.cut(Cv1["FF"], bins=FFbreaks).astype(str)
    Cv1["MM0"] = pd.cut(Cv1["MM"], bins=MMbreaks).astype(str)
    Cv1["BB"] = [(Cv1["Df"][k]-Cv1["D0"][k]).days/(Cv1["FF"][k]-1)
                 if (Cv1["FF"][k] != 1) else np.nan for k in np.arange(Cv1.shape[0])]
    Cv1["BB0"] = pd.cut(Cv1["BB"], bins=BBbreaks).astype(str)
    Cv1["yq0"] = pd.PeriodIndex(Cv1.D0, freq='Q')
    Cv1["yqf"] = pd.PeriodIndex(Cv1.Df, freq='Q')
    #              FF     MM  TT         D0  ...            MM0     BB   yq0   yqf
    # customer                               ...
    # A0000000036   3  10488   6 2021-01-31  ...  (9999, 49999]  188.0  21Q1  22Q1
    # A0000000038   5  20735   8 2020-08-04  ...  (9999, 49999]   93.5  20Q3  21Q3
    return Cv1
def NES3(Ck, K, M):                          ##== status = NES3(Ck, K, M): 定義NES3狀態
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
def addCvNES3(Cv, Tnow, KK, MM, RRbreaks):   ##== 將 NES3 加入 Cv: Cv = addCvNES3(Cv, Tnow, KK, MM, RRbreaks)
    Cv["R0"] = [(Tnow - pd.to_datetime(d)).days for d in Cv.D0]
    Cv["Rf"] = [(Tnow - pd.to_datetime(d)).days for d in Cv.Df]
    Cv["R00"] = pd.cut(Cv["R0"], bins=RRbreaks).astype(str)
    Cv["Rf0"] = pd.cut(Cv["Rf"], bins=RRbreaks).astype(str)
    Cv["status"] = Cv.apply(NES3, K=KK, M=MM, axis=1)
    return Cv

#%%== (3A).streamlit基本運作程序 ==##
#    -- (1).操作每個元件: 都會重新運行 整個應用程式
#    -- (2).全部填寫好,再一次觸發(提交submit): 使用 st.form()表單元件
#    -- (3).快取(cache)先前運算過的結果:      使用 @st.cache_..: 下次遇到一樣的輸入值，會直接使用先前的結果
##== (3B).兩種快取(cache)機制: 使用裝飾器(Decorator) ==##
#    -- @st.cache_data：    快取計算後的結果、如: 載入DataFrame、查詢 API 等等
#    -- @st.cache_resource：全域資源，如 ML 模型或資料庫連接等等
##== (3).客戶圖像分析函式庫 ==##
@st.cache_data
def getX(Xname):     ##== X = getX(Xname): 自 X.csv 讀取 X (KDD1), 並設定標籤 (KDD3) ==##
    X = pd.read_csv(Xname)
    # -- 還有很多其他產生此標籤的方法, 這裡只是取其中較方便的一種
    X["date"] = pd.to_datetime(X["datetime"]).dt.date
    X["year"] = pd.to_datetime(X["datetime"]).dt.year
    X["yq"] = pd.PeriodIndex(X.date, freq='Q')
    X["ym"] = pd.PeriodIndex(X.date, freq='M')
    return(X)
@st.cache_data
def buildCvRDS(X,FFbreaks,MMbreaks,BBbreaks,RRbreaks,Tnow):  ##== Cv = buildCvRDS(X,..): 由交易數據 X 求取客戶數據框 Cv (KDD3)
    Cv = buildCv(X,FFbreaks,MMbreaks,BBbreaks)
    print(Cv.shape);   print(Cv[2:4])  # -- (52217, 17)
    KK = np.nanmean(Cv["BB"]);    print(KK)  # -- 43.070694784611675
    MM = np.nanmean(Cv["MM"]);    print(MM)  # -- 46998.990443725226
    Cv = addCvNES3(Cv, Tnow, KK, MM, RRbreaks);   print(Cv.shape);   print(Cv[2:4])  # -- (52217, 22)
    # Cv = pd.read_csv("cvv.csv")
    return(Cv)

#%%##===== (W2).儀表板函式庫: 前台(a)navbar,(b)sidebar,(c)canvas,後台(d) =====#####
def 擷取交易(fname):  ##== (KDD1)擷取交易儀表板: X = 擷取交易(fnameX) ==##
    ##== (d).後台 ==##
    X = getX(wkDir+fname)
    print("\n\n>>>>> 擷取交易數據 (-->XXX) -----")  # -- 偵錯用
    # == (b).前台-sidebar ==##  #==> [[AIp04/C4)(2)垂直流程]]
    st.sidebar.header("== (KDD1).擷取交易數據 -- ")
    st.sidebar.write("* 交易檔名 = "+fname)
    st.sidebar.write("* 記錄筆數 = ", X.shape[0])
    ##== (c).前台-canvas ==##
    st.header("== (KDD1).s擷取交易數據"+fname+" --")
    st.subheader("* 記錄筆數 = "+str(X.shape[0]))
    st.dataframe(X.head(3))
    return X
def 轉換客戶圖像(X,FFbreaks,MMbreaks,BBbreaks,RRbreaks,Tnow):
    ##== (d).後台 ==##
    Cv = buildCvRDS(X,FFbreaks,MMbreaks,BBbreaks,RRbreaks,Tnow)
    TFM = pd.crosstab(Cv["FF0"], Cv["MM0"], margins=True);   print(TFM)
    TFMs = pd.DataFrame(TFM);     print(TFMs)  #-- (B).客戶價值模型
    TFMs_styled = TFMs.style.format(formatter="{:,}", na_rep=".").bar(cmap="cool", axis=None)  #==> 表格加上style.format更豐富
    FFdata = pd.DataFrame(Cv.FF0.value_counts()).reset_index()              #-- (C1).客戶FF漏斗   
    figFF = px.funnel(x=list(FFdata.FF0), y=list(FFdata.index.astype(str)), title="造訪頻次客戶漏斗") #==> px有漏斗的元件
    MMdata = pd.DataFrame(Cv.MM0.value_counts()).reset_index().sort_values(by='index');   print(MMdata) #-- (C2).客戶MM漏斗
    figMM = px.funnel(x=list(MMdata.MM0), y=list(MMdata["index"].astype(str)), title="消費金額客戶漏斗")
    ##== (c1).前台-canvas: 客戶價值模型 ==##
    st.header("== (KDD1).客戶模型 --")
    st.subheader("* (A) 共有" + str(Cv.shape[0]) + "位客戶")
    st.subheader("* (B) (KDD4) 客戶價值模型")
    st.table(data=TFMs_styled)   #-- st.dataframe(TFM)
    ##== (c2).前台-canvas: 客戶漏斗 (Cv.FF0-->FFdata, Cv.MM0-->MMdata) ==##
    cols = st.columns([1, 1])
    cols[0].subheader("* (C1) 造訪次數(FF)客戶漏斗");   cols[0].plotly_chart(figFF, theme="streamlit", use_container_width=True)
    cols[1].subheader("* (C2) 消費金額(MM)客戶漏斗");   cols[1].plotly_chart(figMM, theme="streamlit", use_container_width=True)
    ##== (b).前台-sidebar ==##  
    st.sidebar.write("* 共有" + str(Cv.shape[0]) + "位客戶")
    return Cv
def TFM_客群選取(Cv):
    ##== (d).後台: 設定 TFMs, TFMvs, TFMv0s 之style ==##
    TFM = pd.crosstab(Cv["FF0"], Cv["MM0"], margins=True);   print(TFM)
    TFMs = pd.DataFrame(TFM);     print(TFMs)  #-- (B).客戶價值模型
    TFMs_styled = TFMs.style.format(formatter="{:,}", na_rep=".").bar(cmap="cool", axis=None)  #==> 表格加上style.format更豐富
    TFMv = pd.crosstab(index=Cv.FF0, columns=Cv.MM0,values=Cv.MM, aggfunc='sum', margins=True)
    TFMvs = pd.DataFrame(TFMv);   print(TFMvs)
    TFMvs_styled = TFMvs.style.format(formatter="{:,}", na_rep=".").bar(cmap="Wistia", axis=None)
    TFMv0s = pd.DataFrame(100*TFMv/TFMv["All"]["All"]);    print(TFMv0s)
    TFMv0s_styled = TFMv0s.style.format(formatter="{:,.1f}", na_rep=".").bar(cmap="Wistia", axis=None)
    #== (c).前台-canvas: 客戶價值模型 ==##
    st.subheader("* (A) 共有" + str(Cv.shape[0]) + "位客戶")
    st.subheader("* (B) 客戶價值模型")
    st.table(data=TFMs_styled)   # st.dataframe(TFM)
    cols = st.columns([3, 2])    #== 營業額模型 (TFMvs,TFMv0s)
    cols[0].subheader("- (B1) 客戶價值營業額模型");       cols[0].table(data=TFMvs_styled)    # st.dataframe(TFM)
    cols[1].subheader("- (B2) 客戶價值營業額佔比模型");   cols[1].table(data=TFMv0s_styled)   # st.dataframe(TFM)
    ##== (b).前台-sidebar ==##  
    ##-- (b1).客群選擇
    CvTA = Cv
    st.sidebar.subheader("- (3D) 選擇客群,以觀看客戶圖像/旅程")
    FF0A = st.sidebar.multiselect('>> 請選擇客戶的造訪頻次區間(FF0):', list(Cv.FF0.unique()))
    FF0 = [str(x) for x in FF0A];     print(">>> "+str(FF0))
    MM0A = st.sidebar.multiselect('>> 請選擇客戶的消費金額區間(MM0):', list(Cv.MM0.unique()))
    MM0 = [str(x) for x in MM0A];     print(">>> "+str(MM0))
    ##-- (b2).客群計算
    cIND1 = np.arange(Cv.shape[0]);   cIND = []
    if len(FF0) > 0:  cIND1 = np.where([(str(Cv.FF0[k]) in FF0) for k in cIND1])[0]
    if len(MM0) > 0:  cIND = np.where([(str(Cv.MM0[k]) in MM0) for k in cIND1])[0]
    st.sidebar.write("-> 目標客戶數 = "+str(len(cIND))+"位")
    if len(cIND)!=0:
        CvTA = Cv.iloc[cIND];   
        # indX = list(np.where([k if X.customer[k] in CvTA.index else None for k in np.arange(X.shape[0])])[0])  # indX
        CvTA.reset_index(inplace=True)
        st.sidebar.write("-> 涵蓋交易數 = "+str(sum(CvTA.FF))+"筆")
    ##-- (b3).客戶圖像+客戶旅程
    if st.sidebar.checkbox("> 確定客群"):
        st.subheader("* (C) 目標客戶: FF0 = "+",".join(FF0) +", MM0 = "+",".join(MM0)+", 共計"+str(len(CvTA))+"位")
        st.subheader("* (D) 客戶圖像")
        st.dataframe(CvTA)
        # st.subheader("* (E) 客戶旅程")
        # st.dataframe(X.iloc[indX])
    return CvTA
def 流失客_客群選取(Cv):
    ##== (d).後台: 設定 TFMs, TFMvs, TClosts ==##
    TFM = pd.crosstab(Cv["FF0"], Cv["MM0"], margins=True);   print(TFM)
    TFMs = pd.DataFrame(TFM);     print(TFMs)
    TFMs_styled = TFMs.style.format(formatter="{:,}", na_rep=".").bar(cmap="cool", axis=None)
    TFMv = pd.crosstab(index=Cv.FF0, columns=Cv.MM0,values=Cv.MM, aggfunc='sum', margins=True)
    TFMvs = pd.DataFrame(TFMv);   print(TFMvs)
    TClost = pd.crosstab(Cv["yqf"], Cv["BB0"], margins=True);   print(TClost)
    TClosts = pd.DataFrame(TClost)
    TClosts_styled = TClosts.style.format(formatter="{:,}", na_rep=".").bar(cmap="Wistia", axis=None)
    #== (c).前台-canvas: 客戶價值模型 ==##
    st.subheader("* (A) 共有" + str(Cv.shape[0]) + "位客戶")
    cols = st.columns([3, 2])   #-- 客戶價值模型+流失模型 (TFMvs,TFMv0s)
    cols[0].subheader("* (B) 客戶價值模型");     cols[0].table(data=TFMs_styled)   
    cols[1].subheader("* (C) 客戶流失模型");     cols[1].table(data=TClosts_styled)        
    ##== (b).前台-sidebar ==##  
    ##-- (b1).客群選擇
    CvTA = Cv
    st.sidebar.subheader("* (D) 選擇客群,以觀看客戶圖像/旅程")
    yqfA = st.sidebar.multiselect('>> 請選擇客戶的最後造訪季(yqf):', sorted(list(Cv.yqf.astype(str).unique())));   yqf = [str(x) for x in yqfA];   print(">>> "+str(yqf))
    BB0A = st.sidebar.multiselect('>> 請選擇客戶的回購週期區間(BB0):', sorted(list(Cv.BB0.astype(str).unique())));   BB0 = [str(x) for x in BB0A];   print(">>> "+str(BB0))        
    ##-- (b2).客群計算
    cIND1 = np.arange(Cv.shape[0]);   cIND = []
    if len(yqf) > 0:    cIND1 = np.where([(str(Cv.yqf[k]) in yqf) for k in cIND1])[0]       
    if len(BB0) > 0:    cIND = np.where([(str(Cv.BB0[k]) in BB0) for k in cIND1])[0]
    st.sidebar.write("-> 目標客戶數 = "+str(len(cIND))+"位")
    if len(cIND)!=0:
        CvTA = Cv.iloc[cIND];    CvTA.reset_index(inplace=True)
        st.sidebar.write("-> 涵蓋交易數 = "+str(sum(CvTA.FF))+"筆")
    ##-- (b3).客戶圖像+客戶旅程
    if st.sidebar.checkbox("> 確定客群"):
        setCvTA = "流失客 BB0="+",".join(BB0) +", yqf="+",".join(yqf)+", 共計"+str(len(CvTA))+"位"
        st.subheader("* (C) 目標客戶: "+setCvTA)
        st.subheader("* (D) 客戶圖像")
        st.dataframe(CvTA)
        ##-- (b4).匯出至EXCEL檔
        if st.sidebar.checkbox("* (E) 匯出客群名單 至EXCEL檔"):
            CvTA.to_excel(setCvTA+".xlsx" )      #==> pandas 提供匯出 EXCEL 的方法 .to_excel()
            st.sidebar.subheader("* (F)"+str(len(cIND))+"筆記錄已輸出至檔案"+setCvTA+".xlsx")
    return CvTA

#%%##===== (W3).導航函式庫 =====#####
def check2log(textStr,log):      ##== check 再將 textStr 納入 log 中, 並中並可以提供建議 ==##
    st.sidebar.markdown('---') 
    st.sidebar.header("== (KDD5)請用戶輸入解讀--")
    st.session_state.username = st.sidebar.text_input("", st.session_state.username, placeholder="輸入用戶名")
    sugg_key = f"log_checkbox_{len(log)}"
    suggestion = st.sidebar.text_area("", value=st.session_state.suggestion, key=sugg_key, height=20, placeholder="輸入建議")
    if st.sidebar.button("LOG操作 / 提交建議"):
        log.append(textStr)
        if st.session_state.username and suggestion:
            log.append(f"<{st.session_state.username}建議>> {suggestion}")
            st.sidebar.success("建议已提交并纳入LOG");     st.session_state.suggestion = "" 
        else:
            st.sidebar.error("不列入建議,只單純LOG操作內容")
    return
def initSSS(variables, pjName):  ##== 初始化 state_session 的各變量 ==##
    sss = st.session_state
    if "LOG" not in sss:        sss.LOG = [pjName]   #-- 初始化 LOG 列表
    if "username" not in sss:   sss.username = ""    #-- 初始化 username 列表
    if "suggestion" not in sss: sss.suggestion = ""  #-- 初始化 suggestion 列表
    for var in variables:       #-- 初始化传入的变量名为 None
        if var not in sss: sss[var] = None
    return sss

#%%##===== (W4).網站架構 =====#####
if __name__ == "__main__":

    ##== (1).設定頁面組態 與 導航列 (前台(a)navbar) ==##
    st.set_page_config(page_title="SPC-S01 RDS系統", page_icon="✅", layout="wide",)  #==> [[AIp04/C4)(5)加上頁註,頁標題等]]
    page = st_navbar(["[擷取交易]", "[轉換客戶圖像]", "[TFM_客群選取]", "[流失客_客群選取]"])

    ##== (2).設定session初始值等 ==##
    Xname = "XXX.csv"
    sss = initSSS(["X", "Cv", "CvTA", "Xname"], "AIp05數據框轉換W"+"--"+Xname);   sss.Xname = Xname

    ##== (3).設定 前台((b)sidebar + (c)canvas)主標題 ==##
    st.title("AIp05數據框轉換: 初步客戶分析儀表板(C01)")
    st.sidebar.title("初步運營分析(C01)控制盤--")

    ##== (4).導航切換: 前台(a)navbar-->儀表板函式(b,c,d) ==##
    match page:
        case "[擷取交易]":
            sss.X = 擷取交易(sss.Xname);    check2log(f"擷取交易: {sss.Xname} to get X with {sss.X.shape[0]} records", sss.LOG)
        case "[轉換客戶圖像]":
            if sss.X is None:    st.write("尚未擷取交易數據，請先擷取交易數據！");    sss.LOG.append("尚未擷取交易數據，請先擷取交易數據！")
            else:  sss.Cv = 轉換客戶圖像(sss.X,FFbreaks,MMbreaks,BBbreaks,RRbreaks,Tnow);    check2log(f"客戶圖像: Cv with {sss.Cv.shape} shape", sss.LOG)
        case "[TFM_客群選取]":
            if sss.Cv is None: st.write("尚未轉換客戶圖像，請先計算客戶圖像！");   sss.LOG.append("尚未轉換客戶圖像，請先轉換客戶圖像！")
            else:    sss.CvTA = TFM_客群選取(sss.Cv);    check2log("TFM_客群選取 = CvTA with {sss.CvTA.shape[0]} 位", sss.LOG)
        case "[流失客_客群選取]":
            if sss.Cv is None: st.write("尚未轉換客戶圖像，請先計算客戶圖像！");   sss.LOG.append("尚未轉換客戶圖像，請先轉換客戶圖像！")
            else:    sss.CvTA = 流失客_客群選取(sss.Cv);    check2log("流失客_客群選取 = CvTA with {sss.CvTA.shape[0]} 位", sss.LOG)

    ##== (5).操作日誌 ==##
    st.sidebar.markdown('<h2 style="color: blue;">操作LOG日誌</h2>', unsafe_allow_html=True)
    for i, log in enumerate(sss.LOG, 1): st.sidebar.write(f"({i}). {log}")


#%%######################################################
# def 流失客分析(Cv):
#     #== (4a).設定 TFMs, TClosts 之style
#     TFMs = pd.DataFrame(TFM)
#     print(TFMs)
#     TFMs_styled = TFMs.style.format(formatter="{:,}", na_rep=".").bar(cmap="cool", axis=None)
#     TFMvs = pd.DataFrame(TFMv)
#     print(TFMvs)
#     TClosts = pd.DataFrame(TClost)
#     TClosts_styled = TClosts.style.format(formatter="{:,}", na_rep=".").bar(cmap="Wistia", axis=None)
#     #== (4b).子網頁抬頭
#     st.sidebar.divider()
#     st.sidebar.header("> [4流失客分析] 子控制盤")
#     st.header("[4流失客分析] 子網頁")
#     if st.sidebar.checkbox("* (KDD4) 4.流失客分析 (Cv)"):
#         #== (4c).子網頁抬頭 + TFMs
#         st.subheader("- (4A) 共有" + str(Cv.shape[0]) + "位客戶")
#         #== (4d).客戶價值模型+流失模型 (TFMvs,TFMv0s)
#         cols = st.columns([3, 2])
#         cols[0].subheader("- (4B) 客戶價值模型")
#         cols[0].caption("-- 客戶價值模型")
#         cols[0].table(data=TFMs_styled)   
#         cols[1].subheader("- (4C) 客戶流失模型")
#         cols[1].caption("-- 客戶流失模型")
#         cols[1].table(data=TClosts_styled)        
#         #== (4e).客群選擇
#         CvTA = Cv
#         st.sidebar.subheader("- (4D) 選擇客群,以觀看客戶圖像,再匯出...")
#         yqfA = st.sidebar.multiselect('>> 請選擇客戶的造訪頻次區間(FF0):', sorted(list(Cv.yqf.astype(str).unique())));   yqf = [str(x) for x in yqfA];   print(">>> "+str(yqf))
#         BB0A = st.sidebar.multiselect('>> 請選擇客戶的消費金額區間(MM0):', sorted(list(Cv.BB0.astype(str).unique())));   BB0 = [str(x) for x in BB0A];   print(">>> "+str(BB0))        
#         #== (4f).客群計算
#         cIND1 = np.arange(Cv.shape[0]);   cIND = []
#         if len(yqf) > 0:    cIND1 = np.where([(str(Cv.yqf[k]) in yqf) for k in cIND1])[0]       
#         if len(BB0) > 0:    cIND = np.where([(str(Cv.BB0[k]) in BB0) for k in cIND1])[0]
#         st.sidebar.write("-> 目標客戶數 = "+str(len(cIND))+"位")
#         if len(cIND)==0:
#             st.subheader("<- (4D) 請從左方控制盤選取客群")
#         else:    
#             CvTA = Cv.iloc[cIND]
#             indX = list(np.where(
#                 [k if X.customer[k] in CvTA.index else None for k in np.arange(X.shape[0])])[0])  # indX
#             CvTA.reset_index(inplace=True)
#             st.sidebar.write("-> 涵蓋交易數 = "+str(len(indX))+"筆")
#         #== (4g).客戶圖像+客群之價值模型
#         if st.sidebar.checkbox("確定此客群設定"):
#             setCvTA = "BB0 = "+",".join(BB0) +", yqf = "+",".join(yqf)+", 共計"+str(len(CvTA))+"位"
#             st.subheader("- (4D) 目標客戶: "+setCvTA)
#             st.subheader("- (4E) 客戶圖像")
#             st.caption("-- 客戶圖像")
#             st.dataframe(CvTA)
#             TTA = pd.crosstab(CvTA["FF0"], CvTA["MM0"], margins=True);    print(TTA)
#             TTA.index = [str(x) for x in TTA.index]
#             TTA.columns = [str(x) for x in TTA.columns]
#             st.subheader("- (4F) 所選客群之價值模型")
#             st.caption("-- 所選客群之價值模型")
#             st.dataframe(TTA)
#         #== (4h).匯出至EXCEL檔
#         if st.sidebar.checkbox("(4G) 匯出客群名單 至EXCEL檔"):
#             CvTA.to_excel(setCvTA+".xlsx" )      #==> pandas 提供匯出 EXCEL 的方法 .to_excel()
#             st.subheader("(4H)"+str(len(cIND))+"筆記錄已輸出至 EXCEL檔")
#     return Cv

