# AIp04空間與網站X.py: AI python 實作 - 04: 空間與網站 with 網站架構
# Jia-Sheng Heh, 10/03/2024, revised from AIp03圖形可視化W.py
# Usage: streamlit run AIp04空間與網站X.py --> http:/localhost:8501

import streamlit as st  # ==> [[AIp04/C0-st啟動]]
from streamlit_navigation_bar import st_navbar
import plotly_express as px
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import os
wkDir = "c:/Users/jsheh/Desktop/postWorking/1131courses/1131AIpractices/"
os.chdir(wkDir)
print(os.getcwd())

#%%####### (W).網站系統基本架構 ##########

#%%##===== (W1).自定公用函式庫 =====#####
def getX(Xname):       ##== X=getX(Xname): 自交易檔(KDD1)讀取交易數據並(KDD3)設定標籤
    X = pd.read_csv(Xname)
    # -- 還有很多其他產生此標籤的方法, 這裡只是取其中較方便的一種
    X["date"] = pd.to_datetime(X["datetime"]).dt.date
    X["year"] = pd.to_datetime(X["datetime"]).dt.year
    X["yq"] = pd.PeriodIndex(X.date, freq='Q')
    X["ym"] = pd.PeriodIndex(X.date, freq='M')
    return(X)
def 成交結構(XX, yy):  ##== Sv=成交結構(XX,yy): 計算(KDD3)針對特定時段(yy)交易(XX)的成交結構Sv
    Sv = XX.groupby(yy).agg({"customer": "nunique", "invoiceNo": "nunique",
                             "amount": "sum", "quantity": "sum"}).reset_index()
    Sv.columns = [yy, "來客數", "交易數", "營業額", "銷售量"]
    Sv["客單價"] = round(Sv["營業額"] / Sv["交易數"], 1)
    Sv["客單件"] = round(Sv["銷售量"] / Sv["交易數"], 2)
    print(Sv.shape)
    print(Sv.head(3))
    return(Sv)

#%%##===== (W2).儀表板函式庫: 前台(a)navbar,(b)sidebar,(c)canvas,後台(d) =====#####
def 擷取交易(Xname):  ##== (KDD1)擷取交易儀表板: X = 擷取交易(fnameX) ==##
    ##== (d).後台 ==##
    X = getX(wkDir+Xname)
    print("\n\n>>>>> 擷取交易數據 (-->XXX) -----")  # -- 偵錯用
    # == (b).前台-sidebar ==##  #==> [[AIp04/C4)(2)垂直流程]]
    st.sidebar.header("== (KDD1).擷取交易數據 -- ")
    st.sidebar.write("* 交易檔名 = "+Xname)
    st.sidebar.write("* 記錄筆數 = ", X.shape[0])
    ##== (c).前台-canvas ==##
    st.header("== (KDD1).擷取交易數據"+Xname+" --")
    st.subheader("* 記錄筆數 = "+str(X.shape[0]))
    st.dataframe(X.head(3))
    return X
def 總成交結構(X):    ##== (KDD3)總成交結構儀表板: Svyq = 總成交結構(X) ==##
    ##== (d).後台 ==##
    Svyq = 成交結構(X, "yq")
    # == (b).前台-sidebar ==##  #==> [[AIp04/C4)(2)垂直流程]]
    st.sidebar.header("== (KDD3).總成交結構 --")
    st.sidebar.write("* 共有"+str(X.shape[0])+"筆交易記錄")
    st.sidebar.write("* 營業額共計"+str(sum(X.amount))+"元")
    # == (c).前台-canvas ==##   #==> [[AIp04/C4)(3)流程中的文字說明]]
    st.header("== (KDD3).總成交結構 --")
    st.dataframe(Svyq)
    return Svyq
def 非會員標籤(X):    ##== (KDD2)非會員標籤: X = 非會員標籤(X) ==##
    #== (d).後台 ==##   #==> [[AIp04/C3)(1)後台]]
    print(type(X));    print(X.shape);    print(X.head(3))
    CvFF = X.groupby("customer").agg({"invoiceNo": "nunique"}).reset_index().sort_values('invoiceNo', ascending=False)
    Nc = CvFF.shape[0]
    CvFF1 = CvFF[0:len(list(np.where(CvFF.invoiceNo > 5)[0]))]
    fig = px.pie(CvFF1, values="invoiceNo", names="customer", title="客戶交易數")
    fig.update_traces(textposition='inside', textinfo='label+value',pull=[0.2*int(ff > 300) for ff in CvFF1.invoiceNo])
    #== (c).前台-canvas ==##   #==> [[AIp04/C4)(3)流程中的文字說明]]
    st.header("== (KDD2).非會員佔比 --")
    cols = st.columns([1, 1])  # -- (d).前台--canvas
    #==> [[AIp04/C4)(1)水平流程]]
    cols[0].subheader("* (A).共有" + str(X.shape[0]) + "筆交易記錄")
    cols[0].subheader("* (B).共有" + str(len(X.customer.unique())) + "位客戶")
    cols[0].subheader("* (C).(KDD2) 各客戶的造訪次數 (遞減排序前" + str(CvFF1.shape[0]) + "位):")
    cols[0].dataframe(CvFF1)
    cols[1].subheader("* (D) 各客戶的造訪次數佔比的圓形圖")
    cols[1].plotly_chart(fig, theme="streamlit", use_container_width=True)
    #== (b).前台-sidebar ==##  #==> [[AIp04/C4)(2)垂直流程]]
    st.sidebar.header("== (KDD3).設定非會員標籤 --")  #==> 定義 "會員"標籤 [[AIp04/4D-定義 X身分]]
    KnonMember = st.sidebar.slider('- 非會員 數目 = ', 0, Nc, 0)  #==> [[AIp04/C4)(2)垂直流程]]
    CvFF["會員"] = ["N" if k < KnonMember else "Y" for k in np.arange(CvFF.shape[0])];   print(CvFF.head(2))
    CvFF.set_index("customer", inplace=True);    CvFF.head(10)
    X["會員"] = [CvFF["會員"][x] if x in CvFF.index else None for x in X["customer"]];   print(X.head(2))
    CvFF.reset_index(inplace=True)
    TnonMember = sum(list(CvFF.invoiceNo[CvFF["會員"] == "N"]))
    Tall = len(X.invoiceNo.unique())
    st.sidebar.write("* 非會員客戶設為: ", ",".join(list(CvFF.customer[CvFF["會員"] == "N"])))
    st.sidebar.write("* 非會員交易數 = ", TnonMember, "筆, 佔總交易數的", round(100*TnonMember/Tall, 2), "%")
    if st.sidebar.checkbox(">> (KDD5).確認 非會員 範圍,並存檔...", value=False):  #==> [[AIp04/C4)(2)垂直流程]]
        X.to_csv("X1.csv", encoding='BIG5', index=False)         #==> [[AIp04/D0)(2)表單]]
        st.sidebar.write("* 非會員相關數據已存檔")
    return X
def 成交結構分析(X):  ##== (KDD4)成交結構分析(X-->yq,ym) ==##
    #== (d).後台 ==##   #==> [[AIp04/C3)(1)後台]]
    Svyq = 成交結構(X, "yq")
    Svym = 成交結構(X, "ym")
    FIGym = go.Figure(go.Scatter(x=Svym['ym'].dt.to_timestamp(), y=Svym['營業額'], ))
    FIGym.update_traces(mode='markers+lines')
    FIGym.update_xaxes(tickformat='%Y-%m', dtick='M1')
    #== (c).前台-canvas ==##   #==> [[AIp04/C4)(3)流程中的文字說明]]
    st.header("== (KDD4).成交結構分析 --")
    cols = st.columns([1, 1])  # -- (d).前台--canvas
    cols[0].subheader("* (A) 各年季成交結構")
    cols[0].dataframe(Svyq)
    cols[1].subheader("* (B) 各年月營業額趨勢")
    cols[1].plotly_chart(FIGym, theme="streamlit", use_container_width=True)
    #== (b).前台-sidebar ==##  #==> [[AIp04/C4)(2)垂直流程]]
    #==> 定義 "會員"標籤 [[AIp04/4D-定義 X身分]]
    st.sidebar.header("== (KDD4).成交結構分析 --")
    st.sidebar.write("* 共有"+str(X.shape[0])+"筆交易記錄")
    st.sidebar.write("* 營業額共計"+str(sum(X.amount))+"元")
    return X
def 成交結構比較(X):  ##== (KDD4)成交結構比較(X["會員"]) ==##
    #== (d).後台 ==##   #==> [[AIp04/C3)(1)後台]]
    Svyq = 成交結構(X, "yq")
    SvyqM = 成交結構(X.loc[X["會員"] == "Y"], "yq")
    SvyqN = 成交結構(X.loc[X["會員"] == "N"], "yq")
    #== (c).前台-canvas ==##   #==> [[AIp04/C4)(3)流程中的文字說明]]
    st.header("== (KDD4).成交結構比較 --")
    cols = st.columns([1, 1, 1])  # -- (d).前台--canvas
    cols[0].subheader("* (A) 總成交結構");     cols[0].dataframe(Svyq)
    cols[1].subheader("* (B) 會員成交結構");   cols[1].dataframe(SvyqM)
    cols[2].subheader("* (C) 非會員成交結構"); cols[2].dataframe(SvyqN)
    #== (b).前台-sidebar ==##  #==> [[AIp04/C4)(2)垂直流程]]
    st.sidebar.header("== (KDD4).成交結構比較 --")  # ==> [[AIp04/D0)(2)表單]]
    st.sidebar.write("* 共有 "+str(len(list(sss.X.customer[sss.X["會員"] == "N"].unique())))+" 位非會員")
    st.sidebar.write("* 非會員 = "+",".join(list(sss.X.customer[sss.X["會員"] == "N"].unique())))
    if st.sidebar.checkbox(">> (KDD5).將成交結構比較表 存檔...", value=False):    #==> [[AIp04/C4)(2)垂直流程]]
        pd.concat([Svyq, SvyqM, SvyqN], axis=1).to_csv("成交結構比較.csv", encoding='BIG5', index=False)  # ==> [[AIp04/D0)(2)表單]]
        st.sidebar.write("* 成交結構比較表已存檔--成交結構比較.csv")
    return X

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
    # st.set_option('deprecation.showPyplotGlobalUse', False)
    st.set_page_config(page_title="SPC-S01 RDS系統", page_icon="✅", layout="wide",)  #==> [[AIp04/C4)(5)加上頁註,頁標題等]]
    page = st_navbar(["[擷取交易]", "[總成交結構]", "[非會員標籤]", "[成交結構分析]", "[成交結構比較]"])

    ##== (2).設定session初始值等 ==##
    Xname = "XXX.csv"
    sss = initSSS(["X", "TWH", "Svyq", "Xname"], "AIp03圖形可視化W"+"--"+Xname);   sss.Xname = Xname

    ##== (3).設定 前台((b)sidebar + (c)canvas)主標題 ==##
    st.title("AIp04空間與網站: 初步運營分析儀表板(S01)")
    st.sidebar.title("初步運營分析(S01)控制盤--")

    ##== (4).導航切換: 前台(a)navbar-->儀表板函式(b,c,d) ==##
    match page:
        case "[擷取交易]":
            sss.X = 擷取交易(sss.Xname)
            check2log(f"擷取交易: {sss.Xname} to get X with {sss.X.shape[0]} records", sss.LOG)
        case "[總成交結構]":
            if sss.X is None:
                st.write("尚未擷取交易數據，請先擷取交易數據！")
                sss.LOG.append("尚未擷取交易數據，請先擷取交易數據！")
            else:
                sss.Svyq = 總成交結構(sss.X)
                check2log(f"總成交結構: Svyq with {sss.Svyq.shape} shape", sss.LOG)
        case "[非會員標籤]":
            if sss.X is None:
                st.write("尚未擷取交易數據，請先擷取交易數據！")
                sss.LOG.append("尚未擷取交易數據，請先擷取交易數據！")
            else:
                sss.X = 非會員標籤(sss.X)
                check2log("非會員標籤: 非會員 = "+",".join(list(sss.X.customer[sss.X["會員"] == "N"].unique())), sss.LOG)
        case "[成交結構分析]":
            if sss.X is None:
                st.write("尚未擷取交易數據，請先擷取交易數據！")
                sss.LOG.append("尚未擷取交易數據，請先擷取交易數據！")
            else:
                sss.X = 成交結構分析(sss.X)
                check2log(f"成交結構分析: {sss.Xname} to get X with {sss.X.shape[0]} records", sss.LOG)
        case "[成交結構比較]":
            if sss.X is None:
                st.write("尚未擷取交易數據，請先擷取交易數據！")
                sss.LOG.append("尚未擷取交易數據，請先擷取交易數據！")
            else:
                sss.X = 成交結構比較(sss.X)
                check2log("成交結構比較: 非會員 = "+",".join(list(sss.X.customer[sss.X["會員"] == "N"].unique())), sss.LOG)

    ##== (5).操作日誌 ==##
    st.sidebar.markdown('<h2 style="color: blue;">操作LOG日誌</h2>', unsafe_allow_html=True)
    for i, log in enumerate(sss.LOG, 1): st.sidebar.write(f"({i}). {log}")
