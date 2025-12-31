# AIp09時序與回歸A.py: AI python 實作 - 09: 時間序列、回歸與滾動式預測 -- 教材部份
# Jia-Sheng Heh, 11/02/2024, revised from bda107s/I710.R
    

#%%####### Import ##########
import os   
wkDir = os.path.dirname(os.path.abspath(__file__));   os.chdir(wkDir);   print(os.getcwd())
import pandas as pd
import numpy as np
from io import StringIO
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser";   
pd.options.display.max_columns = None  #-- 顯示所有列
pd.options.display.width = None        #-- 自動調整列寬

#%%####### (A) 回歸 (Regression) ##########

#%%##===== (A1) 數據模型(Data Model)符號 =====#####

##== 系統/模型/函數(System/Model/Function, M):  因變數/輸出數據y = M( 引數/輸入數據u ) ==##

#--  (1) 訓練階段(Training/Learning/Modeling/Estimation Phase): (u, y) -> M
#        -- 由輸入/輸出 u與y，求取(估測estimate)模型M# ................................. 聚類 linkage()/fcluster()
#                                                                                       關連 apriori()/association_rules()
#                                                                                       回歸 LinearRegression()/LR.fit(X, y)
#--  (2) 預測/應用階段(Prediction/Estimation/Production/Application Phase): (u_new, M#) -> y_predict
#        -- 以所估測的模型M#與新的輸入 u_new，求取(估測)新的輸出 y_predict .............. 回歸 LR.predict()

##== 機器學習(Machine Learning) ==##

#    -- (1) 監督式學習 (Supervised learning): 具範例(u,y), y為教師(teacher, desired output) y, 以求得y=M(u)
#           (1A)回歸(regression):     y 為連續數據   --> 本單元 AIp09: LinearRegression()/LR.fit(X, y) --> LR.predict()
#           (1B)分類(classification): y 為離散數據   --> 下一單元 AIp10   
#    -- (2) 無監督式學習 (Unsupervised learning): 無輸出y, 目標在於發掘輸入(u)的隱含特徵 --> 資料採擷(Data Mining)
#           (2A)聚類(clustering):           計算數據u的相似度，以產生其分類。 --> 上兩單元AIp07: 聚類 linkage()/fcluster()
#           (2B)關聯規則(association rule): 計算多數據(ui-uj)間的關連。       --> 前一單元AIp08: 關連 apriori()/association_rules()
#           (2C)數據序列(data sequencing):  計算多數據(ui-uj)間的時序關係。   --> 待會的(B)與(C)

##== 名詞比對 ==##

#    -- (1).聚類(clustering): 把數據物件集(u, 無y)，劃分成多個組/簇(求取M(u))的過程，使得簇內的對具有很高的相似性
#       -- 與分類類似，但每個客戶(物件)的類標號是未知(無y)的，需要發現這些分組(簇)(求取M(u))

#    -- (2).分類(classification): 找出一組離散數據物件(y)的共同特點，按照分類模式(y=M(u))將其劃分為不同的類y
#                             : M(u)=y, y為離散值
#       -- 預測分類: 找出描述和區分數據類/概念的模型(y=M(u))，以預測未知類標號對象(u_new)的類標號(y_predict)
#                    : M(u_new)=y_predict

#    -- (3).回歸(regression): 通過函數(M)表達連續數據映射(u->y)的關係，來發現屬性值之間的依賴關係
#                         : M(u)=y, y為連續值 ...... LinearRegression()/LR.fit(X, y) 
#       -- 預測回歸: 建立連續值函數的預測模型(y=M(u))，可預測缺失的/難以獲得的數值數據值(u_new)
#                    : M(u_new)=y_predict    ...... LR.predict()

#####===== (A2) 線性回歸(Linear Regressioin) =====#####
# 這部分線性回歸，就是從二維數據的分布去看趨勢，畫出趨勢線。
# 線性回歸 = 用一條直線「擬合」散點數據的趨勢 📈

#%%####### (1).車速(speed)與停(刹)車距離(dist)的關係 ##########
#    -- [1920s, https://www.key2stats.com/data-set/view/357]
cars = pd.read_csv("cars.csv");   print(cars.shape);   print(cars.head(3))  #-- (50, 3)
#    Unnamed: 0  speed  dist
# 0           1      4     2
# 1           2      4    10
# 2           3      7     4
fig = go.Figure()
fig.add_trace(go.Scatter( x=cars['speed'], y=cars['dist'], mode='markers', name='Cars Points'))
fig.update_layout( title="Linear Regression: Speed vs Distance", xaxis_title="Speed", yaxis_title="Distance")
fig.show()

#%%== (2).線性回歸(Simple Linear Regression, SLR)模型 ==##
from sklearn.linear_model import LinearRegression
#    -- 1).假設: yi = b0 + b1*xi + ei, 其中 b0,b1: 回歸係數, ei: 隨機雜訊
X = cars[["speed"]].values.reshape(-1, 1).astype(float) ;   print(X[0:7])  #-- [[ 4.][ 4.][ 7.][ 7.][ 8.][ 9.][10.]]  #== 自變量 (speed)
y = cars["dist"].values.astype(float) ;                     print(y[0:7])  #-- [ 2. 10.  4. 22. 16. 10. 18.]  #== 因變量 (dist) --> y的真實數據
#%%  -- 2A).訓練階段(Training): (x, y) -> M
LR = LinearRegression()
LR.fit(X, y)   #==> 求取回歸模型LR
#    -- 2B).預測階段(Prediction): (u_new, M#) -> y_predict
y_predict = LR.predict(X)
#           --> 內插(interpolation): 預測在原數據範圍內的資料物件,有一定的參考性
#           --> 外插(extrapolation): 預測超過原數據範圍的資料物件,除非必要,儘量避免 (但比較有價值)
#    -- 3).回歸線(Regression Line)
fig.add_trace(go.Scatter( x=cars['speed'], y=y_predict, mode='lines', name='Regression Line'))  #==> 前圖加上回歸線(Regression Line)
fig.show()

#%%== (3).線性回歸的原理: LSE (最小平方誤差Least Square Error) ==##
#    -- 回歸係數:
print(f"截距 (intercept) = { np.round(LR.intercept_,3 )}, 斜率 (slope) = { np.round(LR.coef_[0],3) }")   
           #-- 截距 (intercept) = -17.579, 斜率 (slope) = 3.932
#    -- 線性回歸係數的公式
#       1).將誤差平方(Squared Error) E = sum( ( y - (w0+w1*u) )^2 ) 最小化
#       2).也就是將誤差 E 針對 w1, w0 微分, 使 dE/dw1 = 0 及 dE/dw0 = 0
#       3).可以求得下列回歸係數 w1 和 w0 的公式:
mean_X = np.mean(X);    print(mean_X)   #-- 15.4
mean_y = np.mean(y);    print(mean_y)   #-- 42.98
w1 = np.sum((X.flatten()-mean_X)*(y-mean_y)) / np.sum((X.flatten()-mean_X)**2);  print(f"w1={w1}")   #-- [1] 3.932408759124088
w0 = mean_y - w1 * mean_X;  print(f"w0={w0}")                                                        #-- [1] -17.57909489051096 

#%%##===== (A3) 非線性回歸(NonLinear Regression, NLR) [殷,6.4] =====#####

##== (1).非線性回歸的可能關係 ==##
#    -- y = b0 + b1*exp(x)
#    -- y =  + b1*ln(x)
#    -- y = b0 + b1*x + b2*x^2 + ... + bn*x^n
#    ==> 非線性回歸 轉為 線性回歸: y = y = b0 + b1*u1 + b2*u2 + ... + bm*um
##== (2).彩色顯影的非線性回歸例 ==##
#    -- 銀的光學密度zeta, 形成燃料eta的光學密度 ==> 求eta關於zeta的回歸方程
#    -- 試驗資料
zeta = np.array([0.05, 0.06, 0.07, 0.1, 0.14, 0.2, 0.25, 0.31, 0.38, 0.43, 0.47])
eta = np.array([0.1, 0.14, 0.23, 0.37, 0.59, 0.79, 1, 1.12, 1.19, 1.25, 1.29])
#    -- 圖形關係
fig = go.Figure()
fig.add_trace(go.Scatter(x=zeta, y=eta, mode='markers+lines', name='Original Data'))
fig.update_layout( title="Nonlinear Regression Fit", xaxis_title="zeta", yaxis_title="eta",
    legend_title="Legend", template="plotly_white")
fig.show()
#    ==> 設定之回歸方程 (通常需要領域知識來建構): eta = y = A*exp(b/x) = A*exp(b/zeta),  b<0
#%%== (3).非線性回歸式轉換為線性回歸式來求解 ==##
#    -- 非線性回歸方程，等號兩方共取對數，可以得到 線性回歸方程
#       ln(y) = ln(A) + b/x = ln(eta) = ln(A) + b/zeta
#          Y     = a  + b*U, 其中...
#    -- 線性回歸方程 的 (轉換後)資料
Y = np.log(eta);   print(np.round(Y,3))  #-- [-2.303 -1.966 -1.47  -0.994 -0.528 -0.236  0.     0.113  0.174  0.223  0.255]
U = 1/zeta;        print(np.round(U,3))  #-- [20.    16.667 14.286 10.     7.143  5.     4.     3.226  2.632  2.326  2.128]

#%%== (4).檢視(U,Y)的相關係數(correlation coefficient) ==##
correlation_coefficient = np.corrcoef(U,Y)[0,1];   print(correlation_coefficient)  #-- -0.9982764002046934
#-- = Ruy/sqrt(Ruu*Ryy) = (sum(U*Y)-k*mean(U)*mean(Y)) / ( sum((U-mean(U))^2) * sum((Y-mean(Y))^2) ) 
#-- [殷,p.154] n=2 之 (n-2)=(11-2)=9 自由度 的相關性係數顯著性為0.602/0.735 (對顯著性水準=0.05/0.01)
#%%== (5).代入 線性回歸模型 的數據準備(形成數據框) ==##
lin_reg = LinearRegression()
U = U.reshape(-1, 1)  #-- 轉換為 2D 形狀以符合 scikit-learn 的輸入要求
lin_reg.fit(U, Y);    #-- Y = a + b * U = 0.5476 - 0.1459 * U
b = lin_reg.coef_[0];     print(b)  #-- -0.1459291159370808 -- 斜率 b = Ruy/Ruu
a = lin_reg.intercept_;   print(a)  #-- 0.5476497293362032  -- 截距 a = mean(Y) - b * mean(U)

#%%== (6).線性回歸式 轉換回 非線性回歸式 並進行預測 ==##
A = np.exp(a);   print(A)   #-- 1.7291841874458929
eta_pred = A * np.exp(b/zeta);   print(np.round(eta_pred,3))  #-- [0.093 0.152 0.215 0.402 0.61  0.834 0.965 1.08  1.178 1.232 1.268]
#     #==> 非線性回歸式: y = A*exp(b/x) = 1.729184*exp(-0.1459/zeta)
#%%== (7).圖形關係
fig.add_trace(go.Scatter(x=zeta, y=eta_pred, mode='markers+lines', name='Nonlinear Regression Fit', line=dict(dash='dash')))
fig.show()

#%%##===== (A4) 連續型監督式學習的評估 =====#####
# Model evaluation for Continuous Supervised Learning (Regression) 

##== (1).回歸評估準則 [殷,6.5] ==##
#    -- 準確率: 正確地預測或先前未見過的數據的屬性值
#    -- 魯棒性(韌性): 給定雜訊數據或缺失數據時的正確預測能力
#    -- 速度: 使用的計算花費
#    -- 可伸縮性: 給定大量數據時的有效建構能力
#    -- 可解釋性: 提供的理解和洞察能力，因為主觀很難評估

#== (2).真實輸出/預測輸出: 以cars為例 (cars-->actual/predicted) ==##
y = cars["dist"].values.astype(float) ;                     print(y[0:7])  #-- [ 2. 10.  4. 22. 16. 10. 18.]  #== 因變量 (dist) --> y的真實數據
actual = y;   print(actual[0:7])         #-- [ 2 10  4 22 16 10 18] #== 真實的輸出 
predicted = np.round(y_predict,2);   print(predicted[0:7]) 
                           #-- [-1.85 -1.85  9.95  9.95 13.88 17.81 21.74] #== 預測的輸出

#%%== (3).單預測量的評估 ==##
E  = actual-predicted;          #== 誤差(error) E: 真實樣本值與估計值的誤差也稱為“殘差(Residual)”
aE = np.abs(actual-predicted);  #== 絕對誤差(absolute error) aE: 此值是越小越好
SE = (actual-predicted) ** 2;   #== 平方誤差(squared error) SE
e  = (actual-predicted)/actual; #== 相對誤差(relative error) e: 一般此值應控制在1與-1之間
ERR_df = pd.DataFrame({'actual': actual, 'predicted': predicted,
                       'E (Residual)': E, 'aE (Absolute Error)': aE,
                       'SE (Squared Error)': SE,'e (Relative Error)': e })
ERR_df.iloc[0:4]
#    actual  predicted  E (Residual)  aE (Absolute Error)  SE (Squared Error)  e (Relative Error)  
# 0       2      -1.85          3.85                 3.85             14.8225            1.925000  
# 1      10      -1.85         11.85                11.85            140.4225            1.185000  
# 2       4       9.95         -5.95                 5.95             35.4025           -1.487500  
# 3      22       9.95         12.05                12.05            145.2025            0.547727  

#%%== (4).多預測量的評估--絕對誤差: 檢驗誤差/泛化誤差 ==##
MAE  = np.mean(aE);               print(MAE)  #-- [1] 11.58012  #== 均值絕對誤差(Mean Absolute Error/Deviation) MAE/MAD
MAPE = np.mean(np.abs(e)) * 100;  print(MAPE) #-- [1] 38.36881  #== 平均絕對百分誤差(Mean Absolute Percentage Error) MAPE: 一般小於10%的可被接受
MSE  = np.mean(SE);               print(MSE)  #-- [1] 227.0704  #== 均方誤差(Mean Squared Error) MSE: 可以放大誤差的作用

# 最常見的評估指標之一: 均方根誤差(Root Mean Squared Error, RMSE)
# 歐式距離(Euclidean Distance)的概念: RMSE = sqrt( sum( (yi - y_predicti)^2 ) / n )
RMSE = np.sqrt(MSE);              print(RMSE) #-- [1] 15.06886  #== 均方根誤差(Root Mean Squared Error) RMSE

#%%== (5) 多預測量的評估--相對誤差: 檢驗誤差/泛化誤差 ==##
RAE = np.sum(aE)/np.sum(np.abs(actual - np.mean(actual)));   print(RAE)  #-- [1] 0.5595125  ##== 相對絕對誤差(Relative Absolute Error)
NMSE = np.sum(SE)/np.sum((actual - np.mean(actual)) ** 2);   print(NMSE) #-- [1] 0.3489206  
##== 相對平方誤差(Relative Squared Error) / 正規化均方誤差(Normalized Mean Squared Error) NMSE = SSe/SSt: 越小越好

#%%== (6).R Squared, 決定係數, 擬合度, 誤差百分比(percentage of reduced error, PRE) 
#    -- R^2 = 1 - SSe/SSt = 1 - 殘差變異量/總變異量 = SSregression/SSt = 回歸變異量/總變異量
#    -- 表示是預測值對實際值的解釋程度，越接定1, 預測值越接近真實值 
# R_square 為真實值與預測值的差異程度
# 數據要求出值，數據上證明模型可行，才能去解讀。

R_squared = 1 - (np.sum(SE) / np.sum((actual - np.mean(actual)) ** 2));  
r_squared = LR.score(X, y)
print(f"判定係數: {R_squared} / {r_squared}")  #-- 判定係數: 0.6510401155783003 / 0.6510793807582509
#    -- R_squared > 0.75:    回歸模型擬合度好, 可解釋程度較高
#    -- R_squared: 0.5-0.75: 回歸模可擬合度可接受, 但需再修正模型
#    -- R_squared < 0.5:     回歸模型擬合有問題, 需調整自變量再進行回歸


#%%####### (B) 時間序列 (Time Series) ########## 

#%%##===== (B1) 下載時間序列檔(web-->file) =====##### 

##== [R语言时间序列初探!, (Avril Coghlan, 2016), A Little Book of R For Time Series, 0.2]
kingsURL = "http://robjhyndman.com/tsdldata/misc/kings.dat";    kingsFILE = "kings.dat"
birthsDIR = "http://robjhyndman.com/tsdldata/data/";    
birthsFILE = "nybirths.dat";     fancyURL = "fancy.dat";    souvenirFILE = "souvenir.dat"

##== (1).R語言中的寫法: scan() 直接獲取數據 ==##
# kings = scan(kingsURL)    #==> 看看! 多簡單...

#%%##===== (2).python最常見的作法: request() 直接獲取數據 =====##### 
import requests
response = requests.get(kingsURL)  #-- requests.get(url): 從給定的 URL 下載數據
print(response.text)               #-- 403 - Forbidden | Access to this page is forbidden.
#-- 可能是因為伺服器允許特定的工具訪問(對R的請求不設限),而限制了其他類型的自動化工具(對其他工具更為嚴格)

#%%== (3).使用作業系統中的網頁下載工具: wget/curl 指令 ==##
os.system(f"wget -O {kingsFILE} {kingsURL}")   #-- 1 ---> 未安裝 wget 工具
os.system(f"curl -o {kingsFILE} -H 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36' {kingsURL}")
                                               # 403 - Forbidden | Access to this page is forbidden. ---> 還是被擋

#%%== (4).以 Selenium 模擬瀏覽器下載(繞過伺服器的限制)，再以BeautifulSoup來移除<html>標籤 ==##
#== Selenium: 網頁的自動化測試工具，模擬用戶在瀏覽器中的操作 (如加載網頁、點擊按鈕、填寫表單、滾動頁面、抓取數據等)，可抓取動態生成的網頁內容
#   -- 主要功能: 模擬瀏覽器操作 (包括點擊、滾動、鍵盤輸入、獲取網頁內容等), 適合動態網頁數據抓取：可以執行 JavaScript 後抓取最終顯示的內容。
#   -- 注意事項: 1)效率相對較低，對於大規模的數據抓取不太合適。 2)網站的反爬蟲系統可能會檢測到並限制操作，需要結合代理、隨機等待等技術來避免被封鎖。
def download_with_selenium(url, filepath):   ##== 下載url網址的內容到filepath
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    from bs4 import BeautifulSoup
    try:
        options = webdriver.ChromeOptions()             #== (1).selenium/webdriver模擬瀏覽器
        options.add_argument("--headless")              #   -- 瀏覽器 無窗口顥示
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options) #-- 以webdriver_manager來安裝和設置ChromeDriver
        driver.get(url)                                 #   -- 讀取url
        page_source = driver.page_source   
        driver.quit()                                   #   -- 關閉瀏覽器
        soup = BeautifulSoup(page_source,'html.parser') #== (2).以BeautifulSoup移除HTML tags
        text = soup.get_text()                          #   -- 提取文字
        with open(filepath, 'w') as file:  file.write(text) #== (3).寫檔
        print(f"Successfully downloaded {filepath} using Selenium.")
    except Exception as e:
        print(f"Error downloading data using Selenium: {e}")
# download_with_selenium(kingsURL, kingsFILE)                  #-- Successfully downloaded kings.dat using Selenium.
# download_with_selenium(birthsDIR + birthsFILE, birthsFILE)   #-- Successfully downloaded nybirths.dat using Selenium.
# download_with_selenium(birthsDIR + fancyURL,  souvenirFILE)  #-- Successfully downloaded sourvenir.dat using Selenium.


#%%##===== (B2) 讀取時間序列檔(file-->TS) =====##### 
def readTS(filepath, freq=None, start=None, reset_index=True):   ##== 讀取時間序列檔 ==##
    data = pd.read_csv(filepath, header=None, comment='#', skip_blank_lines=True, skiprows=3)
    if start: timeseries = pd.Series(data[0].values, index=pd.date_range(start=start, periods=len(data), freq=freq))
    else:     timeseries = pd.Series(data[0].values)
    if reset_index: df = timeseries.reset_index();   df.columns = ['index','values'];   return df
    else:           return timeseries
            
#%%== (1).kings.dat: 從威廉一世開始的英國國王的去世年齡資料。(原始出處:Hipel and Mcleod, 1994) ==##
kings  = readTS("kings.dat");   print(kings.shape);   print(kings[0:3])   #-- (42, 2)
#    index values
# 0      0     60
# 1      1     43
# 2      2     67
px.line(kings, x="index", y="values", title='kingstimeseries')
#%%== (2).births.dat: 紐約 1946/1-1959/12 的每月出生人口數量(由牛頓最初收集)資料集 ==##
births  = readTS("nybirths.dat", freq='ME', start='1946-01');   print(births.shape);   print(births[0:3])   #-- (165, 2)
#        index  values
# 0 1946-01-31  24.740
# 1 1946-02-28  25.806
# 2 1946-03-31  24.364
px.line(births, x='index', y="values", title='birthstimeseries')
#%%== (3).souvenir.dat: 昆士蘭海濱度假勝地紀念品商店 1987/1-1987/12 的每月銷售資料(原始資料來源於 Wheelwright and Hyndman, 1998) ==##
souvenir = readTS("souvenir.dat", freq='ME', start='1987-01');   print(souvenir.shape);   print(souvenir[0:3])
px.line(souvenir, x='index', y="values", title='souvenirtimeseries')   #-- (81, 2)
#        index   values
# 0 1987-01-31  3547.29
# 1 1987-02-28  3752.96
# 2 1987-03-31  3714.74
#%%== (4).AirPasseengers.csv:  1949-1960 美國航空公司每月乘客總數 ==##
# [Sadrach Pierre. (2024). A Guide to Time Series Analysis in Python. https://builtin.com/data-science/time-series-python]
AP = pd.read_csv("AirPassengers.csv");   
AP['Month1'] = pd.to_datetime(AP['Month'],format='%Y-%m'); print(AP.shape); print(AP.head(3))  #-- (144, 3)
#      Month  #Passengers     Month1
# 0  1949-01          112 1949-01-01
# 1  1949-02          118 1949-02-01
# 2  1949-03          132 1949-03-01
px.line(AP, x='Month1', y="#Passengers", title='Monthly #Passengers')

#%%##===== (B3) 時間序列的分解(decomposition) =====#####
from statsmodels.tsa.seasonal import seasonal_decompose

##== (1).序列數據分解為三個主要部分 ==##
#      -- 1).趨勢(Trend): 描述數據隨時間變化的長期走勢。
#      -- 2).季節性(Seasonal): 反映出資料中短期內重複出現的模式，通常與季節或週期性因素有關。
#      -- 3).隨機(Random或Residual): 代表資料中無法通過趨勢或季節性解釋的變動，這些變動通常是不規則的或不可預測的。

decomposition = seasonal_decompose(AP['#Passengers'],model='additive', period=7)
decomposition.plot()
##== (2).時間序列分解圖：展示了觀察值（Observed）、趨勢（Trend）、季節性（Seasonal）、和隨機（Residual）成分 ==##
#      -- 1).趨勢線(trend)顯示了隨著時間增加，乘客數量可能呈現穩步上升的趨勢。
#      -- 2).季節性成分(seasonal)展示了每年特定月份的乘客數量可能因假期或旅遊旺季而上升，這便是季節性成分--每年的週期性變化。
#      -- 3).隨機成分(resid)顯示了不可預測的波動。

#%%==(3).趨勢和季節性成分的平均值 ==> 了解數據中這些成分的中心趨勢 ==##
trend_mean = decomposition.trend.mean()     
seasonal_mean = decomposition.seasonal.mean()
print(f"趨勢成分的平均值: {trend_mean:.4f}, 季節性成分的平均值: {seasonal_mean:.4f}")
            #-- 趨勢成分的平均值: 279.7547, 季節性成分的平均值: 0.0054

#%%##===== (B4) 時間序列的數據穩定性(stationarity) =====#####

##== (1).差分去趨勢: 以一階差分(每個值減去前一個值) 來移除趨勢，來檢查數據平穩性，以進一步模型建構 ==##
diff_data = AP['#Passengers'].diff().dropna()   #-- 移除差分產生的 NaN 值（因為第一個元素沒有前一個值可供計算差分）
diff_fig  = px.line(x=diff_data.index, y=diff_data.values, labels={'x': '時間', 'y': '差分值'}, title='一階差分後的數據')
diff_fig.update_layout(xaxis_title='時間', yaxis_title='差分值');   diff_fig.show()

##== (2).數據的穩定性(stationarity) ==##
#      -- 重要性: 許多時間序列模型（如 ARIMA 模型）假設數據是平穩的。
#      -- 平穩性: 是指數據的均值、方差和自相關結構不隨時間變化。
#         - 平穩數據：時間序列的統計特性（如均值和方差）隨時間保持穩定。
#         - 非平穩數據：數據隨時間變化且具有趨勢、週期或隨機遊走行為，統計特性並不穩定。

##== (3).ADF (Augmented Dickey-Fuller) 測試 : 檢查時間序列數據是否平穩的統計方法 [Dickey and Fuller, 1979] ==##
# 時變系統：為非平穩系統
# 非時變系統：為平穩系統

# 社會科學：抽樣兩個標準差(95%) 來檢驗顯著性
# 自然科學：抽樣三個標準差(99%) 來檢驗顯著性

from statsmodels.tsa.stattools import adfuller
#      -- 目的: 檢查時間序列是否具有單位根(unit root)，單位根的存在表示數據不是平穩的--> 原假設(H0):序列具有單位根，即數據非平穩 (拒絕=序列穩定)
#      -- 單位根(unit root): 時間序列的「強記憶性質」, 也就是數值的變動會影響未來的數值，這使得數據具有趨勢或隨機遊走的特性 (因此統計特性會變化)
adf_result = adfuller(diff_data);   print(adf_result)
# (-2.829266824169992, 0.0542132902838265, 12, 130, 
#  {'1%': -3.4816817173418295, '5%': -2.8840418343195267, '10%': -2.578770059171598}, 
#  988.5069317854085)
adf_statistic = adf_result[0];     p_value = adf_result[1];     critical_values = adf_result[4]
# Print ADF results
print(f"ADF 統計值: {adf_statistic}")   #-- ADF 統計值: -2.829266824169992  ---> 根據 ADF公式(此處略) 計算出的統計值
print(f"p 值: {p_value}")               #-- p 值: 0.0542132902838265       ---> p 值小於常用顯著性水平(0.05,0.01)，則可以拒絕原假設 (拒絕=序列穩定)
print("臨界值:")   
for key, value in critical_values.items():  print(f"    {key}: {value}")
                                        #-- 臨界值:  1%: -3.4816817173418295,  5%: -2.8840418343195267,  10%: -2.578770059171598
##---> ADF 統計值為 -2.83，p 值為 0.054。這略高於 0.05，意味著數據的平穩性較弱（接近顯著性邊界）


#%%##===== (B5) 自相關函數(Auto-Correlation Function, ACF) =====#####
from statsmodels.tsa.stattools import acf 
from statsmodels.graphics.tsaplots import plot_acf
##== (1).ACF 的基本概念 ==##
#       -- 自相關：時間序列在不同延遲下的相關性。延遲越小，相關性一般越高；隨延遲增加，自相關通常會減弱。
#       -- 延遲(lag): ACF 使用的滯後階數，表示相隔的時間步長。如，延遲 1 表示相鄰兩個時間點的相關性，延遲 12（在每月資料中）則表示相隔 12 個月的相關性。
#       -- 適合檢測數據中的季節性和趨勢性特徵
##== (2).ACF的用途 ==##
#       -- 檢測週期性：如果 ACF 圖在固定間隔（如每 12 個月）出現峰值，則表明數據可能存在季節性模式。
#       -- 檢測趨勢：在有趨勢的數據中，ACF 圖會顯示出隨延遲增加而緩慢衰減的情況，這說明數據中的歷史值會影響未來值。
#       -- 模型識別：在 ARIMA 模型構建中，ACF 幫助識別適合的滯後數。透過觀察 ACF，可以確定數據適合哪些自回歸模型或移動平均模型。

#%%##===== (3).繪製 ACF圖 =====#####
acf_values = acf(AP['#Passengers'], nlags=24);   print(np.round(acf_values,3))
# [1.    0.948 0.876 0.807 0.753 -- 0.714 0.682 0.663 0.656 0.671 -- 0.703 0.743
#  0.76  0.713 0.646 0.586 0.538 -- 0.5   0.469 0.45  0.442 0.457 -- 0.482 0.517 0.532]
acf_fig = go.Figure()
acf_fig.add_trace(go.Bar(x=list(range(len(acf_values))), y=acf_values, name='ACF'))
acf_fig.update_layout(title='Auto-Correlation Function (ACF) of Air Passengers',
                      xaxis_title='Lags', yaxis_title='ACF', legend_title='Auto-Correlation')
acf_fig.show()
##== (4).ACF的解讀 ==##
# 季節性峰值：若 ACF圖在每 12 個月（每年）呈現峰值，則顯示出數據的年度季節性模式，表明每年乘客數量有類似的增減週期。
# 緩慢衰減：如果 ACF圖中的自相關在前幾個延遲後緩慢衰減，則可能是由數據的長期趨勢導致的。

#%%##===== (B6) 基本滾動式預測 =====#####

##== (1).滾動式預測(Rolling Forecast)的定義 ==##
#     -- 時間序列的預測技術，用於在不同的時間點隨著新的觀測數據加入而不斷更新模型。
#     -- 每次移動一個時間點（通常是向前一個時間步長），重新建構模型並生成預測，
#     -- 因此滾動式預測適合用於分析持續更新的動態資料

##== (2).滾動式預測演算法的步驟 ==##
# def rolling_forecast_演算法(data, window_size):  #--(A).分段數據集：將數據分為訓練集和測試集。使用滾動窗口（例如前 12 個月或過去一段時間的數據）來訓練模型。
#     predictions = [];   true_values = []
#     for i in range(window_size, len(data)):      #--(B).滾動更新：預測一個時間步長後，將窗口向前移動一個單位，再用新的數據重訓模型並生成下一個預測。
#         train_data = data[i - window_size:i]
#         forecast = 滾動式預測演算法( train_data, 參數 )
#         predictions.append(forecast);   true_values.append(data[i]) #==(C).重複預測：重複這一過程，直到涵蓋整個測試集。
#     return predictions, true_values

#%%##===== (3A).線性回歸(LR)：每次基於新的訓練集重新訓練回歸模型 =====#####
from sklearn.linear_model import LinearRegression
def rolling_forecast_LR(data, window_size):    #--(A).分段數據集
    predictions = [];   true_values = []
    for i in range(window_size, len(data)):    #--(B).滾動更新
        train_data = data[i - window_size:i]
        X = np.arange(window_size).reshape(-1, 1);   y = train_data.values #=== LR
        LR = LinearRegression().fit(X, y)                                  #=== LR
        next_step = np.array([[window_size]])                              #=== LR
        forecast = LR.predict(next_step)[0]        #=========================== LR
        predictions.append(forecast);   true_values.append(data[i]) #--(C).重複預測
    return predictions, true_values

##== (3B).簡單移動平均(Simple Moving Average, SMA): 以固定窗口計算移動平均，用於平滑數據。滾動式預測中也可以使用 SMA 的方式進行多步預測 ==##
def rolling_forecast_SMA(data, window_size):   #--(A).分段數據集
    predictions = [];   true_values = []
    for i in range(window_size, len(data)):    #--(B).滾動更新
        train_data = data[i - window_size:i]
        forecast = train_data.mean()        #========================= SMA(t) = sum_n,y(t-i)
        predictions.append(forecast);   true_values.append(data[i]) #--(C).重複預測
    return predictions, true_values

##== (3C).加權移動平均(Weighted MA, WMA)指數平滑法：對窗口內的數據給予權重 ==##
def rolling_forecast_WMA(data, window_size):   #--(A).分段數據集
    predictions = [];   true_values = []
    weights = np.arange(1, window_size + 1) #========================= WMA(t) = sum_n,w(i)y(t-i)
    for i in range(window_size, len(data)):    #--(B).滾動更新
        train_data = data[i - window_size:i]
        forecast = np.dot(train_data, weights) / weights.sum()   #==== WMA
        predictions.append(forecast);   true_values.append(data[i]) #--(C).重複預測
    return predictions, true_values

##== (3D).指數平滑法(Exponential Smoothing, ES)：對越靠近當前的數據賦予更高的權重。適用於平滑短期的非季節性預測 ==##
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
def rolling_forecast_ES(data, window_size, smoothing_level):    #--(A).分段數據集
    predictions = [];   true_values = []
    for i in range(window_size, len(data)):                     #--(B).滾動更新
        train_data = data[i - window_size:i]
        ES = SimpleExpSmoothing(train_data).fit(smoothing_level=smoothing_level) #=== ES
        forecast = ES.forecast(1).values[0]         #=================== ES: F(t) = a*y(t)+(1-a)*F(t-1)
        predictions.append(forecast);   true_values.append(data[i])  #--(C).重複預測
    return predictions, true_values

##== (3E).Holt-Winters法：指數平滑法 的擴展，可處理具有 趨勢 和 季節性 的時間序列數據==##
# HA 方法可以最看得出效果，因為航空乘客數據具有明顯的趨勢和季節性模式
from statsmodels.tsa.holtwinters import ExponentialSmoothing
def rolling_forecast_HW(data, window_size, seasonal_periods, trend='add', seasonal='add', 
                        smoothing_level=None, smoothing_slope=None, smoothing_seasonal=None):    
    predictions = [];   true_values = []                             #--(A).分段數據集
    for i in range(window_size, len(data)):                          #--(B).滾動更新
        train_data = data[i - window_size:i]
        HW = ExponentialSmoothing( train_data, seasonal_periods=seasonal_periods, trend=trend, seasonal=seasonal
             ).fit(smoothing_level=smoothing_level, #=== HW加法型: 水準(level) Lt = a*(yt-St-s) + (1-a)*(Lt-1+Tt-1))
                   smoothing_slope=smoothing_slope,           #=== 趨勢(trend) Tt = b*(Lt-Lt-1) + (1-b)*Tt-1
                   smoothing_seasonal=smoothing_seasonal )    #=== 季節(seasonality): St = c*(yt-Lt) + (1-c)St-s
        forecast = HW.forecast(1).values[0]         #=================== ES: F(t) = a*y(t)+(1-a)*F(t-1)
        predictions.append(forecast);   true_values.append(data[i])  #--(C).重複預測
    return predictions, true_values

##== (4).參數設定與執行預測 ==##
window_size = 12;   smoothing_level = 0.2;    seasonal_periods = 12
LR_preds, LR_true = rolling_forecast_LR(AP['#Passengers'], window_size);   
SMA_preds, SMA_true = rolling_forecast_SMA(AP['#Passengers'], window_size)
WMA_preds, WMA_true = rolling_forecast_WMA(AP['#Passengers'], window_size)
ES_preds, ES_true = rolling_forecast_ES(AP['#Passengers'], window_size, smoothing_level)
HW_preds, HW_true = rolling_forecast_HW(AP['#Passengers'], window_size*2, seasonal_periods, trend="add", seasonal="mul")
                                                           #--> ExponentialSmoothing 的初始化需要至少兩個完整的季節性周期
##== (5).將各預測值調整放在一個數據框(forecast_df) ==##
#-- 截取所有預測結果至相同長度
min_length = min(len(LR_preds), len(SMA_preds), len(WMA_preds), len(ES_preds), len(HW_preds))  #-- 最短結果長度
print(np.round(LR_preds[0:15],1))   #-- [126.2 119.2 117. 122.4 124.7--121.8 129.7 146.6 162.6 171.==166.5 151.8 148.3 144.9 144.1]
LR_preds  = LR_preds[-min_length:];   print(np.round(LR_preds[0:15],1))  #-- [148.3 144.9 144.1 154.7 157.3 159.6 166.8 184.2 200.2 207.3 200.7 183.4 176.8 171.7 169.5]
SMA_preds = SMA_preds[-min_length:];  WMA_preds = WMA_preds[-min_length:]
ES_preds  = ES_preds[-min_length:];   HW_preds  = HW_preds[-min_length:]
LR_true   = LR_true[-min_length:]
forecast_df = pd.DataFrame({ 'AP_true': LR_true, 'LR_preds': LR_preds, 
                            'SMA_preds':SMA_preds,'WMA_preds':WMA_preds,'ES_preds':ES_preds,'HW_preds':HW_preds});   
print(forecast_df.shape);   print(forecast_df.head(3))   #-- (120, 6)
#    AP_true    LR_preds   SMA_preds   WMA_preds    ES_preds    HW_preds
# 0      145  148.348485  139.666667  142.115385  138.464534  136.179661
# 1      150  144.893939  142.166667  142.935897  140.527541  158.298282
# 2      178  144.075758  144.166667  144.141026  143.452825  167.536042

#%%== (6).繪圖比較 ==##
fig = go.Figure()
fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['AP_true'], mode='lines+markers', name='AP_true'))
fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['LR_preds'], mode='lines+markers', name='LR_preds'))
fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['SMA_preds'], mode='lines+markers', name='SMA_preds'))
fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['WMA_preds'], mode='lines+markers', name='WMA_preds'))
fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['ES_preds'],  mode='lines+markers', name='ES_preds'))
fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['HW_preds'],  mode='lines+markers', name='HW_preds'))
fig.update_layout( title='Forecast Comparison', xaxis_title='Index', yaxis_title='Values', legend_title='Legend', template='plotly_white')
fig.show()
##== (7).各種預測方法比較 ==##
#    	  趨勢處理 季節性處理 非平穩性處理 適合時間範圍 複雜度	 優勢(PROs)	                  劣勢(CONs)
#         -------  --------- -----------  ----------- ------ ---------------------------  -------------------------
# LR	    支持	   不支持	 不支持	      長期	      低	     易解釋，適合線性數據	      無法處理季節性和非線性數據
# SMA	    不支持 不支持	 不支持	      短期	      低	     簡單平滑波動	              不反應趨勢或季節性
# WMA	    不支持 不支持	 不支持	      短期	      低	     對近期數據更敏感	          權重選擇主觀

# ES	    支持	   不支持	 不支持	      中短期	      低	     簡單處理趨勢數據	          季節性數據效果較差
# HW	    支持	    支持	     不支持	      中短期	      中等	 簡單高效，季節性和趨勢處理良好 不適用於非線性趨勢和非平穩數據

# ARIMA	    支持	   不支持	 支持	      中長期	      高	     適合非平穩趨勢數據	          無法直接處理季節性
# SARIMA	支持	    支持	     支持	      中長期	      高	     可處理複雜的趨勢與季節性數據	  模型選擇複雜，計算成本高
#        ==> 參見下節
  
#%%####### (C) 滾動式預測的季節性模型 (ARIMA到SARIMA) ########## 
# [REF] Tanmay Deshpande. (2022). Air Passenger Forecast : ARIMA - SARIMA. https://www.kaggle.com/code/tanmay111999/air-passenger-forecast-arima-sarima

#%%##===== (C1) 序列數據(AP)的準備(file-->AP-->AP_diff) =====#####

#%%== (1).擷取數據 (AirPassengers.csv-->AP) ==##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:.2f}'.format

AP = pd.read_csv('AirPassengers.csv');   print(AP.shape);  print(AP.head(2))   #-- (144, 2)
#      Month  #Passengers
# 0  1949-01          112
# 1  1949-02          118
#%%== (2).並設定索引 (AP) ==##
AP['Month1'] = pd.to_datetime(AP['Month']);   AP = AP.drop(columns='Month');   AP = AP.set_index('Month1');
AP = AP.rename(columns = {'#Passengers':'Passengers'});   print(AP.head(2))            
#             Passengers
# Month1                
# 1949-01-01         112
# 1949-02-01         118
#%%== (3).繪圖 (AP) ==##
plt.figure(figsize = (15,5))
AP['Passengers'].plot();   plt.show()

#%%##===== (C2) 基本序列數據分析: 序列分解、穩定性、自相關性 =====#####

#%%== (1).序列分解 (seasonal_decompose(AP['Passengers'],..)) ==##
import statsmodels.api as sm
sm.tsa.seasonal_decompose(AP['Passengers'], period=12, model='multiplicative').plot()
plt.show()

#%%== (2).穩定性測試(test_stationarity(): ADF (Augumented Dickey-Fuller) Test) ==##
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):  #== 繪製序列/平均值/標準差,並列印 DF測定性測試 ==##
    #-- Determing rolling statistics
    MA = timeseries.rolling(window=12).mean()
    MSTD = timeseries.rolling(window=12).std()
    #-- Plot rolling statistics:
    plt.figure(figsize=(15,5))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(MA, color='red', label='Rolling Mean')
    std = plt.plot(MSTD, color='black', label = 'Rolling Std')
    plt.legend(loc='best');    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    #-- Perform Dickey-Fuller test:
    print('Results of D:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items(): dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    return
test_stationarity(AP['Passengers'])
# Results of D:
# Test Statistic                  0.82
# p-value                         0.99
# #Lags Used                     13.00
# Number of Observations Used   130.00
# Critical Value (1%)            -3.48
# Critical Value (5%)            -2.88
# Critical Value (10%)           -2.58
##==> 可以用ADF檢驗決定差分次數 d:
#     -- p-value < 0.05: 可以拒絕原假設，資料是平穩的，因此可以選擇 d=0。
#     -- p-value > 0.05: 資料是非平穩的，可進行差分來消除趨勢 d=1 ---> (C3)
#     -- 通常一次差分即可，如果資料仍然不平穩，可以嘗試更高階的差分

#%%== (3).自相關性圖形(tsplot(): acf, pacf) ==##
#    -- 自相關函數(Auto-Correlation Function): ACF(k) = Cov(Xt,X_t-k)/sqrt(Var(Xt)*Var(X_t-k))
#       - Cov(Xk,X_t-k): 時間序列在 滯後(lag)k 下的 協方差(co-variance)
#       - 衡量一個時間序列與其自身在不同時間滯後下的相關性，顯示序列的內部結構。
#       - 可以判斷出數據是否具有周期性，以及周期的長度。
#    -- 偏自相關函數(Partial ACF): PACF(k) = Corr(Xt,X_t-k | Xt-1,Xt-1,...,X_t-(k-1))
#       - 測量時間序列(Xt)與其自身在不同時間滯後(X_t-k)下的相關性，但排除了中間滯後項的影響
#	    - 用於確定 AR(自回歸)模型的階數(p) --> AR 模型的階數，通常設為 PACF(k) 在某個 k後迅速衰減到不顯著之值
from statsmodels.tsa.stattools import acf, pacf
import statsmodels.tsa.api as smt
def tsplot(y, lags=None, figsize=(12,7), style='bmh'):  #== 繪製序列/acf/pacf圖 ==##
    if not isinstance(y, pd.Series):  y = pd.Series(y)        
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout,(0,0), colspan=2)
        acf_ax = plt.subplot2grid(layout,(1,0));       pacf_ax = plt.subplot2grid(layout,(1,1))       
        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y,lags=lags,ax=acf_ax);  smt.graphics.plot_pacf(y,lags=lags,ax=pacf_ax)
        plt.tight_layout()
    return 
tsplot(AP['Passengers']);    plt.show()
#    -- 陰影部分的含義
#	    - (95%)顯著性水準：陰影部分代表了自相關係數在給定滯後期下的置信區間。即在陰影部分內的自相關係數在統計上不顯著。
#       - 閾值：在陰影區域內的自相關係數可能是由於隨機波動，而不是有意義的關係。
#    --> 1.	從PACF圖顯著滯後項 確定 AR階數p: 上例 p=2
#    --> 2.	從 ACF圖顯著滯後項 確定 MA階數q: 上例 q=1

#%%##===== (C3) 差分序列數據分析: 序列分解、穩定性、自相關性 =====#####

#%%== (1).差分序列測試(AP-->AP_diff)
AP_diff = AP.diff();     AP_diff = AP_diff.dropna()
dec = sm.tsa.seasonal_decompose(AP_diff, period=12).plot();   plt.show()
test_stationarity(AP_diff)
# Results of D:
# Test Statistic                 -2.83
# p-value                         0.05
# #Lags Used                     12.00
# Number of Observations Used   130.00
# Critical Value (1%)            -3.48
# Critical Value (5%)            -2.88
# Critical Value (10%)           -2.58
# ==> d=1 (因為進行了差分)
tsplot(AP_diff['Passengers']);   plt.show()
# ==> q=1 (根據 ACF 圖)
# ==> p=2 (根據 PACF 圖)

#%%##===== (C4) 一個完整的ARIMA 內/外樣本預測 =====#####

##== (1).ARIMA（自回歸整合移動平均，Autoregressive Integrated Moving Average）模型: U(t)--->Y(t)
#    -- 三個部分：
#       - 自回歸（AR, Autoregressive）：使用序列的前幾個值(Y(t-k))來回歸當前值(Y(t))，描述過去數值對當前值的影響。
#       - 差分積分（I, Integrated）：透過差分處理，將非平穩的序列轉化為平穩序列。這是專門針對趨勢的部分。
#       - 移動平均（MA, Moving Average）：使用序列的過去誤差(U(t-k))來回歸當前值(Y(t))，描述誤差對當前值的影響。
#    -- ARIMA(p,d,q)--
#       - DY(s)/U(s) = z^(-d)B(s)/A(s): DY(t) = a1+a2*DY(t-1)+...+ap*DY(t-p) + U(t)+b1*U(t-1)+b2*U(t-2)+...+bq*U(t-q)
#       - (當d=1) DY(t) = Y(t)-Y(t-1), (當d=2) DY(t) = (Y(t)-Y(t-1)) - (Y(t-1)-Y(t-2))...
#       - d：差分階數(以消除趨勢，使序列平穩)。p：自回歸項(前期數值)階數。q：移動平均項(過去誤差項)階數，
#    -- 能同時考慮數據的趨勢和波動特性，適合用來分析和預測平穩或非平穩的時間序列。
#    -- 特性
#       - 平穩序列假設：ARIMA 模型假設數據在時間上的期望值和方差是恆定的，因此適合平穩序列。
#       - 自相關和偏自相關：根據數據的自相關函數(ACF)和偏自相關函數(PACF)來確定最佳的自回歸和移動平均階數 (即p和𝑞的選擇)。
#    -- 應用: 經濟與財務 (如股票價格、銷售量、GDP 增長率等)、營銷與需求預測 (如產品銷售預測、季節性需求分析等)、
#             工程與環境 (如氣象數據、電力負載預測等具有趨勢和週期特性的數據)。
from statsmodels.tsa.arima.model import ARIMA
from plotly import graph_objs as go

##==(2).準備數據 ==##
AP = pd.read_csv('AirPassengers.csv')  # 假設 CSV 文件中有日期和乘客數
AP['Month'] = pd.to_datetime(AP['Month'])
AP.set_index('Month', inplace=True)

##== (3).建立 ARIMA 模型 ==##
model = ARIMA(AP['#Passengers'], order=(2, 1, 1))
model_fit = model.fit()

##== (4).內樣本預測與繪圖 ==##
AP['Fitted'] = model_fit.fittedvalues
fig_in_sample = go.Figure()
fig_in_sample.add_trace(go.Scatter(x=AP.index, y=AP['#Passengers'], mode='lines', name='實際乘客數'))
fig_in_sample.add_trace(go.Scatter(x=AP.index, y=AP['Fitted'], mode='lines', name='內樣本擬合值'))
fig_in_sample.update_layout(title='內樣本預測', xaxis_title='日期', yaxis_title='乘客數')
fig_in_sample.show()

##== (5).內樣本預測與繪圖 ==##
forecast_steps = 12  # 設定預測的步數，例如預測未來12個月
forecast = model_fit.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=AP.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq='M')
forecast_values = forecast.predicted_mean
conf_int = forecast.conf_int()
fig_out_sample = go.Figure()
fig_out_sample.add_trace(go.Scatter(x=AP.index, y=AP['#Passengers'], mode='lines', name='實際乘客數'))
fig_out_sample.add_trace(go.Scatter(x=forecast_index, y=forecast_values, mode='lines', name='預測值'))
fig_out_sample.add_trace(go.Scatter(x=forecast_index, y=conf_int.iloc[:, 0], fill=None, mode='lines', line_color='lightgrey', name='信賴區間下限'))
fig_out_sample.add_trace(go.Scatter(x=forecast_index, y=conf_int.iloc[:, 1], fill='tonexty', mode='lines', line_color='lightgrey', name='信賴區間上限'))
fig_out_sample.update_layout(title='外樣本預測', xaxis_title='日期', yaxis_title='乘客數')
fig_out_sample.show()

#%%== (6).ARIMA 模型的適用性和效果 ==##
#    -- (6A).AIC 和 BIC：用於比較模型的擬合度，分數越低表示模型越好。
#       - AIC (Akaike Information Criterion) = −2ln(L)+2k: k--模型的參數數量, L--模型的對數似然函數
#         ---> 平衡模型的                     擬合度 和 複雜度 =========> 偏向於選擇擬合效果更好的模型，即便模型稍微複雜一些
#       - BIC (Bayesian Information Criterion) = −2ln(L)+kln(n) -- n 是樣本數量 (偏向選擇較簡單的模型, 對參數的懲罰比AIC更強)
#         ---> 估計模型的                適配程度 並考慮 模型的簡單性 ===> 偏向選擇較為簡單的模型
#       - 模型選擇：在比較不同階數的 ARIMA 模型時，通常選擇 AIC 或 BIC 最小的模型
print(f"AIC: {model_fit.aic}, BIC: {model_fit.bic}")  #-- AIC: 1378.338319598773, BIC: 1390.1896981198126
#         --> 需要在多個不同階數或不同模型之間進行比較，來確定哪個模型的 AIC 和 BIC 最低
#    -- (6B).殘差圖：通過繪製殘差圖（時間序列），可以直觀地檢查模型是否有系統性的偏差或模式 (此處略)
#    -- (6C).Ljung-Box 檢驗: 一種統計檢驗方法，用於檢查時間序列中的殘差是否為白噪聲 (此處略)

#%%##===== (C5) SARIMA =====#####

##== (1).SARIMA（Seasonal ARIMA，季節性自回歸積分移動平均）模型
#    -- 是 ARIMA 模型的擴展版本，適用於具有季節性模式的時間序列
#    -- SARIMA(p,d,q),(P,D,Q,s): 季節性週期（例如，對於月度數據的年度季節性，s=12）
#    -- 三個部分：
#       - 自回歸（AR, Autoregressive）：使用序列的前幾個值(Y(t-k))來回歸當前值(Y(t))，描述過去數值對當前值的影響。
#       - 差分積分（I, Integrated）：透過差分處理，將非平穩的序列轉化為平穩序列。這是專門針對趨勢的部分。
#       - 移動平均（MA, Moving Average）：使用序列的過去誤差(U(t-k))來回歸當前值(Y(t))，描述誤差對當前值的影響。
#    -- ARIMA(p,d,q)(P,D,Q,s)--
#       - (1-sum_p(Ai*z^(-i))) * (1-sum_P(AAi*z^(-is))) * (1-z^(-1))^d * (1-z^(-s))^D * Y(t) = B0 + (1+sum_q(Bj*z^(-j))) * (1+sum_Q(BBj*z^(-js))) 
#       - (當d=1) DY(t) = Y(t)-Y(t-1), (當d=2) DY(t) = (Y(t)-Y(t-1)) - (Y(t-1)-Y(t-2))...
#       - d：差分階數(以消除趨勢，使序列平穩)。p：自回歸項(前期數值)階數。q：移動平均項(過去誤差項)階數，

from statsmodels.tsa.statespace.sarimax import SARIMAX
from plotly import graph_objs as go

##== (1).讀取數據 ==##
import pandas as pd
AP = pd.read_csv('AirPassengers.csv')
AP['Month'] = pd.to_datetime(AP['Month']);   AP.set_index('Month', inplace=True)

##== (2).ADF 檢測非平穩性 ==##
from statsmodels.tsa.stattools import adfuller
result = adfuller(AP['#Passengers'])
print('ADF Statistic:', result[0])  #-- ADF Statistic: 0.8153688792060597
print('p-value:', result[1])        #-- p-value: 0.9918802434376411
if result[1] > 0.05:   print("數據非平穩，建議差分處理。")  #===> 數據非平穩，建議差分處理。

##== (3).使用 auto_arima 自動選擇最佳 SARIMA 模型的階數 ==##
# seasonal=True 表示考慮季節性，m=12 表示季節性週期為12（對於月度數據）
from pmdarima import auto_arima
auto_model = auto_arima(AP['#Passengers'], seasonal=True, m=12, trace=True, suppress_warnings=True, stepwise=True)
# Performing stepwise search to minimize aic
#  ARIMA(2,1,2)(1,1,1)[12]             : AIC=1020.048, Time=0.69 sec
#  ARIMA(0,1,0)(0,1,0)[12]             : AIC=1031.508, Time=0.03 sec
#  ARIMA(1,1,0)(1,1,0)[12]             : AIC=1020.393, Time=0.09 sec
#  ARIMA(0,1,1)(0,1,1)[12]             : AIC=1021.003, Time=0.12 sec
#  ARIMA(2,1,2)(0,1,1)[12]             : AIC=1019.935, Time=0.37 sec
#  ARIMA(2,1,2)(0,1,0)[12]             : AIC=1019.290, Time=0.42 sec
#  ARIMA(2,1,2)(1,1,0)[12]             : AIC=1019.546, Time=0.37 sec
#  ARIMA(1,1,2)(0,1,0)[12]             : AIC=1024.160, Time=0.09 sec
#  ARIMA(2,1,1)(0,1,0)[12]             : AIC=1017.847, Time=0.18 sec <--- AIC最小
#  ARIMA(2,1,1)(1,1,0)[12]             : AIC=1017.914, Time=0.41 sec
#  ARIMA(2,1,1)(0,1,1)[12]             : AIC=1018.359, Time=0.39 sec
#  ARIMA(2,1,1)(1,1,1)[12]             : AIC=1018.248, Time=0.79 sec
#  ARIMA(1,1,1)(0,1,0)[12]             : AIC=1022.393, Time=0.08 sec
#  ARIMA(2,1,0)(0,1,0)[12]             : AIC=1022.393, Time=0.06 sec
#  ARIMA(3,1,1)(0,1,0)[12]             : AIC=1019.084, Time=0.20 sec
#  ARIMA(1,1,0)(0,1,0)[12]             : AIC=1020.393, Time=0.05 sec
#  ARIMA(3,1,0)(0,1,0)[12]             : AIC=1023.666, Time=0.08 sec
#  ARIMA(3,1,2)(0,1,0)[12]             : AIC=1021.083, Time=0.34 sec
#  ARIMA(2,1,1)(0,1,0)[12] intercept   : AIC=inf, Time=0.39 sec
# Best model:  ARIMA(2,1,1)(0,1,0)[12]          
# Total fit time: 5.200 seconds
##-- 查看最佳模型的參數
print(auto_model.summary())
#                                       SARIMAX Results                                      
# ===========================================================================================
# Dep. Variable:                                   y   No. Observations:                  144
# Model:             SARIMAX(2, 1, 1)x(0, 1, [], 12)   Log Likelihood                -504.923--->用於AIC/BIC
#                         非季節性參數 季節性參數 季節  
# Date:                             Fri, 08 Nov 2024   AIC                           1017.847 (越小越好)
# Time:                                     22:29:21   BIC                           1029.348 (越小越好)
# Sample:                                 01-01-1949   HQIC                          1022.520 (另一種模型選擇準則，越小越好)
#                                       - 12-01-1960                                         
# Covariance Type:                               opg                                         
# ==============================================================================      模型參數
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# ar.L1          0.5959      0.085      6.987      0.000       0.429       0.763
# ar.L2          0.2143      0.091      2.343      0.019       0.035       0.394
# ma.L1         -0.9819      0.038    -25.599      0.000      -1.057      -0.907
# sigma2       129.3074     14.555      8.884      0.000     100.780     157.835
# ===================================================================================
# Ljung-Box (L1) (Q):                   0.00   Jarque-Bera (JB):                 7.68 模型評估
# Prob(Q):                              0.98   Prob(JB):                         0.02
# Heteroskedasticity (H):               2.33   Skew:                            -0.01
# Prob(H) (two-sided):                  0.01   Kurtosis:                         4.19
# ===================================================================================

##== (4).使用最佳參數構建 SARIMA 模型 ==##
order = auto_model.order;                     print(order)           #-- 非季節性部分 (p, d, q) = (2, 1, 1)
seasonal_order = auto_model.seasonal_order;   print(seasonal_order)  #-- 季節性部分 (P, D, Q, s) = (0, 1, 0, 12)
model = SARIMAX(AP['#Passengers'], order=order, seasonal_order=seasonal_order)
model_fit = model.fit()
# RUNNING THE L-BFGS-B CODE  <== 演算法
#            * * *
# Machine precision = 2.220D-16   N = 4     M = 10
# At X0         0 variables are exactly at the bounds
# At iterate    0    f=  3.57610D+00    |proj g|=  3.04448D-01
# At iterate    5    f=  3.52912D+00    |proj g|=  2.12220D-03
# At iterate   10    f=  3.52819D+00    |proj g|=  7.65367D-03
# At iterate   15    f=  3.52187D+00    |proj g|=  2.59423D-02
# At iterate   20    f=  3.51016D+00    |proj g|=  2.40143D-02
# At iterate   25    f=  3.50645D+00    |proj g|=  4.64214D-03
# At iterate   30    f=  3.50641D+00    |proj g|=  1.28938D-05
# ...
#    N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
#     4     30     34      1     0     0   1.289D-05   3.506D+00
#   F =   3.50641244859507     
# CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH   <== 表示優化過程已成功收斂，收斂條件是相對於目標函數值 
#-- 模型評估
print(f"AIC: {model_fit.aic}, BIC: {model_fit.bic}")   #-- AIC: 1017.8467851953812, BIC: 1029.347574488186

##== (5).內樣本擬合 (內插) 並繪圖 ==##
AP['Fitted'] = model_fit.fittedvalues   #-- 內樣本擬合
fig_in_sample = go.Figure()
fig_in_sample.add_trace(go.Scatter(x=AP.index, y=AP['#Passengers'], mode='lines', name='實際乘客數'))
fig_in_sample.add_trace(go.Scatter(x=AP.index, y=AP['Fitted'], mode='lines', name='內樣本擬合值'))
fig_in_sample.update_layout(title='內樣本預測', xaxis_title='日期', yaxis_title='乘客數')
fig_in_sample.show()

##== (6).外樣本預測 (外插) 並繪圖 ==##
forecast_steps = 12  # 預測未來12個月
forecast = model_fit.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=AP.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq='M')
forecast_values = forecast.predicted_mean
conf_int = forecast.conf_int()
fig_out_sample = go.Figure()
fig_out_sample.add_trace(go.Scatter(x=AP.index, y=AP['#Passengers'], mode='lines', name='實際乘客數'))
fig_out_sample.add_trace(go.Scatter(x=forecast_index, y=forecast_values, mode='lines', name='預測值'))
fig_out_sample.add_trace(go.Scatter(x=forecast_index, y=conf_int.iloc[:, 0], fill=None, mode='lines', line_color='lightgrey', name='信賴區間下限'))
fig_out_sample.add_trace(go.Scatter(x=forecast_index, y=conf_int.iloc[:, 1], fill='tonexty', mode='lines', line_color='lightgrey', name='信賴區間上限'))
fig_out_sample.update_layout(title='外樣本預測', xaxis_title='日期', yaxis_title='乘客數')
fig_out_sample.show()


#%%####### (D).滾動式預測的實務 ##########

#%%##===== (D1).參數/函式庫: 分析參數 + 應用函式庫 + streamlit快取機制 =====#####

def getX(Xname):       ##== X=getX(Xname): 自交易檔(KDD1)讀取交易數據並(KDD3)設定標籤
    X = pd.read_csv(Xname)
    # -- 還有很多其他產生此標籤的方法, 這裡只是取其中較方便的一種
    X["date"] = pd.to_datetime(X["datetime"]).dt.date
    X["year"] = pd.to_datetime(X["datetime"]).dt.year
    X["yq"] = pd.to_datetime(X["datetime"]).dt.to_period("Q").dt.to_timestamp()
    X["ym"] = pd.to_datetime(X["datetime"]).dt.to_period("M").dt.to_timestamp()
    return(X)

#%%##===== (D2).讀取交易數據到營業額數據 (X-->Sv-->data) =====#####

##== (1).讀取交易數據(Xname-->X) ==##
Xname = "XXX.csv";   X = getX(Xname);    print(X.shape);   print(X.head(2))   #-- (84008, 15)
#   invoiceNo channel customer product category  price             datetime  quantity  amount category2   cost        date  year          yq          ym  
# 0        N1      s1       c1      p1    kind1   1980  2015-01-07 20:07:11         1    1692      sub1 931.39  2015-01-07  2015  2015-01-01  2015-01-01  
# 1        N2      s1       c2      p2    kind1   1400  2015-01-18 19:56:06         1    1197      sub2 793.36  2015-01-18  2015  2015-01-01  2015-01-01  
import plotly.express as px
#%%== (2).轉換成各月營業額(X-->Sv) ==##
Sv = X.groupby(["ym"]).agg({"amount":"sum"}).reset_index();  print(Sv.shape);   print(Sv.head(3))   #-- (36, 2)
#            ym   amount
# 0  2015-01-01  2164967
# 1  2015-02-01  2708844
# 2  2015-03-01  2176010
fig = px.line(Sv, x="ym", y="amount", text="amount", title="整體營業額趨勢");   fig.show()
#%%== (3).只取前2.5年的數據繪圖 ==##
data = Sv[Sv['ym'] <= pd.to_datetime('2017-06')]
fig = px.line(data, x="ym", y="amount", text="amount", title="前兩年半的營業額趨勢");   fig.show()

#%%##===== (D3).從序列分解到模型選擇 =====#####
from statsmodels.tsa.seasonal import seasonal_decompose
data = Sv[Sv['ym'] <= pd.to_datetime('2017-06')]
decomposition = seasonal_decompose(data["amount"], model='additive', period=7)
decomposition.plot()

##== (1).各成分量級的特徵分析
#    -- 原始數據 (amount)：量級約在 200萬（2e6），反映數據的主體波動幅度。
#    -- 趨勢成分 (Trend)： 量級接近 200萬，幾乎占據了數據的大部分變化。 表明數據的主要變化來自於 長期趨勢。
#    -- 季節性成分 (Seasonal)：量級約在 5萬（5e4），相較趨勢為小量級變化。表明數據存在一定的 季節性波動，但對總體波動影響較小。
#    -- 殘差 (Residual)：波動量級在 50萬（5e5）左右，但相對分散且無明顯結構。表明模型能較好地分解主要結構，剩餘的部分基本為隨機波動。
#    ==> 主要處理趨勢：因為趨勢成分是主導，模型必須捕捉趨勢。
#        兼顧季節性：雖然季節性影響相對小，但存在週期性，需要考慮進去。
#        容忍殘差隨機性：模型不必過度擬合隨機波動，應更關注趨勢和季節性。

##== (2).具體建議
#    -- 優先使用 SARIMA：建議在數據中同時考慮趨勢和季節性，SARIMA 更靈活，適合處理此類數據。
#    -- 次選 Holt-Winters：如果數據處理或建模需求簡單，且季節性幅度小，Holt-Winters 是快速有效的選擇。
#    -- 最後選 ARIMA：如果確認季節性影響微不足道，可以選擇 ARIMA 簡化模型。

##== (3).方法比較
#        方法	   趨勢處理	季節性處理	隨機波動處理	複雜度	適用場景
#        SARIMA	      支持	  支持	       支持	    較高	    趨勢和季節性影響明顯，長期準確預測
#        Holt-Winters 支持	  支持	     不處理	    中等	    趨勢主導，季節性影響相對較小或穩定的場景
#        ARIMA	      支持	 不支持	       支持	    較低	    季節性影響小或可以忽略，趨勢明顯的短期預測場景


#%%##===== (D4).三種模型求取與繪圖 =====#####

##== (1).SARIMA模型訓練函式: 使用 auto_arima 訓練 SARIMA 模型，並包含手動參數作為後備 ==##
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
def train_sarima_model(data, manual_order=(1,1,1), manual_seasonal_order=(1,1,1,12)): ##== 訓練SARIMA模型，優先使用auto_arima自動選擇參數 ==##
    """
    訓練 SARIMA 模型，優先使用 auto_arima 自動選擇參數。
    :param data: 時間序列數據 (pd.Series)
    :param manual_order: 手動設置的非季節性 ARIMA 模型參數 (p, d, q)
    :param manual_seasonal_order: 手動設置的季節性 SARIMA 模型參數 (P, D, Q, m)
    :return: 已訓練的 SARIMA 模型
    """
    try:
        auto_model = auto_arima( data, seasonal=True, m=12, trace=True, suppress_warnings=True, stepwise=True, error_action='ignore')
        order = auto_model.order
        seasonal_order = auto_model.seasonal_order
        print(f"Auto-selected SARIMA parameters: order={order}, seasonal_order={seasonal_order}")
    except Exception as e:
        print(f"auto_arima failed: {e}. Falling back to manual parameters.")
        order = manual_order
        seasonal_order = manual_seasonal_order
    model = SARIMAX(data, order=order, seasonal_order=seasonal_order).fit(disp=False)
    return model

##== (2).生成內/外樣本預測值函式, 及其繪圖函式 ==##
def generate_predictions(model, data, forecast_steps=6, model_type="sarima"):   ##== 根據模型生成內樣本和外樣本預測 ==##
    """
    根據模型生成內樣本和外樣本預測。
    :param model: 訓練好的模型（SARIMA, ARIMA, HW）
    :param data: 時間序列數據 (pd.Series)
    :param forecast_steps: 外樣本預測步數
    :param model_type: 模型類型 ("sarima", "arima", "hw")
    :return: 包含預測結果的 DataFrame
    """
    #-- 初始化合併數據框，確保包含實際數據
    combined_prediction = pd.DataFrame(data.copy())
    combined_prediction.columns = ['amount']   #-- 確保實際數據欄位名為 'amount'
    #-- 內樣本預測
    if model_type in ["sarima", "arima"]:
        in_sample_forecast = model.get_prediction(start=data.index[0], end=data.index[-1])
        in_sample_forecast_values = in_sample_forecast.predicted_mean
        out_sample_forecast = model.get_forecast(steps=forecast_steps)
        out_sample_forecast_values = out_sample_forecast.predicted_mean
    elif model_type == "hw":
        in_sample_forecast_values = model.fittedvalues
        forecast_index = pd.date_range(start=data.index[-1] + pd.offsets.MonthBegin(), periods=forecast_steps, freq='MS')
        out_sample_forecast_values = model.forecast(forecast_steps)
    else:
        raise ValueError("Invalid model_type. Choose from 'sarima', 'arima', or 'hw'.")
    #-- 合併內樣本和外樣本預測
    forecast_index = pd.date_range(start=data.index[-1] + pd.offsets.MonthBegin(), periods=forecast_steps, freq='MS')
    combined_prediction = combined_prediction.reindex(combined_prediction.index.union(forecast_index))
    combined_prediction['In-Sample Forecast'] = in_sample_forecast_values
    combined_prediction['Out-Sample Forecast'] = out_sample_forecast_values
    return combined_prediction
def plot_predictions(combined_prediction, title="Model Forecast"):              ##== 繪製內樣本與外樣本的預測結果 ==##
    """
    繪製內樣本與外樣本的預測結果。
    :param combined_prediction: 預測結果 DataFrame
    :param title: 圖表標題
    :return: Plotly 圖形對象
    """
    fig = go.Figure()
    actual_data = combined_prediction['amount'].dropna()   #-- 實際數據-->繪製
    fig.add_trace(go.Scatter( x=actual_data.index, y=actual_data, mode='lines+markers', name='Actual Data'))
    in_sample_forecast = combined_prediction['In-Sample Forecast'].dropna()    #-- 內樣本預測-->繪製
    fig.add_trace(go.Scatter( x=in_sample_forecast.index, y=in_sample_forecast, mode='lines+markers', name='In-Sample Forecast', line=dict(dash='dot')))
    out_sample_forecast = combined_prediction['Out-Sample Forecast'].dropna()  #-- 外樣本預測-->繪製
    fig.add_trace(go.Scatter( x=out_sample_forecast.index, y=out_sample_forecast, mode='lines+markers', name='Out-Sample Forecast',line=dict(dash='dash')))
    fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Amount", template="plotly_white" )   #-- 更新圖表布局
    return fig

##== (3).SARIMA 訓練與繪圖例 ==##
sarima_model = train_sarima_model(data['amount'])
print(sarima_model.params)
# ar.L1      -5.856917e-01 -- SARIMA(p,d,q)(P,D,Q,s) 中的 a1
# ma.L1       5.981616e-01 -- SARIMA(p,d,q)(P,D,Q,s) 中的 b1
# ar.S.L12   -6.611255e-01 -- SARIMA(p,d,q)(P,D,Q,s) 中的 A1
# ma.S.L12    8.392960e-01 -- SARIMA(p,d,q)(P,D,Q,s) 中的 B1
# sigma2      1.308517e+11
##-- (1+0.6611255B^12)(1+0.5856917B)(1−B)(1−B^12) y(t) = (1+0.8392960B^12)(1+0.5981616B)U(t)​
combined_prediction_sarima = generate_predictions(sarima_model, data['amount'], forecast_steps=6, model_type="sarima")
print(combined_prediction_sarima.shape);   print(combined_prediction_sarima.tail(8))   #-- (36, 3)
#                amount  In-Sample Forecast  Out-Sample Forecast
# 2017-05-01  2032576.0        1.858286e+06                  NaN
# 2017-06-01  1619493.0        2.007209e+06                  NaN
# 2017-07-01        NaN                 NaN         1.875258e+06
# 2017-08-01        NaN                 NaN         1.707007e+06

# 2017-09-01        NaN                 NaN         1.159431e+06
# 2017-10-01        NaN                 NaN         1.773990e+06
# 2017-11-01        NaN                 NaN         1.929345e+06
# 2017-12-01        NaN                 NaN         2.382478e+06
fig_sarima = plot_predictions(combined_prediction_sarima, title="SARIMA Forecast")
fig_sarima.show()

#%%== (4).ARIMA模型訓練函式: 呼叫SARIMA直接使用 auto_arima 訓練 ARIMA 模型，並包含手動參數作為後備 ==##
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
def train_arima_model(data, manual_order=(1, 1, 1)):   ##== 訓練 ARIMA 模型，優先使用 auto_arima 自動選擇參數 ==##
    """
    訓練 ARIMA 模型，優先使用 auto_arima 自動選擇參數。
    如果 auto_arima 無法運行，則使用手動設置的參數。    
    :param data: 時間序列數據 (pd.Series)
    :param manual_order: 手動設置的 ARIMA 模型參數 (p, d, q)
    :return: 已訓練的 ARIMA 模型
    """
    try:
        print("Using auto_arima to determine ARIMA parameters...")
        auto_model = auto_arima( data, seasonal=False, trace=True, suppress_warnings=True, stepwise=True, error_action='ignore' )  #-- 忽略錯誤以繼續運行
        order = auto_model.order
        print(f"Auto-selected ARIMA parameters: order={order}")
    except Exception as e:
        print(f"auto_arima failed with error: {e}")
        print("Falling back to manual parameters.")
        order = manual_order
    # 使用自動選擇或手動設置的參數來訓練 ARIMA 模型
    model = SARIMAX(data, order=order, seasonal_order=(0, 0, 0, 0)).fit(disp=False)
    return model
# 訓練 ARIMA 模型
arima_model = train_arima_model(data['amount'])
print(arima_model.params)
# ar.L1     8.822177e-01 -- ARIMA(p,d,q)中的 b1
# ar.L2     1.077300e-01 -- ARIMA(p,d,q)中的 b2
# sigma2    8.517773e+10
##--> (1−0.8822177B−0.1077300B^2)(1−B) y(t) = U(t)
combined_prediction_arima = generate_predictions(arima_model, data['amount'], forecast_steps=6, model_type="arima")
fig_arima = plot_predictions(combined_prediction_arima, title="ARIMA Forecast")
fig_arima.show()

#%%== (5).Holt-Winter模型訓練函式 ==##
from statsmodels.tsa.holtwinters import ExponentialSmoothing
def train_hw_model(data, seasonal_periods=12, trend="add", seasonal="mul"):  #---> 水平和趨勢: 加性模式, 季節性"乘性模式
    """
    訓練 Holt-Winters 模型。
    :param data: 時間序列數據 (pd.Series)
    :param seasonal_periods: 季節性周期（如一年 12 個月）
    :param trend: 趨勢類型 ("add" or "mul" or None)
    :param seasonal: 季節性類型 ("add" or "mul" or None)
    :return: 已訓練的 Holt-Winters 模型
    """
    model = ExponentialSmoothing(data, seasonal_periods=seasonal_periods, trend=trend, seasonal=seasonal).fit()
    return model
# 訓練 Holt-Winters 模型
hw_model = train_hw_model(data['amount'])
print(hw_model.params)
# {'smoothing_level': 0.005, 'smoothing_trend': 0.0001, 'smoothing_seasonal': 0.0001, 
#  'damping_trend': nan, 'initial_level': 2073193.0361111101, 'initial_trend': -5280.408080807982, 
#  'initial_seasons': array([1.13890605, 1.31416611, 0.99564678, 0.97030378, 0.98761188, 0.9382119 , 0.99769508, 
#                            0.98080042, 0.95695394, 0.85237169, 0.88778247, 0.9795499 ]), 
#  'use_boxcox': False, 'lamda': None, 'remove_bias': False}
combined_prediction_hw = generate_predictions(hw_model, data['amount'], forecast_steps=6, model_type="hw")
fig_hw = plot_predictions(combined_prediction_hw, title="HW Forecast")
fig_hw.show()

#%%== (6).Holt-Winter公式說明 ==##

#   -- 模型公式
#      - 水平更新公式: L(t) = a*y(t)/s(t-m) + (1-a)*(L(t=1)+B(t-1)) 
#      - 趨勢更新公式: B(t) = b*(L(t)-L(t-1)) + (1-b)*B(t-1)
#      - 季節性更新式: S(t) = c*y(t)/(L(t)+B(t)) + (1-c)*S(t-m)
#      - 預測公式:  y(t+h) = (L(t)+h*B(t))*s(t+h-m | (h-1)/m )
#   -- 初始值
#      - t  = 1（即模型的第一個時間點）
#      - y1 = 2164967（1 月的實際數據）
#      - L0 = 2073193.036 = 第一周期的觀測值總和 / m
#      - B0 = −5280.408 = 第一周期與第二周期的平均差 / m
#      - S0 = 1.13890605 = 第 t 期的觀測值 / 對應的水平估計值
#   -- 代入第一步計算
#      - L1 = 0.005 * (2164967/1.13890605) + (1−0.005)⋅(2073193.036−5280.408) = 2077078..
#      - B1 = 0.0001⋅(2077078.55−2073193.036)+(1−0.0001)⋅(−5280.408) = -5280..
#      - S1 = 0.0001⋅(2164967 / (2077078.55−5279.967) ) + (1−0.0001)⋅1.13890605 = 1.138896..
#      - y2 = (2077078.55−5279.967)⋅1.31416611 = 2723141...
#   -- 數據解讀
#      - 趨勢和水平：初始水平（L0=2073193）和初始趨勢（B0=-5280）表示整體處於高基準，但略微下降的趨勢。
#      - 季節性影響：每個月份有獨立的季節性因子，1 月為基準（S0=1.1389），2 月的預測值受到更強烈的季節性影響（S1=1.3142）。
#      - 預測值：2 月的預測值為 2723141.35，表示季節性提升的影響。
