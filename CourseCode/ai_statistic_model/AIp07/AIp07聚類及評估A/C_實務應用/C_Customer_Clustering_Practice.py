# C_Customer_Clustering_Practice.py: AI Python 實作 - 07C: 客戶聚類實務應用
# Jia-Sheng Heh, 10/23/2024, revised from AIp07聚類及評估A.py
# Usage: 將聚類技術應用於客戶分群分析

import numpy as np
import pandas as pd
import os

##== (O1) 設定工作目錄
wkDir = "AIp07\\AIp07聚類及評估A\\C_實務應用"
os.chdir(wkDir)
print(os.getcwd())

#%%####### (C) 數據聚類的實務 ##########

#%%##===== (C1).參數/函式庫: 分析參數 + 應用函式庫 + streamlit快取機制 =====#####

##== (1).定義相關分析參數: 各家企業的以下參數不盡相同 ==##
FFbreaks = [0, 1, 9, 99, 999, 19999]
MMbreaks = [-5000, 0, 999, 9999, 99999, 999999, 19999999]
BBbreaks = [0, 1, 7, 30, 99, 300, 1999]
RRbreaks = [0, 7, 30, 60, 99, 180, 360, 499, 700, 1999]               #--> 用於客戶漏斗 (下述)
Tnow = pd.to_datetime("2017/12/31", format="%Y/%m/%d");  print(Tnow)  # -- 數據分析點: 2023-07-01 00:00:00

#%%== (2).應用函式庫 (Cv): buildCv(),NES3(),addCvNES3() ==##
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

#%%== (3).數據st.cache函式庫: getX(),buildCvRDS() ==##
def getX(Xname):     ##== X = getX(Xname): 自 X.csv 讀取 X (KDD1), 並設定標籤 (KDD3) ==##
    X = pd.read_csv(Xname)
    # -- 還有很多其他產生此標籤的方法, 這裡只是取其中較方便的一種
    X["date"] = pd.to_datetime(X["datetime"]).dt.date
    X["year"] = pd.to_datetime(X["datetime"]).dt.year
    X["yq"] = pd.PeriodIndex(X.date, freq='Q')
    X["ym"] = pd.PeriodIndex(X.date, freq='M')
    return(X)

def buildCvRDS(X,FFbreaks,MMbreaks,BBbreaks,RRbreaks,Tnow):  ##== Cv = buildCvRDS(X,..): 由交易數據 X 求取客戶數據框 Cv (KDD3)
    Cv = buildCv(X,FFbreaks,MMbreaks,BBbreaks)
    print(Cv.shape);   print(Cv[2:4])  # -- (52217, 17)
    KK = np.nanmean(Cv["BB"]);    print(KK)  # -- 43.070694784611675
    MM = np.nanmean(Cv["MM"]);    print(MM)  # -- 46998.990443725226
    Cv = addCvNES3(Cv, Tnow, KK, MM, RRbreaks);   print(Cv.shape);   print(Cv[2:4])  # -- (52217, 22)
    # Cv = pd.read_csv("cvv.csv")
    return(Cv)

#%%##===== (C2).從交易數據到客戶價值模型 (X-->Cv-->TFM) =====#####

##== (1).讀取交易數據(Xname-->X)
Xname = "XXX.csv";   X = getX(Xname);    print(X.shape);   print(X.head(2))
##== (2).轉換為客戶數據框(X-->Cv)
Cv = buildCvRDS(X,FFbreaks,MMbreaks,BBbreaks,RRbreaks,Tnow);   print(Cv.shape);   print(Cv.head(2))
##== (3).生成客戶價值模型(Cv.FF0/MM0-->TFM)
TFM = pd.crosstab(Cv["FF0"], Cv["MM0"], margins=True);   print(TFM)
# FF0 \ MM0     (-5000, 0]  (0, 999]  (999, 9999]  (9999, 99999]  (99999, 999999]  (999999, 19999999]   All
# (0, 1]                14       901         3219             35                0                   0  4169
# (1, 9]                 6        36         2367            816                0                   0  3225
# (9, 99]                0         0            1            342               25                   0   368
# (99, 999]              0         0            0              0                8                   0     8
# (999, 19999]           0         0            0              0                0                   4     4
# All                   20       937         5587           1193               33                   4  7774

#%%##===== (C3).從價值模型選取客群交易 (Cv-->CvTA-->XX),清理後轉成客戶品類表(CP_matrix) =====#####

##== (1A).選取特定客群: 方法A (Cv.FF0/MM0-->CvTA-->XX)
CvTA = Cv.loc[Cv["FF0"].isin(["(9, 99]","(99, 999]"])];   print(CvTA.shape);   print(CvTA.head(2))   #-- (376, 17)
XX = X.loc[X["customer"].isin(CvTA.index)];               print(XX.shape);     print(XX.head(2))     #-- (19228, 17)
#   invoiceNo channel customer product category  price             datetime  quantity  amount category2    cost        date  year      yq       ym      CvFF0            CvMM0
# 6        N6      s1       c6      p3    kind1   1600  2015-01-29 20:10:56         1    1216      sub1  846.72  2015-01-29  2015  2015Q1  2015-01  (99, 999]  (99999, 999999]
# 7        N7      s1       c7      p3    kind1   1600  2015-01-17 14:26:13         1    1360      sub1  846.72  2015-01-17  2015  2015Q1  2015-01    (9, 99]  (99999, 999999]

##== (1B).選取特定客群: 方法B (Cv.FF0/MM0--(標籤投射)-->X.CvFF0/CvMM0-->XX)
X["CvFF0"] = [ Cv["FF0"][x] if x in Cv.index else None for x in X["customer"] ]
X["CvMM0"] = [ Cv["MM0"][x] if x in Cv.index else None for x in X["customer"] ];   print(X.head(2))
#   invoiceNo channel customer product category  price             datetime  quantity  amount category2    cost        date  year      yq       ym         CvFF0               CvMM0
# 0        N1      s1       c1      p1    kind1   1980  2015-01-07 20:07:11         1    1692      sub1  931.39  2015-01-07  2015  2015Q1  2015-01        (0, 1]         (999, 9999]
# 1        N2      s1       c2      p2    kind1   1400  2015-01-18 19:56:06         1    1197      sub2  793.36  2015-01-18  2015  2015Q1  2015-01  (999, 19999]  (999999, 19999999]
XX = X.loc[X["CvFF0"].isin(["(9, 99]","(99, 999]"])];   print(XX.shape);   print(XX.head(2))   #-- (19228, 17)
#   invoiceNo channel customer product category  price             datetime  quantity  amount category2    cost        date  year      yq       ym      CvFF0            CvMM0
# 6        N6      s1       c6      p3    kind1   1600  2015-01-29 20:10:56         1    1216      sub1  846.72  2015-01-29  2015  2015Q1  2015-01  (99, 999]  (99999, 999999]
# 7        N7      s1       c7      p3    kind1   1600  2015-01-17 14:26:13         1    1360      sub1  846.72  2015-01-17  2015  2015Q1  2015-01    (9, 99]  (99999, 999999]
### XX[["customer","category"]].to_csv("c:/Users/jsheh/Desktop/XX.csv")

#%%== (2).清理數據,並選取前十大品類(XX-->XX_clean-->top_categories-->filtered_XX)
XX_clean       = XX   #-- XX.drop(columns=["Unnamed: 0"])                 #-- 移除無用的 (目前沒有,讀檔進來才可能有)
top_categories = XX_clean['category'].value_counts().nlargest(10).index   #-- 選取前十大品類
filtered_XX    = XX_clean[XX_clean['category'].isin(top_categories)];   print(filtered_XX.shape);   print(filtered_XX.head(2))   #-- (17869, 17)
#   invoiceNo channel customer product category  price             datetime  quantity  amount category2    cost        date  year      yq       ym      CvFF0            CvMM0
# 6        N6      s1       c6      p3    kind1   1600  2015-01-29 20:10:56         1    1216      sub1  846.72  2015-01-29  2015  2015Q1  2015-01  (99, 999]  (99999, 999999]
# 7        N7      s1       c7      p3    kind1   1600  2015-01-17 14:26:13         1    1360      sub1  846.72  2015-01-17  2015  2015Q1  2015-01    (9, 99]  (99999, 999999]

#%%== (3)建立客戶-品類的交叉表(CP矩陣,CP_matrix)
CP_matrix = pd.crosstab(filtered_XX['customer'], filtered_XX['category']);   print(CP_matrix.shape);   print(CP_matrix.head(2))   #-- (376, 10)
# category  kind1  kind11  kind12  kind16  kind17  kind2  kind27  kind3  kind56  kind6
# customer
# c1017        17       2       1       0       0     11       0      1       0      3
# c1049         1       0       0       8       4     12       1      0       0      0

#%%##===== (C4).客戶品類表(CP_matrix)的聚類 =====#####

#%%== (1).聚類使用之系統函式庫
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.font_manager as fm
from sklearn.metrics import silhouette_score

##== (2).聚類使用的自定函式庫
scaler = StandardScaler()
font_path = 'C:/Users/jsheh/Desktop/newWorking/RDsys/RDSgpt/微軟正黑體-1.ttf'   #-- # 加載微軟正黑體字型

def plotDendrogram(Z, font_path):    ##== 依據聚類好的linkage(Z)繪製樹圖(dendrogram),以font_path為中文字體 ==##
    prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()       #-- 使用該字體進行繪圖
    # 繪製 Dendrogram 圖
    plt.figure(figsize=(10, 7));   plt.title("Dendrogram 用於聚類數判斷", fontproperties=prop)
    dendrogram(Z)
    plt.xlabel('客戶', fontproperties=prop);   plt.ylabel('距離', fontproperties=prop)
    fig = plt.gcf();   plt.close()   #  plt.show()
    return fig

def plotSilhouette(Z,CP):            ##== 測試聚類linkage(Z<-CP) 2-10個聚類數，計算其輪廓係數(silhouette_scores)繪圖並求最佳聚類數(best_k) ==##
    silhouette_scores = []
    K_range = range(2, 11)
    for k in K_range:
        cluster_labels = fcluster(Z, k, criterion='maxclust')   #-- 使用 fcluster 來獲取聚類結果
        silhouette_avg = silhouette_score(CP, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    ##-- 畫出聚類數與輪廓係數的關係圖
    plt.figure(figsize=(8, 5));                     plt.plot(K_range, silhouette_scores, 'bx-')
    plt.xlabel('聚類數量 (k)');                      plt.ylabel('輪廓係數 (Silhouette Score)')
    plt.title('層次聚類的聚類數量與輪廓係數的關係');   fig = plt.gcf();   plt.show() # plt.close()   # plt.show()
    ##-- 顯示最佳聚類數
    best_k = K_range[silhouette_scores.index(max(silhouette_scores))]
    print(f"最佳聚類數量為: {best_k}")
    print(f"silhouette_scores = {np.round(silhouette_scores,3)}")
    return fig, best_k, silhouette_scores

def clusterFeature(CP,Z,n_clusters): ##= 求取CP矩陣之linkage Z, 以n_clusters為聚類, 求取聚類CP,各類客戶數customer_clusters,及各類品類特徵cluster_characteristics ==##
    CP['cluster'] = fcluster(Z, n_clusters, criterion='maxclust')   #-- 使用 fcluster 來根據聚類數為 6 進行聚類
    ##-- 查看每個類別中的客戶數量
    customer_clusters = CP.groupby('cluster').size()
    print("各類別的客戶數量：");   print(list(customer_clusters))     #-- [4, 356, 7, 7, 1, 1]
    ##-- 分析各類別的品類特徵
    cluster_characteristics = CP.groupby('cluster').sum()
    print("\n各類別的品類特徵：");   print(cluster_characteristics)   #-- 查看各類別的品類特徵
    # category  kind1  kind11  kind12  kind16  kind17  kind2  kind27  kind3  kind56  kind6
    # cluster
    # 1             0       0       0       0       0      0       0      0     252      0
    # 2          3249     525    1456     658     811   5578     175   1375      58    129
    # 3           749      20     129       3      15     77      41     28       0     49
    # 4            93     103      59     273      57    666      23    176       0      8
    # 5            35      48     104      10       6    190       1    105       0      7
    # 6             5       9      35       0     405     60       0     14       0      0
    return CP, customer_clusters, cluster_characteristics

def generate_cluster_descriptions(cluster_characteristics, customer_clusters, top_n=3): ##== 由各類客戶數customer_clusters,及各類品類特徵cluster_characteristics, 求取各類描述descriptions ==##
    descriptions = []
    for cluster_num in customer_clusters.index:         ##== 遍歷每個類別
        row = cluster_characteristics.loc[cluster_num]       #-- 獲取該類別的特徵數據
        customer_count = customer_clusters[cluster_num]      #-- 獲取該類別的客戶數
        avg_counts = (row / customer_count).sort_values(ascending=False)  #-- 計算平均每位客戶的消費數量，並轉換為 DataFrame 以便排序
        top_categories = avg_counts.head(top_n)              #-- 只取前 top_n 大的品類
        description_parts = []                               #-- 初始化描述列表
        for category, avg_count in top_categories.items():   #== 遍歷前 top_n 的品類
            if avg_count >= 10:  description_parts.append(f"數十件*{category}")
            elif avg_count >= 1: description_parts.append(f"{avg_count:.1f}件*{category}")
            else:                description_parts.append(f"少量*{category}")
        descriptions.append(" + ".join(description_parts))   #-- 將合併的描述加入列表
    return descriptions

#%%== (3).CP矩陣的標準化(CP_matrix-->CP_matrix_scaled)
CP_matrix_scaled = scaler.fit_transform(CP_matrix);   print(np.round(CP_matrix_scaled[0:2],2))  #-- 標準化數據
# [[ 0.29  0.03 -0.45 -0.28 -0.15 -0.29 -0.37 -0.41 -0.12  1.85 -0.16]
#  [-0.49 -0.42 -0.57  0.61  0.02 -0.25  0.21 -0.53 -0.12 -0.38 -0.16]]

#%%== (4).標準化CP矩陣的聚類蟹爪圖/樹圖(CP_matrix-->Z-->dendrogram fig)
Z = linkage(CP_matrix_scaled, method='ward')        #-- 使用層次聚類中的 linkage 方法，使用 'ward' 方法進行聚類
fig1 = plotDendrogram(Z,font_path);   fig1

#%%== (5).聚類評估以求取最佳聚類數(best_k)
fig2, best_k, silhouette_scores = plotSilhouette(Z,CP_matrix_scaled);   fig2;   print(best_k)   #-- 2
print(np.round(silhouette_scores,3))   #--  [0.724 0.675 0.685 0.687 0.688 0.288 0.291 0.292 0.294]

#%%== (6).硬取聚類數=6,來求取各類別特徵
CP_matrixC, clusterSize, clusterFeatures = clusterFeature(CP_matrix,Z,n_clusters=6);   #<-- 硬取聚類數=6
print(CP_matrixC[0:2])
# category  kind1  kind11  kind12  kind16  kind17  kind2  kind27  kind3  kind56  kind6  cluster
# customer
# c1017        17       2       1       0       0     11       0      1       0      3        2
# c1049         1       0       0       8       4     12       1      0       0      0        2
print(list(clusterSize));  #-- [4, 356, 7, 7, 1, 1]
print(clusterFeatures)
# 1             0       0       0       0       0      0       0      0     252      0
# 2          3249     525    1456     658     811   5578     175   1375      58    129
# 3           749      20     129       3      15     77      41     28       0     49 --> 155.97
# 4            93     103      59     273      57    666      23    176       0      8
# 5            35      48     104      10       6    190       1    105       0      7
# 6             5       9      35       0     405     60       0     14       0      0

#%%
cluster_descriptions = generate_cluster_descriptions(clusterFeatures, clusterSize);   print(cluster_descriptions)
# ['數十件*kind56 + 少量*kind1 + 少量*kind11', '數十件*kind2 + 9.1件*kind1 + 4.1件*kind12',
#  '數十件*kind1 + 數十件*kind12 + 數十件*kind2', '數十件*kind2 + 數十件*kind16 + 數十件*kind3',
#  '數十件*kind2 + 數十件*kind3 + 數十件*kind12', '數十件*kind17 + 數十件*kind2 + 數十件*kind12']

#%%== (7).各類別特徵/大小/描述對照
AAA = clusterFeatures;   AAA["size"] = clusterSize;   AAA["description"] = cluster_descriptions;   print(AAA)
# category  kind1  kind11  kind12  kind16  kind17  kind2  kind27  kind3  kind56  kind6  size                                description
# cluster
# 1             0       0       0       0       0      0       0      0     252      0     4    數十件*kind56 + 少量*kind1 + 少量*kind11
# 2          3249     525    1456     658     811   5578     175   1375      58    129   356   數十件*kind2 + 9.1件*kind1 + 4.1件*kind12
# 3           749      20     129       3      15     77      41     28       0     49     7  數十件*kind1 + 數十件*kind12 + 數十件*kind2
# 4            93     103      59     273      57    666      23    176       0      8     7  數十件*kind2 + 數十件*kind16 + 數十件*kind3
# 5            35      48     104      10       6    190       1    105       0      7     1  數十件*kind2 + 數十件*kind3 + 數十件*kind12
# 6             5       9      35       0     405     60       0     14       0      0     1 數十件*kind17 + 數十件*kind2 + 數十件*kind12

#%%== (8).某類別(3)特徵/大小/描述對照
print(CP_matrixC.loc[CP_matrixC["cluster"]==3])
# category  kind1  kind11  kind12  kind16  kind17  kind2  kind27  kind3  kind56  kind6  cluster
# customer
# c2839        68       0      12       0       2      2       0      1       0      6        3
# c287         95      10      28       2       3     40       6     11       0     10        3
# c300        121       2       3       1       2      4       7      0       0      2        3
# c331         48       8      10       0       2     24       1      4       0     10        3
# c661        219       0      50       0       3      5      15      2       0     10        3
# c721         81       0      14       0       2      1       6      2       0      6        3
# c977        117       0      12       0       1      1       6      8       0      5        3

#%%##===== (C5).進一步將第2類(356位)再進一步聚類(CP_matrix-->CP_matrix1) =====#####

#%%== (1).抽取第2類客戶特徵(CP_matrix-->CP_matrix1)
CP_matrix1 = CP_matrix.loc[CP_matrixC["cluster"]==2][CP_matrixC.columns[0:10]];   print(CP_matrix1.shape)   #-- (356, 10)
CP_matrix1_scaled = scaler.fit_transform(CP_matrix1);   #<==(3).CP矩陣的標準化
Z1 = linkage(CP_matrix1_scaled, method='ward')          #<==(4).標準化CP矩陣的聚類蟹爪圖/樹
plotDendrogram(Z1,font_path)

#%%== (2).聚類評估類別數 <== (5)
fig2A, best_k2, silhouette_scores2 = plotSilhouette(Z1,CP_matrix1_scaled);   fig2A          #<== (5).聚類評估以求取最佳聚類數
# 最佳聚類數量為: 4
# silhouette_scores = [0.269 0.285 0.308 0.281 0.286 0.232 0.181 0.114 0.122]

#%%== (3).硬取聚類數=6,來求取各類別特徵，以各類別特徵/大小/描述對照 <== (6)(7)
CP_matrixC1, clusterSize1, clusterFeatures1 = clusterFeature(CP_matrix1,Z1,n_clusters=8);   #<== (6).硬取聚類數=6,來求取各類別特徵
cluster_descriptions1 = generate_cluster_descriptions(clusterFeatures1, clusterSize1);   print(cluster_descriptions1)
AAA1 = clusterFeatures1;   AAA1["size"] = clusterSize1;                                     #<== (7).各類別特徵/大小/描述對照
AAA1["description"] = cluster_descriptions1;   print(AAA1)
# category  kind1  kind11  kind12  kind16  kind17  kind2  kind27  kind3  kind56  kind6  size                                description
# cluster
# 1          1537      46      86      10      33    222     123     43       0     23    41   數十件*kind1 + 5.4件*kind2 + 3.0件*kind27
# 2             0       0       0       0       0      0       0      0      56      0     2    數十件*kind56 + 少量*kind1 + 少量*kind11
# 3           149      26      91      16     586    324      14     48       0      9    17  數十件*kind17 + 數十件*kind2 + 8.8件*kind1
# 4           100      40      88     361      18    311      10    140       0      6    25  數十件*kind16 + 數十件*kind2 + 5.6件*kind3
# 5           259      40      70       9      28    224       2     88       0     50    17   數十件*kind1 + 數十件*kind2 + 5.2件*kind3
# 6           148     208     185      19      75    623       7     56       0      1    32  數十件*kind2 + 6.5件*kind11 + 5.8件*kind12
# 7           312      36     548     111       9   1715       2    232       0      7    57   數十件*kind2 + 9.6件*kind12 + 5.5件*kind1
# 8           744     129     388     132      62   2159      17    768       2     33   165    數十件*kind2 + 4.7件*kind3 + 4.5件*kind1

#%%== (4).某類別(3)特徵/大小/描述對照 <== (8)
CP_matrix1.loc[CP_matrix1["cluster"]==3]                                                     #<==(8).某類別(3)特徵/大小/描述對照
# category  kind1  kind11  kind12  kind16  kind17  kind2  kind27  kind3  kind56  kind6  cluster
# customer
# c1529        27       8      19       5      10     74       5     13       0      1        3
# c339         18      10       0       0      42     48       2      0       0      0        3
# c3838         5       1       0       1      34     19       0      1       0      0        3
# c390         18       1       3       1      25     11       1      0       0      1        3
# c4549         0       0       7       0      52      0       0      0       0      0        3
# c4696         0       0       3       0      32      0       0      0       0      0        3
# c4699         0       0       0       0      54      0       0      0       0      0        3
# c4709         6       0       0       4      45     12       0     12       0      0        3
# c4876        44       6       5       3      35     79       4      9       0      0        3
# c5023         0       0       0       0      26      0       0      0       0      0        3
# c5190        24       0       6       0      23      2       2      1       0      2        3
# c5248         7       0       3       0      32      1       0      1       0      0        3
# c5765         0       0       5       0      11      0       0      0       0      0        3
# c5895         0       0       0       0      73      2       0      0       0      0        3
# c5999         0       0      15       2      50     76       0     11       0      5        3
# c6016         0       0      12       0      24      0       0      0       0      0        3
# c6158         0       0      13       0      18      0       0      0       0      0        3
### CP_matrix.to_csv("c:/Users/jsheh/Desktop/CP_matrix.csv")
### CP_matrix1.to_csv("c:/Users/jsheh/Desktop/CP_matrix1.csv")

print("\n=== C部分完成 ===")
print("你已經學習了:")
print("1. 客戶價值分析 (RFM模型)")
print("2. 交易數據到客戶數據框的轉換")
print("3. 客戶-品類矩陣的建構")
print("4. 階層式聚類的實務應用")
print("5. 聚類特徵分析與商業解讀")
print("6. 二次聚類細分策略")
print("\n下一步: 執行 D_聚類評估/D_Clustering_Evaluation.py")
