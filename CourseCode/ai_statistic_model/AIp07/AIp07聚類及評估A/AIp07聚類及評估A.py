# AIp07聚類及評估A.py: AI python 實作 - 07: 聚類及評估 (除了C聚類可以使用,其餘部份應該都要移到 AIp09回歸)
# Jia-Sheng Heh, 10/23/2024, revised from AIp1123

#%%####### (A) AI模型與機器學習 ########## ===   
import numpy as np   
import pandas as pd
import os            ##== (O1) 設定工作目錄
wkDir = "D:\AboutCoding\CourseCode\Artificial_Intelligence_Practice_CourseCode\CourseMaterials\AIp07\AIp07聚類及評估A"
os.chdir(wkDir)
print(os.getcwd())

#%%####### (A) AI模型與機器學習 ########## ===>>> 
##== AI (Artificial Intelligence, 人工智能) [Chollet, Ch.1]
#    -- 源起：20世紀 50年代
#    -- 定義：努力将通常由人类完成的智力任务自动化
#    -- 图靈测试 (Turing test)

#%%##===== (A1) AI模型/系統(Model/System, M) =====#####

##== (1).系統/模型(System/Model, M): 輸出y = M( 輸入u )
##== (2).操作程序：AI是數據分析的第四階段(KDD4)
#        (phase-1).訓練階段(Learning/Modeling/Estimation/Training Phase): (u, y) -> M#
#                  -- 由輸入/輸出 u與y，求取(估測estimate)模型M#
#        (phase-2).預測階段(Prediction/Production/Application Phase): (u_new, M#) -> y_predict
#                  -- 以所估測的模型M#與新的輸入 u_new，求取(估測)新的輸出 y_predict

#%%##===== (A2) 機器學習(Machine Learning) =====#####
##== 機器學習定義: 從數據(x,y) 求取 知識(模型M#)  

##== (1).監督式學習 (Supervised learning): 具範例(u,y), y為教師(teacher, desired output) y, 以求得y=M(u)
#        -- (AIp09).迴歸(regression): y 為連續數據 -- 統計與機率 的最後重點
#        -- (AIp10).分類(classification): y 為離散數據 --> 決策樹(符號AI)，神經網絡(數值AI)
#                   ----> 神經網路(neural network): 自 2014年後，進入深度學習(deep learning)
#                   ----> 所以，現在的 AI，是機器學習／大數據分析的一環 ***
 
##== (2).非監督式學習 (Unsupervised learning): 無輸出y, 目標在於發掘輸入(u)的隱含特徵 --> 數據挖掘(Data Mining)
#        -- (AIp07).聚類(clustering):           計算數據u的相似度，以產生其分類。 -- intrinsic (內隱式) mining
#        -- (AIp08).關連規則(association rule): 計算多數據(ui-uj)間的關連。      -- extrinsic (外顯式) mining
#        -- (納入AIp09).數據序列(data sequencing):  計算多數據(ui-uj)間的時序關係。 

#%%##===== (A3) 鳶尾花 (iris) 數據說明 (-->iris.data/target) =====#####
# 美國加州大學歐文分校的機械學習數據庫http://archive.ics.uci.edu/ml/datasets/Iris 

##== (1).系統模型:  輸出y = iris.target = M (輸入X = iris.data ) ==###

##== (2).(KDD1) 鳶尾花數據取得
from sklearn import datasets  #-- import some data to play with
iris = datasets.load_iris()   #-- 數據的筆數為150筆，共有五個欄位(前四個單位為公分)
print(iris.keys())   #-- dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])

#%%== (3).(KDD2) 輸入X -- 鳶尾花的四項特徵: iris.data[iris.feature_names]
print("iris.data.shape=",iris.data.shape)         #-- (150, 4)
print("iris.feature_names=",iris.feature_names)   #-- 鳶尾花的四項特徵
#-- ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
#--  花萼長度(Sepal Length),花萼寬度(Sepal Width),花瓣長度(Petal Length),花瓣寬度(Petal Width),
print("iris.data[0:3,]=",iris.data[0:3,])
# [[5.1 3.5 1.4 0.2]
#  [4.9 3.  1.4 0.2]
#  [4.7 3.2 1.3 0.2]]
X = iris.data

#%%== (4).(KDD2) 輸出y -- 鳶尾花的的三個品種: iris.target_names)
print("iris.target.shape=",iris.target.shape)  #-- (150,)
print("iris.target_names=",iris.target_names)  #-- array(['setosa', 'versicolor', 'virginica'], dtype='<U10')
                          #-- 類別(Class)：鳶尾花的三個品種Setosa，Versicolor和Virginica
print("iris.target[1:70]=",iris.target[1:70])
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
y_true = iris.target

#%%##===== (A4) (KDD5) 鳶尾花類別以三種顏色作分布圖(iris.data/target-->plot_iris(feature1,feature2)) =====#####

##== (1).繪出 iris的分類
def plot_iris(feature1,feature2,cmap):    ##== 以 iris.data 中的(feature1,feature2)特徵作圖 ==##
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.datasets import load_iris
    iris = load_iris()    #-- 加載鳶尾花數據集
    irisX = pd.DataFrame(iris.data, columns=iris.feature_names)
    X = irisX[[feature1,feature2]];    y = iris.target
    x_min, x_max = X[feature1].min() - .5, X[feature1].max() + .5
    y_min, y_max = X[feature2].min() - .5, X[feature2].max() + .5
    plt.figure(2,figsize=(8,6));   plt.clf()
    p1 = plt.scatter(X[feature1][0:49], X[feature2][0:49], c=cmap[0], edgecolor='k')
    p2 = plt.scatter(X[feature1][50:99], X[feature2][50:99], c=cmap[1], edgecolor='k')
    p3 = plt.scatter(X[feature1][100:149], X[feature2][100:149], c=cmap[2], edgecolor='k')
    plt.xlabel(feature1);          plt.ylabel(feature2)
    plt.xlim(x_min, x_max);        plt.ylim(y_min, y_max)
    plt.legend([p1,p2,p3],iris.target_names,loc="upper left")
    return(plt)
feature1 = 'petal length (cm)';   feature2 = 'petal width (cm)'
plt1 = plot_iris(feature1,feature2,np.array(["blue","green","red"]));   plt1.show()

#%%== (2).繪出iris分類,並加上聚類標籤(y_predict)
def plot_iris_with_cluster_labels(feature1, feature2, cmap, y_predict):   ##== 繪出iris分類,並加上聚類標籤(y_predict) ==##
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.datasets import load_iris
    iris = load_iris()         #-- 加載鳶尾花數據集
    irisX = pd.DataFrame(iris.data, columns=iris.feature_names)
    X = irisX[[feature1, feature2]];    y = iris.target
    x_min, x_max = X[feature1].min()-.5, X[feature1].max()+.5
    y_min, y_max = X[feature2].min()-.5, X[feature2].max()+.5
    plt.figure(2, figsize=(8, 6));    plt.clf()
    for i in range(len(X)):    #-- 在數據點上標註聚類結果，用原始分類的顏色(iris.target)來顯示數字(y_predict)
        plt.text(X[feature1][i], X[feature2][i], str(y_predict[i]), fontsize=12, color=cmap[y[i]], ha='center', va='center')
    plt.xlabel(feature1);    plt.ylabel(feature2);    plt.xlim(x_min, x_max);    plt.ylim(y_min, y_max)
    from matplotlib.lines import Line2D    #-- 繪製顏色對應的圖例，顏色依據原始類別
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=iris.target_names[i], 
                              markerfacecolor=cmap[i], markersize=10) for i in range(3)]
    plt.legend(handles=legend_elements, loc="upper left")
    return plt
feature1 = 'petal length (cm)';   feature2 = 'petal width (cm)'
y_predict = np.random.randint(0, 3, size=150)   #-- 模擬的聚類結果
plt1 = plot_iris_with_cluster_labels(feature1, feature2, np.array(["blue", "green", "red"]), y_predict);   plt1.show()


#%%####### (B) 數據聚類 (Data Clustering) ########## 

#####===== (B1) 聚類的基本概念 [殷,8.1] =====#####

##== (1).聚類: 把數據物件集,劃分成多個組或簇(cluster)的無監督學習方法，
#              目的是將數據點分為若干個相似的組（稱為簇）
#        -- 簇內的對象: 具有很高的相似性(距離distance較短)
#        -- 不同簇的物件: 很不相似

##== (2).聚類分析系統
#        -- 輸入: 一組樣本和一個度量樣本間的相似性(距離distance)
#        -- 輸出: 簇集(cluster) ---> 對每個簇進行綜合描述 (但在本單元不強調)

#%%##===== (B2) 聚類的類型 [殷,8.1] =====#####

##== (1).聚類演算法的類型
#        -- 劃分型聚類(partitional clustering): 將數據物件集 劃分成不重迭的子集(簇),以最小化某種簇內相似度（通常是距離）指標
#           --> k-means, k-medoids -->> ### (2) 劃分方法 [殷,8.2] ###
#        -- 層次型聚類(hierarchical clusterinng): 採用自底向上或自頂向下的方式構建簇，生成一種層次結構
#           --> kNN, 層次式聚類（Hierarchical Clustering） -->> ### (3) 層次方法 [殷,8.3] ###

##== (2).資料物件與簇組的關係
#        -- 互斥型: 每個對象都指派到單個簇 (目前多為此型)
#        -- 重迭或非互斥型: 一個物件同時屬於多個簇
#        -- 模糊型: 在模糊聚類中,每個物件以一個0-1之間的隸屬權值(membership value)屬於每個簇(模糊簇)
#                   ---> 這也含 完全的/部分的 區分
##== (3).簇的類型
#        -- 明顯分離的: 每個物件到同簇旳其他物件的距離,比到不同簇中的任意物件的距離,都近(更相似)
#           --> kNN -->> ### (3) 層次方法 [殷,8.3] ###
#        -- 基於原型(prototype)的: 每個物件到定義該簇的原型的距離,比到其他簇的原型的距離,更近(更相似)
#           --> k-means, k-medoids -->> ### (2) 劃分方法 [殷,8.2] ###
#        -- 基於圖的: 簇可定義為連通分支,也就是簇內互相連通但不與組外物件連通
#        -- 基於密度的: 簇是物件的稠密區域, 不同簇之間的密度比較低
#           --> DBSCAN -->> ### (4) 基於密度的方法 [殷,8.4] ###
#        -- 概念簇(共同性質的): 簇為具有共同性質的物件的集合

#%%##===== (B3) 常見的聚類演算法 =====#####

##== 聚類評估: 留待(D)中說明
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix

##== (1).k-means 聚類（k-means Clustering）===> 適用於球形簇，但需要預設 k 值，且對異常值敏感
#        -- 一種分割式聚類方法，用於將資料點分為 k 個簇。
#        -- 演算法: 通過選擇 k 個初始質心，然後不斷更新每個簇的質心位置，直到簇中心穩定。
#           1).設隨機中心點：隨機選取 k (=3) 個數據樣本為起始類別/簇(cluster)代表點
#           2).歸類：其餘數據樣本歸入相似度最高代表點所在的類別/簇
#              2.1).新中心點：再確立當前簇中樣本坐標的均值(mean)為新的中心點
#              2.2).重新歸類
#              2.3).重新計算中心點
#           3).直到中心點及歸類不再變化
#        -- complexity~O(N*log(N))
#        -- k-means 尤其適用於球形簇，且每個數據點都屬於一個簇。
#        -- 缺點是需要預先確定 k 值，對雜訊和異常值敏感。
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X);   print(kmeans_labels)
# [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 
#  0 2 0 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 0 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 
#  0 2 0 0 0 0 2 0 0 0 0 0 0 2 2 0 0 0 0 2 0 2 0 2 0 0 2 2 0 0 0 0 0 2 0 0 0 0 2 0 0 0 2 0 0 0 2 0 0 2 ]
kmeans_score = silhouette_score(X, kmeans_labels)
kmeans_ari = adjusted_rand_score(y_true, kmeans_labels)
kmeans_cm = confusion_matrix(y_true, kmeans_labels)
print(f'k-means 聚類的輪廓係數 (Silhouette Score): {kmeans_score:.2f}')  #-- 0.55
print(f'k-means 聚類的調整蘭德指數 (ARI): {kmeans_ari:.2f}')             #-- 0.72
print(f'k-means 聚類的混淆矩陣 (Confusion Matrix):\n{kmeans_cm}')
# [[ 0 50  0]
#  [ 3  0 47]
#  [36  0 14]]
plt = plot_iris_with_cluster_labels(feature1, feature2, np.array(["blue", "green", "red"]), kmeans_labels)
plt.show()
#%%== (2).k 近鄰演算法（k-Nearest Neighbors, kNN）==> 是一種分類演算法，適用於預測，但不直接用於聚類
#        -- kNN 其實是一種監督學習方法，用於分類(classification)而非聚類(clustering)，但在某些場景中也可用於聚類前的資料準備。
#        -- 它基於最近的 k 個鄰居來分類或估計資料點的標籤。
#        -- 在聚類前可用來衡量資料點間的相似性。

#%%== (3).層次式聚類（Agglomerative/Hierarchical Clustering）==> 可以產生不同層次的聚類結構，適合小型資料集，但需要指定簇的数量
#        -- 一種聚合式方法，有兩種策略：自底向上（凝聚法）和自頂向下（分裂法）。
#        -- 自底向上方法將每個數據點最初視為單獨的簇，然後逐漸將最相似的簇合併。
#        -- 最終可以生成一個樹狀圖（稱為樹狀圖或 dendrogram），用於分析簇的層次關係。
from sklearn.cluster import AgglomerativeClustering
hierarchical = AgglomerativeClustering(n_clusters=3)  #-- 需事先給予 類別數目
hierarchical_labels = hierarchical.fit_predict(X);   print(hierarchical_labels)
# [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
#  2 0 2 2 2 2 0 2 2 2 2 2 2 0 0 2 2 2 2 0 2 0 2 0 2 2 0 0 2 2 2 2 2 0 0 2 2 2 0 2 2 2 0 2 2 2 0 2 2 0]
hierarchical_score = silhouette_score(X, hierarchical_labels)        
hierarchical_ari = adjusted_rand_score(y_true, hierarchical_labels)  
hierarchical_cm = confusion_matrix(y_true, hierarchical_labels)
print(f'層次式聚類的輪廓係數 (Silhouette Score): {hierarchical_score:.2f}')   #-- 0.55
print(f'層次式聚類的調整蘭德指數 (ARI): {hierarchical_ari:.2f}')               #-- 0.73
print(f'層次式聚類的混淆矩陣 (Confusion Matrix):\n{hierarchical_cm}')
# [[ 0 50  0]
#  [49  0  1]
#  [15  0 35]]
plt = plot_iris_with_cluster_labels(feature1, feature2, np.array(["blue", "green", "red"]), hierarchical_labels);  
plt.show()
#%%== (4).DBSCAN（Density-Based Spatial Clustering of Applications with Noise）==> 無需指定簇的數量，能夠識別任意形狀的簇，對異常值robust，但難以應對簇密度差異較大的情況
#        -- 基於密度的聚類方法，適用於具有雜訊的資料集。兩個參數--
#           > 特定點的密度: 通過該點eps半徑之內的點計數(包括點本身) ---- eps越大,密度越大
#           > MinPts: 稠密區域的密度閾值
#        -- 通過尋找密度較高的資料區域（簇）來定義簇，並可以自動識別簇的數量。所有的點可分類為
#           > 核心點(core points): 該點的eps鄰域內的點的個數, 超過密度閾值MinPts
#           > 邊界點(border points): 不是核心點，但落在某個核心點的鄰域內
#           > 雜訊點(noise): 不是核心點，也不是邊界點
#        -- DBSCAN 可以處理任意形狀的簇，且對雜訊和異常值較不敏感，但對於不同密度的簇可能效果較差。
# -1 為離異點(noise)

from sklearn.cluster import DBSCAN

# EPS的值是定的，所以很難調整、很難可以說出為何這樣的定的所以然。
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X);   print(dbscan_labels)
# [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 -1  0  0  0  0  0  0  0  0  
#   1  1  1  1  1  1  1 -1  1  1 -1  1  1  1  1  1  1  1 -1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1 -1  1  1  1  1  1 -1  1  1  1  1 -1  1  
#   1  1  1  1  1 -1 -1  1 -1 -1  1  1  1  1  1  1  1 -1 -1  1  1  1 -1  1  1  1  1  1  1  1  1 -1  1  1 -1 -1  1  1  1  1  1  1  1  1  1  1  1  1  1  1]
if len(set(dbscan_labels)) > 1 and -1 in dbscan_labels:
    filtered_X = X[dbscan_labels != -1]
    filtered_labels = dbscan_labels[dbscan_labels != -1]
    dbscan_score = silhouette_score(filtered_X, filtered_labels)
else:
    dbscan_score = -1
dbscan_ari = adjusted_rand_score(y_true, dbscan_labels)
dbscan_cm = confusion_matrix(y_true, dbscan_labels)
print(f'DBSCAN 聚類的輪廓係數 (Silhouette Score): {dbscan_score:.2f}')   #-- 0.74
print(f'DBSCAN 聚類的調整蘭德指數 (ARI): {dbscan_ari:.2f}')              #-- 0.52
print(f'DBSCAN 聚類的混淆矩陣 (Confusion Matrix):\n{hierarchical_cm}')
#       -1   0   1  無
# 0   [[ 0,  0,  0,  0],
# 1    [ 1, 49,  0,  0],
# 2    [ 6,  0, 44,  0],
# 噪聲 [10,  0, 40,  0]]
plt1 = plot_iris_with_cluster_labels(feature1, feature2, np.array(["blue", "green", "red"]), dbscan_labels)
plt1.show()
#%%####### (C) 數據聚類的實務 ########## (11/6 pause)

#%%== (5).層次聚類（使用 Scipy）==> 以不同的距離閾值來生成簇，具靈活性，亦可以 linkage 彈性計算聚類的層次關係
#         -- 使用 Scipy 提供的 linkage 和 fcluster 方法進行層次式聚類。
#         -- 鄰近度矩陣(linkage): 度量兩個簇(Ci,Cj)之間的距離 
#            > 最遠鄰(farthest neighbor)聚類演算法 (全鏈演算法, complete linkage)
#            > 最近鄰(Nearest Neighbor, kNN)聚類演算法 (單鏈演算法, single linkage)
#            > 組平均聚類演算法 (平均鏈演算法, average linkage)
#            > 質心聚類演算法 (質心鏈演算法, centroid linkage)
#            > ward聚類演算法 (ward鏈演算法, ward linkage): 兩個簇合併時,導致的平方誤差的增量 (和average很相似)
#         -- 演算法: complexity~O(N^2)
#            1).計算任兩數據樣本的距離(distance, 參數名為metric, 最常見為Euclidean distance)
#            2).依linkage結合最近鄰的兩個簇
#               2.1) 將最近的兩數據樣本(類別/簇cluster)聚為一新的類別/簇 (聚類法method, 預設為 最近鄰single)
#               2.2) 找出此類別/簇的代表點
#            3).直到全部數據樣本都被歸為同一類為止 
#         -- fcluster 用於根據距離閾值將資料劃分為簇。
#         -- 適合視覺化層級關係，特別是生成樹狀圖。
# 層次型聚類比較常使用，因為它可以生成不同層次的聚類結構，並且不需要預先指定簇的數量。
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
linkage_matrix = linkage(X, method='ward')
hierarchical_scipy_labels_iris = fcluster(linkage_matrix, t=3, criterion='maxclust');   print(hierarchical_scipy_labels_iris)
# [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 
#  3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 2 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 
#  2 3 2 2 2 2 3 2 2 2 2 2 2 3 3 2 2 2 2 3 2 3 2 3 2 2 3 3 2 2 2 2 2 3 3 2 2 2 3 2 2 2 3 2 2 2 3 2 2 3]
hierarchical_scipy_score = silhouette_score(X, hierarchical_scipy_labels_iris)
hierarchical_scipy_ari = adjusted_rand_score(y_true, hierarchical_scipy_labels_iris)
hierarchical_scipy_cm = confusion_matrix(y_true, hierarchical_scipy_labels_iris)
print(f'Scipy 層次聚類的輪廓係數 (Silhouette Score): {hierarchical_scipy_score:.2f}')  #-- 0.55
print(f'Scipy 層次聚類的調整蘭德指數 (ARI): {hierarchical_scipy_ari:.2f}')             #-- 0.73
print(f'Scipy 層次聚類的混淆矩陣 (Confusion Matrix):\n{hierarchical_scipy_cm}')
# [[ 0, 50,  0,  0],   多的行/列: 未分類點
#  [ 0,  0,  1, 49],
#  [ 0,  0, 35, 15],
#  [ 0,  0,  0,  0]]
CF = hierarchical_scipy_cm[1:3, 2:4];   print(CF)
# [[ 1 49]
#  [35 15]]   ===> 在(D4)討論時使用
plt1 = plot_iris_with_cluster_labels(feature1, feature2, np.array(["blue", "green", "red"]), hierarchical_scipy_labels_iris)
plt1.show()
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
##== (1B).選取特定客群: 方法B較有彈性，因為怎麼投射都可以 (Cv.FF0/MM0--(標籤投射)-->X.CvFF0/CvMM0-->XX)
# CV 有做過Index，所以較為快速，進行投影、標籤投射。
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
### 聚類(靜態分析): 可以分析出資料的特性

#%%== (2).清理數據,並選取前十大品類(XX-->XX_clean-->top_categories-->filtered_XX)
XX_clean       = XX   #-- XX.drop(columns=["Unnamed: 0"])                 #-- 移除無用的 (目前沒有,讀檔進來才可能有)
top_categories = XX_clean['category'].value_counts().nlargest(10).index   #-- 選取前十大品類
filtered_XX    = XX_clean[XX_clean['category'].isin(top_categories)];   print(filtered_XX.shape);   print(filtered_XX.head(2))   #-- (17869, 17)
# filtered_XX.shape
# Out[33]: (17869, 17)==> 這是372常貴客, 購買前十大品類的交易 

#   invoiceNo channel customer product category  price             datetime  quantity  amount category2    cost        date  year      yq       ym      CvFF0            CvMM0
# 6        N6      s1       c6      p3    kind1   1600  2015-01-29 20:10:56         1    1216      sub1  846.72  2015-01-29  2015  2015Q1  2015-01  (99, 999]  (99999, 999999]
# 7        N7      s1       c7      p3    kind1   1600  2015-01-17 14:26:13         1    1360      sub1  846.72  2015-01-17  2015  2015Q1  2015-01    (9, 99]  (99999, 999999]

#%%== (3)建立客戶-品類的交叉表(CP矩陣,CP_matrix)
# customer品類表(CP_matrix): 行=customer, 列=category, 值=購買次數
# CP矩陣是將每位常貴客(372位)向量化，依其購買的前10大商品類別，轉成10維的向量表示

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
# 設定中文字體 - 使用系統預設字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei', 'sans-serif']  #-- 優先使用微軟正黑體
plt.rcParams['axes.unicode_minus'] = False  #-- 解決負號顯示問題

def plotDendrogram(Z):    ##== 依據聚類好的linkage(Z)繪製樹圖(dendrogram) ==##
    # 繪製 Dendrogram 圖
    plt.figure(figsize=(10, 7));   plt.title("Dendrogram 用於聚類數判斷")
    dendrogram(Z)
    plt.xlabel('客戶');   plt.ylabel('距離');   
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
# 樹圖/蟹爪圖，是聚類過程的表示。

Z = linkage(CP_matrix_scaled, method='ward')        #-- 使用層次聚類中的 linkage 方法，使用 'ward' 方法進行聚類
fig1 = plotDendrogram(Z);   fig1

#%%== (5).聚類評估以求取最佳聚類數(best_k)
# 此圖就會告訴你該選幾類 ? 經過看此圖，就知道該選6類。

fig2, best_k, silhouette_scores = plotSilhouette(Z,CP_matrix_scaled);   fig2;   print(best_k)   #-- 2
print(np.round(silhouette_scores,3))   #--  [0.724 0.675 0.685 0.687 0.688 0.288 0.291 0.292 0.294]

#%%== (6).硬取聚類數=6,來求取各類別特徵
CP_matrixC, clusterSize, clusterFeatures = clusterFeature(CP_matrix,Z,n_clusters=6);   #<-- 硬取聚類數=6
print(CP_matrixC[0:2]);  
# 這是六類的客數人數, 及各類的特徵。
# category  kind1  kind11  kind12  kind16  kind17  kind2  kind27  kind3  kind56  kind6  cluster
# customer                                                                                     
# c1017        17       2       1       0       0     11       0      1       0      3        2
# c1049         1       0       0       8       4     12       1      0       0      0        2
print(list(clusterSize));  #-- [4, 356, 7, 7, 1, 1]

# 356 --> general cluster 
# 1,1 ---> outlier clusters

print(clusterFeatures)
# 1             0       0       0       0       0      0       0      0     252      0
# 2          3249     525    1456     658     811   5578     175   1375      58    129
# 3           749      20     129       3      15     77      41     28       0     49 --> 155.97
# 4            93     103      59     273      57    666      23    176       0      8
# 5            35      48     104      10       6    190       1    105       0      7
# 6             5       9      35       0     405     60       0     14       0      0
#%%
cluster_descriptions = generate_cluster_descriptions(clusterFeatures, clusterSize);   print(cluster_descriptions)
# 選最多的三個，作為特徵。

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
plotDendrogram(Z1)

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


#%%####### (D) 聚類評估 (Clustering Evaluation) ########## 

#####===== (D1) 定義與類型 =====#####

##== (1).聚類評估的定義：
#    - 對聚類結果的質量進行評估的過程，依據聚類內部的緊密性和聚類之間的區別程度來衡量聚類質量
#    - 這一過程可以是依據數據本身的結構（內部評估,D2-D3），或者將聚類結果與已知的真實標籤進行比較（外部評估,D4）
#    - 其目的在於確定聚類結果是否具有高質量的分群效果，從而反映出數據內部的潛在結構。

##== (2).聚類評估的意義：
#    - 數據科學中常見的無監督學習任務中用來衡量聚類質量的一個重要過程。
#    - 可協助判斷:  聚類結果是否合理？ 聚類結果是否能夠反映數據的潛在結構？ 當改變聚類算法或參數時，哪個方案效果更好？ 

#%%##===== (D2) 內部評估(Internal Clustering Evaluation)方法 =====#####

##== (0).只依賴於聚類本身的結構，不需要已知的真實標籤。根據聚類內部的緊密性和聚類之間的分離度來評估聚類的質量

##== (1).Silhouette Score (輪廓分數)：衡量每個資料點在其聚類內的緊密性與其他聚類的分離度。
#        -- 公式: s(i) = (b(i)−a(i)) / max(a(i),b(i))
#           - s(i): 第 i 個資料點的輪廓分數。
#           - a(i): 資料點 i 與其所屬 聚類內其他資料點  的平均距離，這是聚類內的緊密性。
#           - b(i): 資料點 i 與最接近的 其他聚類的資料點 的平均距離，這是與最相似的其他聚類的分離度。
#        -- 物理意義：
#           - 當 s(i) 接近 1，說明資料點與自己的聚類緊密相似，且與其他聚類有明顯區別（聚類效果好）。
#           - 當 s(i) 接近 0，說明資料點位於兩個聚類的邊界，難以區分（效果一般）。
#           - 當 s(i) 接近 -1，說明資料點更接近其他聚類，聚類效果不理想。
#        -- 範圍：-1 到 1。數值越接近 1，聚類效果越好。
from sklearn.metrics import silhouette_score
silhouette_avg_1 = silhouette_score(CP_matrix[CP_matrix.columns[0:10]], CP_matrix["cluster"]);    print(silhouette_avg_1)  #-- 0.5500056272136359
#        ==> CP_matrix的聚類質量比較好，聚類內的資料點彼此之間比較緊密，且與其他聚類的分離度較好。
silhouette_avg_2 = silhouette_score(CP_matrix1[CP_matrix.columns[0:10]], CP_matrix1["cluster"]);  print(silhouette_avg_2)  #-- 0.08718453674281891
#        ==> CP_matrix1的值較低，表示該聚類的質量不高，聚類內的資料點可能分佈得較為分散。
#        -- 輪廓係數可視化
#           - 聚類數與平均輪廓係數圖: 隨著聚類數 k 的變化，聚類的平均 Silhouette Score 的變化趨勢 ==> 尋找能夠最大化 Silhouette Score 的聚類數。
fig2A, best_k2, silhouette_scores2 = plotSilhouette(Z1,CP_matrix1_scaled);   fig2A          #<== 參考上述的 (C5)(2)
print( np.round(silhouette_scores2,3) )   #-- [0.269 0.285 0.308 0.281 0.286 0.232 0.181 0.114 0.122]
#           - 輪廓係數圖(Silhouette Plot): 展示每個數據點的 Silhouette分數，並按聚類進行排序,可以觀察每個聚類內資料點的質量，從而判斷聚類的效果 ()

#%%== (2).Davies-Bouldin Index（戴維斯-鮑丁指數）：衡量每個聚類與其他聚類之間的相似度。數值越低表示聚類效果越好。
#         -- 公式: DB = sum(1:K, max( (di+dj)/(d(ci,cj)) ) )/k
#            - di/dj: 第 i 和第 j 聚類內資料點與聚類中心的平均距離
#            - d(ci,cj): 第 i 和 第 j 聚類中心之間的距離
#            - K: 聚類數
#         -- 物理意義： 衡量聚類之間的分離度與每個聚類內部的緊密性之間的相對關係。
#                      數值越小，說明聚類之間的距離較大，聚類內部緊密，效果較好。
#         -- 範圍：0 到無限。數值越低表示聚類效果越好。
from sklearn.metrics import davies_bouldin_score
DB_1 = davies_bouldin_score(CP_matrix[CP_matrix.columns[0:10]], CP_matrix["cluster"]);    print(DB_1)  #-- 0.717 ==> 值較小，表示聚類內緊密度較高，且聚類之間的分離度較好。
DB_2 = davies_bouldin_score(CP_matrix1[CP_matrix.columns[0:10]], CP_matrix1["cluster"]);  print(DB_2)  #-- 2.151 ==> 值較大，表示聚類內部不夠緊密，聚類之間的分離度較差。

#%%== (3).Dunn Index（鄧恩指數）
#         -- 公式: DI = min(d(ci,cj) / max( Dk )
#            - Dk: 第 k 個聚類內的最大距離（聚類內距離）,  di/dj, K: 同上
#         -- 物理意義： 衡量聚類間距與聚類內距離之間比率的指標。旨在最大化聚類之間的距離並最小化聚類內的變異。
#         -- 範圍：越大越好。
#         -- 無內建函式,需另寫程式 (略) ==> Dunn Index 均為 0，這表明在計算過程中，存在聚類之間的最小距離與聚類內的最大距離無法產生有效的比值，可能是由於數據分佈過於緊密或分散不均。

#%%##===== (D3) 聚類的變異數分析 =====#####

##== (1).變異數函式定義:
def calculate_within_between_variance(data, cluster_labels):   ##== 計算數據點/類別中心 的類別內/間變異數, 及各類別的類別內變異數 ==##
    overall_mean = np.mean(data, axis=0)          #-- 計算總體數據的均值（整體中心）
    unique_clusters = np.unique(cluster_labels)   #-- 獲取唯一的類別標籤，並計算總的類別數
    num_clusters = len(unique_clusters)
    within_cluster_variance = 0                   #-- 初始化類別內變異數和類別間變異數
    between_cluster_variance = 0
    within_cluster_variances = [0] * num_clusters #-- 用來存儲每個類別的類別內變異數
    for cluster in unique_clusters:    #== 計算每個類別的均值
        cluster_data = data[cluster_labels == cluster]  #-- 提取當前類別的數據點
        cluster_mean = np.mean(cluster_data, axis=0)    #-- 計算該類別的均值（中心）
        cluster_within_variance = np.sum((cluster_data - cluster_mean) ** 2)  #-- 計算類別內變異數（每個點與其類別均值的距離平方和）
        within_cluster_variance += cluster_within_variance        
        # 使用 list 儲存每個類別的類別內變異數
        within_cluster_variances[cluster - 1] = cluster_within_variance  # 假設 cluster 是從 1 開始編號        
        between_cluster_variance += len(cluster_data) * np.sum((cluster_mean - overall_mean) ** 2)  #-- 計算類別間變異數（類別均值與總體均值的距離平方和）
    return within_cluster_variance, between_cluster_variance, within_cluster_variances
#   -- (1a).類別內變異數 (Within-cluster variance): 衡量同一個聚類內部資料點之間的變異性。通常透過計算每個聚類內的資料點與該聚類中心（均值）的距離平方和來進行。
#   -- (1b).類別間變異數 (Between-cluster variance): 衡量不同聚類中心之間的距離平方和，表示各聚類之間的差異性。

##== (2).對CP_matrix/CP_matrix1的應用:
wcv, bcv, pwcv = calculate_within_between_variance(scaler.fit_transform(CP_matrix[CP_matrix.columns[0:10]]), CP_matrix["cluster"].to_numpy());    
print(f"類別內變異數={wcv:.3f}");   print(f"類別間變異數={bcv:.3f}");   print(np.round(pwcv,3))  #-- 0.717 ==> 值較小，表示聚類內緊密度較高，且聚類之間的分離度較好。
# 類別內變異數=1867.611,   類別間變異數=1892.389,   [  17.06  1270.349  155.972  424.23     0.       0.   ]

wcv1, bcv1, pwcv1 = calculate_within_between_variance(scaler.fit_transform(CP_matrix1[CP_matrix1.columns[0:10]]), CP_matrix1["cluster"].to_numpy());    
print(f"類別內變異數={wcv1:.3f}");   print(f"類別間變異數={bcv1:.3f}");   print(np.round(pwcv1,3))  #-- 0.717 ==> 值較小，表示聚類內緊密度較高，且聚類之間的分離度較好。
# 類別內變異數=1834.113,   類別間變異數=1725.887,   [282.999   4.054 252.096 199.568 116.057 192.98  323.074 463.284]

##== (3).wcv/bcv數據解讀:
#    ==> (3a).類別內變異數: CP_matrix 的類別內變異數稍高於 CP_matrix1，這表示在 CP_matrix 中，同一聚類內的資料點之間的距離較大。
#        (3b).類別間變異數: CP_matrix 的類別間變異數也高於 CP_matrix1，這表示不同聚類之間的差異較大，聚類效果可能較好。
##== (4).pwcv數據解讀:
#    ==> (4a).CP_matrix第 2 類 的類別內變異數(1270.35)明顯比其他類別高，這表示第 2 類內的資料點彼此之間的距離較大，內部結構較為分散。
#        (4b).CP_matrix1中所有子類別的總變異數(283.00+4.05+252.10+199.57+116.06+192.98+323.07+463.28 ≈ 1834.11)接近於CPmatrix第 2 類的內部變異數(1270.35)
#             這說明，CP_matrix1在進一步細分 CP_matrix第 2 類時，雖然降低了每個子類別的內部變異數，但整體的類別內變異數仍然較大，這可能是聚類評估結果（如輪廓分數）較差的原因,
#             因此進一步細分後並未顯著提升聚類效果。

#%%== (5).聚類品質 (Cluster Quality, Cluster Separation Index, F-ratio) = 類別間變異數 / 類別內變異數
#        -- 比值越大，表示類別間的差異相對於類別內的差異越明顯，聚類質量越好
#        ==> CP_matrix:  F-ratio = 1892.389 / 1867.611 = 1.013 ==> 聚類效果較好。類別間的分離度大於類別內的緊密性，即 聚類間的差異比聚類內部的差異更顯著
#        ==> CP_matrix1: F-ratio = 1725.887 / 1834.113 = 0.941 ==> 聚類間的差異不明顯，類別內變異與類別間變異相當或類別內變異更大，可能意味著聚類質量較差

##== (6).使用 F-ratio 決定聚類數目，但通常不會單獨使用， 可和其他指標（如 Silhouette Score、Davies-Bouldin Index）結合起來進行綜合評估
#        - Elbow Method (肘部法): 通過計算不同聚類數下的 F-ratio，繪製聚類數量和 F-ratio 的圖形，當聚類數目增加時，F-ratio 通常會隨著類別內變異的減少而增大
#                                選擇 F-ratio 開始增長趨勢變緩的點，即所謂的「肘部」，作為理想的聚類數量

#%%##===== (D4) 外部聚類評估(External Clustering Evaluation)方法 + 混淆矩陣 =====#####

##== (1).混淆矩陣（Confusion Matrix） ==> 用於分類任務，以評估模型預測結果的性能
#        -- 將模型的預測結果與真實標籤進行對比，以便了解模型在哪些類別上表現良好，哪些類別上存在誤判。
CF = hierarchical_scipy_cm[1:3, 2:4];   print(CF)   #<== 來自(B3)(5)
# [[ 1 49]
#  [35 15]]
#%%== 雙分類混淆矩陣 (Two-class confusion matrix)
#                              ACTUAL CLASS 真實值
#                               positive          negative
# PREDICTED    positive   True Positives   False Positives <-- type-I error (alpha)
# CLASS                             (TP)              (FP)
# 預測值        negative  False Negatives    True Negatives
#                                   (FN)              (TN)
#               type-II error (beta) --^
#                 power of test = 1-beta
TP, FP = CF[0, 0], CF[0, 1]
FN, TN = CF[1, 0], CF[1, 1]
print(TP,FP,FN,TN)   #-- 1 49 35 15
#%%== (2).常見的分類模型評估指標: 基於混淆矩陣
#        -- 準確率(Accuracy)--表示模型預測正確的比例 
Accuracy = (TP+TN) / (TP+TN+FP+FN);   print(Accuracy)   #-- 0.16
#        -- 精確率(Precision)--模型對於正例預測的可信度 
P = TP / (TP+FP);    print(P)     #-- 0.02
#        -- 召回率(Recall)或 靈敏度(Sensitivity)--模型捕捉到正例的能力 
R = TP / (TP+FN);    print(R)     #-- 0.027777777777777776
#        -- F1 分數(F1-Score)--精確率與召回率的調和平均(平衡) 
F1 = 2*P*R / (P+R);  print(F1)    #-- 0.023255813953488372
#        -- 特異性(Specificity)--模型對負例的區分能力，與靈敏度相對應 
Specificity = TN / (TN+FP);   print(Specificity)        #-- 0.234375

#%%==(3).外部聚類評估: 評估聚類結果與真實標籤的匹配度
#         -- 這些方法包括 Rand Index、Adjusted Rand Index、V-measure (4-9)等
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score
y_pred = hierarchical_scipy_labels_iris   #<== (B2)(3)
#        -- Rand Index（蘭德指數）: 衡量聚類結果與真實標籤之間的一致性。它計算有多少對樣本在真實標籤和聚類結果中屬於相同的類別或不同的類別。
#           - 取值範圍是 [0, 1]，值越接近 1，表示聚類效果越好。
RandIndex = (TP+TN) / (TP+TN+FP+FN);   print(RandIndex)                   #-- 0.16 ==> 就是Accuracy 
#        -- Adjusted Rand Index (ARI): 表示模型的聚類結果與真實標籤之間的一致性，已考慮隨機結果的調整。1 表示完全一致，0 表示隨機聚類結果。
ari = adjusted_rand_score(y_true, y_pred);   print(ari)                   #-- 0.7311985567707746   
#        -- Normalized Mutual Information (NMI): 表示模型聚類結果與真實標籤之間的相互信息量，範圍為 [0, 1]，1 表示完全相關。
nmi = normalized_mutual_info_score(y_true, y_pred);   print(nmi)          #-- 0.770083661648787
#        -- Homogeneity（同質性）: 表示每個聚類中的樣本幾乎都來自同一個真實標籤類別。
homogeneity = homogeneity_score(y_true, y_pred);   print(homogeneity)     #-- 0.7608008469718723
#        -- Completeness: 表示每個真實類別的樣本幾乎都被分配到同一個聚類中。
completeness = completeness_score(y_true, y_pred);   print(completeness)  #-- 0.7795958005591144
#        -- V-measure: 是 Homogeneity 和 Completeness 的調和平均，用來綜合評估聚類的質量。
v_measure = v_measure_score(y_true, y_pred);   print(v_measure)           #-- 0.7700836616487869
