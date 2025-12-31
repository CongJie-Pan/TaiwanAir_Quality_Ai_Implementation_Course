# B_Clustering_Algorithms.py: AI Python 實作 - 07B: 聚類演算法
# Jia-Sheng Heh, 10/23/2024, revised from AIp07聚類及評估A.py
# Usage: 學習常見的聚類演算法及其應用

import numpy as np
import pandas as pd
import os

##== (O1) 設定工作目錄
wkDir = "AIp07\\AIp07聚類及評估A\\B_聚類演算法"
os.chdir(wkDir)
print(os.getcwd())

##== 載入Iris數據集
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y_true = iris.target
feature1 = 'petal length (cm)'
feature2 = 'petal width (cm)'

##== 載入視覺化函數 (從A部分複製)
def plot_iris_with_cluster_labels(feature1, feature2, cmap, y_predict):
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.datasets import load_iris
    iris = load_iris()
    irisX = pd.DataFrame(iris.data, columns=iris.feature_names)
    X = irisX[[feature1, feature2]];    y = iris.target
    x_min, x_max = X[feature1].min()-.5, X[feature1].max()+.5
    y_min, y_max = X[feature2].min()-.5, X[feature2].max()+.5
    plt.figure(2, figsize=(8, 6));    plt.clf()
    for i in range(len(X)):
        plt.text(X[feature1][i], X[feature2][i], str(y_predict[i]), fontsize=12, color=cmap[y[i]], ha='center', va='center')
    plt.xlabel(feature1);    plt.ylabel(feature2);    plt.xlim(x_min, x_max);    plt.ylim(y_min, y_max)
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=iris.target_names[i],
                              markerfacecolor=cmap[i], markersize=10) for i in range(3)]
    plt.legend(handles=legend_elements, loc="upper left")
    return plt

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
plt = plot_iris_with_cluster_labels(feature1, feature2, np.array(["blue", "green", "red"]), hierarchical_labels)
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
from sklearn.cluster import DBSCAN
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
print(f'DBSCAN 聚類的混淆矩陣 (Confusion Matrix):\n{dbscan_cm}')
#       -1   0   1  無
# 0   [[ 0,  0,  0,  0],
# 1    [ 1, 49,  0,  0],
# 2    [ 6,  0, 44,  0],
# 噪聲 [10,  0, 40,  0]]
plt1 = plot_iris_with_cluster_labels(feature1, feature2, np.array(["blue", "green", "red"]), dbscan_labels)
plt1.show()

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

print("\n=== B部分完成 ===")
print("你已經學習了:")
print("1. 聚類的基本概念與分類")
print("2. k-means聚類演算法")
print("3. 層次式聚類演算法")
print("4. DBSCAN密度聚類演算法")
print("5. Scipy層次聚類的應用")
print("\n下一步: 執行 C_實務應用/C_Customer_Clustering_Practice.py")
