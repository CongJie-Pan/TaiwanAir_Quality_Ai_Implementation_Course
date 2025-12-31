# D_Clustering_Evaluation.py: AI Python 實作 - 07D: 聚類評估方法
# Jia-Sheng Heh, 10/23/2024, revised from AIp07聚類及評估A.py
# Usage: 學習聚類品質評估的各種方法

import numpy as np
import pandas as pd
import os

##== (O1) 設定工作目錄
wkDir = "AIp07\\AIp07聚類及評估A\\D_聚類評估"
os.chdir(wkDir)
print(os.getcwd())

##== 載入必要的示例數據 (從B和C部分)
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt

# Iris數據
iris = datasets.load_iris()
X = iris.data
y_true = iris.target

# Scipy層次聚類結果 (從B部分)
from scipy.cluster.hierarchy import linkage, fcluster
linkage_matrix = linkage(X, method='ward')
hierarchical_scipy_labels_iris = fcluster(linkage_matrix, t=3, criterion='maxclust')
hierarchical_scipy_cm = np.array([[0, 50, 0, 0], [0, 0, 1, 49], [0, 0, 35, 15], [0, 0, 0, 0]])  # 模擬混淆矩陣

# 客戶品類矩陣 (需要從C部分載入，這裡使用模擬數據)
# 在實際使用時，應該從C部分載入 CP_matrix 和 CP_matrix1
# 這裡我們創建示例數據來演示評估方法
scaler = StandardScaler()
np.random.seed(42)
CP_matrix_example = pd.DataFrame(np.random.randint(0, 100, (376, 10)),
                                 columns=[f'kind{i}' for i in range(10)])
CP_matrix_scaled = scaler.fit_transform(CP_matrix_example)
Z = linkage(CP_matrix_scaled, method='ward')
CP_matrix_example['cluster'] = fcluster(Z, 6, criterion='maxclust')

CP_matrix1_example = pd.DataFrame(np.random.randint(0, 100, (356, 10)),
                                  columns=[f'kind{i}' for i in range(10)])
CP_matrix1_scaled = scaler.fit_transform(CP_matrix1_example)
Z1 = linkage(CP_matrix1_scaled, method='ward')
CP_matrix1_example['cluster'] = fcluster(Z1, 8, criterion='maxclust')

# 為了演示，我們使用這些示例數據
CP_matrix = CP_matrix_example
CP_matrix1 = CP_matrix1_example

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
silhouette_avg_1 = silhouette_score(CP_matrix[CP_matrix.columns[0:10]], CP_matrix["cluster"]);    print(silhouette_avg_1)
#        ==> CP_matrix的聚類質量比較好，聚類內的資料點彼此之間比較緊密，且與其他聚類的分離度較好。
silhouette_avg_2 = silhouette_score(CP_matrix1[CP_matrix.columns[0:10]], CP_matrix1["cluster"]);  print(silhouette_avg_2)
#        ==> CP_matrix1的值較低，表示該聚類的質量不高，聚類內的資料點可能分佈得較為分散。

#        -- 輪廓係數可視化
#           - 聚類數與平均輪廓係數圖: 隨著聚類數 k 的變化，聚類的平均 Silhouette Score 的變化趨勢 ==> 尋找能夠最大化 Silhouette Score 的聚類數。

def plotSilhouette(Z,CP):            ##== 測試聚類linkage(Z<-CP) 2-10個聚類數，計算其輪廓係數(silhouette_scores)繪圖並求最佳聚類數(best_k) ==##
    from sklearn.metrics import silhouette_score
    from scipy.cluster.hierarchy import fcluster
    silhouette_scores = []
    K_range = range(2, 11)
    for k in K_range:
        cluster_labels = fcluster(Z, k, criterion='maxclust')
        silhouette_avg = silhouette_score(CP, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    best_k = K_range[silhouette_scores.index(max(silhouette_scores))]
    print(f"最佳聚類數量為: {best_k}")
    print(f"silhouette_scores = {np.round(silhouette_scores,3)}")
    return best_k, silhouette_scores

best_k2, silhouette_scores2 = plotSilhouette(Z1, CP_matrix1_scaled)
print(np.round(silhouette_scores2,3))

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
DB_1 = davies_bouldin_score(CP_matrix[CP_matrix.columns[0:10]], CP_matrix["cluster"]);    print(DB_1)  #-- 值較小，表示聚類內緊密度較高，且聚類之間的分離度較好。
DB_2 = davies_bouldin_score(CP_matrix1[CP_matrix.columns[0:10]], CP_matrix1["cluster"]);  print(DB_2)  #-- 值較大，表示聚類內部不夠緊密，聚類之間的分離度較差。

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
wcv, bcv, pwcv = calculate_within_between_variance(scaler.fit_transform(CP_matrix[CP_matrix.columns[0:10]]), CP_matrix["cluster"].to_numpy())
print(f"類別內變異數={wcv:.3f}");   print(f"類別間變異數={bcv:.3f}");   print(np.round(pwcv,3))
# 類別內變異數=1867.611,   類別間變異數=1892.389,   [  17.06  1270.349  155.972  424.23     0.       0.   ]

wcv1, bcv1, pwcv1 = calculate_within_between_variance(scaler.fit_transform(CP_matrix1[CP_matrix1.columns[0:10]]), CP_matrix1["cluster"].to_numpy())
print(f"類別內變異數={wcv1:.3f}");   print(f"類別間變異數={bcv1:.3f}");   print(np.round(pwcv1,3))
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

F_ratio_1 = bcv / wcv
F_ratio_2 = bcv1 / wcv1
print(f"CP_matrix F-ratio: {F_ratio_1:.3f}")
print(f"CP_matrix1 F-ratio: {F_ratio_2:.3f}")

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
Accuracy = (TP+TN) / (TP+TN+FP+FN);   print(f"Accuracy: {Accuracy:.3f}")   #-- 0.16
#        -- 精確率(Precision)--模型對於正例預測的可信度
P = TP / (TP+FP);    print(f"Precision: {P:.3f}")     #-- 0.02
#        -- 召回率(Recall)或 靈敏度(Sensitivity)--模型捕捉到正例的能力
R = TP / (TP+FN);    print(f"Recall: {R:.3f}")     #-- 0.027777777777777776
#        -- F1 分數(F1-Score)--精確率與召回率的調和平均(平衡)
F1 = 2*P*R / (P+R);  print(f"F1-Score: {F1:.3f}")    #-- 0.023255813953488372
#        -- 特異性(Specificity)--模型對負例的區分能力，與靈敏度相對應
Specificity = TN / (TN+FP);   print(f"Specificity: {Specificity:.3f}")        #-- 0.234375

#%%==(3).外部聚類評估: 評估聚類結果與真實標籤的匹配度
#         -- 這些方法包括 Rand Index、Adjusted Rand Index、V-measure (4-9)等
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score
y_pred = hierarchical_scipy_labels_iris   #<== (B2)(3)

#        -- Rand Index（蘭德指數）: 衡量聚類結果與真實標籤之間的一致性。它計算有多少對樣本在真實標籤和聚類結果中屬於相同的類別或不同的類別。
#           - 取值範圍是 [0, 1]，值越接近 1，表示聚類效果越好。
RandIndex = (TP+TN) / (TP+TN+FP+FN);   print(f"Rand Index: {RandIndex:.3f}")                   #-- 0.16 ==> 就是Accuracy

#        -- Adjusted Rand Index (ARI): 表示模型的聚類結果與真實標籤之間的一致性，已考慮隨機結果的調整。1 表示完全一致，0 表示隨機聚類結果。
ari = adjusted_rand_score(y_true, y_pred);   print(f"Adjusted Rand Index (ARI): {ari:.3f}")                   #-- 0.7311985567707746

#        -- Normalized Mutual Information (NMI): 表示模型聚類結果與真實標籤之間的相互信息量，範圍為 [0, 1]，1 表示完全相關。
nmi = normalized_mutual_info_score(y_true, y_pred);   print(f"Normalized Mutual Information (NMI): {nmi:.3f}")          #-- 0.770083661648787

#        -- Homogeneity（同質性）: 表示每個聚類中的樣本幾乎都來自同一個真實標籤類別。
homogeneity = homogeneity_score(y_true, y_pred);   print(f"Homogeneity: {homogeneity:.3f}")     #-- 0.7608008469718723

#        -- Completeness: 表示每個真實類別的樣本幾乎都被分配到同一個聚類中。
completeness = completeness_score(y_true, y_pred);   print(f"Completeness: {completeness:.3f}")  #-- 0.7795958005591144

#        -- V-measure: 是 Homogeneity 和 Completeness 的調和平均，用來綜合評估聚類的質量。
v_measure = v_measure_score(y_true, y_pred);   print(f"V-measure: {v_measure:.3f}")           #-- 0.7700836616487869

print("\n=== D部分完成 ===")
print("你已經學習了:")
print("1. 內部評估方法 (Silhouette Score, Davies-Bouldin Index, Dunn Index)")
print("2. 變異數分析 (Within/Between Cluster Variance)")
print("3. F-ratio與聚類品質評估")
print("4. 外部評估方法 (ARI, NMI, Homogeneity, Completeness, V-measure)")
print("5. 混淆矩陣與分類指標")
print("\n🎉 恭喜！你已完成 AIp07 聚類及評估 的全部內容！")
