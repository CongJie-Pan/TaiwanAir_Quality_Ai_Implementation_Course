# A_AI_ML_Basics_IrisData.py: AI Python 實作 - 07A: AI基礎與Iris數據準備
# Jia-Sheng Heh, 10/23/2024, revised from AIp07聚類及評估A.py
# Usage: 學習AI與機器學習基本概念，並熟悉Iris鳶尾花數據集

import numpy as np
import pandas as pd
import os

##== (O1) 設定工作目錄
wkDir = "AIp07\\AIp07聚類及評估A\\A_AI基礎與數據準備"
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

print("\n=== A部分完成 ===")
print("你已經學習了:")
print("1. AI與機器學習的基本概念")
print("2. 監督式vs非監督式學習的差異")
print("3. Iris數據集的結構與特徵")
print("4. 數據視覺化的基礎函數")
print("\n下一步: 執行 B_聚類演算法/B_Clustering_Algorithms.py")
