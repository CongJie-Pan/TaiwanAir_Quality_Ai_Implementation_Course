#== AIp10決策與分類A.py: AI python 實作 - 10: 1決策與分類 
#== Jia-Sheng Heh, 11/12/2024, revised from HUT08.R

#%%####### import ##########
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
wkDir = os.path.dirname(os.path.abspath(__file__)) + "/";   os.chdir(wkDir);   print(os.getcwd())

#%%####### (A) AI模型與機器學習 ##########

#%%##===== (A1) 人工智慧 (Artificial Intelligence, AI) =====#####

##== (1).AI 定義  ==##

#    -- 概念源起: 古代哲學家們就開始思考機器是否能夠模仿人類的思維。
#       - AI研究的正式開端: 達特茅斯會議 [McCarthy, 1956] 首次提出了“人工智慧”術語 ---> 圖靈測試 (Turing test)
#    -- 設計開發 能執行通常需要人類智慧的任務的計算機系統，包括學習、推理、問題解決、感知和語言理解等。
#       - 如: 基於規則的（如專家系統），或 基於數據驅動的（如機器學習和深度學習）。

##== (2).AI 發展年代與特性  ==##
#    -- 1950年代-1960年代: AI早期研究主要集中在符號推理和問題解決。
#       - 許多早期的AI程序，如邏輯理論家(Logic Theorist)和通用問題解決器(General Problem Solver)
#    -- 1970年代-1980年代: 專家系統成為人工智慧研究的主要焦點。
#       - 專家系統(expert system): 基於規則的系統，能夠模仿人類專家的決策過程,包括 MYCIN和 DENDRAL。
#    -- 1990年代-2000年代: 研究開發許多機器學習方法，應用擴展到更多領域，如語音識別和計算機視覺。
#       - 機器學習(machine learning): 基於數據驅動的方法,如神經網絡和支持向量機。
#    -- 2010年代至今: 深度學習技術的突破使得人工智慧取得了顯著進展，應用包括自動駕駛、自然語言處理和圖像識別等
#       - 深度學習(deep learning): 基於多層神經網絡的機器學習方法，能夠處理大量數據並自動提取特徵。

##== (3).人工智慧的最新發展: ChatGPT的發展 ---> 展示了大型語言模型(llM)的潛力，提升了人機交互的質量
#    -- OpenAI 開發的一種大型語言模型，基於 GPT（生成預訓練變換器）架構，在自然語言處理（NLP）扮演了重要角色
#    -- GPT-1（2018）: OpenAI 發布了第一個 GPT 模型，基於變換器架構的語言模型  --> 能生成連貫的文本。
#    -- GPT-2（2019）: GPT-1 的改進版本，具有更大的模型參數和更強的文本生成能力 --> 能生成高質量的文本，在多種 NLP 任務表現出色。
#    -- GPT-3（2020）: 目前最先進的 GPT 模型，擁有 1750 億個參數 --> 能生成極其自然和連貫的文本，在多種語言任務表現出色
#      --- 標誌著語言模型生成能力的一個重要里程碑。
#    -- ChatGPT（2020）: 基於 GPT-3 的應用，專門用於對話系統 --> 能理解和生成自然語言，進行多輪對話，提供有用的信息和建議。

#%%##===== (A2) 數據模型(Data Model) =====#####

##== (1).系統/模型(System/Model, M):  輸出y = M( 輸入u )
#    -- (phase-1) 訓練階段(Training/Learning/Modeling/Estimation Phase): (u, y) -> M
#             由輸入/輸出 u與y，求取(估測estimate)模型M#
#    -- (phase-2) 預測階段(Prediction/Estimation/Production/Application Phase): (u_new, M#) -> y_predict
#             以所估測的模型M#與新的輸入 u_new，求取(估測)新的輸出 y_predict

##== (2).機器學習(Machine Learning): 從數據(x,y) 求取 知識(模型M#)

#    -- (K1)無監督式學習 (Unsupervised learning): 無輸出y, 目標在於發掘輸入(u)的隱含特徵 --> 數據挖掘(Data Mining)
#           (AIp07)聚類(clustering):           計算數據u的相似度，以產生其分類。
#           (AIp08)關聯規則(association rule): 計算多數據(ui-uj)間的關連。
#           (AIp09)數據序列(data sequencing):  計算多數據(ui-uj)間的時序關係。

#    -- (K2)監督式學習 (Supervised learning): 具範例(u,y), y為教師(teacher, desired output) y, 以求得y=M(u)
#           (AIp09)回歸(regression):     y 為連續數據
#           (AIp10)分類(classification): y 為離散數據
#                  ----> 神經網路(neural network): 自 2014年後，進入深度學習(deep learning)
#                  ----> 所以，現在的 AI，是機器學習／大數據分析的一環 ***

#%%##===== (A3) 主要的分類模型 [殷,7.1] =====#####

##== 決策樹分類: ID3, C4.5, Cart演算法 [殷,7.2, 7.9.1-7.9.2]     --> (C)
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
##== 組合方法 [殷,7.7, 7.9.7]                                   --> (D)
from sklearn.ensemble import RandomForestClassifier  # == 創建 RF 模型
from sklearn.ensemble import AdaBoostClassifier  # == 創建 Bagging 模型 並訓練
from sklearn.ensemble import BaggingClassifier  # == 創建 Bagging 模型 並訓練
##== kNN (k最近鄰)分類 [殷,7.3, 7.9.3]: 是聚類的衍生,訓練/預測時調用, 本單元略過
##== 貝葉斯分類: 最樸素貝葉斯和貝葉斯信念網路分類法 [殷,7.4, 7.9.4] --> (E)
from scipy.optimize import minimize
from sklearn.naive_bayes import GaussianNB
##== 人工神經網路: 無監督學習網路和有監督學習網路 [殷,7.5, 7.9.5]   --> (F)
from sklearn.neural_network import MLPClassifier
##== 支持向量機: 線性SVM分類 [殷,7.6, 7.9.6]                     --> (F)
from sklearn.svm import SVC

#%%##===== (A4) 相關數據議題 =====#####
##== (1).實驗比較數據: iris
from sklearn import datasets  # -- import some data to play with
from sklearn.datasets import load_iris
##== (2).數據編碼
##-- 先進行 pip install category_encoders
from category_encoders import BinaryEncoder
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from category_encoders import TargetEncoder
from sklearn.preprocessing import OrdinalEncoder
##== (3).數據規畫
from sklearn.model_selection import train_test_split
##== (4).數據評估
from sklearn.metrics import confusion_matrix, classification_report

#%%##===== (A5).實驗數據 =====#####
iris = load_iris()
#%%##===== (A4) 鳶尾花 (iris) 數據說明 (-->iris.data/target) [AIp07(A3)] =====#####
# 美國加州大學歐文分校的機械學習數據庫http://archive.ics.uci.edu/ml/datasets/Iris

##== (1).系統模型:  輸出y = iris.target = M (輸入X = iris.data ) ==###

##== (2).(KDD1) 鳶尾花數據取得
iris = datasets.load_iris()  # -- 數據的筆數為150筆，共有五個欄位(前四個單位為公分)
print(iris.keys())
#-- dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])

#%%== (1).(KDD2) 輸入X -- 鳶尾花的四項特徵: iris.data[iris.feature_names]
print("iris.data.shape=", iris.data.shape)        #-- (150, 4)
print("iris.feature_names=", iris.feature_names)  #-- 鳶尾花的四項特徵
#-- ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
#--  花萼長度(Sepal Length),花萼寬度(Sepal Width),花瓣長度(Petal Length),花瓣寬度(Petal Width),
print("iris.data[0:3,]=", iris.data[0:3,])
# [[5.1 3.5 1.4 0.2]
#  [4.9 3.  1.4 0.2]
#  [4.7 3.2 1.3 0.2]]
X = iris.data

#%%== (2).(KDD2) 輸出y -- 鳶尾花的的三個品種: iris.target_names)
print("iris.target.shape=", iris.target.shape)  # -- (150,)
#-- array(['setosa', 'versicolor', 'virginica'], dtype='<U10')
print("iris.target_names=", iris.target_names)
#-- 類別(Class)：鳶尾花的三個品種Setosa，Versicolor和Virginica
print("iris.target[1:70]=", iris.target[1:70])
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
y = y_true = iris.target

#%%##===== (A6) (KDD5) 鳶尾花類別以三種顏色作分布圖(iris.data/target-->plot_iris(feature1,feature2)) [AIp07(A3)] =====#####
##== (1).繪出 iris的分類
def plot_iris(feature1, feature2, cmap):  ##== 以 iris.data 中的(feature1,feature2)特徵作圖 ==##
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.datasets import load_iris
    iris = load_iris()  # -- 加載鳶尾花數據集
    irisX = pd.DataFrame(iris.data, columns=iris.feature_names)
    X = irisX[[feature1, feature2]]
    y = iris.target
    x_min, x_max = X[feature1].min() - .5, X[feature1].max() + .5
    y_min, y_max = X[feature2].min() - .5, X[feature2].max() + .5
    plt.figure(2, figsize=(8, 6))
    plt.clf()
    p1 = plt.scatter(X[feature1][0:49], X[feature2]
                     [0:49], c=cmap[0], edgecolor='k')
    p2 = plt.scatter(X[feature1][50:99], X[feature2]
                     [50:99], c=cmap[1], edgecolor='k')
    p3 = plt.scatter(X[feature1][100:149], X[feature2]
                     [100:149], c=cmap[2], edgecolor='k')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend([p1, p2, p3], iris.target_names, loc="upper left")
    return (plt)

feature1 = 'petal length (cm)';   feature2 = 'petal width (cm)'
plt1 = plot_iris(feature1, feature2, np.array(["blue", "green", "red"]));   plt1.show()

#%%== (2).繪出iris分類,並加上聚類標籤(y_predict)

##== 繪出iris分類,並加上聚類標籤(y_predict) ==##
def plot_iris_with_cluster_labels(feature1, feature2, cmap, y_predict):
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.datasets import load_iris
    iris = load_iris()  # -- 加載鳶尾花數據集
    irisX = pd.DataFrame(iris.data, columns=iris.feature_names)
    X = irisX[[feature1, feature2]]
    y = iris.target
    x_min, x_max = X[feature1].min()-.5, X[feature1].max()+.5
    y_min, y_max = X[feature2].min()-.5, X[feature2].max()+.5
    plt.figure(2, figsize=(8, 6))
    plt.clf()
    for i in range(len(X)):  # -- 在數據點上標註聚類結果，用原始分類的顏色(iris.target)來顯示數字(y_predict)
        plt.text(X[feature1][i], X[feature2][i], str(y_predict[i]),
                 fontsize=12, color=cmap[y[i]], ha='center', va='center')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    from matplotlib.lines import Line2D  # -- 繪製顏色對應的圖例，顏色依據原始類別
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=iris.target_names[i],
                              markerfacecolor=cmap[i], markersize=10) for i in range(3)]
    plt.legend(handles=legend_elements, loc="upper left")
    return plt

feature1 = 'petal length (cm)';   feature2 = 'petal width (cm)'
y_predict = np.random.randint(0, 3, size=150)  # -- 模擬的聚類結果
plt1 = plot_iris_with_cluster_labels(feature1, feature2, np.array(["blue", "green", "red"]), y_predict)
plt1.show()


#%%####### (B) 數據規畫 ##########

#%%##===== (B1) 數據規畫 =====#####

##== (1).數據規畫 (data planning)
#   -- 將數據分成不同部分 (如訓練集、驗證集、測試集)，使模型在不同階段進行訓練、調參和評估，以保證模型的泛化能力。

##== (2) 數據分割 (Data Partition)
#    -- (A).訓練集（Train Set, Seen Data）
#         - 用於訓練機器學習模型，通過這部分數據更新模型的參數。
#         - 數據量：通常佔總數據的 60%-80%。
#    -- (B).測試集（Test Set, Unseen Data）
#         - 用於評估模型的最終性能，這部分數據不參與模型訓練。
#         - 數據量：通常佔總數據的 10%-20%。
#    -- (C).驗證集（Validation/Tuning Set）
#         - 用於調整模型超參數，評估模型在未知數據上的表現，幫助選擇最佳模型。
#         - 數據量：通常佔總數據的 10%-20%。

##== (3).驗證集與測試集的比較
#      -- 項目	      驗證集（Validation Set）	        測試集（Test Set）
#      -- 主要用途	  用於調整超參數、選擇最佳模型	        測試模型的最終泛化性能
#      -- 是否參與訓練 不直接參與參數更新，但參與模型選擇流程	完全不參與訓練與調參
#      -- 使用次數	  可能多次使用	                    僅使用一次，避免數據「洩露」
#      -- 重要性	      幫助選擇最佳模型結構或超參數	        用於衡量模型的真實性能

#%%##===== (B1) 數據分割方法: 留出法 =====#####

##== (*).準備數據: 將iris.data轉換為DataFrame ==##
irisX = pd.DataFrame(iris.data, columns=iris.feature_names)
X = irisX[['petal length (cm)', 'petal width (cm)']]
y = iris.target

##== (*).留出法(hold-out): 抽取80%的數據用以建構模式, 剩下的20%用於模式的效度檢驗 train_test_split
#      -- 1).將數據集 X 劃分為 訓練集X_train 及 測試集 X_test
#      -- 2).通常以 2/3~4/5 的數據用於訓練，test_size 預設值為 0.25
#      -- 3).亦可設定 test_size 為測試數量，另亦可設定訓練數據 train_size

##== (1).隨機分割(Random Split)法: 隨機將數據分為訓練集、驗證集和測試集 ==##
#    -- (PROs) 簡單易用，適合數據分佈均勻的情況。
#    -- (CONs) 若數據有時間順序，可能會導致未來數據「洩露」。
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.index)
# Int64Index([131,  71, 142,  72,  64,  36,  90, 125,  30,  53, ...
#              26,  68,  18, 107,  56,   1,  16,  69, 128,  76], dtype='int64', length=120)
print(X_test.index)
# Int64Index([ 74,  25,   8,  58, 122, 120,  59, 145,  11, 113,  62, 127,  61,
#              87,  43, 102,   2, 143,  52,  91, 141, 112, 100, 104, 126,  94,  99,  32,   5, 135], dtype='int64')print(X2_train.shape);
print(X_train.head(2))  # --(120, 2)
#      petal length (cm)  petal width (cm)
# 131                6.4               2.0
# 71                 4.0               1.3
print(y_test)
# [1 0 0 1 2 2 1 2 0 2 1 2 1 1 0 2 0 2 1 1 2 2 2 2 2 1 1 0 0 2]

##== (2).分層抽樣(Stratified Sampling)法 ==##
#    -- 按照某個類別標籤的比例，將數據分為訓練集和測試集
#    -- (PROs) 適合分類問題，保證數據的類別分佈一致。
#    -- (CONs) 不適用於回歸問題或無標籤數據。
X_trainS, X_testS, y_trainS, y_testS = train_test_split(
    X, y, test_size=0.2, stratify=y)
# -- [1 0 0 1 1 1 2 1 2 2 2 0 2 2 0 2 1 1 0 0 0 2 1 0 0 1 0 2 2 1]
print(y_testS)

##== (3).時間序列分割（Time-Based Split）法 ==##
#    -- 按時間順序分割數據。早期數據作為訓練集，後期數據作為測試集或驗證集
#    -- (PROs) 適合處理時序數據，保證數據的時間依賴性。
#    -- (CONs) 訓練集和測試集的分佈可能不同，模型難以泛化。

## == (4).Bootstrap 分割 ==##
#    -- 從數據集中有放回地抽樣生成訓練集，未抽到的數據用作測試集。
#    -- (PROs) 適合小數據集，能提供模型性能的統計分佈。
#    -- (CONs) 訓練集和測試集可能有重複數據。

#%%##===== (B2) 數據分割方法: 交叉驗證 =====#####

##== (1).k-fold交叉驗證法(cross-validation,CV) ==##
#    -- 將數據分成k個等分，取一部分作為驗證集，其餘作為訓練集，重複k次
#    -- (方法1A).K-Fold CV：數據均勻分為 k 部分。
#    -- (方法1B).Stratified K-Fold：類似 K-Fold，但保證每部分的類別分佈一致。
#    -- (方法1C).Time-Series CV：保證訓練集總在驗證集之前。
#    -- (PROs) 用於小數據集，能充分利用所有數據。
#    -- (CONs) 計算成本高，尤其在大數據集上。

##== (2).留一法 (Leave-One-Out Cross-Validation, LOOCV) ==##
#    -- 每次僅用一個樣本作為驗證集，其餘作為訓練集
#    -- (PROs) 適用於小數據集，保證每個數據點都被用作驗證。
#    -- (CONs) 計算成本極高，對大數據集不現實。

##== (3).滑動驗證（Rolling Validation，Sliding Window CV） ==##
#    -- 使用前 n 條數據作為訓練集，下一段作為驗證集，重複多次
#    -- 常見於時序數據。
#    -- (PROs) 保證數據的時間順序。
#    -- (CONs) 訓練集數據量可能不足。


#%%####### (C) 決策樹 (Data Clustering) ##########

#%%##===== (C1) 決策樹(decision tree)分類基本概念 [殷,7.2, 7.9.2] =====#####
##== (1).決策樹: 由結節和有向邊組成的層次結構:
#    -- 樹狀結構, 包括(A)(待分類的)決策結點, (B)(可能分類結困的)葉結點或終結點, (C)(不同決策取值的)分支
#    -- 三種資料屬性: (A)標稱屬性(single,married,divorced), (B)序列屬性(e.g.Hot,Mild,Cool), (C)連續屬性(e.g.Petal.Width)
#    -- 兩種資料分裂(split): (A)二元劃分, (B)多路劃分

##== (2).三種數據不純度(impurity)度量: (A)Gini, (B)熵/亂度(Entropy), (C)分類誤差(Error) [請參見(3A)]

##== (3).三種常見決策樹演算法
#    -- (A).ID3: 利用增益率, 採用二叉樹 -- 演法偽代碼 [殷,7.2, p.176]
#    -- (B).C4.5: ID3改進版, 採用多重分支和剪枝技術 -- 演法偽代碼 [殷,7.2, pp.181-182]
#    -- (C).CART: 利用Gini係數，採用二元遞迴劃分方法 -- 演法偽代碼 [殷,7.2, p.188]
#       [分類回歸樹(CART, Classification And Regression Tree), Breiman (1984)]

#%%##===== (C2) 決策樹的實務操作 =====#####

##== (1).數據準備 (iris-->X,y) ==##
iris = load_iris(as_frame=True)
X = iris.data
print(X.head(2))
#    sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
# 0                5.1               3.5                1.4               0.2
# 1                4.9               3.0                1.4               0.2
y = iris.target
print(y[48:52])
# 48    0
# 49    0
# 50    1
# 51    1
features = iris.feature_names
# -- ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
features

#%%== (2).訓練決策樹(X,y-->tree_model-->tree_rules) ==##
tree_model = DecisionTreeClassifier(
    max_depth=3, random_state=0)  # -- 不純度預設為 Gini
tree_model.fit(X, y)
tree_rules = export_text(
    tree_model, feature_names=features, class_names=iris.target_names)
print(tree_rules)
# |--- petal width (cm) <= 0.80
# |   |--- class: setosa
# |--- petal width (cm) >  0.80
# |   |--- petal width (cm) <= 1.75
# |   |   |--- petal length (cm) <= 4.95
# |   |   |   |--- class: versicolor
# |   |   |--- petal length (cm) >  4.95
# |   |   |   |--- class: virginica
# |   |--- petal width (cm) >  1.75
# |   |   |--- petal length (cm) <= 4.85
# |   |   |   |--- class: virginica
# |   |   |--- petal length (cm) >  4.85
# |   |   |   |--- class: virginica

#%%== (3).視覺化決策樹(tree_model-->plot_tree()) ==##
plot_tree(tree_model, feature_names=features,
          class_names=iris.target_names, filled=True)
plt.title("Decision Tree Visualization")
plt.show()

#%%== (4).預測與模型評估: 混淆矩陣 (y->y_tree, y_predict) ==##
y_predict = tree_model.predict(X)
y_true = y
print(pd.crosstab(y_true, y_predict))
# col_0    0   1   2
# target
# 0       50   0   0
# 1        0  47   3
# 2        0   1  49
##== 分類報告(classification report)摘要:
print(classification_report(y_true, y_predict, target_names=iris.target_names))
#               precision    recall  f1-score   support
#       setosa       1.00      1.00      1.00        50
#   versicolor       0.98      0.94      0.96        50
#    virginica       0.94      0.98      0.96        50
#     accuracy                           0.97       150
#    macro avg       0.97      0.97      0.97       150
# weighted avg       0.97      0.97      0.97       150

#%%== (5).數據空間及決策樹規則的直角分割(rectangular partitions) ==##
# 此為數據空間的展示
# 決策樹骨子裡面就是去做空間的切割

plt1 = plot_iris_with_cluster_labels(
    features[2], features[3], np.array(["blue", "green", "red"]), y_predict)
# -- 規則 1: petal width (cm) <= 0.80 -> 水平線
plt1.plot([0.5, 7.5], [0.80, 0.80], color='red',   linestyle='--')
# -- 規則 2: petal width (cm) <= 1.75 -> 水平線
plt1.plot([0.5, 7.5], [1.75, 1.75], color='orchid', linestyle='--')
# -- 規則 3: petal length (cm) <= 4.95 -> 垂直線
plt1.plot([4.95, 4.95], [0.8, 1.75], color='pink',  linestyle='--')
# -- 規則 4: petal length (cm) <= 4.85 -> 垂直線
plt1.plot([4.85, 4.85], [1.75, 3.00], color='orange', linestyle='--')
plt1.show()

#%%##===== (C4) 不純度的計算: Gini [殷,7.2] =====#####

##== (1).三種數據不純度(impurity)度量:
#      -- (A).Gini = 1-sum(p(xi|x)^2)
#      -- (B).熵/亂度(Entropy) = Entropy(x) = -sum(p(xi|x)*log2(p(xi|x))
#      -- (C).分類誤差(Error) = Error(x) = 1 - max(p(xi|x)
##== (2).三種數據不純度(impurity)度量:
#      -- 當分類完成時,不純度達到最小值0
#      -- 增益(gain) = 數據分裂前的不純度 - 數據分裂後的不純度

##== (3).Gini, 第一種不純度(impurity): Gini(x) = 1-sum(p(xi|x)^2) ==##
#       -- (1a) 練習數據 [殷,p.169]
bank = pd.DataFrame({
    "house": ["yes", "no", "no", "yes", "no", "no", "yes", "no", "no", "no"],
    "marriage": ["single", "married", "single", "married", "divorced", "married",
                 "divorced", "single", "married", "single"],
    "income": [125, 100, 70, 120, 95, 60, 220, 85, 75, 90],
    "debt": ["no", "no", "no", "no", "yes", "no", "no", "yes", "no", "yes"]
})
bank['house'] = bank['house'].astype('category')  # -- 將類別型欄位轉為分類型
bank['marriage'] = bank['marriage'].astype('category')
bank['debt'] = bank['debt'].astype('category')
print(bank)
#   house  marriage  income debt
# 0   yes    single     125   no
# 1    no   married     100   no
# 2    no    single      70   no
# 3   yes   married     120   no
# 4    no  divorced      95  yes
# 5    no   married      60   no
# 6   yes  divorced     220   no
# 7    no    single      85  yes
# 8    no   married      75   no
# 9    no    single      90  yes

#%%== (4).Gini 係數及增益函式與測試 ==##

def Gini(tblX):  total = np.sum(tblX); return 1 - np.sum((tblX/total)**2)  ##== 計算 Gini 係數 ==##
print("Gini (全體):", Gini(bank['debt'].value_counts()))   # -- Gini (全體): 0.42000000000000004

def Gini2(X, A, a1, a2):  ##== 計算條件分割後的 Gini 係數 ==##
    mask = A.isin(a1)
    AX = pd.crosstab(mask, X)  # -- 創建分割條件
    gini_values = AX.apply(Gini, axis=1)  # -- 計算條件分割後的 Gini 系數
    weighted_gini = np.sum(gini_values * AX.sum(axis=1) / AX.values.sum())
    return weighted_gini
print("Gini2 (house):", Gini2(bank['debt'], bank['house'], ["yes"], ["no"]))  #-- Gini2 (house): 0.3428571428571429

#%%== (5).Gini 增益計算 ==##
gini_all = Gini(bank['debt'].value_counts())
gini_gain_house = gini_all - Gini2(bank['debt'], bank['house'], ["yes"], ["no"])
print("Gini 增益 (house):", gini_gain_house)   #-- Gini 增益 (house): 0.07714285714285712
gini_gain_marriage1 = gini_all - Gini2(bank['debt'], bank['marriage'], ["married", "divorced"], ["single"])
print("Gini 增益 (marriage1):", gini_gain_marriage1)   #-- Gini 增益 (marriage1): 0.053333333333333455
gini_gain_marriage2 = gini_all - Gini2(bank['debt'], bank['marriage'], ["single", "divorced"], ["married"])
print("Gini 增益 (marriage2):", gini_gain_marriage2)   # -- Gini 增益 (marriage2): 0.12000000000000005
gini_gain_marriage3 = gini_all - Gini2(bank['debt'], bank['marriage'], ["single", "married"], ["divorced"])
print("Gini 增益 (marriage3):", gini_gain_marriage3)   #-- Gini 增益 (marriage3): 0.020000000000000018
#-- 使用分割的收入範圍計算 Gini 增益 --#
bank['income_bin1'] = pd.cut( bank['income'], bins=[0, 65, 300], right=True).astype(str)
gini_gain_income1 = gini_all - Gini2(bank['debt'], bank['income_bin1'], ["(0, 65]"], ["(65, 300]"])
print("Gini 增益 (income_bin1):", gini_gain_income1)   #-- Gini 增益 (income_bin1): 0.020000000000000018
bank['income_bin2'] = pd.cut( bank['income'], bins=[0, 92.5, 300], right=True).astype(str)
gini_gain_income2 = gini_all - Gini2(bank['debt'], bank['income_bin2'], ["(0, 92.5]"], ["(92.5, 300]"])
print("Gini 增益 (income_bin2):", gini_gain_income2)   #-- Gini 增益 (income_bin2): 0.0

#%%##===== (C5) 組合方法分類 [殷,7.7] =====#####

#%%==(0).組合方法/整體學習/匯總學習 (Ensemble Learning):
#      -- 通過多個分類器的預測，以各基分類器的預測進行投票，來提高分類準確度的技術
#      -- 加載 iris 數據集
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

#%%==(1).裝袋Bagging (Bootstrap Aggregating) 自助聚集: 匯總學習 [Breiman,1994][黃文, Chap.10]
#      -- 根據均勻概率分佈，從資料中重複抽樣，各訓練一個基分類器，再依其預測結果投票
bagging_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(), n_estimators=5, random_state=42)
bagging_model.fit(X, y)  # == PHASE I-模型訓練 (固定隨機數生成器的初始化，確保結果可重現)
print("Base estimators (trees):", len(bagging_model.estimators_))   #-- Base estimators (trees): 5
print(pd.DataFrame({feature_names[i]: [tree.feature_importances_[i] for tree in bagging_model.estimators_]
                    for i in range(len(feature_names))}).mean(axis=1))
                    # -- 各特徵重要性 的 平均值:  0.25   0.25   0.25    0.25   0.25
print(export_text(bagging_model.estimators_[0], feature_names=feature_names))
# |--- petal width (cm) <= 0.80
# |   |--- class: 0
# |--- petal width (cm) >  0.80
# |   |--- petal length (cm) <= 4.85
# |   |   |--- petal width (cm) <= 1.70
# |   |   |   |--- class: 1
# |   |   |--- petal width (cm) >  1.70
# |   |   |   |--- sepal width (cm) <= 3.10
# |   |   |   |   |--- class: 2
# |   |   |   |--- sepal width (cm) >  3.10
# |   |   |   |   |--- class: 1
# |   |--- petal length (cm) >  4.85
# |   |   |--- petal width (cm) <= 1.75
# |   |   |   |--- petal width (cm) <= 1.65
# |   |   |   |   |--- sepal width (cm) <= 3.05
# |   |   |   |   |   |--- class: 2
# |   |   |   |   |--- sepal width (cm) >  3.05
# |   |   |   |   |   |--- class: 1
# |   |   |   |--- petal width (cm) >  1.65
# |   |   |   |   |--- class: 1
# |   |   |--- petal width (cm) >  1.75
# |   |   |   |--- class: 2
predictions = bagging_model.predict(X)  #== PHASE II-模型預測
print(confusion_matrix(y, predictions))
# [[50  0  0]
#  [ 0 49  1]
#  [ 0  0 50]]

#%%==(2).提升Adaboost (Adaptive Boosting) 自我調整增加模型 [Breiman,1996]
#      -- 每一個訓練樣本一個權值，依是否正確分類調整權重，越難分類權重越高，使某些分類器更對其分類
adaboost_model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1),
                                    n_estimators=5,  random_state=42)
adaboost_model.fit(X, y)  #== PHASE I-模型訓練 (固定隨機數生成器的初始化，確保結果可重現)
#-- 模型中第一棵決策樹的結構:
print(export_text(adaboost_model.estimators_[0], feature_names=feature_names))
# |--- petal width (cm) <= 0.80
# |   |--- class: 0
# |--- petal width (cm) >  0.80
# |   |--- class: 1
# -- 模型中第二棵決策樹的結構:
print(export_text(adaboost_model.estimators_[1], feature_names=feature_names))
# |--- petal width (cm) <= 1.75
# |   |--- class: 1
# |--- petal width (cm) >  1.75
# |   |--- class: 2

#%%==(3).Random Forest (RF) 隨機森林 [Breiman,2001]
#      -- 每一棵決策樹依賴於獨立抽樣，所有的樹具有相同的分佈的隨機向量的值，依各樹的預測結果進行投票
#-- 固定隨機數生成器的初始化，確保結果可重現在 --> 多數實驗中，建議同時設置，特別是同時使用 NumPy 和 scikit-learn 時
np.random.seed(777)
rf_model = RandomForestClassifier(
    n_estimators=500, max_features=2, random_state=777, oob_score=True)  # -- oob_score:設置袋外錯誤
rf_model.fit(X, y)  #== PHASE-I: 訓練模型
#-- 特徵重要性
feature_importances = pd.DataFrame({"Feature": feature_names, "Importance": rf_model.feature_importances_}).sort_values(by="Importance", ascending=False)
print(feature_importances)
#              Feature  Importance
# 3   petal width (cm)        0.45
# 2  petal length (cm)        0.42
# 0  sepal length (cm)        0.10
# 1   sepal width (cm)        0.03
tree_sizes = [estimator.tree_.node_count for estimator in rf_model.estimators_]
print(tree_sizes)  #-- 500棵樹, 每棵樹的大小
# [15, 11, 11, 9*, 17, 17, 11, 19, 21, 11, 9, 17, 17, 23, 15, 19, 17, 19, 23, 17, 15, 13, 15, 25, 15, 21, 13, 19, 17, 17, 21, 13, 13, 15, 13, 15, 17, 23, 25, 21, 21, 7, 19, 21, 17, 15, 17, 29, 19, 23,
#  11, 21, 13, 15, 19, 13, 13, 17, 21, 19, 13, 11, 19, 19, 19, 11, 17, 13, 23, 23, 21, 15, 9, 11, 13, 15, 23, 15, 13, 15, 13, 19, 13, 13, 15, 21, 19, 13, 19, 11, 13, 19, 13, 11, 17, 25, 19, 13, 23,
#  25, 15, 13, 17, 15, 15, 17, 15, 19, 13, 19, 17, 23, 15, 13, 15, 7, 21, 19, 21, 15, 29, 25, 19, 19, 19, 11, 23, 21, 23, 15, 21, 21, 27, 13, 17, 7, 15, 21, 9, 9, 19, 11, 17, 13, 21, 13, 15, 17, 19,
#  21, 29, 19, 15, 9, 11, 13, 23, 5, 19, 13, 9, 19, 19, 13, 17, 15, 17, 11, 23, 21, 15, 19, 19, 19, 21, 15, 15, 17, 15, 17, 13, 11, 17, 15, 15, 13, 9, 7, 13, 13, 29, 17, 23, 9, 19, 15, 25, 17, 17, 21,
#  21, 21, 17, 17, 27, 15, 19, 19, 17, 15, 27, 23, 13, 15, 17, 19, 11, 15, 23, 15, 11, 15, 17, 9, 17, 11, 15, 15, 21, 19, 15, 13, 15, 11, 11, 17, 11, 9, 23, 21, 13, 13, 13, 11, 23, 13, 17, 27, 9, 11,
#  21, 9, 17, 19, 17, 15, 13, 23, 29, 17, 11, 13, 21, 15, 15, 15, 19, 19, 9, 9, 15, 17, 11, 17, 11, 15, 23, 15, 21, 13, 25, 15, 21, 15, 19, 15, 11, 19, 19, 11, 9, 23, 17, 17, 17, 17, 15, 19, 21, 15,
#  23, 11, 11, 23, 23, 21, 23, 11, 13, 15, 7, 15, 11, 15, 15, 19, 21, 17, 19, 15, 19, 7, 9, 15, 13, 17, 15, 17, 21, 17, 15, 29, 17, 17, 21, 23, 17, 21, 15, 27, 19, 17, 11, 19, 17, 15, 9, 19, 23, 15,
#  15, 19, 17, 11, 15, 25, 13, 19, 11, 11, 13, 11, 15, 23, 21, 21, 19, 27, 15, 7, 25, 23, 13, 25, 17, 21, 15, 21, 23, 15, 13, 19, 13, 13, 11, 15, 21, 19, 17, 27, 25, 19, 15, 21, 21, 19, 21, 15, 11,
#  23, 25, 11, 23, 11, 15, 21, 21, 19, 13, 17, 5, 13, 17, 7, 9, 11, 19, 17, 19, 17, 13, 19, 13, 13, 17, 17, 21, 21, 13, 13, 27, 23, 23, 23, 19, 19, 15, 19, 23, 21, 15, 21, 19, 11, 11, 21, 11, 29, 19,
#  11, 21, 15, 23, 19, 21, 25, 13, 19, 23, 11, 11, 21, 7, 21, 19, 23, 11, 19, 15, 11, 17, 15, 21, 13, 23, 9, 25, 21, 15, 9, 13, 21, 11, 21, 17, 13, 17, 19, 13, 17, 15, 23, 13, 17, 23, 15, 17, 17, 21, 19]
#-- 第3號樹的結構 (4個決策點*,5個終端點$)
print(export_text(rf_model.estimators_[3], feature_names=feature_names))
# |--- petal width (cm) <= 0.80*
# |   |--- class: 0.0$
# |--- petal width (cm) >  0.80
# |   |--- petal width (cm) <= 1.65*
# |   |   |--- sepal width (cm) <= 2.85*
# |   |   |   |--- petal length (cm) <= 5.00*
# |   |   |   |   |--- class: 1.0$
# |   |   |   |--- petal length (cm) >  5.00
# |   |   |   |   |--- class: 2.0$
# |   |   |--- sepal width (cm) >  2.85
# |   |   |   |--- class: 1.0$
# |   |--- petal width (cm) >  1.65
# |   |   |--- class: 2.0$
#%%      -- 混淆矩陣
y_pred = rf_model.predict(X)
print(pd.DataFrame(confusion_matrix(y, y_pred),
      index=target_names, columns=target_names))
#             setosa  versicolor  virginica
# setosa          50           0          0
# versicolor       0          50          0
# virginica        0           0         50

#%%==(3A).袋外誤差(Out-of-Bag Error, OOB Error)或袋外分數(OOB Score)
#        -- 隨機森林模型的一個重要評估指標，主要用於衡量模型的泛化性能。
#        -- Bootstrap Sampling: 在隨機森林中，每棵決策樹的訓練過程 會隨機從訓練數據中進行有放回的抽樣
#           - 袋內樣本: 大約 63% 的數據 會被抽樣用於訓練一棵決策樹
#           - 袋外樣本(Out-of-Bag Samples): 剩下的 37% 數據 不會被抽樣，對該棵決策樹，可用來測試該樹的預測性能
#        -- 袋外誤差(OOB Error)計算:
#           - 1)對每個樣本，檢查是否是某棵樹的袋外樣本。                        2)使用此袋外樣本來測試對應樹的預測結果。
#           - 3)對各訓練樣本的袋外預測取眾數（對分類問題）或平均值（對回歸問題） 4)計算袋外預測結果與實際結果之間的錯誤率。
#           - 特點: 1)不需要額外的驗證集, 2)模型泛化性能的(交叉驗證)指標, 3)比單一訓練集測試更可靠
#        -- 袋外分數 (OOB Score):  OOB Score = 1 − OOB Error... 如:袋外分數為 96%，袋外誤差即為 4%。
#-- 袋外誤差 (OOB Score): 4.00%  --> 表示模型在未見數據上的預期錯誤率。
print(f"袋外誤差 (OOB Score): {1 - rf_model.oob_score_:.2%}")


#%%####### (D) 數據編碼(Data Encoding) + 決策樹的實作例 ##########

#%%##===== (D1) 數據編碼(Data Encoding) =====#####

# == (1).數據編碼 (Data Encoding)
#     -- 將原始數據（通常是非數值型數據，例如文字或類別型數據）轉換為模型可以處理的數值型數據的過程
#     -- 是機器學習中預處理數據的重要步驟之一，因大多數機器學習(如決策樹、線性回歸等)僅能處理數值型數據。

# == (2).數據編碼的原因
#      -- (A).機器學習模型的需求：大部分機器學習模型無法直接處理文字或類別型數據，需先轉換為數值型數據。
#      -- (B).捕捉數據間的關係：如類別型數據的大小或頻率可以用數值表示，這樣更容易讓模型理解數據的含義。
#      -- (C).統一數據格式：混合數據類型的特徵需統一為數值型數據，這樣才能進行數學運算和特徵工程。

#%%##===== (D2) 數據編碼的種類1: 類別型數據 =====#####

##== (0).目標數據：如 ["男", "女"], ["是", "否"] 或 ["高", "中", "低"] 等非數值型類別變數。

#%%==(1).Label Encoding: 將每個類別標籤直接映射為數值
#     -- 適用場景： 當類別之間存在 自然順序（如 低、中、高）時。
#     -- 優點：簡單、快速，適合數據量較大的情況。
#     -- 缺點：當類別間沒有順序關係時，模型可能會錯誤地解讀數值之間的大小關係（如 1 比 0 大）。
data = ['cat', 'dog', 'mouse', 'dog', 'cat']  #-- 這種數據原則上亦不該用Label Encoding
label_encoder = LabelEncoder()
encoded_data = label_encoder.fit_transform(data)
print("Encoded data:", encoded_data)  #-- Encoded data: [0, 1, 2, 1, 0]

#%%==(2).One-Hot Encoding：為每個類別創建二元特徵。
#     -- 適用場景：當類別之間沒有順序關係時（如 cat、dog、mouse）。
#     -- 優點：消除了類別之間的大小關係問題。
#     -- 缺點：當類別數量多時，會產生大量的特徵，導致維度爆炸（稱為 "curse of dimensionality"）。
data = np.array(['cat', 'dog', 'mouse', 'dog', 'cat']).reshape(-1, 1)
one_hot_encoder = OneHotEncoder(sparse_output=False)
encoded_data = one_hot_encoder.fit_transform(data)
#-- One-Hot Encoded data: [[1. 0. 0.]
print("One-Hot Encoded data:\n", encoded_data)
#  [0. 1. 0.]
#  [0. 0. 1.]
#  [0. 1. 0.]
#  [1. 0. 0.]]

#%%==(3).Ordinal Encoding：類似於 Label Encoding，但允許指定類別之間的順序關係, 如 低->0,中->1,高->2
#      -- 適用場景：當類別之間有邏輯順序時（如 low、medium、high）。
#      -- 缺點: 與模型選擇高度相關
data = [['low'], ['medium'], ['high'], ['medium']]
ordinal_encoder = OrdinalEncoder(categories=[['low', 'medium', 'high']])
encoded_data = ordinal_encoder.fit_transform(data)
print("Ordinal Encoded data:", encoded_data)  # -- Ordinal Encoded data: [[0.]
#                        [1.]
#                        [2.]
#                        [1.]]

#%%==(4).Binary Encoding：先將類別映射為數值，再轉換為二進制，最後將每個二進制位作為一個新的特徵
#     -- 適用場景: 適合類別數量較多且無順序的情況，比 One-Hot Encoding 更加節省空間。
#     -- 缺點: 解釋性較弱，需要額外安裝 category_encoders 套件。
data = pd.DataFrame({'category': ['A', 'B', 'C', 'D']})
binary_encoder = BinaryEncoder(cols=['category'])
encoded_data = binary_encoder.fit_transform(data)
print(encoded_data)
#    category_0  category_1  category_2
# 0           0           0           1
# 1           0           1           0
# 2           0           1           1
# 3           1           0           0

#%%==(5).Target Encoding: 將類別標籤替換為目標變數的平均值（適用於監督學習）。
#      -- 適用場景: 適合類別數量多且有監督數據(監督學習、有目標變數)的情況。
#      -- 易受數據洩露（data leakage）影響，需要交叉驗證。
data = pd.DataFrame({'category': ['A', 'B', 'A', 'C'], 'target': [1, 2, 1, 3]})
target_encoder = TargetEncoder(cols=['category'])
encoded_data = target_encoder.fit_transform(data['category'], data['target'])
print(encoded_data)
#    category
# 0  1.643612
# 1  1.782527
# 2  1.643612
# 3  1.912636

#%%==(6).Frequency Encoding: 用每個類別在數據中的出現頻率進行編碼。
#       -- 適用場景: 簡單且有效，適合處理類別數量多但頻率差異大的情況。
#       -- 缺點: 無法直接捕捉類別間的差異
data = pd.Series(['A', 'B', 'A', 'C', 'B', 'A'])
frequency_encoded = data.map(data.value_counts(normalize=True))
print(frequency_encoded)
# 0    0.500000
# 1    0.333333
# 2    0.500000
# 3    0.166667
# 4    0.333333
# 5    0.500000

#%%##===== (D3) 數據編碼的種類2: 數值、時間性，與文本數據 =====#####

#%%==(1).數值型數據的分箱
#     -- 目標數據：對於數值型數據（如年齡、價格）進行分組以簡化模型處理。
#     -- 編碼方法：
#        - (1A).分箱（Binning）：將連續數據分為固定範圍的區間，例如 [0-10], [10-20]。
#        - (1B).等距分箱：根據固定間隔劃分數據。
#        - (1C).等頻分箱：根據數據的分佈劃分為大小相等的區間。

#%%==(2).時間型數據的編碼
#      -- 目標數據：將日期和時間型數據轉換為模型可處理的數值型數據。
#      -- 編碼方法：
#        - (2A).提取特徵：例如提取年份、月份、星期、季節等。
#        - (2B).周期性編碼：將時間數據轉換為正弦和餘弦值以表示周期性。

#%%==(3).文本數據的編碼
#      -- 目標數據：處理自然語言數據（如產品評論或文章）。
#      -- 編碼方法：
#        - (3A).詞袋模型（Bag of Words, BoW）：將文本轉換為詞語的出現頻率向量。
#        - (3B).詞向量（Word Embedding）：使用深度學習模型生成詞語的數值表示，如 Word2Vec, GloVe。
#        - (3C).TF-IDF：基於詞語的重要性對其進行加權。

#%%##===== (D4) 實作例之數據準備 [AIp08(D)] =====#####
#%%== (0A).定義相關分析參數: 各家企業的以下參數不盡相同 ==##
FFbreaks = [0, 1, 9, 99, 999, 19999]
MMbreaks = [-5000, 0, 999, 9999, 99999, 999999, 19999999]
BBbreaks = [0, 1, 7, 30, 99, 300, 1999]
RRbreaks = [0, 7, 30, 60, 99, 180, 360, 499, 700, 1999]  # --> 用於客戶漏斗 (下述)
Tnow = pd.to_datetime("2017/12/31", format="%Y/%m/%d")
print(Tnow)  # -- 數據分析點: 2023-07-01 00:00:00

#%%== (0B).應用函式庫 (Cv): buildCv(),NES3(),addCvNES3() ==##
def buildCv(XX, FFbreaks, MMbreaks, BBbreaks): ##== 建構客戶價值數據框: Cv = buildCv(X,FFbreaks,MMbreaks,BBbreaks)
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
def NES3(Ck, K, M):  ##== status = NES3(Ck, K, M): 定義NES3狀態
    if Ck["R0"] < 0:
        status = "U尚未消費"
    elif Ck["R0"] < 2*K:
        if Ck["MM"] > M:
            status = "N1新貴客"
        else:
            status = "N2新客"
    else:
        if Ck["Rf"] < 2*K:
            if Ck["R0"]/Ck["FF"] < 0.75*K:
                status = "A1較活躍客"
            else:
                status = "A2活躍客"
        elif Ck["Rf"] < 3*K:
            status = "S1瞌睡客"
        elif Ck["Rf"] < 4*K:
            status = "S2半睡客"
        elif Ck["FF"] < 10:
            status = "S3沈睡客"
        else:
            status = "S4沈睡忠誠客"
    return status
def addCvNES3(Cv, Tnow, KK, MM, RRbreaks):     ##== 將 NES3 加入 Cv: Cv = addCvNES3(Cv, Tnow, KK, MM, RRbreaks)
    Cv["R0"] = [(Tnow - pd.to_datetime(d)).days for d in Cv.D0]
    Cv["Rf"] = [(Tnow - pd.to_datetime(d)).days for d in Cv.Df]
    Cv["R00"] = pd.cut(Cv["R0"], bins=RRbreaks).astype(str)
    Cv["Rf0"] = pd.cut(Cv["Rf"], bins=RRbreaks).astype(str)
    Cv["status"] = Cv.apply(NES3, K=KK, M=MM, axis=1)
    return Cv

#%%== (0C).數據st.cache函式庫: getX(),buildCvRDS() ==##
def getX(Xname):  ##== X = getX(Xname): 自 X.csv 讀取 X (KDD1), 並設定標籤 (KDD3) ==##
    X = pd.read_csv(Xname)
    # -- 還有很多其他產生此標籤的方法, 這裡只是取其中較方便的一種
    X["date"] = pd.to_datetime(X["datetime"]).dt.date
    X["year"] = pd.to_datetime(X["datetime"]).dt.year
    X["yq"] = pd.PeriodIndex(X.date, freq='Q')
    X["ym"] = pd.PeriodIndex(X.date, freq='M')
    return (X)
def buildCvRDS(X, FFbreaks, MMbreaks, BBbreaks, RRbreaks, Tnow):   ##== Cv = buildCvRDS(X,..): 由交易數據 X 求取客戶數據框 Cv (KDD3)
    Cv = buildCv(X, FFbreaks, MMbreaks, BBbreaks)
    print(Cv.shape)
    print(Cv[2:4])  # -- (52217, 17)
    KK = np.nanmean(Cv["BB"])
    print(KK)  # -- 43.070694784611675
    MM = np.nanmean(Cv["MM"])
    print(MM)  # -- 46998.990443725226
    Cv = addCvNES3(Cv, Tnow, KK, MM, RRbreaks)
    print(Cv.shape)
    print(Cv[2:4])  # -- (52217, 22)
    # Cv = pd.read_csv("cvv.csv")
    return (Cv)
#%%== (1A).讀取交易數據(Xname-->X)
Xname = "XXX.csv"
X = getX(Xname);   print(X.shape);   print(X.head(2))
##== (1B).轉換為客戶數據框(X-->Cv)
Cv = buildCvRDS(X, FFbreaks, MMbreaks, BBbreaks, RRbreaks, Tnow);   print(Cv.shape);   print(Cv.head(2))
##== (1C).生成客戶價值模型(Cv.FF0/MM0-->TFM)
TFM = pd.crosstab(Cv["FF0"], Cv["MM0"], margins=True);   print(TFM)
##== (1D).選取特定客群: 方法A (Cv.FF0/MM0-->CvTA-->XX)
CvTA = Cv.loc[Cv["FF0"].isin(["(9, 99]", "(99, 999]"])]; print(CvTA.shape); print(CvTA.head(2))  # -- (376, 17)
XX = X.loc[X["customer"].isin(CvTA.index)];   print(XX.shape);  print(XX.head(2))  # -- (19228, 15)
#   invoiceNo channel customer product category  price             datetime   quantity  amount category2    cost        date  year      yq       ym
# 6        N6      s1       c6      p3    kind1   1600  2015-01-29 20:10:56          1    1216      sub1  846.72  2015-01-29  2015  2015Q1  2015-01
# 7        N7      s1       c7      p3    kind1   1600  2015-01-17 14:26:13          1    1360      sub1  846.72  2015-01-17  2015  2015Q1  2015-01

#%%##===== (D5) 決策樹的實例 =====#####

#%%== (1).生成所需標籤(XX["price0","quarter"]) ==##
XX["price0"] = pd.cut(XX["price"], bins=[0, 1000, 5000, 9000], labels=["低", "中", "高"], right=True)
XX["yq"] = XX["yq"].astype(str)
XX["quarter"] = XX["yq"].str.extract(r"(Q\d)")

#%%== (2).訓練決策樹(X,y-->tree_model-->tree_rules) ==##
XX1 = XX[["channel", "category", "category2", "price0"]]
yy1 = XX["quarter"]
print(XX1[17670:17674])
#       channel category category2 price0
# 57029      s5   kind17      sub4      中
# 57030      s5   kind17      sub4      中
# 57077      s1   kind11      sub2      中
# 57140      s2   kind18      sub2      中
print(yy1[17670:17674])
# 57029    Q2
# 57030    Q2
# 57077    Q3
# 57140    Q3

#%%== (3).LABEL-Encoding (XX1,yy1-->XX1L,yy1L) ==##
#     -- (A1).建立 LabelEncoder (XX1-->XX1L)
label_encoders = {}
XX1L = XX1
for column in XX1L.columns:
    le = LabelEncoder()
    XX1L[column] = le.fit_transform(XX1L[column])
    label_encoders[column] = le  # 保存每個欄位的編碼器
print(XX1L[17670:17674])
#        channel  category  category2  price0
# 57029        4         8          3       0
# 57030        4         8          3       0
# 57077        0         2          1       0
# 57140        1         9          1       0
#     -- (A2).將目標變數也進行編碼(yy1-->yy1L)
yy1L_encoder = LabelEncoder()
yy1L = yy1
yy1L = yy1L_encoder.fit_transform(yy1L)
print(yy1L[17670:17674])  # -- array([1, 1, 2, 2])

#%%   -- (B).決策樹模型與規則 (XX1L,yy1L-->tree_modelL-->tree_rulesL) ==##
tree_modelL = DecisionTreeClassifier(max_depth=3, random_state=0)
tree_modelL.fit(XX1L, yy1L)
tree_rulesL = export_text(tree_modelL, feature_names=XX1L.columns.tolist(), class_names=list(yy1L_encoder.classes_))
print(tree_rulesL)
# |--- category <= 40.50       <---x 這種category 很難解讀
# |   |--- price0 <= 1.50      <---v 這種price0 還算有意義
# |   |   |--- channel <= 3.50 <---x 這種channel 很難解讀f
# |   |   |   |--- class: Q2
# |   |   |--- channel >  3.50
# |   |   |   |--- class: Q2
# |   |--- price0 >  1.50
# |   |   |--- category2 <= 3.50
# |   |   |   |--- class: Q1
# |   |   |--- category2 >  3.50
# |   |   |   |--- class: Q1
# |--- category >  40.50
# |   |--- category <= 46.50
# |   |   |--- category <= 43.50
# |   |   |   |--- class: Q3
# |   |   |--- category >  43.50
# |   |   |   |--- class: Q4
# |   |--- category >  46.50
# |   |   |--- category <= 47.50
# |   |   |   |--- class: Q1
# |   |   |--- category >  47.50
# |   |   |   |--- class: Q4

#%%   -- (C).可視化決策樹 (tree_modelL) ==##
plt.figure(figsize=(25, 15))
plot_tree(tree_modelL, feature_names=XX1L.columns, class_names=yy1L_encoder.classes_,
          filled=True, fontsize=8)
plt.title("以 LABEL_encoding 求得的決策樹")
plt.show()

#%%== (4).ONE-HOT Encoding (XX1,yy1-->XX1H,yy1H) ==##
#     -- (A1).使用 One-Hot Encoding 將特徵轉換為數值類型 (XX1-->XX1H)
encoder = OneHotEncoder(sparse_output=False, drop='first')   #-- sparse_output=False` 取代 `sparse=False`
label_encoders = {}
XX1H = encoder.fit_transform(XX1);   print(XX1H[17670:17674])
# [[0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 1.]]
#     -- (A2).將目標變數也進行編碼(yy1-->yy1H)
yy1H_encoder = LabelEncoder()
yy1H = yy1H_encoder.fit_transform(yy1);   print(yy1H[17670:17674])  # -- [1 1 2 2]

#%%   -- (B).決策樹模型與規則 (XX1H,yy1H-->tree_modelH-->tree_rulesH) ==##
tree_modelH = DecisionTreeClassifier(max_depth=3, random_state=0)
tree_modelH.fit(XX1H, yy1H)
tree_rulesH = export_text(tree_modelH, feature_names=encoder.get_feature_names_out(), class_names=list(yy1H_encoder.classes_))
print(tree_rulesH)
# |--- category_44 <= 0.50
# |   |--- price0_3 <= 0.50
# |   |   |--- category_45 <= 0.50
# |   |   |   |--- class: Q1
# |   |   |--- category_45 >  0.50
# |   |   |   |--- class: Q4
# |   |--- price0_3 >  0.50
# |   |   |--- category2_4 <= 0.50
# |   |   |   |--- class: Q1
# |   |   |--- category2_4 >  0.50
# |   |   |   |--- class: Q1
# |--- category_44 >  0.50
# |   |--- category2_3 <= 0.50
# |   |   |--- price0_1 <= 0.50
# |   |   |   |--- class: Q4
# |   |   |--- price0_1 >  0.50
# |   |   |   |--- class: Q4
# |   |--- category2_3 >  0.50
# |   |   |--- price0_1 <= 0.50
# |   |   |   |--- class: Q4
# |   |   |--- price0_1 >  0.50
# |   |   |   |--- class: Q3

#%%   -- (C).可視化決策樹 (tree_modelH) ==##
plt.figure(figsize=(25, 15))
plot_tree(tree_modelH, feature_names=encoder.get_feature_names_out(), class_names=yy1H_encoder.classes_,
          filled=True, fontsize=8)
plt.title("以 ONE-HOT_encoding 求得的決策樹")
plt.show()

#%%== (5).獨熱編碼(ONC)決策樹包成函式庫(ONCtree, plotONCtree) ==##
def prepDTdata(XX):  ##== 準備決策樹數據 ==##
    XX["price0"] = pd.cut(XX["price"], bins=[0, 1000, 5000, 9000], labels=["低", "中", "高"], right=True)
    XX["yq"] = XX["yq"].astype(str)
    XX["quarter"] = XX["yq"].str.extract(r"(Q\d)")
    XX1 = XX[["channel", "category", "category2", "price0"]]
    yy1 = XX["quarter"]
    print(XX1[17670:17674])
    return XX1, yy1
def ONCtree(XX1, yy1, nDepth=3):  ##== 獨熱編碼(ONC)決策樹 ==##
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder
    from sklearn.tree import DecisionTreeClassifier, export_text
    #-- (A1).使用 One-Hot Encoding 將特徵轉換為數值類型 (XX1-->XX1H)
    encoder = OneHotEncoder(sparse_output=False, drop='first')  #-- sparse_output=False` 取代 `sparse=False`
    XX1H = encoder.fit_transform(XX1)
    print(XX1H[17670:17674])
    #-- (A2).將目標變數也進行編碼(yy1-->yy1H)
    yy1H_encoder = LabelEncoder()
    yy1H = yy1H_encoder.fit_transform(yy1)
    print(yy1H[17670:17674])  # -- [1 1 2 2]
    #-- (B).決策樹模型與規則 (XX1H,yy1H-->tree_modelH-->tree_rulesH) ==##
    tree_modelH = DecisionTreeClassifier(max_depth=nDepth, random_state=0)
    tree_modelH.fit(XX1H, yy1H)
    tree_rulesH = export_text(tree_modelH, feature_names=encoder.get_feature_names_out(
    ), class_names=list(yy1H_encoder.classes_))
    print(tree_rulesH)
    return tree_modelH, tree_rulesH, encoder, yy1H_encoder
def plotONCtree(tree_modelH, encoder, yy1H_encoder):  # == 可視化決策樹 (tree_modelH) ==##
    from sklearn.tree import plot_tree
    plt.figure(figsize=(25, 15))
    fig = plot_tree(tree_modelH, feature_names=encoder.get_feature_names_out(), class_names=yy1H_encoder.classes_,
                    filled=True, fontsize=8)
    plt.title("以 ONE-HOT_encoding 求得的決策樹")
    plt.show()
    return fig
XX1, yy1 = prepDTdata(XX)
DTmodel, DTrules, encoder, yy1H_encoder = ONCtree(XX1, yy1, nDepth=3);   print(DTrules)
plotONCtree(DTmodel, encoder, yy1H_encoder)

#%%####### (E) 人工神經網路(Artificial Neural Networks, ANN)分類 ##########

#%%##===== (E1) 神經網路原理 =====#####

##== (1).處理單元/神經元(neuron)原理
#    -- 彙集來自其他神經元(i)的資訊(xi)，與本神經元的相互作用強度/權重為wi --> sum(xi*wi) = a (activation激發)
#    -- 大於某閾值 w0 即為產生輸出 y --> y = f(a-w0) = f( sum(xi*wi) - w0 )
#       -- 一個神經元相當於一個感知機(perceptron)，
#                          是一個超平面(hyperplane)，將輸入資料 切成 正/負兩類(positive/negative halfspace)
#    .. 神經元的輸入/輸出 xi/y 可為 0/1 (二元binary), (-1)/1 (bipolar雙極), 任意實數 (linear,線性)

##== (2).神經網路(neural network)架構
#    -- 前向網路(feedforward network, multi-layer perceptron, MLP): 包括 輸入層/隱含層(hidden layer) / 輸出層
#    -- 回饋網路(feedback/recurrent network): 輸出層到輸入層中存在回饋(feedback)

#%%##===== (E2) 多層神經網路(MLP, Multi-Layer Perceptron) =====#####
iris = load_iris()
X = iris.data[:, 2:4];   y = iris.target
feature_names = iris.feature_names;   target_names = iris.target_names

##== (1).訓練 MLP神經網絡 ==##
mlp = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic',
                    max_iter=50000, alpha=0.05, random_state=101)
mlp.fit(X, y)

#%%==(2).列出MLP的神經鍵 ==##
input_to_hidden_weights = mlp.coefs_[0];   print(input_to_hidden_weights)
# [[ 0.34829341 -0.7402417 ]  -->  0.35*PL -3.73*PW + 3.82 = 0 (Hidden-1) 紫線
#  [-3.72562175 -3.6346505 ]] --> -0.74*PL -3.63*PW + 3.99 = 0 (Hidden-2) 橘線
hidden_layer_biases = mlp.intercepts_[0];   print(hidden_layer_biases)
# [3.82162376 3.99471172]     --> (Hidden-1, Hidden-2)
hidden_to_output_weights = mlp.coefs_[1];   print(hidden_to_output_weights)
# [[ 2.09341989  1.52021992 -3.30884575]
#  [ 2.40996669 -3.22882117 -2.49758875]]
output_layer_biases = mlp.intercepts_[1];   print(output_layer_biases)
# [-1.90588759  0.82806828  2.63010554]

#%%==(3).繪出MLP圖形 ==##

#== 繪製 MLP 神經網絡 ==##
def plot_neural_network(input_weights, hidden_weights, input_labels, output_labels):
    import graphviz
    dot = graphviz.Digraph(format='png', engine='dot')
    #-- 添加輸入層/隱藏層/輸出層節點
    for i, label in enumerate(input_labels):
        dot.node(f'I{i}', label=f"Input {label}", shape='circle',
                 style='filled', color='lightblue')
    for j in range(hidden_weights.shape[0]):
        dot.node(f'H{j}', label=f"Hidden {j+1}", shape='circle',
                 style='filled', color='lightgreen')
    for k, label in enumerate(output_labels):
        dot.node(f'O{k}', label=f"Output {label}", shape='circle',
                 style='filled', color='lightpink')
    #-- 添加從輸入到隱藏層的有向邊
    for i in range(input_weights.shape[0]):
        for j in range(input_weights.shape[1]):
            dot.edge(f'I{i}', f'H{j}', label=f"{input_weights[i, j]:.2f}")
    #-- 添加從隱藏層到輸出的有向邊
    for j in range(hidden_weights.shape[0]):
        for k in range(hidden_weights.shape[1]):
            dot.edge(f'H{j}', f'O{k}', label=f"{hidden_weights[j, k]:.2f}")
    return dot

input_labels = ["Petal Length", "Petal Width"]
output_labels = target_names
nn_graph = plot_neural_network(
    input_to_hidden_weights, hidden_to_output_weights, input_labels, output_labels)
nn_graph.render('mlp_neural_network', view=True)  #-- 保存並打開圖像

#%%==(4).混淆矩陣 ==##
y_pred = mlp.predict(X)
print(confusion_matrix(y, y_pred))
# [[50  0  0]
#  [ 0 47  3]
#  [ 0  4 46]]
#%%==(5).數據空間(PL,PW)中的隱藏層超平面(hyperplane) ==##
def plot_split_lines(X, input_weights, biases, title):   ##== 繪製 Petal.Length 和 Petal.Width 中的分割直線 ==##
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    colors = ['purple', 'orange']
    for i, (weight, bias) in enumerate(zip(input_weights.T, biases)):
        slope = -weight[0] / weight[1]
        intercept = -bias / weight[1]
        x_vals = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 100)
        y_vals = slope * x_vals + intercept
        plt.plot(x_vals, y_vals, label=f'Hidden Neuron {i+1}', color=colors[i])
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
    plt.title(title)
    plt.xlabel("Petal Length")
    plt.ylabel("Petal Width")
    plt.legend()
    plt.show()
    return
plot_split_lines(X, input_to_hidden_weights,
                 hidden_layer_biases, "Split Lines for Hidden Layer")


#%%##===== (E3) 支持向量機(Support Vector Machine, SVM) =====#####

##== (1).支持向量機(SVM): 以一組事先選擇的非線性映射(kernel核函數)，將輸入向量映射到高維特徵空間，進行最佳分類 ==##
#    -- 支持向量(Support Vector): 區隔各(正/負)分類的最大間隔超平面
def plot_decision_boundary(X, y, model, title):   ##== 繪製 SVM決策邊界(decision boundary)函數 ==##
    import matplotlib.pyplot as plt
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])  # -- 預測網格上的值
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))  # -- 繪製決策邊界
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.Paired)
    plt.title(title)
    plt.xlabel("Petal Length (cm)")
    plt.ylabel("Petal Width (cm)")
    plt.show()
    return

#%%==(2).線性SVM(Support Vector Machine)支援向量機: kernel=線性函數 ==##
svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
svm_linear.fit(X, y)  #== PHASE-I:訓練階段
plot_decision_boundary(X, y, svm_linear, "SVM with Linear Kernel")   #-- 繪製 decision boundary
y_pred_linear = svm_linear.predict(X)  # == PHASE-II:預測階段
conf_matrix_linear = confusion_matrix(y, y_pred_linear)  #-- 混淆矩陣
# [[50  0  0]
#  [ 0 47  3]
#  [ 0  2 48]]

#%%==(3).RBF SVM(Support Vector Machine)支持向量機: kernel-RBF函數 ==##
svm_rbf = SVC(kernel='rbf', C=10.0, gamma=0.5, random_state=42)
svm_rbf.fit(X, y)  #== PHASE-I:訓練階段
plot_decision_boundary(X, y, svm_rbf, "SVM with RBF Kernel")
#-- 繪製 decision boundary
y_pred_rbf = svm_rbf.predict(X)  #== PHASE-II:預測階段
conf_matrix_rbf = confusion_matrix(y, y_pred_rbf)  #-- 混淆矩陣
# [[50  0  0]
#  [ 0 47  3]
#  [ 0  2 48]]


#%%####### (F) 貝葉斯分類 [殷,7.4, 7.9.4] ##########

#%%##===== (F1) 貝葉斯分類(Bayes classifier) =====#####

##== 貝葉斯分類法的特點 ==##
#    -- 利用領域知識和其他先驗資訊，計算假設概率，分類結果是領域知識和資料樣本資訊的綜合體現
#    -- 利用有向圖，來表現各變數之間的依賴關係，以概率分佈表示依賴關係的強弱 (Bayesian Network)
#    -- 可進行增量學習，資料樣本可以增量地提高或降低某種假設的估計
##== 貝葉斯定理: P(Ci|X) = P(Ci)*P(X|Ci)/P(X) ==##
#    -- P(Ci): Ci分類的先驗概率(a priori probability)
#    -- P(X): 樣本X出現的概率，   P(X|Ci): 在Ci分類下會出現X的機率
#    -- P(Ci|X): 在條件X下(資料為X的條件下)，為Ci分類的後驗概率(a posteriori probability)

##== 素樸貝葉斯分類法(Naive Bayes): 指派為 Ci分類, 若對於所有j<>i, P(Ci)*P(X|Ci) > P(Cj)*P(X|Cj) ==##
#    -- 偽代碼: [殷,p.197]
## == 計算例: (3A)的bank例 ==##
#    -- P(debt=yes) = 3/10, P(debt=no) = 7/10
#       P(house=yes|debt=no) = 3/7, P(house=no|debt=no) = 4/7
#       P(house=yes|debt=yes) = 0,  P(house=no|debt=yes) = 1
#       P(marriage=single|debt=no) = 2/7, P(marriage=divorced|debt=no) = 1/7, P(marriage=married|debt=no) = 4/7
#       P(marriage=single|debt=yes) = 2/3, P(marriage=divorced|debt=yes) = 1/3, P(marriage=married|debt=yes) = 0
#       debt=no:  income均值=110, income方差=2975
#       debt=yes: income均值=90, income方差=25
#    -- X = (house=no, marriage=married, income=120K)
#    --> P(X|debt=no) = P(house=no|no)*P(marriage=married|no)*P(income=120K|no) = (4/7)*(4/7)*0.0072 = 0.0024
#        P(X|debt=yes) = P(house=no|yes)*P(marriage=married|yes)*P(income=120K|yes) = (1)*(0)*1.2*10^(-9) = 0
#    --> P(X|debt=no)*P(debt=no) = 0.0024*(7/10) = 0.00168
#        P(X|debt=yes)*P(debt=yes) = 0*(3/10) = 0
#    --> X 分類為 debt=no

#%%##===== (F2) 素樸貝葉斯分類器(Naive Bayes classifier) [殷,7.9.4] =====#####

##== (1).素樸貝葉斯分類器（Naive Bayes Classifier）==##
#    -- 基於貝葉斯定理（Bayes' Theorem）的簡單概率分類模型，常用於文本分類、醫學診斷等應用。
#    -- 素樸(Naive): 假設特徵之間相互條件獨立，在現實中往往不成立，但該模型仍然在許多場合表現良好。

##== (2).Naive Bayes 模型 的 建立-訓練-預測-混淆矩陣 ==##
naive_bayes = GaussianNB()
naive_bayes.fit(X, y)
y_pred = naive_bayes.predict(X)
print(y_pred)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 1 1 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
print(confusion_matrix(y, y_pred))
# [[50  0  0]
#  [ 0 47  3]
#  [ 0  3 47]]

#%%==(3).模型中的 後驗概率(posterior probability) ==##
posterior_probs = naive_bayes.predict_proba(X)
for i in range(49, 55):
    probabilities = posterior_probs[i]
    predicted_class = target_names[y_pred[i]]
    print(f"[{i:2d}] {probabilities[0]:.6e} {probabilities[1]:.6e} {
          probabilities[2]:.6e} --> {predicted_class}")
# [49] 1.000000e+00 3.581927e-18 1.117096e-25 --> setosa
# [50] 3.213809e-109 8.040377e-01 1.959623e-01 --> versicolor
# [51] 7.273478e-102 9.451696e-01 5.483036e-02 --> versicolor
# [52] 1.871429e-123 4.561513e-01 5.438487e-01*--> virginica*
# [53] 4.262698e-71 9.999688e-01 3.124886e-05 --> versicolor
# [54] 1.033231e-107 9.524418e-01 4.755819e-02 --> versicolor


#%%##===== (F3) 最大似然函式(Maximum Likelihood Estimate, MLE) =====#####

##== (1).似然函式(Likelihood Function): L(theta) = prod f(xi|theta), theta in THETA
#    -- 最大似然函式(Maximum Likelihood Estimate, MLE) theta^: theta的點估測
#    -- 最大值性質: 一次微分 L'(theta^) = 0, 且二次微分 L"(theta^) < 0
#    -- Log似然函式(Log-Likelihood): l(theta) = ln L(theta) can get the same solutions

##== (2).Iris數據, 定義 似然函數 與參數 ==
iris = load_iris()
X = pd.DataFrame(data=np.c_[iris.data, iris.target],
                 columns=['x1', 'x2', 'x3', 'x4', 'y'])

##== 最大似然函數 (Log-Likelihood) 的定義: y-f(w0,w1,w2,w3,w4) 的殘差(residual)
def log_likelihood(params, X):
    from scipy.stats import norm
    w0, w1, w2, w3, w4, sigma = params
    residuals = X['y'] - (w0 + w1 * X['x1'] + w2 *
                          X['x2'] + w3 * X['x3'] + w4 * X['x4'])
    log_likelihood = norm.logpdf(residuals, loc=0, scale=sigma)
    return -np.sum(log_likelihood)  # -- 返回負的 log-likelihood 作為損失

initial_params = [0, 0, 0, 0, 0, 1]  # == # 初始參數設置: w0, w1, w2, w3, w4, sigma

#%%==(3).最大似然(Maximum Likelihood, ML)-->求取 y-f(W) 的最佳w0,w1,w2,w3,w4 ==##
result = minimize(log_likelihood, initial_params, args=(X,), method='L-BFGS-B',
                  bounds=[(None, None)] * 5 + [(1e-6, None)])  # sigma > 0 的約束
w0, w1, w2, w3, w4, sigma = result.x
print  # == 獲取估計的最佳參數 W --> 就形成模型
print(f"w0 = {w0:.4f}, w1 = {w1:.4f}, w2 = {w2:.4f}, w3 = {
      w3:.4f}, w4 = {w4:.4f}, sigma = {sigma:.4f}")
#-- w0 = 0.1865, w1 = -0.1119, w2 = -0.0401, w3 = 0.2286, w4 = 0.6093, sigma = 0.2154

#%%==(4).以 f(W)模型來預測 ==##
def predict(X, w0, w1, w2, w3, w4):
    return w0 + w1 * X['x1'] + w2 * X['x2'] + w3 * X['x3'] + w4 * X['x4']
y_predict = predict(X, w0, w1, w2, w3, w4)
print(np.round(y_predict, 2).values)
# [-0.08, -0.04, -0.05,  0.01, -0.08,  0.06,  0.04, -0.04,  0.02, -0.08, -0.1 ,  0.  , -0.09, -0.1 , -0.23, -0.04, -0.03, -0.02, -0.03, -0.01,
#  -0.04,  0.05, -0.12,  0.18,  0.07, -0.01,  0.1 , -0.07, -0.09,  0.02,  0.01,  0.03, -0.16, -0.16, -0.02, -0.11, -0.15, -0.13, -0.01, -0.06,
#  -0.03,  0.07, -0.02,  0.22,  0.14,  0.03, -0.05, -0.01, -0.09, -0.06,  1.2 ,  1.28,  1.32,  1.19,  1.31,  1.26,  1.4 ,  0.91,  1.18,  1.24,
#   0.96,  1.28,  0.95,  1.32,  1.06,  1.17,  1.38,  0.98,  1.35,  1.02,  1.59,  1.1 ,  1.42,  1.2 ,  1.13,  1.19,  1.26,  1.5 ,  1.34,  0.85,
#   1.01,  0.93,  1.05,  1.55,  1.4 ,  1.38,  1.3 ,  1.19,  1.17,  1.18,  1.2 ,  1.29,  1.08,  0.9 ,  1.2 ,  1.12,  1.18,  1.15,  0.87,  1.17,
#   2.24,  1.75,  1.9 ,  1.74,  2.01,  2.  ,  1.6 ,  1.79,  1.76,  2.15,  1.72,  1.73,  1.84,  1.81,  2.05,  1.96,  1.69,  2.04,  2.2 ,  1.48,
#   1.99,  1.79,  1.96,  1.59,  1.89,  1.72,  1.57,  1.6 ,  1.92,  1.56,  1.8 ,  1.83,  1.98,  1.45,  1.53,  2.  ,  2.09,  1.7 ,  1.59,  1.8 ,
#   2.06,  1.86,  1.75,  2.05,  2.13,  1.91,  1.68,  1.75,  1.99,  1.67])

# %%==(5).混淆矩陣 ==##
rounded_predictions = np.round(y_predict).astype(int)
conf_matrix = confusion_matrix(rounded_predictions, X['y'])
print(conf_matrix)
# [[50  0  0]
#  [ 0 48  2]
#  [ 0  2 48]]

#%%##===== (F4) 各種分類器之間的關係 =====#####
# [Pieter Eykhoff. (1974). System Identification: Parameter and State Estimation. NY:Wiley]

##== (0).事件/量測概率(event probability) P(Xt): 在某時間t發生量測值(measurement)Xt的概率
#    -- 先驗概率(a priori probability) P(Ci): 在量測(measure)前某類別Ci會出現的概率
#    -- 後驗概率(a posteriori probability) P(Ci|Xt): 在量測值為Xt後,類別Ci出現的概率
#    -- 貝葉斯定理(Bayesian Rule): P(Ci|Xt) = P(Ci)*P(Xt|Ct) / P(Xt)

#%%=== 各種分類器(classifiers)的條件與計算式 [Ekyhoff, Table 5-1, p.152] ===###
#
#          分類器 ----- 事先條件(a priori knowledge) ----- 分類器計算式(objective function)
##== (1) Bayes分類器 ----- P(Ci),P(X|Ci),loss(ai|Cj) ----- ai = arg min_ai risk(ai|X) = sum(loss(ai|Cj)*P(Cj|X))
#     |  -- ai: 決策(decision)或動作(action)
#     |
#     |== 當 損失函式loss(ai|Cj) 均相等, 或 沒有損失函式時，Bayes分類器 化簡為 MAP分類器
#     V
##== (2) MAP(Maximum A Posteriori)分類器 --- P(Ci), P(X|Ci) ----- max(P(Ci|X)) = max(P(Ci)*P(Xt|Ct)/P(Xt))
#     |  -- 上述(5A-5B)的素樸貝葉斯分類器，其實是 MAP分類器
#     |
#     |== 當 沒有 先驗概率 P(Ci)時，MAP分類器 化簡為 ML分類器
#     V
##== (3) ML(Maximum Likelihood)分類器 --- P(X,Ci) ----- max(P(X|Ci)) = max(log(P(X,Ci)))
#     |  -- 上述(5C)最大似然函式，不僅可以用在常態分佈，也可用在二項分佈等
#     |
#     |== 當 P(X,Ci)為 常態分佈時，ML分類器 化簡為 Markov分類器
#     V
##== (4) Markov(Minimum Variance)分類器 --- P(X,Ci)=n(0,R) ----- min( t(y-W*X)*inv(R)*(y-W*X) )
#     |  -- 可以求得 W = inv(t(X)*inv(R)*X) * t(X)*inv(R)*y, 即上述(4)中若取無隱含層的神經網路公式
#     |
#     |== 當 常態分佈中的變異數/標準差均相等時，則可化為LS(最小平方)/回歸分類器，即(HUT07)中的回歸
#     V
##== (5) Least Square (最小平方)/回歸分類器 --- R = (sigma^2)*I ----- W = inv(t(X)*X) * t(X)*y


#%%####### (G) AI模型與大數據的全貎 [請參考投影片] ##########

#%%##===== (G1-A2) 數據模型(Data Model) =====#####

##== (1).系統/模型(System/Model, M):  輸出y = M( 輸入u )
#    -- (phase-1) 訓練階段(Training/Learning/Modeling/Estimation Phase): (u, y) -> M
#             由輸入/輸出 u與y，求取(估測estimate)模型M#
#    -- (phase-2) 預測階段(Prediction/Estimation/Production/Application Phase): (u_new, M#) -> y_predict
#             以所估測的模型M#與新的輸入 u_new，求取(估測)新的輸出 y_predict

##== (2).機器學習(Machine Learning): 從數據(x,y) 求取 知識(模型M#)

#    -- (K1)無監督式學習 (Unsupervised learning): 無輸出y, 目標在於發掘輸入(u)的隱含特徵 --> 數據挖掘(Data Mining)
#           (AIp07)聚類(clustering):           計算數據u的相似度，以產生其分類。
#           (AIp08)關聯規則(association rule): 計算多數據(ui-uj)間的關連。
#           (AIp09)數據序列(data sequencing):  計算多數據(ui-uj)間的時序關係。

#    -- (K2)監督式學習 (Supervised learning): 具範例(u,y), y為教師(teacher, desired output) y, 以求得y=M(u)
#           (AIp09)回歸(regression):     y 為連續數據
#           (AIp10)分類(classification): y 為離散數據
#                  ----> 神經網路(neural network): 自 2014年後，進入深度學習(deep learning)
#                  ----> 所以，現在的 AI，是機器學習／大數據分析的一環 ***

##== (3).(KDD4-K0) 商業智能 (BI, Business Intelligene)、OLAP(在線分析,OnLine Analytic Processing)、
#           -- MDA/MDS(多維度分析/系統, Multi-Dimensional Analysis/System)
#              : 以方格(cube), 階層(hierarchy)等維度視角來分析數據
#           -- 2維表格模型: 可以表示(客戶)靜態質量/進出佇留/動態狀態 等模型
#           -- 量值結構: 將(營業額等)量值依其因果關係排序，以表現不同維度切面(cube)的量值變化

##== (4).數據分析的標準步驟：知識發現(KDD, Knowledge Discovery in Databases)
#    .. (原KDD2)數據處理: 一般 包括 數據清理、數據整合(前兩者納入KDD1)、數據轉換(納入KDD3)、數據化約(儘量不用)
#    -- (KDD1)數據擷取(Data Acquisition): 包括數據清理與數據整合，包括檔案格式與編碼調整等
#    -- (KDD2改為)數據探索(Data Exploration): 進行維度量值的數據空間分析
#    -- (KDD3)數據轉換(Data Transformation): 是數據分析的核心，用以產生標籤(tag)
#    -- (KDD4)數據模型(Data Modeling): 數據分析以產生知識，如二維表格(K0)、數據挖掘(K1)，及機器學習(K2)
#    -- (KDD5)數據解讀(Data Interpretation): 結合數據視覺化(Visualization)，才能表現數據洞察(data insight)
#    (=).如果可能的話
#    -- (KDD6)數據決策(Strategy)與行動(Action)
#    -- (KDD7)回饋評估(Evaluation with Feedback)

#%%##===== (G2) 監督式學習模型比較 =====#####

#%%##===== (G3) 網路模型 =====#####

#####===== (G4) 時序模型->(AIp09) =====#####

#####===== (G5) 地圖模型 =====#####

#####===== (G6) 大數據的全面觀 =====#####
