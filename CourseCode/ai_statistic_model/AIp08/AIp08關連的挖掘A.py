# AIp08關連的挖掘A.py: AI python 實作 - 08: 數據挖掘的數據關連 
# Jia-Sheng Heh, 10/28/2024, revised from HUT06s.R

#%%####### (0) 匯入所需的套件 ##########
import numpy as np   
import pandas as pd
import os            ##== (O1) 設定工作目錄
wkDir = os.path.dirname(os.path.abspath(__file__));   os.chdir(wkDir);   print(os.getcwd())
import matplotlib.pyplot as plt
import seaborn as sns

#%%####### (A) 數據挖掘 ##########

#####===== (A1) 數據挖掘(data mining)的魅力 [百度] =====#####
##== 維克多.邁爾-舍恩伯格(Viktor Mayer-Schönberger): 別問為什麼，知道是什麼就夠了
#    -- 牛津大學網路學院互聯網治理與監管專業教授, 最早洞見大數據時代發展趨勢的數據科學家
#    -- [大數據時代, 2012] 大數據時代最大的轉變就是，放棄對因果關係的渴求，而取而代之關注相關關係
#    -- [刪除, 2013] 大數據取捨之道，就是把有意義的留下來，把無意義的去掉

#%%##===== (A2) 數據模型(Data Model) =====#####
##== 系統/模型(System/Model, M):  輸出y = M( 輸入u )
#    -- (1) 訓練階段(Training/Learning/Modeling/Estimation Phase): (u, y) -> M
#             由輸入/輸出 u與y，求取(估測estimate)模型M#
#    -- (2) 預測階段(Prediction/Estimation/Production/Application Phase): (u_new, M#) -> y_predict
#             以所估測的模型M#與新的輸入 u_new，求取(估測)新的輸出 y_predict
##== 機器學習(Machine Learning)
#    -- (1) 監督式學習 (Supervised learning): 具範例(u,y), y為教師(teacher, desired output) y, 以求得y=M(u)
#           (1A)回歸(regression):     y 為連續資料   --> 下一單元 AIp09
#           (1B)分類(classification): y 為離散資料   --> 下二單元 AIp10   
#    -- (2) 無監督式學習 (Unsupervised learning): 無輸出y, 目標在於發掘輸入(u)的隱含特徵 --> 資料採擷(Data Mining)
#           (2A)聚類(clustering):           計算資料u的相似度，以產生其分類。 --> 上一單元 AIp07
#           (2B)關聯規則(association rule): 計算多資料(ui-uj)間的關連。       --> 本單元  AIp08
#           (2C)數據序列(data sequencing):  計算多資料(ui-uj)間的時序關係。   --> 納入 AIp09

#####===== (A3) 關聯分析(association analysis) [殷,5.1] =====#####
##== 關聯分析: 發現隱藏在大型數據集中的令人感興趣的聯繫
#              -- 關聯規則(association rule): 所發現的模式 if A1 and A2... then B
#              -- 關聯分析演算法: 包括 Apriori演算法, FP-growth演算法
#              -- 其餘演算法: 處理分類屬性, 處理連續屬性, 處理概念分層, 序列模式, 子圖模式, 非頻繁模式等 (均不介紹)
from mlxtend.frequent_patterns import apriori, association_rules


#%%####### (B) 數據關聯規則 ##########

#%%##===== (B1) 事務/交易資料(transactions) [殷,5.2.1] =====#####

##== (1).python的交易數據集,沒像R語言那麼好找 (file-->df) ==##
# from mlxtend.data import load_dataset  
# df = load_dataset('groceries', delimiter='\n')        #===> (1a).mlxtend 沒有內建交易數據集
# df = pd.read_csv('grocery_dataset.csv', header=None, names=['items'], keep_default_na=False);  #===> (1b).來自 ShilpaGopal/association-rule-mining: 有很多問題, 不建議使用
df = pd.read_csv('groceries - groceries.csv', header=0) #===> (1c).來自 Devansh-Gupta-Official/Groceries-Dataset
df.fillna('', inplace=True);   print(df.shape)   #-- (9835, 33) --> 9835條購物記錄(transaction事務/交易)
print(df.iloc[0:3][df.columns[0:10]])
#    Item(s)          Item 1               Item 2     Item 3       Item 4 Item 5 Item 6 Item 7 Item 8 Item 9
# 0        4    citrus fruit  semi-finished bread  margarine  ready soups                                   
# 1        3  tropical fruit               yogurt     coffee                                                
# 2        1      whole milk    

#%%== (2).交易數據: 將數據列表(list)化 (df.items-->transactions) ==##
df['items'] = df.apply(lambda row: [row[col].strip() for col in df.columns[1:] if pd.notna(row[col]) and row[col] != ''], axis=1)
print(df.iloc[0:3]["items"])
# 0    [citrus fruit, semi-finished bread, margarine,...
# 1                     [tropical fruit, yogurt, coffee]
# 2                                         [whole milk]
transactions = df['items'].tolist()
print(transactions[0:3])  #-- [['citrus fruit', 'semi-finished bread', 'margarine', 'ready soups'], ['tropical fruit', 'yogurt', 'coffee'], ['whole milk']]
##== 購物籃(basket)事務/交易(transaction)
#    -- 事務/交易資料集: 所有的購物記錄, 即所有的事務/交易  D = {Ti} = {T1,T2,...,TN}
#    -- 事務/交易(transaction)Ti: 每一條購物記錄所購買的物品
#    -- 每個事務/交易是項(item)的集合 Ti included in I = { 1,2,...,im }, 每個事務/交易有一個識別字(TID)
#--> 9835條購物記錄(transaction事務/交易)

#%%== (3).交易數據編(獨熱)碼: One-Hot Encoding (transaction-->oht_df) ==##
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
oht_df = pd.DataFrame(te_ary, columns=te.columns_)
print(oht_df.shape);   #-- (9835, 169) --> 9835條購物記錄(事務/交易transaction), 共有169個物品(項,item)
print(oht_df.iloc[0:3][oht_df.columns[30:40]])
#    chocolate marshmallow  citrus fruit  cleaner  cling film/bags  cocoa drinks  coffee  condensed milk  cooking chocolate  cookware  cream
# 0                  False      --->True    False            False         False   False           False              False     False  False
# 1                  False         False    False            False         False -->True           False              False     False  False
# 2                  False         False    False            False         False   False           False              False     False  False

#%%##===== (B2) 項集(item sets, IS)及頻繁項集(frequent item set, FIS) [殷,5.2] =====#####

##== (1).項集(item set,IS): 包含0個或多個項的集合 ==##
#    -- 1-項集: 包括1項的項集, 如: {whole milk}, {rolls/buns}
#    -- 2-項集: 包括2項的項集, 如: {tropical fruit, yogurt,coffee}, {whole milk, cereals}
#    ...

##== (2).格結構(lattice structure): 列舉所有可能的項集(item sets) == ##
#    -- k個項的資料集，最多產生 2^k - 1 個項集，不含空集合{}
#    -- 如: I = {A,B,C} --> 項集可為 {},{A},{B},{C},{A,B},{A,C},{B,C},{A,B,C} 可形成一個偏序的(partial ordered)的格結構

#%%== (3).頻繁項集(frequent item set, FIS) [殷,5.2.2] ==##
from mlxtend.frequent_patterns import apriori, association_rules
#    -- 支持度(support): 一個項集的 出現次數 與 資料集中所有事務/交易數的百分比
#    -- 頻繁項集(frequent item set, FIS): 一個項集的支持度大於或等於某個閾值(minimal support)
#    -- ECLAT (Equivalence Class Clustering and bottom-up Lattice Traversal) 演算法
#       --- 挖掘關聯規則的頻繁項集
#       --- 深度優先(depth-first)演算法，比 Apriori演算法(寬度優先breadth-first)有效率
FIS = apriori(oht_df, min_support=0.05, use_colnames=True, max_len=5)  #--> 只能用 min_support, 用來求取FIS, 此處不能用 min_confidence
FIS['count'] = FIS['support'] * len(oht_df)
FIS.sort_values(by='count', ascending=False, inplace=True)
# FIS['count'] = FIS['support'] * len(oht_df)
print(f"Number of frequent itemsets: {len(FIS)}")   #-- Number of frequent itemsets: 31
print(pd.concat([FIS.iloc[0:5, :], FIS.iloc[25:30, :]]))
#      support                  itemsets   count
# 26  0.255516              (whole milk)  2513.0 <--
# 15  0.193493        (other vegetables)  1903.0
# 19  0.183935              (rolls/buns)  1809.0 <-- maxFIS
# 23  0.174377                    (soda)  1715.0
# 27  0.139502                  (yogurt)  1372.0 <--
# 29  0.056634  (rolls/buns, whole milk)   557.0 <-- maxFIS
# 30  0.056024      (yogurt, whole milk)   551.0 <-- maxFIS
# 4   0.055414                  (butter)   545.0
# 8   0.053279                    (curd)   524.0
# 0   0.052466                    (beef)   516.0

##== (4) 最大频繁项集(Maximal Frequent Item Sets, maxFIS): 具有最大项目集的频繁项集(FIS) [殷p.110-111]
#    -- maxFIS的直接超集 都不是频繁的 (C:rolls/buns(FIS)---B&C(非FIS,support=0.034)))
#    -- 闭项集(closed item set): 它的直接超集都不具有和它相同的支持度计数 (???)
#       ==> 闭频繁项集: 一个频繁项集满足闭项集

#%%##===== (B3) 關聯規則(association rule) [殷,5.2.3-5.2.5] =====#####

##== (1).關聯規則(association rule), X->Y: 蘊含運算式，表示資料內隱含的關聯性
#    -- X included in I, Y included in I, X intersect Y = {}
#    -- X為前件/先決條件(condition, lhs, left-hand side)，Y為相應的後件/關聯結果(conclusion, rhs, right-hand side)
##== (2).Groceries的關聯規則說明例
# rules = association_rules(FIS, metric="confidence", min_threshold=0.2)
rules = association_rules(FIS, metric="confidence", min_threshold=0.2)   #--> 只能用 min_confidence, 此處不能用 min_support ==> 因此必須 和 apriori() 並用
print(f"Number of rules: {len(rules)}");   print(rules)  #-- Number of rules: 6
#           antecedents         consequents  antecedent support  consequent support   support  confidence      lift  leverage  conviction  zhangs_metric
# 0  (other vegetables)        (whole milk)            0.193493            0.255516  0.074835    0.386758  1.513634  0.025394    1.214013       0.420750
# 1        (whole milk)  (other vegetables)            0.255516            0.193493  0.074835    0.292877  1.513634  0.025394    1.140548       0.455803
# 2        (rolls/buns)        (whole milk)            0.183935            0.255516  0.056634    0.307905  1.205032  0.009636    1.075696       0.208496
# 3        (whole milk)        (rolls/buns)            0.255516            0.183935  0.056634    0.221647  1.205032  0.009636    1.048452       0.228543         P(M&Y)/P(Y)=P(M|Y)
# 4            (yogurt)        (whole milk)            0.139502            0.255516  0.056024    0.401603  1.571735  0.020379    1.244132       0.422732 0.401 = 0.056/0.139
# 5        (whole milk)            (yogurt)            0.255516            0.139502  0.056024    0.219260  1.571735  0.020379    1.102157       0.488608
#    -- Antecedent(X) 和 Consequent(Y)： 關聯規則 為 IF X (antecedents,條件）, THEN Y (consequents, 結果）--項目間的共現規律
#    -- Antecedent Support P(A) / Consequent Support P(B): 前件項目 / 後件項目集在所有交易中出現的比例 -- 衡量前件或後件在交易中的受歡迎程度。

# Support值為count值除以總交易數9835，所以Suppot值為其item佔全數的比例(%)。

##== (3).關聯規則的主要參數(association rule) [殷,5.2.3] 
#    -- (3A).支持度(support):    supp(X->Y) = P(X且Y)            -- 該規則適用於更大比例的交易數據, 最小支持度(minsupp)：support(Z)>=minsupp。    
#    -- (3B).置信度(confidence): conf(X->Y) = P(Y|X)=P(X,Y)/P(X) -- 衡量規則的可靠性, 最小置信度(minconf)。
#    -- (3C).提升度(lift):       lift(X->Y) = P(Y|X)/P(Y) = conf(X->Y)/P(Y) -- 前件和後件之間的關聯程度 lift越大(>1): 表示X對Y的提升作用越大
#    -- (3D).槓杆值(leverage)：  leverage(X->Y) = P(X且Y) - P(X)*P(Y)        -- 衡量前件和後件間額外出現(相對於獨立出現)頻率，
#    -- (3E).信念度(conviction)：conviction(X->Y) = (1-P(Y)) / (1-P(X->Y))   -- 衡量後件不發生時前件發生的比率，越大表示關聯規則越強

##== (4).舉例說明參數: 規則5--IF X (antecedent=whold-milk) THEN Y (conquent=yogurt), 故 antecent support P(X)=0.139502, consequent support P(Y)=0.255516
#    -- support=0.05602440, confidence=0.2192598, lift=1.571735, count=551, leverage=0.020379, conviction=1.244132, zhangs_matric=0.422732

#    -- 支持數count=551  表示 全部 9835個事務/交易中，有 551位元客戶符合本規則
#    -- 支持度support(X->Y) = P(X且Y) = 0.0560244 = 551 / 9835 為全部事務/交易中，本規則的支援比例

#    -- 置信度confidence(X->Y) = P(Y|X)=P(X,Y)/P(X) = 0.2192598 = 551/2513 符合本規則前提的2513位元客戶中，有551位元使得本規則成立
#    -- 信念度(conviction) = (1-P(Y)) / (1-P(X->Y)) = 1.244132 = (1-0.255516)/(1-0.2192598)​
#       ==> 這表示如果 X(whole milk) 不被購買，那麼 yogurt 發生的可能性比隨機情況下高 1.244倍

#    -- 提升度lift(X->Y) = P(Y|X)/P(Y) = conf(X->Y)/P(Y) = 1.571735 = 0.2192598/0.1395018: 
#       -- P(B) = 0.1395018 = 1372/9835 = P({yogurt}) 原本符合 B 的個數有1372位元，占全部比0.1395018
#       -- 加上{whole milk}前提後，占前提比為0.2192598，提升了 1.571735，也就是在前提的資料集合中，結論較明顯(提高為1.57倍)
#    -- 槓杆值(leverage) = P(X且Y) - P(X)*P(Y) = 0.020379 = 0.0560244 - 0.139502*0.255516 
#       ==> 表示購買 yogurt 和 whole milk 同時發生的頻率比它們獨立出現的頻率多了 2%。

#%%== (5).關聯規則挖掘演算法的基本步驟 [殷,5.2.3]
#    -- (5A).產生頻繁項集: 發現滿足最小支持度閾值的所有項集(即 頻繁項集)... FIS = apriori(oht_df, min_support=... )
#    -- (5B).產生規則: 從頻繁項集中提取大於置信度閾值的規則，即強規則   ... rules = association_rules(FIS, metric="confidence", ...) 

##== (6).先驗(Apriori)原理: 如果一個項集是頻繁的，則它的所有子集一定也是頻繁的 [殷,5.3.2]
#    -- 例: (B3)(2)的格結構
#    -- 支持度度量的反單調性: 一個項集的支持度，絕對不會超過它的子集的支持度
#    -- 支持度的剪枝(prune): 一旦發現一個項集是非頻繁的，則其所有子集的子集可以被立即剪枝


#%%####### (C) 數據關聯的評估 ##########

#%%##===== (C1) 關聯模式的評估 [殷,5.5] =====#####

##== (1).興趣度(interestingness)度量: 表示關聯規則模式的有趣性 [殷,5.5.1]
#    -- 客觀興趣度(objective interestingness)度量: 以統計論據建立，可用來排除資料中的獨立項或少數模式的偽聯繫
#    -- 主觀興趣度(subjective interestingness)度量: 以主觀論據建立，可用來無法提供新穎資訊或有益行動的主觀無趣模式
##== (2).(客觀興趣度)支持度(support)度量: 反映關聯規則的普遍性 [殷,5.5.2]
#    -- 越高則代表該規則能適用越多的數據
#    -- 缺點: 若支持度閾值過高，許多潛在有意義的模式會被刪除；支援度閾值過低，則計算代價過高，且會產生大量的關聯模式
##== (3).(客觀興趣度)置信度(confidence): 反映關聯規則的可靠性 [殷,5.5.2]
#    -- 越高則代表前件滿足時,後件成立的可能性越大
#    -- 辛普森悖論: 整體資料分析的關聯規則，可能不適用於分組資料 [殷pp.135-136]
#    -- 缺點: 若忽略規則前件和計的統計獨立性，即便置信度高，可能會是誤導的關聯規則
##== (4).(客觀興趣度)其他基於統計的度量 [殷,5.5.3]
#    -- 提升度lift(X->Y) = P(Y|X)/P(Y) = conf(X->Y)/P(Y): 評估項集X出現是否能促進項集Y出現
#       -- lift(X->Y) = 1: 表示 P(X,Y) = P(X)P(Y)，則X和Y是相互獨立的，否則存在某種依賴關係
#       -- (經驗) 所以通常算出關聯規則後，要依提升度排序，小於1 (常用小於 1.1)　的則捨棄
#    -- 相關係數(correlation coefficient) r = (P(X,Y)-P(X)P(Y)) / sqrt(P(X)(1-P(X))P(Y)(1-P(Y)))
#       -- r = (+1)/(-1)/0: 表示 完全正相關/完全負相關/統計獨立
#    -- 餘弦(cosine) IS = cos(X,Y) = P(X,Y)　/　sqrt(P(X)P(Y)): 越大代表越相似
##== (5).(主觀興趣度) 將主觀資訊加入到模式發現任務 [https://www.twblogs.net/a/5cab8268bd9eee5b1a07c26a]
#    -- 視覺化方法: 以資料視覺化方法呈現出資料中蘊涵的資訊
#    -- 基於範本的方法：限制發現的模式類型，只有滿足指定範本的模式才被認為是有趣的
#    -- 主觀興趣度量: 以基於領域資訊定義某些主觀度量，來過濾顯而易見和沒有實際價值的模式

#%%##===== (C2) 一般關聯規則的圖形化 [殷,5.6.2] =====#####

##== visualizations arulesViz in R (但在python未必可行,且不易使用)
##== (1).Grouped Matrix Plot 
plt.figure(figsize=(10, 8))
support_values = rules['support'];   lift_values = rules['lift'];   confidence_values = rules['confidence']
sns.scatterplot(x=support_values, y=lift_values, size=confidence_values, sizes=(20, 200), alpha=0.6)
plt.xlabel('Support');   plt.ylabel('Lift');   plt.title('Grouped Matrix Plot');   plt.show()

#%%== (2).Heatmap Matrix Plot
plt.figure(figsize=(12, 10))
matrix = pd.crosstab(index=[tuple(rule) for rule in rules['antecedents']], columns=[tuple(rule) for rule in rules['consequents']], values=rules['lift'], aggfunc='mean').fillna(0)
sns.heatmap(matrix, annot=True, fmt='.2f', cmap='Blues')
plt.xlabel('Consequents');   plt.ylabel('Antecedents');   plt.title('Matrix Plot of Lift Values');  plt.show()

#%%== (3).Graph Plot (Network)
##== igraph套裝軟體: 一個非常強大的套裝軟體，可以進行圖形運算與社群網路分析
import igraph as ig
graph_edges = [(list(rule['antecedents'])[0], list(rule['consequents'])[0]) for _, rule in rules.iterrows()]
g = ig.Graph.TupleList(graph_edges, directed=True)
layout = g.layout('kk')  # Kamada-Kawai layout for better spacing
ig.plot(g, layout=layout, vertex_label=g.vs['name'], vertex_size=20, edge_arrow_size=0.5, bbox=(800, 800), margin=50)

#%%##===== (C3) 一般關聯規則的圖形化 [殷,5.6.2] =====#####
##== igraph套裝軟體: 一個非常強大的套裝軟體，可以進行圖形運算與社群網路分析
##== (1).以關聯規則形成圖形
#    -- 圖形(graph): 一種知識標記法的呈現，具有節點(node,vertex)和邊(edge)
graph_edges = [(list(rule['antecedents'])[0], list(rule['consequents'])[0]) for _, rule in rules.iterrows()];   
g = ig.Graph.TupleList(graph_edges, directed=True);   print(g)
# IGRAPH DN-- 4 6 --
# + attr: name (v)
# + edges (vertex names):
# other vegetables->whole milk, whole milk->other vegetables, rolls/buns->whole
# milk, whole milk->rolls/buns, yogurt->whole milk, whole milk->yogurt

#%%== (2).圖形的節點與有向邊
#    -- 節點(node,vertex): 代表 lhs/rhs頻繁項集
#    -- 有向邊(directed edge): 代表 連帶消費之關連規則，其上的數位代表置信度 (confidence)
nodes = g.vs['name'];   print("Nodes:", nodes)
# Nodes: ['other vegetables', 'whole milk', 'rolls/buns', 'yogurt']
edges = g.get_edgelist();   print("Edges:", edges)
# Edges: [(0, 1), (1, 0), (2, 1), (1, 2), (3, 1), (1, 3)]
print(graph_edges)
# [('other vegetables', 'whole milk'), ('whole milk', 'other vegetables'), ('rolls/buns', 'whole milk'),
#  ('whole milk', 'rolls/buns'), ('yogurt', 'whole milk'), ('whole milk', 'yogurt')]
### node_size_dict = {vertex: sum(rules[rules['antecedents'] == {vertex}]['support']) * len(df) for vertex in g.vs['name']}
node_size_dict = {vertex: rules[rules['antecedents'] == {vertex}]['antecedent support'].iloc[0] * len(df) if not rules[rules['antecedents'] == {vertex}].empty else 1 for vertex in g.vs['name']}
print(node_size_dict)  #-- {'other vegetables': 736.0, 'whole milk': 1844.0, 'rolls/buns': 557.0, 'yogurt': 551.0}

#%%== (3).設定節點與邊的屬性, 進行繪圖
visual_style = { "layout": g.layout("fr"), "bbox": (800, 800), "margin": 50,
    "vertex_size": [0.1 * node_size_dict.get(vertex['name'], 1) for vertex in g.vs],
    "vertex_label": [f"{vertex['name']}\n({int(node_size_dict[vertex['name']] )})" for vertex in g.vs],
    "edge_label": [f"{rule['confidence']:.2f}" for _, rule in rules.iterrows()],
    "edge_arrow_size": 2,
}
visual_style['vertex_color'] = 'orange'
visual_style['edge_color'] = 'cyan';   visual_style['edge_label_color'] = 'blue'
ig.plot(g, **visual_style)


#%%####### (D) 數據關連的實務 [AIp07聚類及評估A.py(C)] ##########

#%%##===== (D1).參數/函式庫: 分析參數 + 應用函式庫 + streamlit快取機制 =====#####

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

#%%##===== (D2).從交易數據到客戶價值模型 (X-->Cv-->TFM--(+FF0A,MM0A)-->XX) =====#####

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

##== (4A).選取特定客群: 方法A (Cv.FF0/MM0-->CvTA-->XX)
CvTA = Cv.loc[Cv["FF0"].isin(["(9, 99]","(99, 999]"])];   print(CvTA.shape);   print(CvTA.head(2))   #-- (376, 17)
XX = X.loc[X["customer"].isin(CvTA.index)];               print(XX.shape);     print(XX.head(2))     #-- (19228, 17)   
#   invoiceNo channel customer product category  price             datetime  quantity  amount category2    cost        date  year      yq       ym      CvFF0            CvMM0
# 6        N6      s1       c6      p3    kind1   1600  2015-01-29 20:10:56         1    1216      sub1  846.72  2015-01-29  2015  2015Q1  2015-01  (99, 999]  (99999, 999999]
# 7        N7      s1       c7      p3    kind1   1600  2015-01-17 14:26:13         1    1360      sub1  846.72  2015-01-17  2015  2015Q1  2015-01    (9, 99]  (99999, 999999]
##== (4B).選取特定客群: 方法B (Cv.FF0/MM0--(標籤投射)-->X.CvFF0/CvMM0-->XX)
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

#%%##===== (D3).從所選客群交易 求取關聯規則(XX-->CPlist/transactions--(+min_supp,min_conf)-->rules) =====#####
#===> 本章開發，從這裡開始...

#%%== (1).從選取交易(XX)中生成數據關連中的交易型式(CPlist-->transaction)
# XX = pd.read_csv("c:/Users/jsheh/Desktop/XX.csv")[['customer', 'category']]
CPlist = XX.groupby('customer')['category'].apply(lambda x: list(set(x.unique()))).tolist()
print(len(CPlist));   print(CPlist[0:3])   #-- 376
# [['kind1', 'kind12', 'kind2', 'kind3', 'kind11', 'kind6'], ['kind27', 'kind1', 'kind2', 'kind17', 'kind16'], ['kind1']]
transactions = CPlist

#%%== (2).(依前(B1)(3)所述) 將交易數據轉成獨熱碼(transactions-->oht_df)
# Frequent itemset mining using apriori directly from transactions
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
oht_df = pd.DataFrame(te_ary, columns=te.columns_);   
print(oht_df.shape);   print(oht_df.iloc[0:3][oht_df.columns[0:20]])  #-- (376, 53)
#    kind1  kind10  kind11  kind12  kind13  kind14  kind15  kind16  kind17  kind18  kind19  kind2  kind20  kind21  kind23  kind24  kind25  kind26  kind27  kind28
# 0   True   False    True    True   False   False   False   False   False   False   False   True   False   False   False   False   False   False   False   False
# 1   True   False   False   False   False   False   False    True    True   False   False   True   False   False   False   False   False   False    True   False
# 2   True   False   False   False   False   False   False   False   False   False   False  False   False   False   False   False   False   False   False   False

#%%== (3).(依前(B2)所述) 生成頻繁項目集(oht_df--(min_support=0.4)-->FIS)
FIS = apriori(oht_df, min_support=0.4, use_colnames=True, max_len=5)
FIS['count'] = FIS['support'] * len(oht_df)
FIS.sort_values(by='count', ascending=False, inplace=True)
FIS['count'] = FIS['support'] * len(oht_df)
print(f"Number of frequent itemsets: {len(FIS)}")   #-- Number of frequent itemsets: 14
print(FIS[['itemsets', 'support', 'count']].head())
#           itemsets   support  count
# 3          (kind2)  0.898936  338.0
# 4          (kind3)  0.686170  258.0
# 11  (kind2, kind3)  0.664894  250.0 <-- 會有 2項集的條件喔!!
# 2         (kind12)  0.662234  249.0
# 0          (kind1)  0.630319  237.0

#%%== (4).(依前(B3)所述) 生成關聯規則(FIS--(min_confidence=0.6)-->rules) 
rules = association_rules(FIS, metric="confidence", min_threshold=0.6)
rules = rules[(rules['antecedents'].apply(len) == 1) & (rules['consequents'].apply(len) == 1)]  # <== 去除掉 2-項集等
print(f"Number of filtered rules (with single antecedent and consequent): {len(rules)}")   #-- Number of filtered rules (with single antecedent and consequent): 13
print(rules)
#    antecedents consequents  antecedent support  consequent support   support  confidence      lift  leverage  conviction  zhangs_metric
# 0      (kind2)     (kind3)            0.898936            0.686170  0.664894    0.739645  1.077932  0.048070    1.205392       0.715368
# 1      (kind3)     (kind2)            0.686170            0.898936  0.664894    0.968992  1.077932  0.048070    3.259309       0.230373
# 2      (kind2)    (kind12)            0.898936            0.662234  0.617021    0.686391  1.036477  0.021715    1.077027       0.348230
# 3     (kind12)     (kind2)            0.662234            0.898936  0.617021    0.931727  1.036477  0.021715    1.480288       0.104195
# 4      (kind2)     (kind1)            0.898936            0.630319  0.574468    0.639053  1.013857  0.007851    1.024198       0.135234
# 5      (kind1)     (kind2)            0.630319            0.898936  0.574468    0.911392  1.013857  0.007851    1.140578       0.036970
# 6      (kind3)    (kind12)            0.686170            0.662234  0.484043    0.705426  1.065222  0.029637    1.146627       0.195102
# 7     (kind12)     (kind3)            0.662234            0.686170  0.484043    0.730924  1.065222  0.029637    1.166323       0.181275
# 13    (kind11)     (kind2)            0.478723            0.898936  0.454787    0.950000  1.056805  0.024445    2.021277       0.103115
# 14     (kind1)     (kind3)            0.630319            0.686170  0.425532    0.675105  0.983875 -0.006974    0.965944      -0.042452
# 15     (kind3)     (kind1)            0.686170            0.630319  0.425532    0.620155  0.983875 -0.006974    0.973241      -0.049633
# 21     (kind1)    (kind12)            0.630319            0.662234  0.412234    0.654008  0.987579 -0.005185    0.976226      -0.032902
# 22    (kind12)     (kind1)            0.662234            0.630319  0.412234    0.622490  0.987579 -0.005185    0.979261      -0.035900

#%%== (5).包成函式: 從所選客群交易 求取關聯規則(XX-->CPlist/transactions--(+min_supp,min_conf)-->rules)
def assocRule(XX,min_supp,min_conf):   ##== 求取關聯規則: assocRule(XX,min_supp,min_conf)
    CPlist = XX.groupby('customer')['category'].apply(lambda x: list(set(x.unique()))).tolist()  ##== (1)
    transactions = CPlist
    total_transactions = len(transactions)
    from mlxtend.preprocessing import TransactionEncoder                         ##== (2)
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    oht_df = pd.DataFrame(te_ary, columns=te.columns_);   
    from mlxtend.frequent_patterns import apriori, association_rules
    FIS = apriori(oht_df, min_support=min_supp, use_colnames=True, max_len=5)    ##== (3): min_supp = 0.4
    FIS['count'] = FIS['support'] * len(oht_df)
    FIS.sort_values(by='count', ascending=False, inplace=True)
    FIS['count'] = FIS['support'] * len(oht_df)
    rules = association_rules(FIS, metric="confidence", min_threshold=min_conf)  ##== (4): min_conf = 0.6
    rules['count'] = (rules['antecedent support'] * total_transactions).astype(int)
    rules = rules[(rules['antecedents'].apply(len) == 1) & (rules['consequents'].apply(len) == 1)]  # <== 去除掉 2-項集等
    return rules
rules = assocRule(XX, min_supp=0.4, min_conf=0.6);   print(rules[0:2])   #--> 答案如上方 (rules) --> 多了count欄位 
#   antecedents consequents  antecedent support  consequent support   support  confidence      lift  leverage  conviction  zhangs_metric  count
# 0     (kind2)     (kind3)            0.898936            0.686170  0.664894    0.739645  1.077932   0.04807    1.205392       0.715368    338
# 1     (kind3)     (kind2)            0.686170            0.898936  0.664894    0.968992  1.077932   0.04807    3.259309       0.230373    258

#%%##===== (D4).關聯規則繪圖(rules--(igraph)-->g) =====#####

##== (1).產生 圖形的有向邊序列(graph_edges),再生成圖形(g) (rules-->graph_edges-->g)
graph_edges = [(list(rule['antecedents'])[0], list(rule['consequents'])[0]) for _, rule in rules.iterrows()]
# Display edges as rules in the format: IF antecedent THEN consequent
for edge in graph_edges:   print(f"IF {edge[0]} THEN {edge[1]}")
# IF kind2 THEN kind3
# IF kind3 THEN kind2
# IF kind2 THEN kind12
g = ig.Graph.TupleList(graph_edges, directed=True);   print(g)
# IGRAPH DN-- 5 13 --
# + attr: name (v)
# + edges (vertex names): 
#  kind2 -> kind3, kind12, kind1;   kind1 -> kind2, kind3, kind12;  kind3 -> kind2, kind12, kind1,
#  kind11 -> kind2;                 kind12 -> kind2, kind3, kind1

#%%== (2).圖形的量度: 節點大小 (node_size_dict), 有向邊(edgelist)
node_size_dict = {vertex: rules[rules['antecedents']=={vertex}]['antecedent support'].iloc[0]*XX.customer.nunique() 
                  if not rules[rules['antecedents']=={vertex}].empty else 1 for vertex in g.vs['name']}
for key, value in node_size_dict.items(): print(f"{key}: {value:.1f}")  #-- kind2:338, kind3:258, kind12:249, kind1:237, kind11:180
##-- 有向邊的探索:
edgelist = g.get_edgelist();    print(edgelist)
# [(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0), (1, 2), (2, 1), (4, 0), (3, 1), (1, 3), (3, 2), (2, 3)]
##-- 比較(1)中的節點標記:
#  kind2(0) -> kind3, kind12, kind1;   kind1(3) -> kind2, kind3, kind12;  kind3(1) -> kind2, kind12, kind1,
#  kind11(4) -> kind2;                 kind12(2) -> kind2, kind3, kind1
print(g.vs["name"])   #-- ['kind2', 'kind3', 'kind12', 'kind1', 'kind11']
#                               0        1        2         3        4

#%%== (3).設定繪圖參數並繪圖 (g)
node_labels = [f"{vertex['name']}\n ({int(node_size_dict[vertex['name']] )})" for vertex in g.vs]
node_sizes  = [ 0.5*node_size_dict.get(vertex['name'],1) for vertex in g.vs];  
print(node_sizes)  #-- [169.0, 129.0, 124.5, 118.5, 90.0] ---> 實驗調整節點大小
visual_style = { "layout": g.layout("fr"), "bbox": (900, 700), "margin": 50,
    "vertex_size": node_sizes, "vertex_label": node_labels, 'vertex_color': 'orange',
    "edge_label": [f"{rule['confidence']:.2f}" for _, rule in rules.iterrows()], 'edge_color': 'lightblue',
    "edge_arrow_size": 2,
}
ig.plot(g, **visual_style)

#%%== (4).包成函式: 從關聯規則(rules)創建圖形(g)與繪圖風格(visual_style)再繪圖
def graph_style(rules):   ##== 從關聯規則(rules,及選取數據): graph, style = graph_style(rules,XX)
    import igraph as ig
    graph_edges = [(list(rule['antecedents'])[0], list(rule['consequents'])[0]) for _, rule in rules.iterrows()]
    g = ig.Graph.TupleList(graph_edges, directed=True);   print(g)    ##== (1)
    lenXX = rules["count"][0]/rules["antecedent support"][0]
    node_size_dict = {vertex: rules[rules['antecedents']=={vertex}]['antecedent support'].iloc[0]*lenXX 
                      if not rules[rules['antecedents']=={vertex}].empty else 1 for vertex in g.vs['name']}  ##== (2)
    node_labels = [f"{vertex['name']}\n ({int(node_size_dict[vertex['name']] )})" for vertex in g.vs]
    node_sizes  = [ 0.5*node_size_dict.get(vertex['name'],1) for vertex in g.vs];  
    print(node_sizes)  #-- [169.0, 129.0, 124.5, 118.5, 90.0] ---> 實驗調整節點大小
    visual_style = { "layout": g.layout("fr"), "bbox": (900, 700), "margin": 50,
        "vertex_size": node_sizes, "vertex_label": node_labels, 'vertex_color': 'orange',
        "edge_label": [f"{rule['confidence']:.2f}" for _, rule in rules.iterrows()], 'edge_color': 'lightblue',
        "edge_arrow_size": 2, }
    return g, visual_style
graph, style = graph_style(rules)
print(style)   #-- {'layout': <Layout with 5 vertices and 2 dimensions>, 'bbox': (900, 700), 'margin': 50, 
#                   'vertex_size': [169.0, 129.0, 124.5, 118.5, 90.0], 'vertex_label': ['kind2\n (338)', 'kind3\n (258)', 'kind12\n (249)', 'kind1\n (237)', 'kind11\n (180)'], 
#                   'vertex_color': 'orange', 'edge_label': ['0.74', '0.97', '0.69', '0.93', '0.64', '0.91', '0.71', '0.73', '0.95', '0.68', '0.62', '0.65', '0.62'], 
#                   'edge_color': 'lightblue', 'edge_arrow_size': 2}
ig.plot(graph, **style)


#%%####### (E) CHATGPT 的實驗解讀 ##########

#%%== PROMPT-1: 因為這個數據關連圖...  是針對消費10次以上, 消費1萬元以上的客戶所得到的...   在這情形下, 你能否幫忙再幫忙找出重要的4-6個重點解讀, 謝謝......

# 1.  高價值客戶的品類集中度
# 從圖中可以看到，kind2 這個品類在這些高價值客戶中的支持度非常高，並且與其他品類之間有多條強關聯的邊，表明這些客戶在消費高價值產品時，對這些品類有一致的需求偏好。
# 這種集中度表明 kind2 可以被視為一個核心品類，對於高價值客戶來說，它代表了一個基礎性的購物需求，因此可以考慮進一步擴大這類產品的種類或相關服務。

# 2. 強烈的互補品類
# 圖中顯示出 kind2 與 kind12 之間有高達 0.93 和 0.97 的信心度，表示這兩個品類在高價值客戶的購物行為中存在強烈的互補性。
# 這意味著高價值客戶傾向於同時購買這兩個品類，這樣的洞察可以用於推出針對高價值客戶的捆綁銷售策略。
# 此外，這種高度的互補性意味著將這些品類一同進行展示或者促銷，可能會大大提高顧客滿意度和客單價。

# 3. 關聯性強的品類集群
# 多個節點之間，例如 kind2、kind12、kind3 形成了一個緊密的集群，這些品類之間的信心度均高於 0.7，說明這些品類之間有較強的關聯。
# 這些品類形成的集群表明高價值客戶在購物時往往會選擇多個相互關聯的產品。
# 因此，這些產品可以作為目標組合進行行銷和配貨，例如在門店中將這些產品相鄰陳列，以激發顧客的衝動購買。

# 4. 客戶需求的差異化程度
# 從圖中可以觀察到 kind1 與其他節點之間的關聯性明顯較弱，信心度低於 0.6。
# 這表明 kind1 的購買行為相對不如其他品類那麼穩定，可能具有較大的個性化差異。
# 對於這部分品類，可以考慮進行更多個性化的促銷活動，例如根據客戶的歷史行為推送不同的產品推薦，以滿足多樣化的需求。

# 5. 逆向關聯的意義
# 在某些關聯規則中，例如從 kind2 到 kind1 的信心度為 0.51，而反向從 kind1 到 kind2 則達到 0.95。
# 這意味著 kind1 作為後件時的影響較強，表明高價值客戶購買 kind1 時，經常也會購買 kind2，但購買 kind2 的顧客不一定會選擇 kind1。
# 可以考慮針對這種關聯進行促銷引導，例如當顧客購買 kind1 時提供 kind2 的促銷優惠，以提高整體的購物體驗和銷售額。

# 6. 核心品類的穩定性和策略應用
# 圖中的 kind2 節點表現出很強的穩定性，它與多個節點都有強關聯，而且這些關聯規則的支持度和信心度都較高。
# 這表明 kind2 作為高價值客戶群體中的核心購買品類，具有較大的市場潛力和忠誠度。
# 建議在營銷策略中重點關注 kind2，並將它作為主要促銷活動的一部分，針對高價值客戶群體提供忠誠獎勵或會員優惠，以進一步強化顧客對此品類的偏好。

# 對於消費頻繁且金額較高的高價值客戶，他們的購物行為顯示出某些品類之間的強烈互補關係 (kind2 和 kind12) 以及相對穩定的核心品類 (kind2)。
# 這些關聯可以作為商品開發和營銷的依據，例如推出聯合促銷、提高庫存、優化陳列等，以最大化高價值客戶的購物體驗並增加客單價。

#%%== PROMPT-2: 雖上述來自高價值的客戶, 能否由這樣的銷售關聯狀況, 試給我一個 商品開發/配貨的策略, 及商品行銷的策略

##### 商品開發/配貨策略
#== (1).開發針對高價值客戶的高關聯產品組合：
# 根據圖中的強關聯結果，例如 kind2 和 kind12 之間的高信心度，
# 這表明高價值客戶往往同時購買這些品類。因此，應開發專門針對高價值客戶的聯合產品包，將這些品類打包銷售。
# 這不僅可以增加客單價，還可以讓高價值客戶感受到更多的附加價值。
# 在配貨策略上，應優先確保這些高關聯產品的庫存充足，特別是在高價值客戶常購買的門店和地區，以防止因缺貨而流失潛在的銷售機會。

##== (2).優化商品布局，形成高效購物體驗：
# 將關聯強度高的品類（例如 kind2、kind12）在門店內相鄰陳列，以促進客戶的連帶購買。
# 這些品類的互補性強，將它們放在同一展示區域可以方便客戶一次性找到所需的所有商品，提高購物便利性。
# 針對高價值客戶經常購買的品類進行數量和存放位置的優化。
# 例如，可以根據客戶的購買習慣增加特定品類在門店的擺放比例，或者在門店內的不同區域設置多個擺放位置，確保其隨時可取。

##== (3).集中布局銷售空間：
# 將高度相關的品類放置在相鄰的貨架或同一銷售空間內。這樣的布局能夠有效刺激顧客的衝動購買，當顧客找到其中一種產品時，更容易購買與之高度相關的產品。這特別適用於如 kind2 和 kind12 之類的強關聯品類。
# 商品行銷策略

##== (4).定期更新高價值產品組合：
# 高價值客戶的需求可能會隨季節或其他外部因素發生變化。
# 因此，可以根據季節性或促銷期間，針對這些高關聯品類進行不斷的組合更新和優化，以吸引重複購買。
# 例如，夏季可以推出與飲料相關的品類組合，而在冬季則推出與保暖相關的品類。

##### 商品行銷策略：
##== (1).針對高價值客戶的聯合促銷活動
# 針對圖中顯示的強關聯規則，可以設計特定的聯合促銷活動，例如當顧客購買 kind2 時，提供購買 kind12 的折扣優惠。
# 這樣的策略可以激勵客戶將關聯品類加入購物車，從而提高銷售額。
# 可以考慮進行“買一搭一”或者套餐優惠，特別是針對那些同時購買多個高關聯品類的顧客，
# 這樣的優惠活動會讓顧客覺得更有價值，從而提高對這些品類的忠誠度。

##== (2).個性化推廣和定向行銷：
# 根據高價值客戶的購買行為，對這些客戶進行定向推廣。例如，當客戶購買了 kind1 之後，
# 根據分析結果自動向其推送包含 kind2 的產品推薦，並附上折扣優惠，這樣的交叉銷售策略可以有效提高客戶的購買率。
# 發送定向優惠券給那些曾經購買過強關聯產品的客戶，通過電子郵件或應用通知來告知有關新產品或促銷活動的信息，以增加回購的可能性。

##== (3).會員忠誠度計畫和特權推廣：
# 將高價值客戶納入特定的會員計畫中，為購買強關聯品類的顧客提供專屬的折扣或積分計畫，獎勵他們的忠誠度。
# 例如，針對多次購買 kind2 和 kind12 的客戶設立“黃金會員”特權，提供更多優惠和專屬服務。
# 針對這些高價值客戶的忠誠度行銷活動，可以進一步激勵他們增加購買次數和金額。這樣可以提高客戶的整體終身價值（LTV），使顧客對品牌的粘性更強。

##== (4).加強線上線下整合推廣
# 根據數據關聯結果，可以在線上商城和實體門店之間進行協同推廣。
# 例如，當顧客在線上商城購買了 kind3，則向其推薦高關聯的 kind12，並提供線下門店的取貨折扣。
# 這樣的整合行銷可以提高客戶的全渠道購物體驗。
# 將這些關聯強的品類整合到線上推薦引擎中，利用個性化推薦系統，根據客戶的購物車情況進行精準推薦。

##### 總結
# 在商品開發方面，可以推出聯合打包的產品，並優化高關聯品類的庫存與配貨，集中在門店中的貨架上以方便顧客購買。
# 在行銷策略上，建議使用聯合促銷、交叉銷售、VIP定向行銷和季節性促銷等方法，
# 以最大化利用數據中揭示的品類之間的強關聯性，提升顧客的購物體驗和商家的銷售額。
# 這些策略將有效利用數據分析所得的銷售關聯結果來促進銷售增長和顧客滿意度。

#%%
# FIS = apriori(oht_df, min_support=0.02, use_colnames=True, max_len=5);   print(FIS)
# rules = association_rules(FIS, metric="confidence", min_threshold=0.4);  print(rules)
