import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,classification_report,roc_curve,accuracy_score,roc_auc_score
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import graphviz
import os
import re
from sklearn.model_selection import GridSearchCV

df = pd.read_excel('/Users/gushuai/Desktop/员工离职预测模型.xlsx')
print(df.head())

df = df.replace({'工资': {'低': 0, '中': 1, '高': 2}})
print(df.head())
#换了工资为数字之后的矩阵
plt.rcParams['font.sans-serif'] = ['SimHei']
df[list(df.columns)].hist(layout=(2,4))
plt.show(block=False)
#设置X,y
X = df.drop(columns='离职')
y = df['离职']
#划分训练集和数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
#建模
model = DecisionTreeClassifier(max_depth=3, random_state=1)
model.fit(X_train, y_train)
#预测
y_pred = model.predict(X_test)
a = pd.DataFrame()
a['预测值'] = list(y_pred)
a['实际值'] = list(y_test)
print(a.head(10))

score = accuracy_score(y_pred, y_test)
print('分类准确率为: %f%%' % (score * 100))

print('分类准确率为: %f%%' % (model.score(X_test, y_test) * 100))


y_pred_proba = model.predict_proba(X_test)
b = pd.DataFrame(y_pred_proba, columns=['不离职概率', '离职概率'])
print(b.head(10))

m = confusion_matrix(y_test, y_pred)
a = pd.DataFrame(m, index=['0（实际不离职）', '1（实际离职）'],
                 columns=['0（预测不离职）', '1（预测离职）'])
print(a)

print(classification_report(y_test, y_pred))
#计算fpr，tpr和thres
fpr, tpr, thres = roc_curve(y_test, y_pred_proba[:,1])

a = pd.DataFrame()
a['阈值'] = list(thres)
a['假警报率'] = list(fpr)
a['命中率'] = list(tpr)
print(a.head())

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.plot(fpr, tpr)
plt.title('ROC曲线')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show(block=False)

score = roc_auc_score(y_test, y_pred_proba[:,1])
print('AUC value is %f' % score)

#计算特征重要性
features = X.columns
importances = model.feature_importances_

importances_df = pd.DataFrame()
importances_df['特征名称'] = features
importances_df['特征重要性'] = importances
importances_df.sort_values('特征重要性', ascending=False, inplace=True)
print(importances_df)
#柱状图
n_features = len(features)
plt.barh(range(n_features), importances, align='center')
plt.yticks(range(n_features), features)
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.show()


#修改graphviz的dot
graphviz.backend.ExecutableNotFound('/Users/gushuai/opt/anaconda3/envs/pytorch/bin/dot')
os.environ["PATH"] += os.pathsep + '/Users/gushuai/opt/anaconda3/envs/pytorch/bin'
#保存pdf
feature_names = ['income', 'satisfication', 'score', 'project_num', 'hours', 'year']
dot_data = export_graphviz(model, out_file=None, feature_names=feature_names,
                           class_names=['0', '1'], filled=True)
graph = graphviz.Source(dot_data)
graph.render("result")
print('可视化文件result.pdf已经保存在代码所在文件夹！')


#修改graphviz的dot

os.environ['PATH'] = os.pathsep + '/Users/gushuai/opt/anaconda3/envs/pytorch/bin'

feature_names = X_train.columns
dot_data = export_graphviz(model, out_file=None, feature_names=feature_names,
                           class_names=['不离职', '离职'], rounded=True, filled=True)

f = open('dot_data.txt', 'w')
f.write(dot_data)
f.close()


f_old = open('dot_data.txt', 'r')
f_new = open('dot_data_new.txt', 'w', encoding='utf-8')
for line in f_old:
    if 'fontname' in line:
        font_re = 'fontname=(.*?)]'
        old_font = re.findall(font_re, line)[0]
        line = line.replace(old_font, 'SimHei')
    f_new.write(line)
f_old.close()
f_new.close()

os.system('dot -Tpng dot_data_new.txt -o 决策树模型.png')
print('决策树模型.png已经保存在代码所在文件夹！')

os.system('dot -Tpdf dot_data_new.txt -o 决策树模型.pdf')
print('决策树模型.pdf已经保存在代码所在文件夹！')


# 9.多参数调优**

parameters = {
    'max_depth': [5, 7, 9, 11, 13],
    'criterion':['gini', 'entropy'],
    'min_samples_split':[5, 7, 9, 11, 13, 15]
}
model = DecisionTreeClassifier()
grid_search = GridSearchCV(model, parameters, scoring='roc_auc', cv=5, n_jobs=4)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
model = DecisionTreeClassifier(criterion=best_params['criterion'],
                               max_depth=best_params['max_depth'],
                               min_samples_split=best_params['min_samples_split'])
model.fit(X_train, y_train)

features = X.columns
importances = model.feature_importances_
importances_df = pd.DataFrame()
importances_df['特征名称'] = features
importances_df['特征重要性'] = importances
importances_df.sort_values('特征重要性', ascending=False, inplace=True)
print(importances_df)

y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred, y_test)
print('分类准确率为: %f%%' % (score * 100))

y_pred_proba = model.predict_proba(X_test)
score = roc_auc_score(y_test, y_pred_proba[:,1])
print('AUC value is %f' % score)

