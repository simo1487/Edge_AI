import pandas as pd
import sklearn.model_selection as ms
import sklearn.ensemble as rf
import sklearn.preprocessing as pp
import numpy as np
import matplotlib.pyplot as plt
import sklearn.tree as tree
import emlearn

df = pd.read_csv('data/breast-cancer/data.csv')
df["rnd"] = np.random.rand(len(df))
x = df.drop(['id', 'diagnosis'], axis=1)
y = df['diagnosis'].values

x_train, x_test, y_train, y_test = ms.train_test_split(x, y, test_size=0.2, random_state=42)

scaler = pp.StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape)

model = rf.RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)
print(model.score(x_train, y_train))
print(model.score(x_test, y_test))

print(model.feature_importances_)

plt.figure(figsize=(14, 5))
plt.bar(x.columns, model.feature_importances_)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

np.savetxt('data/breast-cancer/scaler_rf_cancer_mean.csv',  scaler.mean_,  delimiter=',')
np.savetxt('data/breast-cancer/scaler_rf_cancer_scale.csv', scaler.scale_, delimiter=',')

cmodel = emlearn.convert(model, method='inline')
cmodel.save(file='data/breast-cancer/cancer_rf_v2_model.csv', name='rf', format='csv')
cmodel.save(file='data/breast-cancer/cancer_rf_v2_model.h', name='breast_cancer_rf')

tree.export_graphviz(model.estimators_[0], out_file='data/breast-cancer/tree.dot',
                     feature_names=x.columns, class_names=["benign", "malignant"])
