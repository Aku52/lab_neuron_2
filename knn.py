import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


data = pd.DataFrame({
    "students": ["student","student","student","student","student","student","student","student","student","student"],
    "study_hours" : [3,1,0,6,14,8,3,5,1,0],
    "rest_hours" : [5,4,2,5,2,8,5,0,9,2],
    "labels" :['неуставший','неуставший','неуставший','уставший','уставший','неуставший','неуставший','уставший','неуставший','неуставший']
})

encoder = OneHotEncoder()
x_encoded = encoder.fit_transform(data[["students"]])
X = np.hstack([x_encoded.toarray(), data[["study_hours", "rest_hours"]].values])
y = data["labels"]

# Разделение на обучающую и тестовую части
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=45)

# 2) Обучение KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 4) Проверка accuracy
y_pred = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Создание примера
new = encoder.transform([["student"]]).toarray()
new_data = np.hstack([new, [[0, 0]]])
print(knn.predict(new_data))

# 3) Отрисовка decision boundary
# Только числовые признаки: study_hours и rest_hours
X_vis = data[["study_hours", "rest_hours"]].values
y_vis = data["labels"].map({'уставший': 1, 'неуставший': 0}).values

knn_vis = KNeighborsClassifier(n_neighbors=3)
knn_vis.fit(X_vis, y_vis)

# Сетка координат
x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

Z = knn_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Построение графика
plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_vis, edgecolors='k', cmap='coolwarm')
plt.xlabel("study_hours")
plt.ylabel("rest_hours")
plt.title("Decision Boundary (KNN)")
plt.grid(True)
plt.show()




