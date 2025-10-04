# Задание 16. "Уставший и отдохнувший"
# Требование к таблице: часы отдыха и часы учёбы
# Требование к меткам: уставший/неуставший
# Основные задачи:
# 1) Scatter plot
# 2) Обучить KNN
# 3) Нарисовать decision boundary
# 4) Проверить accuracy
# 5) Сравнение с деревом ркшений
#===============================================

# Импорт необходимых библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#===============================================

# Создание Датафрейма (ввод данных) 
data = pd.DataFrame({
    "students": ["student"]*10,
    "study_hours" : [3,1,0,6,14,8,3,5,1,0],
    "rest_hours" : [5,4,2,5,2,8,5,0,9,2],
    "labels" :[
        'неуставший', 'неуставший', 'неуставший', 'уставший', 'уставший',
        'неуставший', 'неуставший', 'уставший', 'неуставший', 'неуставший'
    ]
})

# Преобразует данные в one-hot формат, который удобен для машинного обучения.
# One-hot представление — это способ закодировать категориальные данные в
# числовой формат, понятный алгоритмам машинного обучения
encoder = OneHotEncoder()
students_encoded = encoder.fit_transform(data[["students"]]) 

# Формирование признаков (X) и целевых значений (y) для обучения
X = np.hstack([students_encoded.toarray(), data[["study_hours", "rest_hours"]].values]) 
y = data["labels"]

# Разделение на обучающую и тестовую части
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=45
)

# 2) Обучение KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 4) Проверка accuracy (доля правильных предсказаний среди всех предсказаний)
y_pred_knn= knn.predict(X_test)
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))

# 5) Сравнить с деревом решений(по времени выполнения)
# Обучение дерева решений
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

# Проверка accuracy дерева решений
y_pred_tree = tree.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_tree))

# Создание примера
new_student = encoder.transform([["student"]]).toarray()
new_data = np.hstack([new_student, [[0, 0]]])
print(knn.predict(new_data))

# 3) Отрисовка decision boundary
# Только числовые признаки: study_hours и rest_hours. ".values" првращает данные в массив Numpy"
# Метод map() преобразуют в числовые значения
X_vis = data[["study_hours", "rest_hours"]].values
y_vis = data["labels"].map({'уставший': 1, 'неуставший': 0}).values 

# Обучение модели (3 ближайших соседа нового объекта при классификации)
knn_vis = KNeighborsClassifier(n_neighbors=3) 
knn_vis.fit(X_vis, y_vis)

# Определение границ по осям X и Y (1-отступ)
x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1 # study_hours
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1 # rest_hours

# Создаётся сетка 200 на 200
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)

# xx.ravel() и yy.ravel() превращают сетку в список координат
# np.c_[] объединяет точки на плоскости в пары 
Z = knn_vis.predict(np.c_[xx.ravel(), yy.ravel()]) 
# Преобразованиие обратно в форму сетки
Z = Z.reshape(xx.shape)

# 1,3) Построение графика (Scatter plot and decision boundary)
# X_vis[:, 0] — значения по оси X (часы учёбы)
# X_vis[:, 1] — значения по оси Y (часы отдыха)
# edgecolors='k' — чёрная окантовка точек
# alpha=0.3 — прозрачность заливки (чтобы видно было)
# coolwarm (синяя и красная заливка)
colors = ['green' if label == 'неуставший' else 'red' for label in data["labels"]]

plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm') # 3) функция строющая decision boundary
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_vis, edgecolors='k', cmap='coolwarm') # 1) функция Scatter plot
plt.xlabel("study_hours")
plt.ylabel("rest_hours")
plt.title("Decision Boundary (KNN)")
plt.grid(True)
plt.show()
