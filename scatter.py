# 1) Scatter plot (отрисовка графика на основании анализа введенных данных)

import matplotlib.pyplot as plt
study_hours = [3,1,0,6,14,8,3,5,1,0]
rest_hours = [5,4,2,5,2,8,5,0,9,2]
labels =['неуставший','неуставший','неуставший','уставший','уставший','неуставший','неуставший','уставший','неуставший','неуставший']

colors = ['green' if label in label =='неуставший'
          else 'red' for label in labels]

plt.scatter(study_hours, rest_hours, c = colors)
plt.xlabel ('Часы учебы')
plt.ylabel ('Часы отдыха')
plt.title ('Определение уставшего или неуставшего')

plt.show()







