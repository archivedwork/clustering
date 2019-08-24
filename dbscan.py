import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
from pylab import rcParams

import sklearn
from sklearn.cluster import DBSCAN
from collections import Counter

rcParams['figure.figsize'] = (5, 4)
sb.set_style('whitegrid')

# train your model and identify outliers
df = pd.read_csv('iris.data.csv', header=None, sep=',')
df.columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Species']
data   = df.iloc[:, 0:4]
target = df.iloc[:, 4]

print(df[:5])

# initiate dbscan object
model = DBSCAN(eps=0.8, min_samples=19).fit(data)
print(model)

# visualize your results
outliers_df = pd.DataFrame(data)
print Counter(model.labels_)
print outliers_df[model.labels_==-1]

fig = plt.figure()
ax = fig.add_axes([.1, .1, 1, 1])
colors = model.labels_
ax.scatter(data.iloc[:, 2].values, data.iloc[:, 1].values, c=colors, s=120)

ax.set_xlabel('Petal Length')
ax.set_ylabel('Sepal Width')
plt.title('DBSCAN for outliers detection')
plt.show()