#Adicionar importação de biblioteca.
import pandas as pd
from sklearn import datasets
iris = datasets.load_iris()

#Criar árvore de decisão.
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
x = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(criterion="entropy", random_state=42)


