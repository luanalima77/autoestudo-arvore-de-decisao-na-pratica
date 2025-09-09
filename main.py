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
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

#Avaliar previsões realizadas.
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

def accuracy(yt, yp):
    return (yp==yt).mean()

print(accuracy(y_test, y_pred))

#Imprimir a matriz de confusão.
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
confusion_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(confusion_matrix)
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.show()