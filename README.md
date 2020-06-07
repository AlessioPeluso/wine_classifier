# Wine classification

Supervised analysis on the [wine](https://www.kaggle.com/brynja/wineuci) dataset of kaggle. This dataset was created for testing out different classifiers.
>"This data set is the result of a chemical analysis of wines   grown in the same region in Italy but derived from three different cultivars. The analysis determined the quantities of 13 constituents found in each of the three types of wines"

---

We begin importing the dataset and renaming the columns.

```
import pandas as pd 

wine = pd.read_csv('/Wine.csv')
wine.columns = [
    'Label', 
    'Alcohol',
    'Malic acid',
    'Ash',
    'Alcalinity of ash',
    'Magnesium',
    'Total phenols',
    'Flavanoids',
    'Nonflavanoid phenols',
    'Proanthocyanins',
    'Color intensity',
    'Hue',
    'OD280/OD315 of diluted wines',
    'Proline']
    
# wine.head()
```

Unfortunately the wine labels are just [1,2,3] and not the wine names.

---

## Data Visualization

Now let's plot some graphs to look how those features are related. 

Let's look at the pairplot.

```
import seaborn as sns
import matplotlib.pyplot as plt
 
# basic correlogram
sns.pairplot(wine, hue = 'Label', height=5)
plt.show()
```

![](images/wine.png)

From the plot we can see that the features are:

1. normally distributed
2. visibly apart from each other

---

##Analysis

### - Method #1 the *Naive Bayes Classifier*
Because the features are close to the normal distribution the first method I'm gonna use is the *Naive Bayes Classifier*, this classifier performs well on normally distributed data and if the distributions are situated apart from each other it will be much easier to distinguish among the three different classes.

```
# extract the labels 
wine_temp = wine.copy()

labels = wine_temp.pop('Label')

# split train-test
import sklearn 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(wine_temp, labels, test_size=0.3, random_state=1623)
```

After we have extracted the *labels* and splitted the data into *train* and *test* we can build the model.

```
# building the model 
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
```

Building this model is really easy, then we have to train it on the *train set* and we can see the accuracy in this particular set.

```
# train the model 
clf = clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_predict))
0.9629629629629629
```

We have acheived a good score (0.963) with our seed. Now to have a more generalizable result we compute the accuracy using the *cross-validation*.

```
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, wine_temp, labels, cv=10)

print(scores.mean())
0.9774509803921567
```

The accuracy using the cross-validation is even better.\
Let's try another method.

---

### Method #2 *Support Vector Machine*
Let's implement a support vector machine with hyperparameter tuning.

```
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

param_grid = {'C': [0.01, 0.1, 0.5, 1, 10, 100], 
              'gamma': [1, 0.75, 0.5, 0.25, 0.1, 0.01, 0.001], 
              'kernel': ['rbf', 'poly', 'linear']} 

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=1, cv=10, n_jobs = -1)

grid.fit(X_train, y_train)
pred = clf.predict(X_test)

print(f"accuracy score: {accuracy_score(y_test, pred)}\n")
accuracy score: 0.9629629629629629
```

