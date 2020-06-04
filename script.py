import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
from scipy import stats
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y)



class flexible_naive_bayes():

    def __init__(self,):

        self.my_dict: dict = {}

    def fit(self, df: pd.DataFrame, y: pd.Series):

        n: int = df.shape[0]

        self.cont_columns: list = df.select_dtypes('float64').columns.to_list()
        self.cat_columns: list = df.select_dtypes('category').columns.to_list()

        self.class_list: list[int] = y.astype(int).sort_values().unique()
        self.class_freq: pd.Series = y.astype(int).value_counts() / n

        # construct dict
        for i in self.class_list:

            self.my_dict[i] = {}

            self.my_dict[i]['prior'] = self.class_freq[i]
        
            for col in self.cat_columns:
                
                self.my_dict[i][col] = (df.loc[y == i, col].astype(int).value_counts() / len(y == i)).to_dict()

            for col in self.cont_columns:

                self.my_dict[i][col] = stats.gaussian_kde(df.loc[y == i, col])

        return self.my_dict

    def predict(self, df):

        predictions = []
        for i in range(df.shape[0]):

            class_hood = []

            for j in self.class_list: #classes

                inter = 0

                inter += np.log(self.my_dict[j]['prior'])

                for cat_col in self.cat_columns:

                    try:
                        inter += np.log(self.my_dict[j][cat_col][int(df.loc[j, cat_col])])
                    except KeyError:
                        inter += 0

                for cont_col in self.cont_columns:
                  
                    inter += np.log(self.my_dict[j][cont_col](df.loc[i, cont_col]))
                                      
                class_hood.append(inter.item())

            predictions.append(np.argmax(class_hood))

        return predictions



aa = flexible_naive_bayes()
aa.fit(pd.DataFrame(X_train), pd.Series(y_train))

print('flexible naive bayes accuracy:')
print(accuracy_score(y_test, aa.predict(pd.DataFrame(X_test))))

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)

print('sklearn gaussian naive bayes accuracy:')
print(accuracy_score(y_test, clf.predict(X_test)))











