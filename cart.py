import pandas as pd
from sklearn.datasets import load_iris

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import classification_report, confusion_matrix

def writeCsv(dataset, filename):
    dataset.to_csv(filename)

def train_test():
    iris_dt = load_iris()
    print(iris_dt.data[1:5])
    iris_dt.data[1:5]
    print(iris_dt.target[1:5])
    iris_dt.target[1:5]
    x_train, x_test, y_train, y_test = train_test_split(iris_dt.data, 
                        iris_dt.target, test_size=1/3.0, random_state=5)
    clf = DecisionTreeClassifier()
    # Train Decision Tree Classifer
    clf = clf.fit(x_train,y_train)

    #print(x_train)
    #print(x_test)
    #print(y_train)
    print(y_test)
    
    #Predict the response for test dataset
    y_pred = clf.predict(x_test)
    print(y_pred)
    y_test
    print(clf.predict([[4,4,3,3]]))
    print('Accuracy:',metrics.accuracy_score(y_test, y_pred))                    

def split_data(x, y, test_size, random_state = 0):
    train_pct_index = int((1 - test_size) * len(x))
    x_train, x_test = x[:train_pct_index], x[train_pct_index:]
    y_train, y_test = y[:train_pct_index], y[train_pct_index:]
    return x_train, x_test, y_train, y_test

def train_data(dataset, label):
    print(dataset.head())
    
    x = dataset[dataset.columns]   #features
    #print(x)
    #y = dataset.quality         #target variable
    y = dataset[label]
    #print(y)
    #train with sklearn
    #x_train, x_test, y_train, y_test = train_test_split(x, y, 
    #            test_size=0.3, random_state=1) # 80% training and 20% test
    
    #train with pandas only
    x_train, x_test, y_train, y_test = split_data(x, y, 
                0.3, 0) # 70% training and 30% test  

    #for f problem
    #x_test = x_test[:5]
    #y_test = y_test[:5]
                
    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()
    # Train Decision Tree Classifer
    clf = clf.fit(x_train,y_train)

    #print(x_train)
    #print(x_test)
    #print(y_train)
    #print(y_test)
    
    #Predict the response for test dataset
    y_pred = clf.predict(x_test)
    print('Accuracy:', metrics.accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))    
    

def main():
    dataset = pd.read_csv('winequality-white.csv', header=0, delimiter=';')
    #train_test()
    label = dataset.columns[-1:]
    train_data(dataset, label)

if __name__ == '__main__':
    main();