import pandas as pd
import numpy as np
from pprint import pprint
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import classification_report, confusion_matrix


def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts = True)
    en = 0
    for i in range(len(elements)):
        pi = counts[i] / np.sum(counts)
        temp = -pi*np.log2(pi)
        en = en + temp
    return en

def infoGain(data, split_attribute_name, target_name):
    total_entropy = entropy(data[target_name])
    
    vals, counts = np.unique(data[split_attribute_name], return_counts = True)
    
    total_elements = np.sum(counts)
    
    weighted_entropy = 0
    for i in range(len(vals)):
        weighted_elements = (counts[i]/total_elements)
        
        dt_split_attribute_vals = data[data[split_attribute_name] == vals[i]]
            
        entropy_elements = entropy(dt_split_attribute_vals[target_name])
        
        weighted_entropy = weighted_entropy + weighted_elements*entropy_elements
    
    information_gain = total_entropy - weighted_entropy
    return information_gain
    
def xstr(s):
    if s is None:
        return ''
    else:
        return s
    
def ID3(data, original_data, features, target_attribute_name, parent_node_class):
    unique_instance_num = len(np.unique(data[target_attribute_name]))
    
    if unique_instance_num <= 1:
        return np.unique(data[target_attribute_name])[0]
    else:
        if len(features) == 0:
            return 0
        else:
            unique_data_attribute, indices = np.unique(data[target_attribute_name], return_counts = True)
            argmax_index = np.argmax(indices)
            parent_node_class = np.unique(data[target_attribute_name])[argmax_index]
            
            item_values = [infoGain(data, features, target_attribute_name) for feature in features]
            best_feature_index = np.argmax(item_values)
            best_feature = features[best_feature_index]
            
            tree = {best_feature:{}}
            features = [i for i in features if i != best_feature]
            
            for value in np.unique(data[best_feature]):
                sub_data = data.where(data[best_feature] == value).dropna()
                subtree = ID3(sub_data, original_data, features, target_attribute_name, parent_node_class)
                
                tree[best_feature][value] = subtree
            return(tree)

def writeCsv(dataframe, filename):
    dataframe.to_csv(filename)

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
    
    #Predict the response for test dataset
    y_pred = clf.predict(x_test)
    print('Accuracy:',metrics.accuracy_score(y_test, y_pred)) 
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))                     

def split_data(x, y, test_size, random_state = 0):
    train_pct_index = int((1 - test_size) * len(x))
    x_train, x_test = x[:train_pct_index], x[train_pct_index:]
    y_train, y_test = y[:train_pct_index], y[train_pct_index:]
    return x_train, x_test, y_train, y_test

def train_data(dataset, label):
    x = dataset[dataset.columns]   #features
    x = x.drop(columns = [label])
    y = dataset[label]
                
    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier(criterion='entropy', random_state=100, max_depth=3, min_samples_leaf=5)
    # Train Decision Tree Classifer
    clf = clf.fit(x, y)
    
    y_pred = clf.predict([[135, 39, 1]])
    print(y_pred)
    
def main():
    dataframe = pd.read_csv('data.csv')
    dataframe = dataframe.iloc[:, 1:]
    
    tree = ID3(dataframe, dataframe, dataframe.columns[1:-1], 'Label', None)
    pprint(tree)
    
    columns = dataframe.columns
    label = columns[-1]
    train_data(dataframe, label)
    
if __name__ == "__main__":
    main()