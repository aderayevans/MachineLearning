import pandas as pd
import numpy as np
from pprint import pprint
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import classification_report, confusion_matrix

dataset = pd.read_csv('data.csv')
dataset = dataset.iloc[:, 1:]

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
    print(vals)
    print(counts)
    
    total_elements = np.sum(counts)
    print(total_elements)
    
    weighted_entropy = 0
    for i in range(len(vals)):
        print(vals[i])
        print(split_attribute_name)
        
        weighted_elements = (counts[i]/total_elements)
        print(weighted_elements)
        
        dt_split_attribute_vals = data[data[split_attribute_name] == vals[i]]
        print(dt_split_attribute_vals)
            
        entropy_elements = entropy(dt_split_attribute_vals[target_name])
        
        weighted_entropy = weighted_entropy + weighted_elements*entropy_elements
        print(weighted_entropy)
    
    information_gain = total_entropy - weighted_entropy
    print(information_gain)
    return information_gain
    
def xstr(s):
    if s is None:
        return ''
    else:
        return s
    
def ID3(data, original_data, features, target_attribute_name, parent_node_class):
    unique_instance_num = len(np.unique(data[target_attribute_name]))
    print('** According to ' + target_attribute_name + '/' 
            + xstr(parent_node_class) + ' with unique_instance_num = ' + str(unique_instance_num) + ' **')
    print(data)
    
    if unique_instance_num <= 1:
        print('Du lieu thuan nhat tra ve nut la')
        return np.unique(data[target_attribute_name])[0]
    else:
        if len(features) == 0:
            return 0
        else:
            print(data[target_attribute_name])
            unique_data_attribute, indices = np.unique(data[target_attribute_name], return_counts = True)
            argmax_index = np.argmax(indices)
            print('Unique:', unique_data_attribute)
            print('Indices:', indices)
            print('Argmax_index: ', argmax_index)
            parent_node_class = np.unique(data[target_attribute_name])[argmax_index]
            
            print('Parent node class: ', xstr(parent_node_class))
            print('Tap features: ', features)
            
            item_values = [infoGain(data, features, target_attribute_name) for feature in features]
            print('Item_values: ', item_values)
            best_feature_index = np.argmax(item_values)
            best_feature = features[best_feature_index]
            print('Best_feature', best_feature)
            
            tree = {best_feature:{}}
            features = [i for i in features if i != best_feature]
            
            print('Tap thuoc tinh sau khi loai la: ', features)
            
            for value in np.unique(data[best_feature]):
                print('Xet gia tri: ', value, '\n')
                sub_data = data.where(data[best_feature] == value).dropna()
                
                print('Subdata la: \n', sub_data)
                print('parent_node_class la: ', parent_node_class, '\n')
                subtree = ID3(sub_data, dataset, features, target_attribute_name, parent_node_class)
                
                tree[best_feature][value] = subtree
            return(tree)



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

def train_data(dataset, label, x_testcase, y_testcase):
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
    x_train, x_test, y_train, y_test = split_data(x, y, 0, 0)
    
    x_test = x_testcase
    y_test = y_testcase
    
    print(x_test)
    print(y_test)
    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()
    # Train Decision Tree Classifer
    clf = clf.fit(x_train, y_train)

    #print(x_train)
    #print(x_test)
    #print(y_train)
    #print(y_test)
    
    #Predict the response for test dataset
    y_pred = clf.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)
    if accuracy > 0:
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred)) 
    
    
def main():
    tree = ID3(dataset, dataset, dataset.columns[1:-1], 'Label', None)
    pprint(tree)
    
    label = dataset.columns[-1:]
    
    dataset[label] = dataset[label].replace({'Boy': 1,'Girl': 0})
    for instance in np.unique(dataset[label]):
        x_testcase = [[135, 39, 1, instance]]
        y_testcase = [[instance]]  
        train_data(dataset, label, x_testcase, y_testcase)
    
    
if __name__ == "__main__":
    main()