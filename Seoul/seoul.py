import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


class Node:
    def __init__(self, predict_data, parent, left_child, right_child, label, mean, mse, samples, value, level, leaf=False):
        self.predict_data = predict_data
        self.parent = parent
        self.left_child = left_child
        self.right_child = right_child
        self.label = label
        self.mean = mean
        self.mse = mse
        self.samples = samples
        self.value = value
        self.level = level
        self.leaf = leaf
    
    def display(self):
        for i in range (0, self.level):
            print('|   ', end = '')
        print('|--- ', end = '')
        if self.leaf == False:
            print(self.label, '<=', self.mean)
        else:
            print('value: [', self.value, ']', sep='')
        if self.left_child != None:
            self.left_child.display()
        if self.right_child != None:
            for i in range (0, self.level):
                print('|   ', end = '')
            print('|--- ', end = '')
            print(self.label, '>', self.mean)
            self.right_child.display()
    
    def _display(self):
        for i in range (0, self.level):
            print('|   ', end = '')
        print('|--- ', end = '')
        if self.leaf == False:
            print(self.label, '<=', self.mean, end = ', ')
            print('mse =', self.mse, end = ', ')
            print('samples =', self.samples, end = ', ')
            print('value =', self.value)
        else:
            print('value: [', self.value, ']', sep='')
        if self.left_child != None:
            self.left_child.display()
        if self.right_child != None:
            for i in range (0, self.level):
                print('|   ', end = '')
            print('|--- ', end = '')
            print(self.label, '>', self.mean, end = ', ')
            print('mse =', self.mse, end = ', ')
            print('samples =', self.samples, end = ', ')
            print('value =', self.value)
            self.right_child.display()
    
    def set_left_child(self, left_child):
        self.left_child = left_child
    
    def set_right_child(self, right_child):
        self.right_child = right_child
    
    def predict(self, x_test_instance_dict):
        if self.left_child != None and x_test_instance_dict[self.label] <= self.mean:
            return self.left_child.predict(x_test_instance_dict)
        elif self.right_child != None and x_test_instance_dict[self.label] > self.mean:
            return self.right_child.predict(x_test_instance_dict)
        else: #is a leaf
            #print('$$$', self.predict_data.values.tolist()[0])
            #self.parent.display()
            return self.predict_data.values.tolist()[0]


class DecisionTreeRegressor:
    def __init__(self, max_depth = 0):
        self.data = None
        self.target = None
        self.max_depth = 0
        self.tree = None
        self.depth = 0
        self.max_depth = max_depth
        
    def fit(self, x_train, y_train):
        self.data = x_train
        self.target = y_train
        self.tree = self.build_tree(self.data, self.target, 0, None)
    
    def best_split(self, df, label, best_mse):
        print(df)
        #use to save best feature and its value easily
        best_feature = None
        best_value = None
        
        #use to drop features got only one unique
        drop_data_columns = []
        for feature in df.columns:
            df = df.dropna().sort_values(feature)
            if feature == label: continue
            if len(np.unique(df[feature])) == 1:
                drop_data_columns.append(feature)
                continue
            ### calculate moving average (MA), to split into left and right branch
            means = np.convolve(df[feature].unique(), np.ones(2), 'valid') / 2
            print(means)
            for val in means:
                left_y = df[df[feature] <= val]['Y'].values
                right_y = df[df[feature] > val]['Y'].values
                left_mean = np.mean(left_y)
                right_mean = np.mean(right_y)
                # Getting the left and right residuals 
                res_left = left_y - left_mean 
                res_right = right_y - right_mean
                # Concatenating the residuals 
                residuals = np.concatenate((res_left, res_right), axis=None)
                # Calculate mse when we split with this MA of this FEATURE
                mse_split = np.sum(residuals ** 2) / len(residuals)
                print(feature, val, mse_split)
                # Compare mse, take min
                if mse_split < best_mse:
                    best_feature = feature
                    best_value = val
                    best_mse = mse_split
                    print('*****')
        #print(best_feature)
        #print(best_value)
        #print(best_mse)
        #drop no need feature
        for feature in drop_data_columns:
            df = df.drop(columns = [feature])
        return df, best_feature, best_value
    
    def build_tree(self, data, target, level, parent):
        df = data.copy()
        df['Y'] = target
        
        #Calculate mse_base
        temp_y = (df['Y'] - np.mean(df['Y'])) ** 2
        mse_base = np.sum(temp_y) / len(df['Y'])
        
        #save value for node (just ignore these till create Node
        value = np.mean(df['Y'])
        samples = len(df['Y'])
        if len(df['Y'].unique()) == 1 or (self.max_depth != 0 and level >= self.max_depth):
            print('Making leaf:', round(mse_base, 3), samples, round(value, 3), len(df['Y'].unique()))
            print('###')
            print(df['Y'])
            print('###')
            return Node(df['Y'], parent, None, None, None, None, round(mse_base, 3), samples, round(value, 3), level, True)
        
        #'best' spliter    
        df, best_feature, best_value = self.best_split(df, 'Y', mse_base)
        if best_feature == None: return None
        else: self.depth = max(self.depth, level)
        subset_left = df[df[best_feature] <= best_value]
        subset_right = df[df[best_feature] > best_value]
        left_Node = None
        right_Node = None
        newNode = Node(df['Y'],
                        parent, left_Node, right_Node, 
                        best_feature, 
                        round(best_value, 3), 
                        round(mse_base, 3), 
                        samples, 
                        round(value, 3),
                        level
                       )
        #newNode.display()  
        left_Node = self.build_tree(subset_left, subset_left['Y'], level+1, newNode)
        right_Node = self.build_tree(subset_right, subset_right['Y'], level+1, newNode)
        newNode.set_left_child(left_Node)
        newNode.set_right_child(right_Node)                
        return newNode
    
    def display(self):
        self.tree.display()
        
    def predict(self, x_test):
        y_pred = []
        for index in range (0, len(x_test)):
            y_pred.append(self.tree.predict((x_test.iloc[index, :]).to_dict()))
        return pd.Series(y_pred).values


def read_data(data_name, label, opt=2):
    if opt == 0: #create normalized_data
        dt = pd.read_csv('SeoulBikeData.csv')
        dt, normalize_ar = normalize(dt, label)
        #print(normalize_ar)
        with open('attribute_convert_map.txt', 'w', encoding='utf-8') as f:
            for item in normalize_ar:
                f.write("%s\n" % item)
        writeCsv(dt, 'normalized_data.csv')
        writeCsv(dt.iloc[0:10], 'test.csv')
    elif opt == 1:
        dt = pd.read_csv('test.csv', index_col = 0)
    else: #read test_data
        dt = pd.read_csv('normalized_data.csv', index_col = 0)
        #dt.columns = dt.columns.str.replace(r'[^A-Za-z0-9]', '', regex=True)
    return dt

def write_object_to_file(the_object, file_name):
    with open(file_name, 'wb') as output:
        pickle.dump(the_object, output, pickle.HIGHEST_PROTOCOL)

def read_object_from_file(file_name):
    with open(file_name, 'rb') as input:
        the_object = pickle.load(input)
    return the_object
       
def writeCsv(dataframe, filename):
    dataframe.to_csv(filename)

def show(x_label, y_label):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
    plt.close()

def normalize(dt, label):
    attribute_list = dt.columns
    return_list = []
    for attribute in attribute_list:
        if attribute == label:
            continue
        array = np.unique(dt[attribute])
        list_ = array.tolist()
        if isinstance(array[0], str):
            #for index in list_:
            #    dt[attribute] = dt[attribute].replace({index: list_.index(index)})
            min = 0
            max = (len(list_) - 1)
            ##interval = max - min
            for index in list_:
                #return_list.append({attribute:{index: ((list_.index(index) - min)/(max - min))}})
                #dt[attribute] = dt[attribute].replace({index: ((list_.index(index) - min)/(max - min))}) 
                return_list.append({attribute:{index: list_.index(index)}})
                dt[attribute] = dt[attribute].replace({index: list_.index(index)}) 
                
        #else:
        #    min = np.amin(array)
        #    max = np.amax(array)
        #    ##interval = max - min
        #    for index in list_:
        #        return_list.append({attribute:{index: ((index - min)/(max - min))}})
        #        dt[attribute] = dt[attribute].replace({index: ((index - min)/(max - min))})    
    return dt, return_list

def mean_squared_error(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2) / len(y_true)
    
def predict_with_DecisionTreeRegressor(x, y, label, visualize=True):
    x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.3, random_state=10, shuffle=True)
    x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, 
                                                    test_size=0.5, random_state=10, shuffle=False)                                              
    #print(x_train)
    #print(y_train)
    
    #dr = DecisionTreeRegressor(max_depth = 0)
    #dr.fit(x_train, y_train)
    
    #write_object_to_file(dr, 'trained_data.pkl')
    dr = read_object_from_file('trained_data.pkl')
    
    #dr.display()
    
    y_pred = dr.predict(x_test)
    #for val in y_pred:
    #    print(val)
    
    mse = mean_squared_error(y_test, y_pred)
    
    print('mse =', mse)
    rmse_err = np.sqrt(mse)
    print('root mse =', rmse_err)
    
    if visualize:
        attribute_list = x.columns
        for each in attribute_list:
            plt.scatter(x_train[each], y_train, color='blue')
            plt.scatter(x_test[each], y_test, color='red')
            plt.scatter(x_valid[each], y_valid, color='violet')
            plt.scatter(x_test[each], y_pred, color='green')
        #    #plt.plot(x_train[each], y_train, color='violet')
        #    #plt.plot(x_test[each], y_pred, color='red')
            show(x_label=each, y_label=label)
            plt.close()
 
def main():
    label='Rented_Bike_Count'
    df = read_data('SeoulBikeData.csv', label, opt=2)
    df.columns = df.columns.str.replace(r'[^a-zA-Z0-9]', '_', regex=True)
    
    x = df.drop(columns = [label])
    y = df[label]
    
    predict_with_DecisionTreeRegressor(x, y, label, visualize=False)
    
 
if __name__ == "__main__":
    main()