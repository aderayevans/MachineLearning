#!/usr/bin/env python
# coding: utf-8

# In[1772]:


import pickle
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
from sklearn.model_selection import train_test_split


# In[1773]:


class Node:
    def __init__(self, predict_data, parent, left_child, right_child, 
                     label, mean, mse, samples, value, level, leaf=False):
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
    
    def predict(self, x_test_instance_dict, int_ml_task=0):
        if self.left_child != None and x_test_instance_dict[self.label] <= self.mean:
            return self.left_child.predict(x_test_instance_dict)
        elif self.right_child != None and x_test_instance_dict[self.label] > self.mean:
            return self.right_child.predict(x_test_instance_dict)
        else: #is a leaf
            if int_ml_task == 0:#'classification'
                return self.predict_data.value_counts().index[0]
            else:
                return self.predict_data.mean()
    
    def getKey(self):
        return str(self.label) + ' <= ' + str(self.mean)
    
    def getValue(self):
        if self.leaf:
            return self.predict_data.value_counts().index[0]
        else:
            return {self.getKey(): [self.left_child.getValue(), self.right_child.getValue()]}
    
    def isLeaf(self):
        return self.leaf
    
    def be_leaf(self): #turn to a leaf
        this_node = copy.copy(self)
        left_node = None
        right_node = None
        this_node.set_left_child(left_node)
        this_node.set_right_child(right_node)
        this_node.leaf = True
        return this_node
    
    def prune_by_level(self, max_depth):
        if self.leaf:
            return copy.copy(self)
        elif self.level == max_depth:
            return self.be_leaf()
        else:
            this_node = copy.copy(self)
            left_node = this_node.left_child.prune_by_level(max_depth)
            right_node = this_node.right_child.prune_by_level(max_depth)
            this_node.set_left_child(left_node)
            this_node.set_right_child(right_node)
            return this_node


# In[1774]:


class DecisionTreeRegressor:
    def __init__(self, max_depth = 0, ml_task='classification'):
        self.data = None
        self.target = None
        self.max_depth = 0
        self.tree = None
        self.depth = 0
        self.max_depth = max_depth
        self.dict_tree = None
        self.ml_task = ml_task
        
    def fit(self, x_train, y_train):
        self.data = x_train
        self.target = y_train
        self.tree = self.build_tree(self.data, self.target, 0, None)
        self.dict_tree = self.build_dict(self.tree)
    
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
    
    def build_dict(self, tree):
        ##build tree dictionary
        dict_tree = {tree.getKey() : [tree.left_child.getValue(), tree.right_child.getValue()]}
        return dict_tree
    
    def get_tree(self, max_depth=0):
        if max_depth == 0 or max_depth == self.depth:
            return self.tree, self.dict_tree
        else:
            pruned_tree = copy.copy(self.tree)
            pruned_tree = pruned_tree.prune_by_level(max_depth)
            dict_tree = self.build_dict(pruned_tree)
            return pruned_tree, dict_tree
    
    def get_dict_tree(self):
        return self.dict_tree

    def predict(self, x_test, tree=None):
        if tree == None:
            tree = self.tree
        
        if self.ml_task == 'classification':
            int_ml_task = 0 
        else:
            int_ml_task = 1
        y_pred = x_test.apply(tree.predict, args=(int_ml_task, ), axis=1)
        return y_pred
    
    def prune(self, df_val, label):
        df_train = self.data.copy()
        df_train[label] = self.target
        
        self.tree = self.post_pruning(df_train, df_val, label, tree=None)
    
    def determine_leaf(self, label):
        df_train = self.data.copy()
        df_train[label] = self.target
        
        if self.ml_task == 'regression':
            return df_train[label].mean()
        else:
            #take the most numberous leaf
            return df_train[label].value_counts().index[0]
    
    def determine_errors(self, label, df_val, tree):
        actual_values = df_val[label]
        predictions = self.predict(df_val, tree)
        
        if self.ml_task == 'regression':
            return ((actual_values - predictions) ** 2).mean()
        else:
            return sum(actual_values != predictions)
    
    def post_pruning(self, df_train, df_val, label, tree=None):
        if tree == None:
            tree = self.tree
        pruned_tree = copy.copy(tree)

        yes_answer = pruned_tree.left_child
        no_answer = pruned_tree.right_child

        #its child right below is leaf
        if yes_answer.isLeaf() and no_answer.isLeaf():
            leaf = self.determine_leaf(label)
            
            errors_leaf = self.determine_errors(label, df_val, pruned_tree.be_leaf())
            errors_decision_node = self.determine_errors(label, df_val, pruned_tree)
            print('Mse leaf [',  errors_leaf, '] <= Mse tree [' , errors_decision_node, '] ?', sep='')
            if errors_leaf <= errors_decision_node:
                print('Staring')
                pruned_tree.display()
                print('pruning . . .')
                pruned_tree.be_leaf().display()
                print('Done.')
                return pruned_tree.be_leaf()
            else: 
                return pruned_tree
        else:
            feature = pruned_tree.label
            value = pruned_tree.mean
            df_train_yes = df_train[df_train[feature] <= value]
            df_train_no = df_train[df_train[feature] > value]
            df_val_yes = df_val[df_val[feature] <= value]
            df_val_no = df_val[df_val[feature] > value]
            if len(df_train_yes) == 0 or len(df_train_no) == 0 or len(df_val_yes) == 0 or len(df_val_no) == 0:
                    return pruned_tree
            
            if not yes_answer.isLeaf():
                yes_answer = self.post_pruning(df_train_yes, df_val_yes, label, yes_answer)
                pruned_tree.set_left_child(yes_answer)
            if not no_answer.isLeaf():
                no_answer = self.post_pruning(df_train_no, df_val_no, label, no_answer)
                pruned_tree.set_right_child(no_answer)
            return pruned_tree


# In[1775]:


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


# In[1776]:


def write_object_to_file(the_object, file_name):
    with open(file_name, 'wb') as output:
        pickle.dump(the_object, output, pickle.HIGHEST_PROTOCOL)


# In[1777]:


def read_object_from_file(file_name):
    with open(file_name, 'rb') as input:
        the_object = pickle.load(input)
    return the_object


# In[1778]:


def writeCsv(dataframe, filename):
    dataframe.to_csv(filename)


# In[1779]:


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


# In[1780]:


def mean_squared_error(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2) / len(y_true)


# In[1781]:


def split(df, label):
    x = df.drop(columns = [label])
    y = df[label]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.3, random_state=10, shuffle=True)
    x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, 
                                                    test_size=0.5, random_state=10, shuffle=False)
    return x_train, y_train, x_test, y_test, x_valid, y_valid 


# In[1782]:


def create_plot(the_column, dr, label, kind_int=0):
    the_column = the_column.name
    if the_column == label: 
        return
    df = dr.data.copy()
    df[label] = dr.target
    x = df.sort_values(by=the_column)
    y = x[label]
    
    if kind_int == 0:
        plt.plot(x[the_column], y, color='green')
    else:
        plt.scatter(x[the_column], y, color='green')
    plt.xlabel(the_column)
    plt.ylabel(label)
    plt.show()
    plt.close() 


# In[1783]:


def show_plot(dr, label, kind='plot'):
    if kind=='plot':
        kind_int = 0
    else:
        kind_int = 1
    dr.data.apply(create_plot, args=(dr, label, kind_int), axis=0)


# In[1784]:


def show_predictions_lines(dr, df_test, pruned_tree, label):
    y_test = df_test[label]
    x_test = df_test.sort_values(by='Date')
    
    y_pred = dr.predict(x_test)
    pruned_y_pred = dr.predict(x_test, pruned_tree)
    ##
    mse = mean_squared_error(y_test, y_pred)
    print('mse =', mse)
    rmse_err = np.sqrt(mse)
    print('root mse =', rmse_err)
    
    print('mse after pruned =', mean_squared_error(y_test, pruned_y_pred))
    print('root mse after pruned =', np.sqrt(mean_squared_error(y_test, pruned_y_pred)))
    ##
    #plot_df = pd.DataFrame({'actual':y_test,'predictions':y_pred})
    #plot_df = pd.DataFrame({'actual':y_test,'pruned_predictions':pruned_y_pred})
    plot_df = pd.DataFrame({'actual':y_test,'predictions':y_pred,'pruned_predictions':pruned_y_pred})
    plot_df.plot(figsize=(150,6), color=['black', '#66c2a5', '#fc8d62'], style=['-', '-', '--'])


# In[1785]:


def train_with_DecisionTreeRegressor(x_train, y_train, visualize=True):                                           
    #print(x_train)
    #print(y_train)
    
    dr = DecisionTreeRegressor(max_depth = 0, ml_task='regression')
    dr.fit(x_train, y_train)
    
    write_object_to_file(dr, 'trained_data.pkl')


# In[1786]:


label='Rented_Bike_Count'
df = read_data('SeoulBikeData.csv', label, opt=2)
df.columns = df.columns.str.replace(r'[^a-zA-Z0-9]', '_', regex=True)


# In[1787]:


x_train, y_train, x_test, y_test, x_valid, y_valid = split(df, label)


# In[1788]:


from sklearn import linear_model

lm = linear_model.LinearRegression()
lm.fit(x_train, y_train)

df_test = x_test.copy()
df_test[label] = y_test
x_test = df_test.sort_values(by='Date')
y_test = x_test[label]
x_test = df_test.drop(columns=[label])

y_pred = lm.predict(x_test)
    
##mse
err = mean_squared_error(y_test, y_pred)
print('mse =', err)
##root mse = sqrt(mse)
rmse_err = np.sqrt(err)
print('root mse =', round(rmse_err, 3))
    
plot_df = pd.DataFrame({'actual':y_test,'predictions':y_pred})
plot_df.plot(figsize=(150,6), color=['black', '#66c2a5'], style=['-', '--'])


# In[1789]:


from sklearn.tree import DecisionTreeRegressor as DTR
dtr = DTR()

dtr.fit(x_train, y_train)

df_test = x_test.copy()
df_test[label] = y_test
x_test = df_test.sort_values(by='Date')
y_test = x_test[label]
x_test = x_test.drop(columns=[label])

y_pred = dtr.predict(x_test)

err = mean_squared_error(y_test, y_pred)
print('mse =', err)
##root mse = sqrt(mse)
rmse_err = np.sqrt(err)
print('root mse =', round(rmse_err, 3))

plot_df = pd.DataFrame({'actual':y_test,'predictions':y_pred})
plot_df.plot(figsize=(150,6), color=['black', '#66c2a5'], style=['-', '--'])


# In[1790]:


dr = read_object_from_file('trained_data.pkl')
dr.ml_task = 'regression'

df_test = x_test.copy()
df_test[label] = y_test
x_test = df_test.sort_values(by='Date')
y_test = x_test[label]
x_test = x_test.drop(columns=[label])

y_pred = dr.predict(x_test)

err = mean_squared_error(y_test, y_pred)
print('mse =', err)
##root mse = sqrt(mse)
rmse_err = np.sqrt(err)
print('root mse =', round(rmse_err, 3))

plot_df = pd.DataFrame({'actual':y_test,'predictions':y_pred})
plot_df.plot(figsize=(150,6), color=['black', '#66c2a5'], style=['-', '--'])


# In[1791]:


#train_with_DecisionTreeRegressor(x_train, y_train, visualize=True)
    
dr = read_object_from_file('trained_data.pkl')
dr.ml_task = 'regression'
#dr.display()
tree = dr.get_dict_tree()
#pprint(dr.get_dict_tree())


# In[1792]:


df_train = x_train.copy()
df_train[label] = y_train
df_val = x_valid.copy()
df_val[label] = y_valid
df_test = x_test.copy()
df_test[label] = y_test

pruned_tree_by_level, pruned_dict_by_level = dr.get_tree(max_depth=0)
pprint(pruned_dict_by_level)


# In[1793]:


pruned_tree = dr.post_pruning(df_train, df_val, label, tree=None)


# In[1794]:


dr.predict(df_test)


# In[1795]:


dr.predict(df_test, pruned_tree)


# In[1796]:


show_predictions_lines(dr, df_test, pruned_tree, label)


# In[1797]:


#dr.prune(df_val, label, ml_task='regression')


# In[1798]:


plt.close() 
show_plot(dr, label, kind='scatter')


# In[1799]:


df_train = x_train.copy()
df_train[label] = y_train
df_val = x_valid.copy()
df_val[label] = y_valid
   
y_pred = dr.predict(x_test)
#print(y_pred)
#for val in y_pred:
#    print(val)
       
#show_predictions_lines(dr, x_test, y_test)
    
#show_plot(dr, label)

