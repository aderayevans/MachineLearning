import pickle
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
from sklearn.model_selection import train_test_split


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
    
    def predict(self, x_test_instance_dict):
        if self.left_child != None and x_test_instance_dict[self.label] <= self.mean:
            return self.left_child.predict(x_test_instance_dict)
        elif self.right_child != None and x_test_instance_dict[self.label] > self.mean:
            return self.right_child.predict(x_test_instance_dict)
        else: #is a leaf
            #print('$$$', self.predict_data.values.tolist()[0])
            #self.parent.display()
            return self.predict_data.value_counts().index[0]
    
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


class DecisionTreeRegressor:
    def __init__(self, max_depth = 0):
        self.data = None
        self.target = None
        self.max_depth = 0
        self.tree = None
        self.depth = 0
        self.max_depth = max_depth
        self.dict_tree = None
        
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
    
    def predict_tree(tree, df_test):
        y_pred = df_test.apply(tree.predict, axis=1)
        return y_pred

    def predict(self, x_test, tree=None):
        if tree == None:
            tree = self.tree
        y_pred = x_test.apply(tree.predict, axis=1)
        return y_pred
    
    def prune(self, df_val, label, ml_task='classification'):
        df_train = self.data.copy()
        df_train[label] = self.target
        
        self.tree = self.post_pruning(df_train, df_val, label, tree=None, ml_task=ml_task)
    
    def determine_leaf(self, label, ml_task='classification'):
        df_train = self.data.copy()
        df_train[label] = self.target
        
        if ml_task == 'regression':
            return df_train[label].mean()
        else:
            #take the most numberous leaf
            return df_train[label].value_counts().index[0]
    
    def determine_errors(self, label, df_val, tree, ml_task='classification'):
        actual_values = df_val[label]
        predictions = self.predict(df_val, tree)
        
        if ml_task == 'regression':
            #mse
            return ((actual_values - predictions) ** 2).mean()
        else:
            return sum(actual_values != predictions)
    
    def post_pruning(self, df_train, df_val, label, tree=None, ml_task='classification'):
        if tree == None:
            tree = self.tree
        pruned_tree = copy.copy(tree)

        yes_answer = pruned_tree.left_child
        no_answer = pruned_tree.right_child

        #its child right below is leaf
        if yes_answer.isLeaf() and no_answer.isLeaf():
            leaf = self.determine_leaf(label, ml_task=ml_task)
            
            errors_leaf = self.determine_errors(label, df_val, pruned_tree.be_leaf(), ml_task=ml_task)
            errors_decision_node = self.determine_errors(label, df_val, pruned_tree, ml_task=ml_task)
            if errors_leaf <= errors_decision_node:
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

def show_plot(x_train, y_train, x_test, y_test, x_valid, y_valid, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    print('mse =', mse)
    rmse_err = np.sqrt(mse)
    print('root mse =', rmse_err)
    for each in x_test.columns:
        plt.scatter(x_train[each], y_train, color='blue')
        plt.scatter(x_test[each], y_test, color='red')
        plt.scatter(x_valid[each], y_valid, color='violet')
        plt.scatter(x_test[each], y_pred, color='green')
        plt.xlabel(each)
        plt.ylabel(y_test.name)
        plt.show()
        plt.close()

def show_predictions_lines(dr, x_test, y_test):
    df_test = x_test.copy()
    df_test[y_test.name] = y_test
    df_test.apply(create_plot, args=(dr, df_test, y_test.name), axis=0)

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

def split(df, label):
    x = df.drop(columns = [label])
    y = df[label]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.3, random_state=10, shuffle=True)
    x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, 
                                                    test_size=0.5, random_state=10, shuffle=False)
    return x_train, y_train, x_test, y_test, x_valid, y_valid 

def create_plot(the_column, dr, df_test, label):
    the_column = the_column.name
    if the_column == label: return
    x_test = df_test.sort_values(by=the_column)
    y_test = df_test[label]
            
    y_pred = dr.predict(x_test)
    
    #plot_df = pd.DataFrame({'actual':y_test,'predictions':y_pred})
    #plt.plot(x_test[the_column], plot_df, color='blue')
    
    plot_df = pd.DataFrame({'actual':y_test,'predictions':y_pred})
    plt.plot(x_test[the_column], plot_df)
    plt.show()
    plt.close() 
    
def train_with_DecisionTreeRegressor(x_train, y_train, visualize=True):                                           
    #print(x_train)
    #print(y_train)
    
    dr = DecisionTreeRegressor(max_depth = 0)
    dr.fit(x_train, y_train)
    
    write_object_to_file(dr, 'trained_data.pkl')


def main():
    label='Rented_Bike_Count'
    df = read_data('SeoulBikeData.csv', label, opt=2)
    df.columns = df.columns.str.replace(r'[^a-zA-Z0-9]', '_', regex=True)
    
    x_train, y_train, x_test, y_test, x_valid, y_valid = split(df, label)
    #train_with_DecisionTreeRegressor(x_train, y_train, visualize=True)
        
    dr = read_object_from_file('trained_data.pkl')
    #dr.display()
    tree = dr.get_dict_tree()
    #pprint(dr.get_dict_tree())
    
    df_train = x_train.copy()
    df_train[label] = y_train
    df_val = x_valid.copy()
    df_val[label] = y_valid

    #pruned_tree, pruned_dict = dr.get_tree(max_depth=2)
    #pprint(pruned_dict)
    
    show_predictions_lines(dr, x_test, y_test)
    dr.prune(df_val, label, ml_task='regression')
    show_predictions_lines(dr, x_test, y_test) 
    
    y_pred = dr.predict(x_test)
    #print(y_pred)
    #for val in y_pred:
    #    print(val)
       
    #show_predictions_lines(dr, x_test, y_test)
    
    #show_plot(x_train, y_train, x_test, y_test, x_valid, y_valid, y_pred)
 
if __name__ == "__main__":
    main()