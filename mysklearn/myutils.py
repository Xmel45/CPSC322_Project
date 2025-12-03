from mysklearn.mypytable import MyPyTable 
from mysklearn import myevaluation

"""
Programmer: Xavier Melancon
Class: CPSC 322-01 Fall 2025
Programming Assignment #6
Description: This program is a helper program for PA6 """

def normalize_column(col):
    """Normalizes a columns value
        Args:
            col (list of obj): Column to be normalized
        Returns:
            new_col: Column of normalized values
    
    """
    new_col = []
    for x in col:
        x = (x-min(col))/(max(col)-min(col))
        new_col.append(x)
    return new_col



def labels_poslabel(labels,pos_label,y_true):
    if labels is None:
        labels = list(set(y_true))
    if pos_label is None:
        pos_label = labels[0]
    return(labels,pos_label)

def load_titanic():
    mp = MyPyTable()
    mp.load_from_file("input_data/titanic.csv")
    y = mp.get_column("survived")
    x_data = list(zip((mp.get_column("class")),(mp.get_column("age")),(mp.get_column("sex"))))
    return x_data,y


def scores(X,y,classifier,folds):
    f1 = []
    p = []
    r = []
    accs=[]
    all_y_test=[]
    all_y_pred = []
    for train_idx, test_idx in folds:
            # Train/Test Sets and Classifier Fit
            X_train = [X[i] for i in train_idx]
            y_train = [y[i] for i in train_idx]
            X_test = [X[i] for i in test_idx]
            y_test = [y[i] for i in test_idx]
            classifier.fit(X_train, y_train)

            # Prediction and Accuracy
            y_pred = classifier.predict(X_test)
            f = myevaluation.binary_f1_score(y_test,y_pred,pos_label='yes')
            prec = myevaluation.binary_precision_score(y_test,y_pred,pos_label='yes')
            rec = myevaluation.binary_recall_score(y_test,y_pred,pos_label='yes')
            f1.append(f)
            p.append(prec)
            r.append(rec)
            acc = myevaluation.accuracy_score(y_test, y_pred)
            accs.append(acc)
            all_y_test.extend(y_test)
            all_y_pred.extend(y_pred)
    confusion = myevaluation.confusion_matrix(all_y_test,all_y_pred, labels=["yes","no"]) 
   
    return f1,p,r,accs,confusion