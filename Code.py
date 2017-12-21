import sklearn as sk;
import numpy as np;
import pandas as pd;
from sklearn.model_selection import KFold;
from sklearn import preprocessing;
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn import tree;
from sklearn.naive_bayes import GaussianNB;
from sklearn import linear_model, datasets;
from sklearn import neighbors;
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm

def load_data(dloc):
#List of data points
    data_list = []

#Column labels
    col_labels= ["buying","maint","doors","persons","lug_boot","safety","purchase"]
    # Open the data file
    with open(dloc, "r+") as f:
        # Process each line in the data file
        for line in f:
            # Clean the CSV data line
            line = line.replace(" ","");
            line = line.replace(".","");
            line = line.replace("\"","");
            line = line.strip().split(",");
            # Add the list containing the datapoint on this in the CSV to the
            # list containing all data points
            data_list.append(line);
#Creating a dataframe with the data
    df=pd.DataFrame(data=data_list, columns=col_labels);
    return df;

def preprocess_data(df):
    df = df[df.buying != ""]  # removing empty rows
    df["buying"] = df["buying"].astype('str');
    df["maint"] = df["maint"].astype('str');
    df["doors"] = df["doors"].astype('str');
    df["persons"] = df["persons"].astype('str');
    df["lug_boot"] = df["lug_boot"].astype('str');
    df["safety"] = df["safety"].astype('str');
    
    #converting multivariate target variable into a dichotomous variable
    df.purchase[df.purchase=="unacc"]="no";
    df.purchase[df.purchase=="acc"]="yes";
    df.purchase[df.purchase=="good"]="yes";
    df.purchase[df.purchase=="vgood"]="yes";
    
    
    # Creating a label column for each of the categorical variables
    le = preprocessing.LabelEncoder()
    
    le.fit(df['buying']);
    df['buying'] = le.transform(df['buying']);

    le.fit(df['maint']);
    df['maint'] = le.transform(df['maint']);

    le.fit(df['doors']);
    df['doors'] = le.transform(df['doors']);

    le.fit(df['persons']);
    df['persons'] = le.transform(df['persons']);

    le.fit(df['lug_boot']);
    df['lug_boot'] = le.transform(df['lug_boot']);

    le.fit(df['safety']);
    df['safety'] = le.transform(df['safety']);
    
    le.fit(df['purchase']);
    df['purchase'] = le.transform(df['purchase']);
    
    return df;

if __name__ == "__main__":
    # The data directory
    data_loc="C:\\Users\\Sharad\\Desktop\\AML\\AMLProject\\car.csv";
#    data_loc1="C:\\Users\\Sharad\\Desktop\\AML\\adult.test";
    # Load the data into memory
    df=load_data(data_loc);
    df1=preprocess_data(df);
    #train = preprocess_data(df);
    train, test = train_test_split(df1, test_size = 0.3)#attributes to be selected for training
    train=train.reset_index(); #resets the index
    test=test.reset_index();
    col_labels= ["buying","maint","doors","persons","lug_boot","safety"]
    trainX=train[col_labels];
    #target variable
    trainY=train["purchase"];
    testX=test[col_labels];
    testY=test["purchase"];
   

def learn_naive_bayes(X, Y):
    best_model = [ None, float("-inf") ];
    # Create the object that will split the training set into training and
    # validation sets
    kf = KFold(n_splits=10);
    col_labels= ["buying","maint","doors","persons","lug_boot","safety"]
    #attributes to be selected for training, can be customized for each algorithm
    # Iterate over each of the 10 splits on the data set
    for train, test in kf.split(X):
        # Pull out the records and labels that will be used to train this model     
        train_X=X.loc[train,col_labels];
        train_Y=Y.loc[train];
        valid_X =X.loc[test,col_labels];
        valid_Y =Y.loc[test];    # Create the Naive Bayes object
        clf = GaussianNB();
        # Learn the model on the training data that will be used for this fold
        clf = clf.fit(train_X, train_Y);
        # Evaluate the learned model on the validation set
        accuracy = clf.score(valid_X, valid_Y);
        # Check whether or not this learned model is the most accuracy model
        if accuracy > best_model[1]:
            # Update best_model so that it holds this learned model and its
            # associated accuracy and hyper-parameter information
            best_model = [ clf, accuracy ];
    return best_model;
    

best_naive_bayes = learn_naive_bayes(trainX, trainY);
print("Accuracy of the model using Naive Bayes: ",best_naive_bayes[1]);
naive_bayes_accuracy = best_naive_bayes[0].score(testX, testY);
print("Accuracy on test data using Naive Bayes: ",naive_bayes_accuracy);
    


def learn_logistic_regression(X, Y):
    best_model = [ None, float("-inf") ];
    # Create the object that will split the training set into training and
    # validation sets
    kf = KFold(n_splits=10);
    #attributes to be selected for training, can be customized for each algorithm
    col_labels= ["buying","maint","doors","persons","lug_boot","safety"]    #
    # Iterate over each of the 10 splits on the data set
    for train, test in kf.split(X):
        # Pull out the records and labels that will be used to train this model     
        train_X=X.loc[train,col_labels];
        train_Y=Y.loc[train];
        valid_X =X.loc[test,col_labels];
        valid_Y =Y.loc[test]; 
        # Create the regression tree object
        clf = linear_model.LogisticRegression();
        # Learn the model on the training data that will be used for this
        # fold
        clf = clf.fit(train_X, train_Y);
        # Evaluate the learned model on the validation set
        accuracy = clf.score(valid_X, valid_Y);
        # Check whether or not this learned model is the most accuracy model
        if accuracy > best_model[1]:
            # Update best_model so that it holds this learned model and its
            # associated accuracy and hyper-parameter information
            best_model = [ clf, accuracy ];
    return best_model;

best_logistic_regression = learn_logistic_regression(trainX, trainY);
print("\nAccuracy of the model using Logistic Regression: ",best_logistic_regression[1]);
logistic_regression_accuracy = best_logistic_regression[0].score(testX, testY);
print("Accuracy on test data using Logistic Regression: ",logistic_regression_accuracy);


def learn_decision_tree(X, Y):
    # This list tracks the learned decision tree with the best accuracy
    depths = [ 6,6,8,8,10,10,12,12,14,14];
    best_model = [ None, 0, float("-inf") ];
    # Create the object that will split the training set into training and
    # validation sets
    kf = KFold(n_splits=10);
    col_labels= ["buying","maint","doors","persons","lug_boot","safety"]    #
    # Iterate over each of the 10 splits on the data set
    for (train, test), cdepth in zip(kf.split(X),depths):
        # Pull out the records and labels that will be used to train this model     
        train_X=X.ix[train,col_labels];
        train_Y=Y.ix[train];
        valid_X =X.ix[test,col_labels];
        valid_Y =Y.ix[test];
        # Create the decision tree object
        clf = tree.DecisionTreeClassifier(max_depth=cdepth);
        # Learn the model on the training data that will be used for this
        # fold
        clf = clf.fit(train_X, train_Y);
        # Evaluate the learned model on the validation set
        accuracy = clf.score(valid_X, valid_Y);
        # Check whether or not this learned model is the most accuracy model
        if accuracy > best_model[2]:
            # Update best_model so that it holds this learned model and its
            # associated accuracy and hyper-parameter information
            best_model = [ clf, cdepth, accuracy ];
    return best_model;

best_decision_tree = learn_decision_tree(trainX, trainY);
print("\nAccuracy of the model using Decision Trees: ",best_decision_tree[2], "\nOptimum Depth: ",best_decision_tree[1]);
decision_tree_accuracy = best_decision_tree[0].score(testX, testY);
print("Accuracy on test data using Decision Trees: ",decision_tree_accuracy);

def learn_knn(X, Y):
    #an array of different k values to be tried on the training set
    k_values = [ 2,2,4,4,6,6,8,8,10,10];
    best_model = [ None, 0, float("-inf") ];
    # Create the object that will split the training set into training and
    # validation sets
    kf = KFold(n_splits=10);
    col_labels= ["buying","maint","doors","persons","lug_boot","safety"]     #
    # Iterate over each of the 10 splits on the data set
    for (train, test), k in zip(kf.split(X),k_values):
        # Pull out the records and labels that will be used to train this model     
        train_X=X.ix[train,col_labels];
        train_Y=Y.ix[train];
        valid_X =X.ix[test,col_labels];
        valid_Y =Y.ix[test];
        # Create the classifier object
        clf = neighbors.KNeighborsClassifier(n_neighbors=k);
        # Learn the model on the training data that will be used for this
        # fold
        clf = clf.fit(train_X, train_Y);
        # Evaluate the learned model on the validation set
        accuracy = clf.score(valid_X, valid_Y);
        # Check whether or not this learned model is the most accuracy model
        if accuracy > best_model[2]:
            # Update best_model so that it holds this learned model and its
            # associated accuracy and hyper-parameter information
            best_model = [ clf, k, accuracy ];
    return best_model;

best_knn = learn_knn(trainX, trainY);
print("\nAccuracy of the model using kNN: ",best_knn[2],"\nBest K value: ",best_knn[1]);
knn_accuracy = best_knn[0].score(testX, testY);
print("Accuracy on test data using kNN: ",knn_accuracy);



def learn_randomForest(X, Y):
    noOfTrees = [2,3,4,5, 6, 7, 8]
    best_model = [None, 0, float("-inf"), 0]
    depth = [4,6,8,10, 12, 14,16]
    # Create the object that will split the training set into training and
    # validation sets
    kf = KFold(n_splits=10);
    col_labels= ["buying","maint","doors","persons","lug_boot","safety"]    
    for(train, test), number, d in zip(kf.split(X), noOfTrees, depth):
        # Pull out the records and labels that will be used to train this model
        train_X = X.ix[train, col_labels]
        train_Y = Y.ix[train]
        valid_X = X.ix[test, col_labels]
        valid_Y = Y.ix[test]

        clf = RandomForestClassifier(n_estimators = number)
        clf = clf.fit(train_X, train_Y)
        accuracy = clf.score(valid_X, valid_Y)

        if(accuracy > best_model[2]):
            best_model = [clf, number, accuracy, d]
    return best_model

best_randomForest = learn_randomForest(trainX, trainY)
print("\nAccuracy of the model using Random Forests: ",best_randomForest[2],"\nOptimum no. of trees: ",best_randomForest[1],"\nOptimum depth: ",best_randomForest[3])
randomForestAccuracy = best_randomForest[0].score(testX, testY)
print("Accuracy on test data using Random Forests: ",randomForestAccuracy)

def learn_adaboost(X, Y):
    estimator = [50, 60, 70, 80, 90, 100, 110]
    #depth = []
    best_model = [None, 0, float("-inf")]
    # Create the object that will split the training set into training and
    # validation sets
    kf = KFold(n_splits=10);
    col_labels= ["buying","maint","doors","persons","lug_boot","safety"]    
    for(train, test), number in zip(kf.split(X), estimator):
        # Pull out the records and labels that will be used to train this model
        train_X = X.ix[train, col_labels]
        train_Y = Y.ix[train]
        valid_X = X.ix[test, col_labels]
        valid_Y = Y.ix[test]

        clf = AdaBoostClassifier(n_estimators = number)
        clf = clf.fit(train_X, train_Y)
        accuracy = clf.score(valid_X, valid_Y)

        if(accuracy > best_model[2]):
            best_model = [clf, number, accuracy]
    return best_model

best_adaboost = learn_adaboost(trainX, trainY)
print("\nAccuracy of the model using Adaboost: ",best_adaboost[2],"\nBest estimator value: ",best_adaboost[1])
adaboostAccuracy = best_adaboost[0].score(testX, testY)
print("Accuracy on test data using Adaboost: ",adaboostAccuracy)

def learn_SVM(X, Y):
    #an array of different k values to be tried on the training set
    k_values = [0.25,1,2,3,4,5];
    best_model = [ None, 0, float("-inf") ];
    # Create the object that will split the training set into training and
    # validation sets
    kf = KFold(n_splits=5);
    col_labels= ["buying","maint","doors","persons","lug_boot","safety"]      #
    # Iterate over each of the 10 splits on the data set
    for (train, test), k in zip(kf.split(X),k_values):
        # Pull out the records and labels that will be used to train this model     
        train_X=X.ix[train,col_labels];
        train_Y=Y.ix[train];
        valid_X =X.ix[test,col_labels];
        valid_Y =Y.ix[test];
        # Create the classifier object
        clf = svm.SVC(C=k);
        # Learn the model on the training data that will be used for this
        # fold
        clf = clf.fit(train_X, train_Y);
        # Evaluate the learned model on the validation set
        accuracy = clf.score(valid_X, valid_Y);
        # Check whether or not this learned model is the most accuracy model
        if accuracy > best_model[2]:
            # Update best_model so that it holds this learned model and its
            # associated accuracy and hyper-parameter information
            best_model = [ clf, k, accuracy ];
    return best_model;

best_SVM = learn_SVM(trainX, trainY);
print("\nAccuracy of the model using SVM: ",best_SVM[2],"\nBest K value: ",best_SVM[1] );
SVM_accuracy = best_SVM[0].score(testX, testY);
print("Accuracy on test data using SVM: ",SVM_accuracy);
