import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from loguru import logger
import random
from joblib import dump
import argparse
import json
from timeit import default_timer as timer
import logging

def prepare_features(files):
    '''
    Prepare features and labels for a binary classification task based on a dataset of files.

    Parameters:
    - files (list of str): A list of file paths representing the dataset. Each file is associated with a data sample.
    '''

    logger.info(f"--Feature preparation for training")
    selfeatures1 = [file for file in files if "_gas_" in file.split('/',-1)[-1]]
    selfeatures1.sort()
    selfeatures2 = [file for file in files if "_nongas_" in file.split('/',-1)[-1]]
    random.shuffle(selfeatures2)
    selfeatures2 = selfeatures2[:len(selfeatures1)]
    selfeatures = selfeatures1 + selfeatures2

    X = np.asarray([np.load(feature, mmap_mode='r')['arr_0'].flatten() for feature in selfeatures])
    y = np.asarray([1] * len(selfeatures1) + [0] * len(selfeatures2))

    return X, y

def train_sgdclassifier_partial(files, classifier_params, batch_size):  
    '''
    Incremental training of an SGD (Stochastic Gradient Descent) classifier with specified hyperparameters.

    Parameters:
    - files (list of str): A list of file paths representing the dataset. Each file is associated with a data sample.
    - classifier_params (dictionary): Best parameters found from grid search.
    '''

    logger.info("-Incremental training SGD classifier")
    sgdclassifier = SGDClassifier()
    sgdclassifier.set_params(**classifier_params)
    
    gas_files = [file for file in files if "_gas_" in file.split('/',-1)[-1]]
    gas_files.sort()
    nongas_files = [file for file in files if "_nongas_" in file.split('/',-1)[-1]]
    random.shuffle(nongas_files)
    nongas_files = nongas_files[:len(gas_files)]
    
    for i in range(0, len(gas_files), batch_size):
        batch_gas, batch_nongas = gas_files[i : i + batch_size], nongas_files[i : i + batch_size]
        allfeatures = batch_gas + batch_nongas

        X = np.asarray([np.load(feature, mmap_mode='r')['arr_0'].flatten() for feature in allfeatures])
        y = np.asarray([1] * len(batch_gas) + [0] * len(batch_nongas))

        sgdclassifier.partial_fit(X, y, classes=np.unique(y))

    return sgdclassifier

def train_sgdclassifier(features, classifier_params):  
    '''
    Train an SGD (Stochastic Gradient Descent) classifier with specified hyperparameters.

    Parameters:
    - features (list of str): A list of file paths representing the dataset.
    - classifier_params (dictionary): Best parameters found from grid search.
    '''

    X_train, y_train = prepare_features(features)

    logger.info("-Training SGD classifier")
    sgdclassifier = SGDClassifier()
    sgdclassifier.set_params(**classifier_params)
    sgdclassifier.fit(X_train, y_train)

    return sgdclassifier

def train_adaboost(features, classifier_params, n_estimators=100):
    '''
    Train an AdaBoost classifier using a base classifier and specified number of estimators.

    Parameters:
    - features (list of str): A list of file paths representing the dataset.
    - classifier_params (dictionary): Best parameters found from grid search.
    - n_estimators (int): The number of boosting stages (estimators) in the AdaBoost ensemble.
    '''

    logger.info("-Training Adaboost classifier")
    X_train, y_train = prepare_features(features)

    logger.info("-Using an SGD classifier as base estimator")
    from sklearn.tree import DecisionTreeClassifier
    base_classifier =  DecisionTreeClassifier(max_depth=1) #SGDClassifier()
    # base_classifier.set_params(**classifier_params)

    adaboost_classifier = AdaBoostClassifier(base_estimator=base_classifier, n_estimators=n_estimators, algorithm='SAMME')
    logger.info("--Starting fitting data to classifier")
    adaboost_classifier.fit(X_train, y_train)

    return adaboost_classifier

def train_randomforest(features):
    '''
    Train a Random Forest classifier using the parameters found from grid search.

    Parameters:
    - features (list of str): A list of file paths representing the dataset.
    - classifier_params (dictionary): Best parameters found from grid search.
    '''

    logger.info("-Training RandomForest classifier")

    X_train, y_train = prepare_features(features)

    rf_classifier = RandomForestClassifier()
    # rf_classifier.set_params(**classifier_params)
    rf_classifier.fit(X_train, y_train)

    return rf_classifier

def train_gaussianNB(features):
    '''
    Train a Gaussian Naive Bayes classifier.

    Parameters:
    - features (list of str): A list of file paths representing the dataset.
    '''
    logger.info("-Training Gaussian Naive Bayes classifier")

    X_train, y_train = prepare_features(features)

    gnb_classifier = GaussianNB()
    gnb_classifier.fit(X_train, y_train)

    return gnb_classifier

def calculate_metrics(clf, X_valid, y_valid, metrics_file):
    '''
    Calculate various classification metrics for a given classifier's predictions.

    Parameters:
    - clf: Classifier model for which metrics need to be calculated.
    - X_valid (numpy.ndarray): Features of the validation dataset.
    - y_valid (numpy.ndarray): True labels for the validation dataset.
    '''
    
    logger.info(f"-Metrics Calculation")
    y_pred = clf.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_pred)
    logger.info(f"Accuracy: {accuracy:.2f}")
    precision = precision_score(y_valid, y_pred, average='macro', zero_division=1)
    logger.info(f"Precision: {precision:.2f}")
    recall = recall_score(y_valid, y_pred, average='macro', zero_division=1)
    logger.info(f"Recall: {recall:.2f}")
    f1 = f1_score(y_valid, y_pred, average='macro', zero_division=1)
    logger.info(f"F1 Score: {f1:.2f}")
    roc_auc = roc_auc_score(y_valid, y_pred)
    logger.info(f"ROC AUC Score: {roc_auc:.2f}")

    class_report = classification_report(y_valid, y_pred)
    logger.info("Classification Report:")
    logger.info(class_report)

    with open(metrics_file, 'w') as file:
        file.write(f"Accuracy: {accuracy:.2f}\n")
        file.write(f"Precision: {precision:.2f}\n")
        file.write(f"Recall: {recall:.2f}\n")
        file.write(f"F1 Score: {f1:.2f}\n")
        file.write(f"ROC AUC Score: {roc_auc:.2f}\n")
        file.write("Classification Report:\n")
        file.write(class_report)

def parse_args():
    parser = argparse.ArgumentParser(description="Model training")
    parser.add_argument("-p", "--path", type=str, required=True, help="Path to the generated features (Required)")
    parser.add_argument("-c", "--classifier", choices=["sgd", "sgd_partial", "adaboost", "rf", "gnb"], nargs="+", default=["sgd"], required=True, help="Choose which classifier to train [sgd, sgd_partial, adaboost, rf, gnb]}")
    return parser.parse_args()

def main():
    time_script_start = timer()
    args = parse_args()
    path = Path(args.path)
    path_train = path / "training"
    path_test = path / "test"

    trainfeatures = [str(file) for file in Path(path_train).rglob('*') if file.is_file()]
    testfeatures = [str(file) for file in Path(path_test).rglob('*') if file.is_file()]

    with open('/home/antonkout/Documents/modules/flammable_gas_detection/release/parameters/classifier_params.json', 'r') as json_file:
        classifier_params = json.load(json_file)

    if args.classifier[0] == 'sgd': 
        classifier = train_sgdclassifier(trainfeatures, classifier_params)   
    elif args.classifier[0] == 'sgd_partial':
        classifier = train_sgdclassifier_partial(trainfeatures, classifier_params, batch_size=50)
    elif args.classifier[0] == 'adaboost':
        classifier = train_adaboost(trainfeatures, classifier_params, n_estimators=50)
    elif args.classifier[0] == 'rf':
        classifier = train_randomforest(trainfeatures)
    elif args.classifier[0] == 'gnb':
        classifier = train_gaussianNB(trainfeatures)

    X_test, y_test =  prepare_features(testfeatures)
    logger.info(f'{args.classifier[0]} classifier metrics:')
    
    outfolder = Path("/home/antonkout/Documents/modules/flammable_gas_detection/release/classifier")
    metrics_file = f'classifier_{str(args.path).split("/",-1)[-1]}_{args.classifier[0]}_metrics.txt'
    calculate_metrics(classifier, X_test, y_test,  outfolder / metrics_file)

    clf_name = f'classifier_{str(args.path).split("/",-1)[-1]}_{args.classifier[0]}.joblib'
    dump(classifier, outfolder / clf_name)
    time_script_end = timer()

    logger.debug(
        "---Execution time:\t%2.2f minutes" % np.round((time_script_end - time_script_start) / 60, 2))

if __name__ == "__main__":
    main()