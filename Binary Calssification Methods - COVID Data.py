
# Author: Ryan Kuhn
# Last Edited: 1/8/23
# Purpose: Demonstrate various classification methods, the performance, the pros and cons
#          Methods use: Naive Bayes, Decision Tree, Random Forest, K-Nearest Neighbor, Tensorflow Deep Neural Network

#Generic imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Libraries for calculating metrics, splitting the test/train data
from sklearn.metrics import mean_absolute_error, precision_score, accuracy_score
from sklearn.model_selection import train_test_split
# Libraries for each method
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
# Neural network
from IPython.display import display
from tensorflow import keras
from tensorflow.keras import layers


# ---------------- CSV Data --------------------------------------------------
# Define the file path and crease dataframe
file_path = r'C:\Users\KUHNR1\PycharmProjects\python_learning\venv\Covid Data.csv'
df = pd.read_csv(file_path)

# 1 Yes, 2 No, 97 or 99 No data
# USMER - Indicates whether the patient treated medical units of the first, second or third level.
# MEDICAL_UNIT - type of institution of the National Health System that provided the care.
# SEX - 1 Female, 2 Male
# PATIENT_TYPE - Type of care, 1 returned home, 2 for hospitalization
# DATE_DIED - If patient died, date
# INTUBED, PNEUMONIA, PREGNANT, DIABETES, COPD, ASTHMA, INMSUPR, HYPERTENSION
#     CARDIOVASCULAR, RENAL CHRONIC, OTHER DISEASE, OBESITY,
#     TOBACCO, ICU  - 1 Yes, 2 No
# AGE
# CLASSIFICATION - 1-3 COVID in diff degrees, 4+ not a carrier or inconclusive

df['DIED'] = np.where(df['DATE_DIED'] == '9999-99-99', 2, 1)
# If Male, set pregnant to false
# This is a debatable strategy because it may offset the effect of pregnancy
# The alternative is to use a value in pregnancy column to denote male
# Or remove all NaN rows
df.loc[df['SEX'] == 2, 'PREGNANT'] = 2

# Change the coding system, 0 Yes, 1 No, 97 & 99 NaN
re_map_dict = {97: np.nan, 98: np.nan, 99: np.nan, 1: 0, 2: 1}
df.replace(re_map_dict, inplace=True)

# Scale the age column using normalization
# This technique preserves the distribution and shifts values to a 0 to 1 scale
df['AGE'] = (df['AGE'] - df['AGE'].min()) / (df['AGE'].max() - df['AGE'].min())
# Also drop INTUBED AND ICU because those columns have a high percentage of NAN values
df.drop(['INTUBED', 'ICU'], axis=1, inplace=True)
# Drop rows with NaN values
df.dropna(axis=0, inplace=True)

y = df['DIED']
features_to_drop = ['CLASIFFICATION_FINAL', 'DATE_DIED', 'MEDICAL_UNIT', 'PATIENT_TYPE', 'USMER', 'DIED']
X = df.drop(features_to_drop, axis=1)

# Split data inton training and validation data
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=0)

# ------------------- Preliminary visualizations ----------------------
def visualization(X):
    binary_features = ['SEX', 'AGE', 'PREGNANT', 'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR', 'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY', 'RENAL_CHRONIC', 'TOBACCO']
    binary_feat_series = X.drop("AGE", axis=1)
    count_df = binary_feat_series.apply(pd.value_counts)
    count_df.plot(kind='pie', subplots=True, autopct='%1.1f%%', startangle=270, fontsize=8, layout=(2,7), figsize=(30,30))
    display()

#visualization(X)


# Precision or PPV metric
# This tells ratio of the number of deaths vs the total predicted deaths
# Measure of how much we can trust our positive prediction
def acc_and_prec(test_X, test_y, model):
    pred_y = model.predict(test_X)
    acc_score = accuracy_score(test_y, pred_y)
    ppv = precision_score(test_y, pred_y)

    return acc_score, ppv


# ------------------- NAIVE BAYES --------------------------------------
def naive_bayes(train_X, test_X, train_y, test_y):
    model = MultinomialNB().fit(train_X, train_y)
    score, ppv = acc_and_prec(test_X, test_y, model)
    train_score, train_ppv = acc_and_prec(train_X, train_y, model)

    # .score returns the mean accuracy from the fit
    print("Naive Bayes train data:\t Accuracy %.3f%%" % (train_score*100))
    print("Naive Bayes test data:\t Accuracy %.3f%% \t Precision %.2f%%" % (score*100, ppv*100))

#naive_bayes(train_X, test_X, train_y, test_y)
# Test and training scores are similar, indicating it is not over fit

# ------------------- DECISION TREE CLASSIFIER METHOD --------------------

def decision_tree(train_X, test_X, train_y, test_y):
    model = DecisionTreeClassifier(random_state=1)
    # Fit Model
    model.fit(train_X, train_y)
    score, ppv = acc_and_prec(test_X, test_y, model)

    # Print the MAE
    print("Decision Tree: \t Accuracy %.2f%% \t Precision %.2f%%" % (score*100, ppv*100))
    # Both the accuracy and prediction are reasonable

#decision_tree(train_X, test_X, train_y, test_y)


# -------------------- RANDOM FOREST METHOD ------------------------------------------
def plot_rand_tree_feature_importance(train_X, model):
    importance = model.feature_importances_
    feature_names = train_X.columns
    forest_importances = pd.Series(importance, index=feature_names)
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    plt.show()

# Uses many trees and averages over results
def random_trees(train_X, test_X, train_y, test_y):
    model = RandomForestClassifier(random_state=1)
    model.fit(train_X, train_y)
    score, ppv = acc_and_prec(test_X, test_y, model)
    print("Random Tree without specifying max leaf nodes:\t Accuracy %.3f%% \t Precision %.2f%%" % (score*100, ppv*100))

    #plot_rand_tree_feature_importance(train_X, model)


#random_trees(train_X, test_X, train_y, test_y)


# ---------------------- RANDOM FORESTS - TEST DIFFERENT MAX LEAF SIZE ----------------
def get_acc(max_leaf_nodes, train_X, test_X, train_y, test_y):
    model = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    score, ppv = acc_and_prec(test_X, test_y, model)
    return score

def random_trees_test_max_leaf(train_X, test_X, train_y, test_y):
    candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500, 1000, 10000, 50000]
    # Write loop to find the ideal tree size from candidate_max_leaf_nodes
    acc_dict = dict.fromkeys(candidate_max_leaf_nodes)
    for max_leaf_nodes in candidate_max_leaf_nodes:
        my_acc = get_acc(max_leaf_nodes, train_X, test_X, train_y, test_y)
        acc_dict[max_leaf_nodes] = my_acc
        print("Max leaf nodes: %d  \t\t Accuracy:  %.2f" % (max_leaf_nodes, my_acc*100))

#random_trees_test_max_leaf(train_X, test_X, train_y, test_y)
# Changing the max leaf nodes did not appreciably affect the accuracy of the model
# Max leaf nodes 25-500 was optimal
# Consider changing the features used in the model


# ---------------------------- K-Nearest Neighbors -------------------------------------------
# Not Optimal for large data sets or with lots of features
# May need to reduce number of features for this method

def KNN(train_X, test_X, train_y, test_y):
    # Reduce the number of features to the following list in order to run in reasonable timeframe
    new_features = ['AGE', 'PNEUMONIA', 'HIPERTENSION']
    train_X = train_X[new_features]
    test_X = test_X[new_features]
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(train_X, train_y)
    score, ppv = acc_and_prec(test_X, test_y, model)
    print("KNN:\t Accuracy %.3f%% \t Precision %.2f%%" % (score * 100, ppv * 100))

#KNN(train_X, test_X, train_y, test_y)
# No better performance than other methods but much slower


# ------------------------ TensorFlow Deep Learning Neural Network ----------------------------
def tensorflow_deep_learning(train_X, test_X, train_y, test_y):
    input_shape = train_X.shape[1]
    # Create the model
    # Set number of units to between #inputs and #outputs, rule of thumb
    # relu activation - rectifier + linear unit
    # Classification may require activation on output
    model = keras.Sequential([
        layers.Dense(8, activation='relu', input_shape=[input_shape]),
        layers.Dense(8, activation='relu'),
        #layers.Dense(1, activation='sigmoid'),
        layers.Dense(1),
    ])

    # adam - adaptive learning rate
    # binary accuracy due to binary classification
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['binary_accuracy'],
    )

    # Early stopping to prevent over/under fitting
    early_stopping = keras.callbacks.EarlyStopping(
        patience=10,
        min_delta=0.001,
        restore_best_weights=True,
    )

    # Save history
    history = model.fit(
        train_X, train_y,
        validation_data=(test_X, test_y),
        batch_size=2048,
        epochs=1000,
        callbacks=[early_stopping],
        verbose=0,  # hide the output because we have so many epochs
    )

    history_df = pd.DataFrame(history.history)
    # Start the plot at epoch 5
    history_df.loc[5:, ['loss', 'val_loss']].plot()
    history_df.loc[5:, ['binary_accuracy', 'val_binary_accuracy']].plot()

    print(("Best Validation Loss: {:0.4f}" + \
           "\nBest Validation Accuracy: {:0.4f}") \
          .format(history_df['val_loss'].min(),
                  history_df['val_binary_accuracy'].max()))

tensorflow_deep_learning(train_X, test_X, train_y, test_y)



