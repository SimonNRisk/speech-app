import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve, validation_curve, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import mode
import soundfile as sf
import joblib
import os

#add more tuning to prevent overfitting.

#create the absolute path to the 'outputs' directory
current_directory = os.path.abspath(os.path.dirname(__file__))
outputs_directory = os.path.join(current_directory, 'outputs')
#ensure the 'outputs' directory exists
os.makedirs(outputs_directory, exist_ok=True)

print("Running...")
data = pd.read_csv('all_word_audio_features.csv')
data = data.drop(columns= ['filename','start','end']) #drops from data (removes)

words = ["three", "tree"]

data.shape

def check_duplicates_or_nulls(data):
    duplicates = data.duplicated().sum() > 0
    nulls = data.isnull().sum().sum() > 0
    return duplicates or nulls

if check_duplicates_or_nulls(data):
    print('Duplicates or null values found in data')
else:
    print('No duplicates or null values found in data')

#clean the 'tempo' column
data['tempo'] = data['tempo'].str.strip('[]').astype(float)

#verify the changes
data.describe(include='all')
#maybe the order matters, add randomizer
def apply_moving_average_filter(df, window_size=3):

    feature_columns = df.columns.difference(['word'])
    '''
    df.columns.difference returns a new Index with elements of index not in other.

    the purpose is to identify and extract all columns in your DataFrame df except for the column named 'word'.

    df.columns returns an Index object containing the column names of the DataFrame df. (Index(['tempo', 'pitch', 'loudness', 'word'], dtype='object'))

    The 'word' column  contains the labels / target classes (e.g., 'three' and 'tree'). I don't want this column to be part of the feature set because it represents what I'm trying to predict, not the features (inputs) used to make predictions.


    '''


    df[feature_columns] = df[feature_columns].rolling(window=window_size, min_periods=1).mean()
    return df

data = apply_moving_average_filter(data)

#split the data into features and target variable
X = np.array(data.iloc[:, :-1], dtype=float)
y = data.iloc[:, -1]

#split the dataset with 70% for training set and 30% for test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
print("Data split.")
#scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Features scaled")

#encode the target labels
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)
print("Encoded target labels")

def plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=None, save_path=None):
    plt.figure()
    plt.title(title)
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=np.linspace(.1, 1.0, 5))
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid()
    if save_path:
        plt.savefig(save_path)

    plt.show()

#define the model training and evaluation function with cross-validation
def train_and_evaluate_model(model, param_grid):
    grid_search = GridSearchCV(model, param_grid, cv=StratifiedKFold(n_splits=5), scoring='accuracy', error_score='raise')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    #cross-validation
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=StratifiedKFold(n_splits=5), scoring='accuracy')
    print(f'Cross-validation scores: {cv_scores}')
    print(f'Mean cross-validation score: {np.mean(cv_scores)}')

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=encoder.classes_)
    return best_model, accuracy, report


rf_param_grid = {
    'n_estimators': [100, 250, 1000],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
print("param grid set")

rf_model = RandomForestClassifier()
rf_best_model, rf_accuracy, rf_report = train_and_evaluate_model(rf_model, rf_param_grid)
print("Random Forest Accuracy:", rf_accuracy)
print("Random Forest Classification Report:\n", rf_report)

#create the save path for the plot in the 'outputs' directory
learning_curve_image_path = os.path.join(outputs_directory, 'learning_curve_rf.png')

#call the function and pass the path to save the image
plot_learning_curve(rf_best_model, "Learning Curves (Random Forest)", X, y, cv=5, save_path=learning_curve_image_path)

'''
Stops after this point, not sure why...'''
print('made it past sticking point')
#ensure the 'outputs' directory exists
os.makedirs('outputs', exist_ok=True)
print('made sure path of outputs exists')

joblib.dump(rf_best_model, os.path.join(outputs_directory, 'rf_best_model.pkl'))
joblib.dump(scaler, os.path.join(outputs_directory, 'scaler.pkl'))
joblib.dump(encoder, os.path.join(outputs_directory, 'encoder.pkl'))
print('dumped joblibs')


