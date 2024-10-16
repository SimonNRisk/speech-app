import pandas as pd
import numpy as np
import joblib
import os

def apply_moving_average_filter(df, window_size=3):
    feature_columns = df.columns.difference(['word'])
    df[feature_columns] = df[feature_columns].rolling(window=window_size, min_periods=1).mean()
    return df

#create the absolute path to the 'outputs' directory
current_directory = os.path.abspath(os.path.dirname(__file__))
outputs_directory = os.path.join(current_directory, 'outputs')
#ensure the 'outputs' directory exists
os.makedirs(outputs_directory, exist_ok=True)

#load external (test) data
external_data = pd.read_csv('file_features.csv')
external_data = external_data.drop(columns=['filename', 'start', 'end'])
print('loaded external data')

#clean the 'tempo' column
external_data['tempo'] = external_data['tempo'].str.strip('[]').astype(float)
print('cleaned tempo')

#apply the same moving average filter
external_data = apply_moving_average_filter(external_data)
print('applied same moving average filter')

#separate features
X_external = np.array(external_data.iloc[:, :], dtype=float)
print('seperated features')

#load scaler and transform features
scaler = joblib.load('outputs/scaler.pkl')
X_external = scaler.transform(X_external)
print('loaded scaler, transformed features')

rf_best_model = joblib.load('outputs/rf_best_model.pkl')
print('joblib of rf loaded')

#define a function to evaluate a model on unlabeled external data
def evaluate_model_on_external_data(model, X_ext):
    y_pred = model.predict(X_ext)
    return y_pred

#evaluate rf and gather predictions
rf_predictions = evaluate_model_on_external_data(rf_best_model, X_external)

print('evaluate model')

#create a DataFrame to hold the RF predictions
rf_predictions_df = pd.DataFrame({
    'RF': rf_predictions
})
print('dataframe made')

#map predicted numbers to word names
encoder = joblib.load('outputs/encoder.pkl')
rf_predictions_word = rf_predictions_df['RF'].astype(int).map(lambda x: encoder.classes_[x])
print('mapped predicted numbers to word names')

#save RF predictions per segment
rf_predictions_df_word = pd.DataFrame(rf_predictions_word, columns=['RF Predictions Per Segment'])
print('saved rf predictions per segment')

#output a singular prediction (majority vote)
singular_final_prediction = int(rf_predictions_df['RF'].mode()[0])
print('saved rf predictions per segment')

#map predicted number to word name
singular_final_prediction_word = encoder.classes_[singular_final_prediction]

#print the results
print("Random Forest Model Predictions:")
print(rf_predictions_df_word)
print("\nSingular Final Prediction:")
print(singular_final_prediction_word)

with open(os.path.join(outputs_directory, 'singular_final_prediction.txt'), 'w') as f:
    f.write(singular_final_prediction_word)

#print the path to verify
print(f"Files are saved to: {outputs_directory}")