import pandas as pd
import joblib

# === Step 1: Load the saved (calibrated) XGBoost model ===
xgb_model = joblib.load("alzheimers_xgb_model.pkl")
print("XGBoost model loaded successfully!")

# === Step 2: Load new dataset & remove empty rows ===
new_data = pd.read_csv("new_patients.csv").dropna(how="all")
print(f"New data loaded: {new_data.shape[0]} rows, {new_data.shape[1]} columns")

# === Step 3: Predict probabilities ===
probabilities = xgb_model.predict_proba(new_data)[:, 1]  # Probability of Alzheimer's (class 1)

# === Step 4: Convert probabilities to percentages ===
percentages = (probabilities * 100).round(2)

# === Step 5: Apply threshold (default = 0.5) to get final prediction ===
predictions_numeric = (probabilities >= 0.5).astype(int)

# === Step 6: Map numeric predictions to labels ===
label_mapping = {0: "No Alzheimer's", 1: "Alzheimer's"}
predictions_labels = [label_mapping[p] for p in predictions_numeric]

# === Step 7: Add results to DataFrame ===
new_data['Alzheimers_Probability_%'] = percentages
new_data['Prediction'] = predictions_labels

# === Step 8: Save to Excel ===
output_file = "new_patients_predictions.xlsx"
new_data.to_excel(output_file, index=False)
print(f"Predictions with probabilities saved to '{output_file}'")

# === Step 9: Print summary for each patient ===
for i, (prob, label) in enumerate(zip(percentages, predictions_labels), start=1):
    print(f"Patient {i} → {prob}% chance Alzheimer’s (predicted: {label})")