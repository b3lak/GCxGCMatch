import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os


base_path = "/Users/caleb/Desktop/PurpleOrangeBlue/NonTargetSearch/"

# Paths and metadata for your Excel files
files_details = [
    ('87dR.xlsx', 87, 'Dilute'),
    ('87eR.xlsx', 87, 'Extracted'),
    ('89dR.xlsx', 89, 'Dilute'),
    ('89eR.xlsx', 89, 'Extracted'),
    ('91dR.xlsx', 91, 'Dilute'),
    ('91eR.xlsx', 91, 'Extracted'),
    ('10dR.xlsx', 10, 'Dilute'),
    ('10eR.xlsx', 10, 'Extracted')
]

unknown_file = "/Users/caleb/Desktop/PurpleOrangeBlue/TestFiles/0exR.xlsx"


def sanitize_compound_name(name, idx):
    if not isinstance(name, str) or not name:
        return f"Compound_{idx}"

    replacements = {
        ',': 'comma',
        '[': 'openbracket',
        ']': 'closebracket',
        '<': 'lessthan'
    }

    for char, replacement in replacements.items():
        name = name.replace(char, replacement)

    return name

def unsanitize_compound_name(name):
    replacements = {
        'comma': ',',
        'openbracket': '[',
        'closebracket': ']',
        'lessthan': '<'
    }

    for replacement, char in replacements.items():
        name = name.replace(replacement, char)

    return name

def organize_dataframe(is_unknown=False):
    all_data = []
    
    if is_unknown:
        data_files = [(unknown_file, None, None)]  # No octane rating or sample type for unknown
    else:
        data_files = files_details
    
    for file_name, octane_rating, sample_type in data_files:
        if is_unknown:
            df = pd.read_excel(file_name, sheet_name="Area")
        else:
            df = pd.read_excel(base_path + file_name, sheet_name="Area")
        
        df.fillna(0, inplace=True)  # Filling NaNs with zeros
        
        for col in df.columns[4:]:
            sample_data = {}
            if not is_unknown:
                sample_data = {
                    'Octane_Rating': octane_rating,
                    'Type': sample_type
                }
            
            compounds = []
            for idx, compound in enumerate(df['Compound_Results']):
                compounds.append(sanitize_compound_name(compound, idx))
            df['Compound_Results'] = compounds

            area_values = df[col].tolist()
            
            for idx, compound in enumerate(compounds):
                sample_data[f"{compound}_Area"] = area_values[idx]
                sample_data[f"{compound}_Binary"] = 1 if area_values[idx] > 0 else 0

            all_data.append(sample_data)
    
    return pd.DataFrame(all_data)  # Return the dataframe instead of setting global variable


def octane_predicter(df):
    global model_octane, octane_encoder
    octane_encoder = LabelEncoder()
    df['Octane_Rating_encoded'] = octane_encoder.fit_transform(df['Octane_Rating'])

    X = df.drop(columns=['Octane_Rating', 'Type', 'Octane_Rating_encoded'])
    y = df['Octane_Rating_encoded']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_octane = xgb.XGBClassifier()
    model_octane.fit(X_train, y_train)

    y_pred = model_octane.predict(X_test)
    print("Accuracy for Octane Rating:", accuracy_score(y_test, y_pred))


def type_predicter(df):
    global model_type, analysis_encoder
    analysis_encoder = LabelEncoder()
    df['Type_encoded'] = analysis_encoder.fit_transform(df['Type'])

    X = df.drop(columns=['Octane_Rating', 'Type', 'Type_encoded'])
    y = df['Type_encoded']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Adjusting the parameters
    model_type = xgb.XGBClassifier(
    gamma=0.01,
    alpha=0.1,                 # L1 regularization
    reg_lambda=1,              # L2 regularization
    max_depth=3,               # Maximum depth of a tree was three
    min_child_weight=1,        # Minimum sum of instance weight (hessian) needed in a child
    colsample_bytree=0.8,      # Fraction of columns to be randomly sampled for building each tree
    colsample_bylevel=0.8      # Fraction of columns to be randomly sampled for building each level
)


    model_type.fit(X_train, y_train)
    y_pred = model_type.predict(X_test)
    print("Accuracy for Type Rating:", accuracy_score(y_test, y_pred))


def display_top_features(model, df_sample):
    # Get importance scores
    importances = model.feature_importances_

    # Associate scores with feature names
    feature_importances = [(feature, score) for feature, score in zip(df_sample.columns, importances)]
    
    # Sort based on importance
    sorted_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    
    # Display top 20
    print("\nTop 20 Compounds for Prediction:")
    for i, (feature, importance) in enumerate(sorted_importances[:20]):
        # Extract the compound name without the '_Area' or '_Binary' suffix
        compound_name = feature.rsplit('_', 1)[0]
        unsanitized_name = unsanitize_compound_name(compound_name)
        print(f"{i + 1}. {unsanitized_name}: {importance:.4f}")


def predict_unknown_octane(df_sample):
    # Check if columns exist before attempting to drop
    drop_cols = [col for col in ['Octane_Rating', 'Type'] if col in df_sample.columns]
    df_sample = df_sample.drop(columns=drop_cols)
    
    global model_octane, octane_encoder
    predicted_octane_encoded = model_octane.predict(df_sample)
    predicted_octane = octane_encoder.inverse_transform(predicted_octane_encoded)

    predicted_probs = model_octane.predict_proba(df_sample)
    predicted_octane_prob = predicted_probs[0][predicted_octane_encoded[0]]
    print(f"Predicted Octane Rating: {predicted_octane[0]} {predicted_octane_prob*100:.2f}%")

    display_top_features(model_octane, df_sample)


def predict_unknown_type(df_sample, df_processed):
    # Check if columns exist before attempting to drop
    drop_cols = [col for col in ['Octane_Rating', 'Type'] if col in df_sample.columns]
    df_sample = df_sample.drop(columns=drop_cols)

    # Identify missing columns
    missing_cols = set(df_processed.columns) - set(df_sample.columns) - set(['Octane_Rating', 'Type', 'Type_encoded'])
    #print(f"Missing columns: {missing_cols}")
    
    # Fill in the missing columns with zeros
    for col in missing_cols:
        df_sample[col] = 0

    # Ensure column order matches training data
    df_sample = df_sample[df_processed.drop(columns=['Octane_Rating', 'Type', 'Type_encoded']).columns]

    global model_type, analysis_encoder
    predicted_type_encoded = model_type.predict(df_sample)
    predicted_type = analysis_encoder.inverse_transform(predicted_type_encoded)

    predicted_probs = model_type.predict_proba(df_sample)
    predicted_type_prob = predicted_probs[0][predicted_type_encoded[0]]

    print(f"Predicted Type: {predicted_type[0]} {predicted_type_prob*100:.2f}%")

    display_top_features(model_type, df_sample)


def main():
    # Training data
    df_processed = organize_dataframe()
    octane_predicter(df_processed)
    type_predicter(df_processed)

    # Unknown sample prediction
    if os.path.exists(unknown_file):
        df_sample = organize_dataframe(is_unknown=True)
        predict_unknown_octane(df_sample)
        predict_unknown_type(df_sample, df_processed)  # Pass df_processed here



if __name__ == "__main__":
    main()
