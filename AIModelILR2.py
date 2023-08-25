import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import os
import matplotlib.pyplot as plt




base_path = "/Users/caleb/Desktop/ILRStationAnalysis/"

# Paths and metadata for your Excel files
files_details = [
    ('S1R.xlsx', 1, 'Parkdale Gas Co-op'),
    ('S2R.xlsx', 2, 'Esso'),
    ('S3R.xlsx', 3, 'Centex'),
    ('S4R.xlsx', 4, 'Petro Canada'),

    ('S6R.xlsx', 6, 'Shell'),
    ('S7R.xlsx', 7, 'Esso'),

    ('S9R.xlsx', 9, 'Canpro'),
    ('S10R.xlsx', 10, '7/11'),
    ('S11R.xlsx', 11, 'Co-op'),
    ('S12R.xlsx', 12, 'Husky'),
    ('S13R.xlsx', 13, 'Chevron'),
    ('S14R.xlsx', 14, 'Canadian Tire Gas+'),
    ('S15R.xlsx', 15, 'Mobil'),
    ('S16R.xlsx', 16, 'Fair Deal Gas Bar'),
    ('S17R.xlsx', 17, 'Co-op'),
    ('S18R.xlsx', 18, 'Domo'),

    ('S20R.xlsx', 20, 'Gas Plus'),
    ('S21R.xlsx', 21, '7/11'),
    ('S22R.xlsx', 22, 'Petro Canada'),
    ('S23R.xlsx', 23, 'Safeway Gas'),
    ('S24R.xlsx', 24, 'Canadian Tire Gas+'),
    ('S25R.xlsx', 25, 'G&B Fuels'),
    ('S26R.xlsx', 26, 'Costco'),
    ('S27R.xlsx', 27, 'Tsuu Tina Gas Stop'),
    ('S28R.xlsx', 28, 'Shell'),
    ('S29R.xlsx', 29, 'Chevron'),
    ('S30R.xlsx', 30, 'Safeway Gas'),
    ('S31R.xlsx', 31, 'Husky'),
    ('S32R.xlsx', 32, 'G&B Fuels'),
    ('S33R.xlsx', 33, 'Mobil')
]

unknown_file = "/Users/caleb/Desktop/ILRStationAnalysis/S19R.xlsx"


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
    
    for file_name, station_number, brand_type in data_files:
        # Load data from different sheets
        if is_unknown:
            areas = pd.read_excel(file_name, sheet_name="Area").fillna(0)
            heights = pd.read_excel(file_name, sheet_name="Height").fillna(0)
            signals = pd.read_excel(file_name, sheet_name="SignalToNoise").fillna(0)
            area_percs = pd.read_excel(file_name, sheet_name="Area%").fillna(0)
        else:
            areas = pd.read_excel(base_path + file_name, sheet_name="Area").fillna(0)
            heights = pd.read_excel(base_path + file_name, sheet_name="Height").fillna(0)
            signals = pd.read_excel(base_path + file_name, sheet_name="SignalToNoise").fillna(0)
            area_percs = pd.read_excel(base_path + file_name, sheet_name="Area%").fillna(0)

        # Removing rows with Carbon disulfide
        areas = areas[areas['Compound_Results'] != 'Carbon disulfide']
        heights = heights[heights['Compound_Results'] != 'Carbon disulfide']
        signals = signals[signals['Compound_Results'] != 'Carbon disulfide']
        area_percs = area_percs[area_percs['Compound_Results'] != 'Carbon disulfide']

        # Process each sample column
        for col in areas.columns[4:]:
            sample_data = {}
            if not is_unknown:
                sample_data = {
                    'Station_Rating': station_number,
                    'Brand': brand_type
                }
            
            # Sanitize compound names
            compounds = [sanitize_compound_name(comp, idx) for idx, comp in enumerate(areas['Compound_Results'])]
            areas['Compound_Results'] = compounds
            
            # Apply log10 normalization to the area values
            area_values = np.log10(areas[col] + 1).tolist()

            # Collect other data based on sample name
            height_col = col.replace("Area", "Height")
            signal_col = col.replace("Area", "Signal to Noise")
            area_perc_col = col.replace("Area", "Area %")

            for idx, compound in enumerate(compounds):
                sample_data[f"{compound}_Area"] = area_values[idx]
                sample_data[f"{compound}_Binary"] = 1 if area_values[idx] > 0 else 0
                sample_data[f"{compound}_Height"] = np.log10(heights[height_col].iloc[idx] + 1)
                #sample_data[f"{compound}_SignalToNoise"] = signals[signal_col].iloc[idx]
                #sample_data[f"{compound}_AreaPercent"] = area_percs[area_perc_col].iloc[idx]

            all_data.append(sample_data)

    return pd.DataFrame(all_data)

def model_tuning(X, y):
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'p': [1, 2, 3]
    }

    model = KNeighborsClassifier()
    
    stratified_kfold = StratifiedKFold(n_splits=3)
    grid_search = GridSearchCV(model, param_grid, cv=stratified_kfold, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X, y)

    return grid_search

def Station_predicter(df):
    global model_station, station_encoder, scaler_station
    
    station_encoder = LabelEncoder()
    df['Station_Rating_encoded'] = station_encoder.fit_transform(df['Station_Rating'])

    X = df.drop(columns=['Station_Rating', 'Brand', 'Station_Rating_encoded'])
    scaler_station = StandardScaler()
    X = scaler_station.fit_transform(X)
    y = df['Station_Rating_encoded']

    # Define the parameter grid
    assert not np.isnan(X).any(), "There are NaN values in the dataset!"
    grid_search = model_tuning(X, y)
    print(f"Best parameters for Station Model: {grid_search.best_params_}")
    print(f"Best cross-validation score for Station Model: {grid_search.best_score_:.4f}")
    model_station = grid_search.best_estimator_



def brand_predicter(df):
    global model_brand, brand_encoder, scaler_brand
    
    brand_encoder = LabelEncoder()
    df['Brand_encoded'] = brand_encoder.fit_transform(df['Brand'])

    X = df.drop(columns=['Station_Rating', 'Brand', 'Brand_encoded'])
    scaler_brand = StandardScaler()
    X = scaler_brand.fit_transform(X)
    y = df['Brand_encoded']

    # Define the parameter grid
    assert not np.isnan(X).any(), "There are NaN values in the dataset!"
    grid_search = model_tuning(X, y)
    print(f"Best parameters for Brand Model: {grid_search.best_params_}")
    print(f"Best cross-validation score for Brand Model: {grid_search.best_score_:.4f}")
    model_brand = grid_search.best_estimator_

def predict_unknown_station(df_sample):
    drop_cols = [col for col in ['Station_Rating', 'Brand'] if col in df_sample.columns]
    df_sample = df_sample.drop(columns=drop_cols)
    
    # Standardize the sample data
    df_sample = scaler_station.transform(df_sample)
    
    global model_station, station_encoder
    predicted_station_encoded = model_station.predict(df_sample)
    predicted_station = station_encoder.inverse_transform(predicted_station_encoded)

    predicted_probs = model_station.predict_proba(df_sample)
    predicted_station_prob = predicted_probs[0][predicted_station_encoded[0]]
    print(unknown_file, ":")
    print(f"Predicted Station Rating: {predicted_station[0]} {predicted_station_prob*100:.2f}%")

def predict_unknown_brand(df_sample, df_processed):
    drop_cols = [col for col in ['Station_Rating', 'Brand'] if col in df_sample.columns]
    df_sample = df_sample.drop(columns=drop_cols)

    # Identify missing columns
    missing_cols = set(df_processed.columns) - set(df_sample.columns) - set(['Station_Rating', 'Brand', 'Brand_encoded'])
    
    # Fill in the missing columns with zeros
    for col in missing_cols:
        df_sample[col] = 0

    # Ensure column order matches training data
    df_sample = df_sample[df_processed.drop(columns=['Station_Rating', 'Brand', 'Brand_encoded']).columns]

    # Standardize the sample data
    df_sample = scaler_brand.transform(df_sample)

    global model_brand, brand_encoder
    predicted_brand_encoded = model_brand.predict(df_sample)
    predicted_brand = brand_encoder.inverse_transform(predicted_brand_encoded)

    predicted_probs = model_brand.predict_proba(df_sample)
    predicted_brand_prob = predicted_probs[0][predicted_brand_encoded[0]]

    print(f"Predicted Brand: {predicted_brand[0]} {predicted_brand_prob*100:.2f}%")





def main():
    # Training data
    df_processed = organize_dataframe()
    Station_predicter(df_processed)
    brand_predicter(df_processed)

    # Unknown sample prediction
    if os.path.exists(unknown_file):
        df_sample = organize_dataframe(is_unknown=True)
        predict_unknown_station(df_sample)
        predict_unknown_brand(df_sample, df_processed)  # Pass df_processed here



if __name__ == "__main__":
    main()