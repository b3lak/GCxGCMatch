import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors



brand_encoder = LabelEncoder()


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
    ('S10R.xlsx', 10, '7eleven'),
    ('S11R.xlsx', 11, 'Co-op'),
    ('S12R.xlsx', 12, 'Husky'),
    ('S13R.xlsx', 13, 'Chevron'),
    ('S14R.xlsx', 14, 'Canadian Tire Gas+'),
    ('S15R.xlsx', 15, 'Mobil'),
    ('S16R.xlsx', 16, 'Fair Deal Gas Bar'),
    ('S17R.xlsx', 17, 'Co-op'),
    ('S18R.xlsx', 18, 'Domo'),
    ('S20R.xlsx', 20, 'Gas Plus'),
    ('S21R.xlsx', 21, '7eleven'),
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
            areas = pd.read_excel(file_name, sheet_name="Area")
            heights = pd.read_excel(file_name, sheet_name="Height")
            signals = pd.read_excel(file_name, sheet_name="SignalToNoise")
            area_percs = pd.read_excel(file_name, sheet_name="Area%")
        else:
            areas = pd.read_excel(base_path + file_name, sheet_name="Area")
            heights = pd.read_excel(base_path + file_name, sheet_name="Height")
            signals = pd.read_excel(base_path + file_name, sheet_name="SignalToNoise")
            area_percs = pd.read_excel(base_path + file_name, sheet_name="Area%")

        # Removing rows with Carbon disulfide
        unwanted_compounds = ['Carbon disulfide', 'Benzene', 'Toluene', 'Ethylbenzene', 'p-Xylene', 'Cyclotrisiloxane, hexamethyl-']
        for compound in unwanted_compounds:
            areas = areas[areas['Compound_Results'] != compound]
            heights = heights[heights['Compound_Results'] != compound]
            signals = signals[signals['Compound_Results'] != compound]
            area_percs = area_percs[area_percs['Compound_Results'] != compound]

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

                # Add the new features and apply log10 normalization if it's not a binary column
                sample_data[f"{compound}_Height"] = np.log10(heights[height_col].iloc[idx] + 1)


            all_data.append(sample_data)

    return pd.DataFrame(all_data)


def visualize_pca_3d(pca_results, labels, title='PCA of Station Data'):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    labels_str = labels.astype(str)
    unique_labels = np.unique(labels_str)
    
    base_colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(unique_labels)))
    color_dict = {label: color for label, color in zip(unique_labels, base_colors)}

    # If 'unknown' is present, set its color to black
    if 'unknown' in color_dict:
        color_dict['unknown'] = (0, 0, 0, 1)

    # Create an array of colors for each label
    point_colors = np.array([color_dict[label] for label in labels_str])

    scatter = ax.scatter(pca_results[:, 0], pca_results[:, 1], pca_results[:, 2], c=point_colors, s=100, alpha=0.7)

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title(title)

    handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, markersize=10, markerfacecolor=color_dict[label]) for label in unique_labels]

    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    plt.show()


def encode_brands(df):
    """
    Encode 'Brand' column to numerical values.
    """
    df['Brand'] = brand_encoder.fit_transform(df['Brand'])
    return df

def decode_brands(df):
    """
    Decode 'Brand' column back to original values for visualization.
    """
    unknown_mask = df['Brand'] != "Unknown"  # Identify rows where Brand is not "Unknown"
    df.loc[unknown_mask, 'Brand'] = brand_encoder.inverse_transform(df.loc[unknown_mask, 'Brand'].astype(int))
    return df



def main():
    # Set up PCA
    pca = PCA(n_components=5)

    # Organize the processed dataframe
    df_processed = organize_dataframe()
    df_processed = encode_brands(df_processed)
    df_processed.fillna(0, inplace=True)

    # Organize the unknown dataframe
    df_unknown = organize_dataframe(is_unknown=True)
    df_unknown.fillna(0, inplace=True)

    # Combine the two dataframes
    df_all = pd.concat([df_processed, df_unknown])

    # Fill NaN values again to be sure
    df_all['Brand'].fillna("Unknown", inplace=True)
    df_all['Station_Rating'].fillna("Unknown", inplace=True)

    # Ensure that there are no NaN values
    if df_all.isna().sum().sum() != 0:
        print("Data still contains NaN values!")
        return

    # Ensure all data types are numeric for features, not labels
    features = df_all.drop(columns=['Brand', 'Station_Rating'])
    non_numeric_cols = features.select_dtypes(exclude=['int64', 'float64']).columns
    if len(non_numeric_cols) > 0:
        print("Following columns are non-numeric:", non_numeric_cols)
        return

    # Extract features and scale them
    scaler = StandardScaler()
    scaled_features_all = scaler.fit_transform(features)

    # PCA for grouping by Station_Rating
    pca_result_station = pca.fit_transform(scaled_features_all)
    labels_station = df_all['Station_Rating']  # "Unknown" values are already set
    visualize_pca_3d(pca_result_station[:, :3], labels_station, 'PCA Grouped by Station')

    # PCA for grouping by Brand
    pca_result_brand = pca.fit_transform(scaled_features_all)
    labels_brand = decode_brands(df_all.copy())['Brand']  # Decode brands for visualization
    visualize_pca_3d(pca_result_brand[:, :3], labels_brand, 'PCA Grouped by Brand')

    # Print the explained variance
    explained_var = pca.explained_variance_ratio_
    print("Explained variation per principal component: {}".format(explained_var))

if __name__ == "__main__":
    main()
