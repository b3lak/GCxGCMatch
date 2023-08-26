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

unknown_files = [
    "/Users/caleb/Desktop/ILRStationAnalysis/S5R.xlsx",
    "/Users/caleb/Desktop/ILRStationAnalysis/S8R.xlsx",
    "/Users/caleb/Desktop/ILRStationAnalysis/S19R.xlsx"
]

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



def organize_dataframe(is_unknown=False, unknown_files=None, normalization_type='log10'):
    all_data = []
    
    if is_unknown:
        # Assign unique labels to unknown samples
        counter = 1
        data_files = []
        for file in unknown_files:
            data_files.append((file, f"Unknown {counter}", "Unknown"))
            counter += 1
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
            if is_unknown:
                sample_data = {
                    'Station_Rating': station_number,
                    'Brand': station_number
                }
            else:
                sample_data = {
                    'Station_Rating': station_number,
                    'Brand': brand_type
                }
            
            # Sanitize compound names
            compounds = [sanitize_compound_name(comp, idx) for idx, comp in enumerate(areas['Compound_Results'])]
            areas['Compound_Results'] = compounds
            
            # Apply log10 normalization to the area values
            area_values = normalize_data(areas[col], normalization_type).tolist()

            # Collect other data based on sample name
            height_col = col.replace("Area", "Height")
            signal_col = col.replace("Area", "Signal to Noise")
            area_perc_col = col.replace("Area", "Area %")

            for idx, compound in enumerate(compounds):
                sample_data[f"{compound}_Area"] = area_values[idx]
                sample_data[f"{compound}_Binary"] = 1 if area_values[idx] > 0 else 0

                # Add the new features and apply log10 normalization if it's not a binary column
                sample_data[f"{compound}_Height"] = normalize_data(heights[height_col].iloc[idx], normalization_type)


            all_data.append(sample_data)

    return pd.DataFrame(all_data)


def percent_normalize(data):
    """Perform a min-max normalization scaled to 0-100%"""
    # Create a mask for values that are 0 or NaN
    mask = (data == 0) | np.isnan(data)  # Using numpy's isnan method
    
    # Handle the case if data is a singular value (like a float)
    if isinstance(data, (int, float, np.float64)):
        if mask:
            return 0
        return data  # or any other normalization you want for a single value
    
    # Copy data to avoid in-place modifications
    normalized_data = data.copy()
    
    # Only normalize non-zero and non-NaN values
    min_val = data[~mask].min()
    max_val = data[~mask].max()
    normalized_data[~mask] = (data[~mask] - min_val) / (max_val - min_val) * 100
    
    return normalized_data

def zscore_normalize(data):
    """Perform a z-score normalization"""
    
    # Create a mask for values that are 0 or NaN
    mask = (data == 0) | np.isnan(data)
    
    # Handle the case if data is a singular value (like a float)
    if isinstance(data, (int, float, np.float64)):
        if mask:
            return 0
        return data  # or any other normalization you want for a single value
    
    # Copy data to avoid in-place modifications
    normalized_data = data.copy()
    
    # Only normalize non-zero and non-NaN values
    mean_val = data[~mask].mean()
    std_val = data[~mask].std()
    normalized_data[~mask] = (data[~mask] - mean_val) / std_val
    
    return normalized_data

def no_normalize(data):
    return data
    
def normalize_data(data, normalization_type='log10'):
    if normalization_type == 'log10':
        return np.log10(data + 1)
    elif normalization_type == 'percent':
        return percent_normalize(data)
    elif normalization_type == 'zscore':
        return zscore_normalize(data)
    elif normalization_type == 'nonorm':
        return no_normalize(data)
    else:
        return data

def visualize_pca_3d(pca_results, labels, title='PCA of Station Data', emphasize_unknowns=True):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    labels_str = labels.astype(str)
    unique_labels = np.unique(labels_str)

    base_colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(unique_labels)))
    color_dict = {label: color for label, color in zip(unique_labels, base_colors)}

    # Separate point size and color for unknowns
    unknown_size = 150 if emphasize_unknowns else 100
    known_size = 100

    for label in unique_labels:
        label_mask = labels_str == label
        current_color = [color_dict[label]] * len(pca_results[label_mask, 0])  # List of the same color repeated
        if "Unknown" in label:
            ax.scatter(pca_results[label_mask, 0], pca_results[label_mask, 1], pca_results[label_mask, 2], 
                    c=current_color, s=unknown_size, alpha=0.7, label=label, edgecolors='black', linewidths=1.5)
            if emphasize_unknowns:
                for x, y, z in zip(pca_results[label_mask, 0], pca_results[label_mask, 1], pca_results[label_mask, 2]):
                    ax.text(x, y, z, label, fontsize=9)
        else:
            ax.scatter(pca_results[label_mask, 0], pca_results[label_mask, 1], pca_results[label_mask, 2], 
                    c=current_color, s=known_size, alpha=0.7, label=label)

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
    df['Brand'] = df['Brand'].astype(str)  # Convert the column to string type

    unknown_mask = df['Brand'].str.startswith("Unknown")
    known_mask = ~unknown_mask

    df.loc[known_mask, 'Brand'] = brand_encoder.inverse_transform(df.loc[known_mask, 'Brand'].astype(int))

    return df

def print_top_features_for_each_pc(pca, feature_names, n=1000):
    """
    Print the top features for each principal component.
    
    Parameters:
    - pca: The fitted PCA object.
    - feature_names: List of feature names.
    - n: Number of top features to print.
    """
    components = pca.components_
    for idx, component in enumerate(components):
        # Sorting the feature names based on the loadings (in absolute value)
        sorted_feature_idx = np.argsort(np.abs(component))[::-1]
        
        top_features = [feature_names[i] for i in sorted_feature_idx[:n]]
        
        print(f"Principal Component {idx + 1} top {n} features:")
        for i, feature in enumerate(top_features, 1):
            unsanitized_feature = unsanitize_compound_name(feature)  # Unsanitizing the feature name
            print(f"{i}. {unsanitized_feature} ({component[sorted_feature_idx[i-1]]:.4f})")
    
        print('-'*50)
        


def main():
    # Set up PCA
    pca = PCA(n_components=3)

    # Organize the processed dataframe
    df_processed = organize_dataframe(normalization_type='nonorm')
    df_processed = encode_brands(df_processed)
    df_processed.fillna(0, inplace=True)

    # Organize the unknown dataframe
    df_unknown = organize_dataframe(is_unknown=True, unknown_files=unknown_files, normalization_type='nonorm')
    df_unknown.fillna(0, inplace=True)

    # Drop Brand and Station_Rating columns
    features_processed = df_processed.drop(columns=['Brand', 'Station_Rating']).values
    features_unknown = df_unknown.drop(columns=['Brand', 'Station_Rating']).values

    # Train PCA on df_processed and transform it
    pca_result_processed_station = pca.fit_transform(features_processed)
    labels_processed_station = df_processed['Station_Rating']

    # After you've computed the PCA, print the top features for each principal component
    feature_names = df_processed.drop(columns=['Brand', 'Station_Rating']).columns
    print_top_features_for_each_pc(pca, feature_names)

    # Transform df_unknown using the same PCA
    pca_result_unknown_station = pca.transform(features_unknown)
    labels_unknown_station = df_unknown['Station_Rating']

    # Combine the PCA results and labels for visualization
    pca_result_station = np.vstack([pca_result_processed_station, pca_result_unknown_station])
    labels_station = pd.concat([labels_processed_station, labels_unknown_station])

    visualize_pca_3d(pca_result_station[:, :3], labels_station, 'PCA Grouped by Station')

    # Use the same PCA to transform df_processed for brands
    pca_result_processed_brand = pca_result_processed_station
    labels_processed_brand = decode_brands(df_processed.copy())['Brand']

    # Transform df_unknown using the same PCA for brands
    pca_result_unknown_brand = pca_result_unknown_station
    labels_unknown_brand = decode_brands(df_unknown.copy())['Brand']

    # Combine the PCA results and labels for visualization
    pca_result_brand = np.vstack([pca_result_processed_brand, pca_result_unknown_brand])
    labels_brand = pd.concat([labels_processed_brand, labels_unknown_brand])

    visualize_pca_3d(pca_result_brand[:, :3], labels_brand, 'PCA Grouped by Brand')

    # Print the explained variance
    explained_var = pca.explained_variance_ratio_ * 100
    print("Explained variation per principal component: {}".format(explained_var))

if __name__ == "__main__":
    main()
