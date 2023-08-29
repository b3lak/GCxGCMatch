import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QTextEdit


base_path = "/Users/caleb/Desktop/ILRStationAnalysis/"

files_details = [
    ('S1R.xlsx'),
    ('S2R.xlsx'),
    ('S3R.xlsx'),
    ('S4R.xlsx'),
    ('S5R.xlsx'),
    ('S6R.xlsx'),
    ('S7R.xlsx'),
    ('S8R.xlsx'),
    ('S9R.xlsx'),
    ('S10R.xlsx'),
    ('S11R.xlsx'),
    ('S12R.xlsx'),
    ('S13R.xlsx'),
    ('S14R.xlsx'),
    ('S15R.xlsx'),
    ('S16R.xlsx'),
    ('S17R.xlsx'),
    ('S18R.xlsx'),
    ('S4R.xlsx'),
    ('S20R.xlsx'),
    ('S21R.xlsx'),
    ('S22R.xlsx'),
    ('S23R.xlsx'),
    ('S24R.xlsx'),
    ('S25R.xlsx'),
    ('S26R.xlsx'),
    ('S27R.xlsx'),
    ('S28R.xlsx'),
    ('S29R.xlsx'),
    ('S30R.xlsx'),
    ('S31R.xlsx'),
    ('S32R.xlsx'),
    ('S33R.xlsx'),
]

def process_data():
    master_df = None
    compound_name_map = {}


    # Initialize an empty DataFrame to store occurrence per file
    occurrence_per_file_df = pd.DataFrame()

    for file_name in files_details:
        df = pd.read_excel(base_path + file_name, sheet_name="Area")
        
        for idx, row in df.iterrows():
            sanitized_name = f"Compound_{idx+1}"
            if pd.isna(row['Compound_Results']):
                compound_name_map[sanitized_name] = sanitized_name
            else:
                compound_name_map[sanitized_name] = row['Compound_Results']
            df.at[idx, 'Compound_Results'] = sanitized_name

        
        # Binary classification
        area_columns = [col for col in df if col.startswith('Area_')]
        binary_df = df[area_columns].applymap(lambda x: 1 if x > 0 else 0)

        file_code = file_name.split('.')[0]
        binary_df_with_index = binary_df.set_index(df['Compound_Results'])
        occurrence_per_file_df[file_code] = binary_df_with_index.sum(axis=1)


        # For the first file, get the common columns
        if master_df is None:
            common_columns = ["Compound_Results", "RT1_Results", "RT2_Results", "Major_Results"]
            master_df = df[common_columns].join(binary_df)
        else:
            # Ensure the rows (compounds) align between master and the current df
            binary_df = binary_df.set_index(df['Compound_Results']).reindex(master_df['Compound_Results']).reset_index()
            master_df = pd.concat([master_df, binary_df.drop('Compound_Results', axis=1)], axis=1)

    # Calculate the total occurrences across all files
    master_df['Total'] = master_df.drop(["Compound_Results", "RT1_Results", "RT2_Results", "Major_Results"], axis=1).sum(axis=1)

    # Remove rows with Compound_Results of 'Carbon disulfide'
    master_df = master_df[master_df['Compound_Results'] != 'Carbon disulfide']

    return master_df, occurrence_per_file_df, compound_name_map

def create_heatmap(data, occurrence_per_file_df, compound_name_map):
    # 1. Bottom 50 compounds by total occurrences
    plt.figure(figsize=(10, 15))
    bottom_50_total = data.sort_values(by="Total", ascending=True).head(50)
    sns.heatmap(bottom_50_total.set_index("Compound_Results")[["Total"]], annot=True, cmap="YlGnBu", cbar_kws={'label': 'Occurrences'})
    plt.title("Bottom 50 Compounds by Total Occurrence Across All Samples")
    plt.show()

    # 2. Use the same bottom 50 compounds identified by total occurrences to show their occurrence per file
    bottom_50_compounds = bottom_50_total['Compound_Results'].tolist()
    
    unsanitized_compound_names = [compound_name_map[name] for name in bottom_50_compounds]
    bottom_50_data_by_file = occurrence_per_file_df.loc[bottom_50_compounds]
    bottom_50_data_by_file.index = unsanitized_compound_names

    # Visualize bottom 50 compounds by occurrence per file
    plt.figure(figsize=(20, 10))
    sns.heatmap(bottom_50_data_by_file, annot=False, cmap="YlGnBu", cbar_kws={'label': 'Occurrences'}, vmin=0, vmax=3)
    plt.title("Bottom 50 Compounds by Occurrence in Each File")
    plt.xticks(fontsize=8)  # Adjust the font size of the x-axis labels
    plt.yticks(fontsize=8)  # Adjust the font size of the y-axis labels
    plt.show()

def main():
    data, occurrence_per_file_df, compound_name_map = process_data()
    
    # Create a new Excel writer object
    with pd.ExcelWriter("output_binary_table.xlsx") as writer:
        # Save the main data to the Excel file
        data.to_excel(writer, sheet_name="Binary Table", index=False)
        
        # Filter compounds with an occurrence of 99
        compounds_with_99_occurrences = data[data['Total'] == 99]["Compound_Results"]
        compounds_with_99_occurrences.to_excel(writer, sheet_name="Compounds with 99 Occurrences", index=False)
        
        # Get the bottom 200 compounds by occurrence
        sorted_data = data[["Compound_Results", "Total"]].sort_values(by="Total", ascending=True).head(200)
        sorted_data.to_excel(writer, sheet_name="Bottom 200 Compounds", index=False)
    
    create_heatmap(data, occurrence_per_file_df, compound_name_map)

if __name__ == '__main__':
    main()