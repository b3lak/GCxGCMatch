import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter



base_path = "/Users/caleb/Desktop/ILRStationAnalysis/"

files_details = [f"S{i}R.xlsx" for i in range(1, 34)]  # Simplified file name list creation

def process_data():
    master_df = None
    master_area_df = None  # To store the original area values

    # Initialize an empty DataFrame to store occurrence per file
    occurrence_per_file_df = pd.DataFrame()

    for file_name in files_details:
        df = pd.read_excel(base_path + file_name, sheet_name="Area")
        
        # Fill NaN values in 'Compound_Results' column
        for idx, val in enumerate(df['Compound_Results']):
            if pd.isna(val):
                df.at[idx, 'Compound_Results'] = f"Compound_{idx + 1}"
        
        # Binary classification
        area_columns = [col for col in df if col.startswith('Area_')]
        binary_df = df[area_columns].applymap(lambda x: 1 if x > 0 else 0)

        # Process for the original area values
        if master_area_df is None:
            common_columns = ["Compound_Results", "RT1_Results", "RT2_Results", "Major_Results"]
            master_area_df = df[common_columns].join(df[area_columns])
        else:
            temp_area_df = df[area_columns].set_index(df['Compound_Results']).reindex(master_area_df['Compound_Results']).reset_index()
            master_area_df = pd.concat([master_area_df, temp_area_df.drop('Compound_Results', axis=1)], axis=1)

        # For the first file, get the common columns for binary values
        if master_df is None:
            common_columns = ["Compound_Results", "RT1_Results", "RT2_Results", "Major_Results"]
            master_df = df[common_columns].join(binary_df)
        else:
            binary_df = binary_df.set_index(df['Compound_Results']).reindex(master_df['Compound_Results']).reset_index()
            master_df = pd.concat([master_df, binary_df.drop('Compound_Results', axis=1)], axis=1)

    # Calculate the total occurrences across all files for binary values
    master_df['Total'] = master_df.drop(["Compound_Results", "RT1_Results", "RT2_Results", "Major_Results"], axis=1).sum(axis=1)

    # Remove rows with Compound_Results of 'Carbon disulfide'
    master_df = master_df[master_df['Compound_Results'] != 'Carbon disulfide']

    return master_df, master_area_df  # Return both the binary and area DataFrames

def plot_average_area_for_heptane(master_area):
    #CHANGE COMPOUND NAME HERE

    # Filter for rows where Compound_Results is "Heptane"
    heptane_data = master_area[master_area["Compound_Results"] == "Nonadecane"]
    
    # Ensure that it is always a DataFrame (even if one row)
    if len(heptane_data.shape) == 1:  # It's a Series
        heptane_data = heptane_data.to_frame().T  # Convert it to a DataFrame
    
    # Directly drop the non-area columns
    heptane_areas = heptane_data.drop(columns=["Compound_Results", "RT1_Results", "RT2_Results", "Major_Results"])
    
    # Flatten the DataFrame and extract the column names
    area_columns = heptane_areas.columns
    
    # Get the file names by splitting column names
    file_names = [col.split("_")[1].split("-")[0] for col in area_columns]
    
    # Calculate average areas per file
    average_areas = heptane_areas.mean(axis=0)  # Mean along the rows for each column
    average_areas.index = file_names  # Rename the index with file names
    
    #CHANGE COMPOUND NAME HERE

    # Plot the bar graph
    average_areas.plot(kind='bar', figsize=(14,6), edgecolor="k")  # Increased figure width
    plt.title("Average Area Values for Nonadecane Across Files")
    
    # Set labels with specified font sizes
    plt.xlabel("File Names", fontsize=12)
    plt.ylabel("Average Area Value", fontsize=12)
    plt.xticks(rotation=90, fontsize=10)  # Increased rotation and further decreased font size
    
    # Adjust y-axis to not use scientific notation
    ax = plt.gca()  # Get current axis
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    ax.ticklabel_format(style='plain', axis='y')  # Turn off scientific notation for y-axis

    plt.tight_layout()
    plt.show()

def find_minimum_area_value(master_area):
    # Flatten the DataFrame and keep only Area related columns
    flat_df = master_area.melt(id_vars=["Compound_Results"], 
                               value_vars=[col for col in master_area if col.startswith('Area_')])

    # Find the minimum non-zero area value
    min_entry = flat_df[flat_df['value'] > 0].nsmallest(1, 'value').iloc[0]

    # Print the result
    compound_name = min_entry['Compound_Results']
    min_value = min_entry['value']
    print(f"The compound with the smallest individual area value is '{compound_name}' with a value of {min_value}.")

def main():
    data, master_area = process_data()
    plot_average_area_for_heptane(master_area)  # Plot the bar graph for Heptane's average area

    find_minimum_area_value(master_area)

    # Create a new Excel writer object
    with pd.ExcelWriter("binaryandarea_table.xlsx") as writer:
        # Save the main data to the Excel file
        data.to_excel(writer, sheet_name="Binary Table", index=False)
        master_area.to_excel(writer, sheet_name="Master Area", index=False)  # Save the Master Area DataFrame
        
        # Filter compounds with an occurrence of 99
        compounds_with_99_occurrences = data[data['Total'] == 99]
        compounds_with_99_occurrences.to_excel(writer, sheet_name="Compounds with 99 Occurrences", index=False)
        
        # Get the bottom 50 compounds by occurrence
        bottom_50_total = data.sort_values(by="Total", ascending=True).head(50)
        bottom_50_total.to_excel(writer, sheet_name="Bottom 50 Compounds", index=False)

    # Find the compound with the smallest individual area value
    flattened_master_area = master_area.drop(columns=["Compound_Results", "RT1_Results", "RT2_Results", "Major_Results"])
    min_value = flattened_master_area.min().min()
    compound_for_min_value = master_area[flattened_master_area.isin([min_value])]['Compound_Results'].iloc[0]

if __name__ == '__main__':
    main()
