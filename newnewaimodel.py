import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import ttest_ind
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from scipy.stats import mannwhitneyu
from collections import Counter


columns_details = [
    (['Area_S1-1', 'Area_S1-2', 'Area_S1-3'], 1, 'Parkdale Gas Co-op'),
    (['Area_S2-1', 'Area_S2-2', 'Area_S2-3'], 2, 'Esso'),
    (['Area_S3-1', 'Area_S3-2', 'Area_S3-3'], 3, 'Centex'),
    (['Area_S4-1', 'Area_S4-2', 'Area_S4-3'], 4, 'Petro Canada'),
    (['Area_S6-1', 'Area_S6-2', 'Area_S6-3'], 6, 'Shell'),
    (['Area_S7-1', 'Area_S7-2', 'Area_S7-3'], 7, 'Esso 2'),
    (['Area_S9-1', 'Area_S9-2', 'Area_S9-3'], 9, 'Canpro'),
    (['Area_S10-1', 'Area_S10-2', 'Area_S10-3'], 10, '7/11'),
    (['Area_S11-1', 'Area_S11-2', 'Area_S11-3'], 11, 'Co-op'),
    (['Area_S12-1', 'Area_S12-2', 'Area_S12-3'], 12, 'Husky'),
    (['Area_S13-1', 'Area_S13-2', 'Area_S13-3'], 13, 'Chevron'),
    (['Area_S14-1', 'Area_S14-2', 'Area_S14-3'], 14, 'Canadian Tire Gas+'),
    (['Area_S15-1', 'Area_S15-2', 'Area_S15-3'], 15, 'Mobil'),
    (['Area_S16-1', 'Area_S16-2', 'Area_S16-3'], 16, 'Fair Deal Gas Plus'),
    (['Area_S17-1', 'Area_S17-2', 'Area_S17-3'], 17, 'Co-op 2'),
    (['Area_S18-1', 'Area_S18-2', 'Area_S18-3'], 18, 'Domo'),
    (['Area_S20-1', 'Area_S20-2', 'Area_S20-3'], 20, 'Gas Plus'),
    (['Area_S21-1', 'Area_S21-2', 'Area_S21-3'], 21, '7/11 2'),
    (['Area_S22-1', 'Area_S22-2', 'Area_S22-3'], 22, 'Petro Canada 2'),
    (['Area_S23-1', 'Area_S23-2', 'Area_S23-3'], 23, 'Safeway Gas'),
    (['Area_S24-1', 'Area_S24-2', 'Area_S24-3'], 24, 'Canadian Tire Gas+ 2'),
    (['Area_S25-1', 'Area_S25-2', 'Area_S25-3'], 25, 'G&B Fuels'),
    (['Area_S26-1', 'Area_S26-2', 'Area_S26-3'], 26, 'Costco'),
    (['Area_S27-1', 'Area_S27-2', 'Area_S27-3'], 27, 'Tsuu Tina Gas Stop'),
    (['Area_S28-1', 'Area_S28-2', 'Area_S28-3'], 28, 'Shell 2'),
    (['Area_S29-1', 'Area_S29-2', 'Area_S29-3'], 29, 'Chevron'),
    (['Area_S30-1', 'Area_S30-2', 'Area_S30-3'], 30, 'Safeway Gas 2'),
    (['Area_S31-1', 'Area_S31-2', 'Area_S31-3'], 31, 'Husky 2'),
    (['Area_S32-1', 'Area_S32-2', 'Area_S32-3'], 32, 'G&B Fuels 2'),
    (['Area_S33-1', 'Area_S33-2', 'Area_S33-3'], 33, 'Mobil 2')
]

unknown_column_details = [
    (['Area_S5-1', 'Area_S5-2', 'Area_S5-3'], None, 'Unknown 1'),
    (['Area_S8-1', 'Area_S8-2', 'Area_S8-3'], None, 'Unknown 2'),
    (['Area_S19-1', 'Area_S19-2', 'Area_S19-3'], None, 'Unknown 3')
]
# 1. Reading the Excel file
file_path = "/Users/caleb/Desktop/ILRStationAnalysis/NonTargetSearchOCT16.xlsx"
df = pd.read_excel(file_path, engine='openpyxl')  # Assumes the first sheet is the one we want.

# Extract all individual column names
columns_to_normalize = [col for sublist in columns_details + unknown_column_details for col in sublist[0]]

# Get the overall average across all samples for each compound
#overall_avg = df[columns_to_normalize].mean(axis=1)

# Get the top 200 compounds by average area
#top_200_compounds = overall_avg.nlargest(200).index

# Filter the main DataFrame to only include the top 200 compounds
#df = df.loc[df.index.isin(top_200_compounds)]

# 2. Data Preprocessing
# Fill NaN values with 0
df.fillna(10567, inplace=True)

# percent normalization
df[columns_to_normalize] = MinMaxScaler().fit_transform(df[columns_to_normalize])

# Calculate mean values for each set of replicates
all_columns = columns_details + unknown_column_details
mean_values = {name: df[columns].mean(axis=1) for columns, _, name in all_columns}

# Pairwise t-test and store results in a DataFrame
all_names = [name for _, _, name in all_columns]
comparison_results = pd.DataFrame(index=all_names, columns=all_names)

for name1 in mean_values:
    for name2 in mean_values:
        t_stat, p_val = ttest_ind(mean_values[name1], mean_values[name2])
        comparison_results.at[name1, name2] = p_val  # store p-values in the DataFrame

# Convert to standard float before visualizing
comparison_results = comparison_results.astype(float)

# Visualizing the table as a heatmap
plt.figure(figsize=(20, 15))
sns.heatmap(comparison_results, annot=True, cmap="YlGnBu", cbar_kws={'label': 'p-value'})
plt.title("Pairwise T-test P-values")
plt.tight_layout()
plt.show()

# Create X_train and y_train for known data
X_train_list = []
y_train_list = []

for columns, label, _ in columns_details:
    station_data = df[columns]
    X_train_list.append(station_data.T)
    y_train_list.extend([label] * station_data.shape[1])

X_train = pd.concat(X_train_list)
y_train = y_train_list

X_unknown_list = []

for columns, _, _ in unknown_column_details:
    unknown_station_data = df[columns]
    X_unknown_list.append(unknown_station_data.T)

X_unknown = pd.concat(X_unknown_list)


n_components = 5
pca = PCA(n_components=n_components)

# Fit PCA on training data and transform both training and unknown data
X_train_pca = pca.fit_transform(X_train)
X_unknown_pca = pca.transform(X_unknown)

print(f"Explained variance by first {n_components} components: {np.sum(pca.explained_variance_ratio_)}")

# Continue with KNN and hyperparameter tuning using the PCA transformed data
knn = KNeighborsClassifier()
kf = KFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {
    'n_neighbors': range(1, 31),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

grid_search = GridSearchCV(knn, param_grid, cv=kf, scoring='accuracy', return_train_score=True)
grid_search.fit(X_train_pca, y_train)  # Use PCA transformed training data here

best_knn = grid_search.best_estimator_

print("Best Hyperparameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

# Make predictions using the best model
predictions = best_knn.predict(X_unknown_pca)  # Use PCA transformed unknown data here

# Averaging predictions
avg_predictions = []
for i in range(0, len(predictions), 3):
    most_common_prediction = Counter(predictions[i:i+3]).most_common(1)[0][0]
    avg_predictions.append(most_common_prediction)

label_to_station = {item[1]: item[2] for item in columns_details}
predicted_stations = [label_to_station.get(label, "Unknown") for label in avg_predictions]

print(predicted_stations)

# Extract data for each station for Mann-Whitney U test
station_data_dict = {name: df[columns].mean(axis=1) for columns, _, name in all_columns}

# Iterate over the unknown samples and the station they are predicted as to perform the test
for i, (columns, _, name) in enumerate(unknown_column_details):
    station_name = predicted_stations[i]
    # Here we're using the averaged replicates for both the unknown sample and the predicted station
    u_stat, p_val = mannwhitneyu(df[columns].mean(axis=1), station_data_dict[station_name])
    print(f"Comparison of {name} with {station_name}: U-statistic={u_stat}, p-value={p_val}")
