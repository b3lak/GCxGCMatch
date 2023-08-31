import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os

# 1. Read the data from the Excel file
file_path = "/Users/caleb/Desktop/ILRStationAnalysis/RImasterlistSTN10W50AllColumnBleed.xlsx"
df = pd.read_excel(file_path)

# Sanitize the 'Compound' column
for idx, value in enumerate(df['Compound']):
    if pd.isna(value):
        df.at[idx, 'Compound'] = f"Compound_{idx + 1}"

# Extract the filename without extension
file_name = os.path.basename(file_path)
base_file_name = os.path.splitext(file_name)[0]

# 2. Compute the KDE
x = df['RT1']
y = df['RT2']
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

# 3. Plot the data
fig, ax = plt.subplots()
scatter = ax.scatter(x, y, c=z, s=50, edgecolor='Black', cmap='inferno', picker=10)
plt.colorbar(scatter)
plt.xlabel('RT1')
plt.ylabel('RT2')
plt.title(f'Scatter plot of RT1 vs RT2 with KDE coloring ({base_file_name})\nPress "a" to enable point selection after zoom/pan.')

# This variable will keep track of the annotation
annot = ax.annotate("", xy=(0,0), xytext=(20,20),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def update_annot(ind):
    # Get the index of the clicked point from the `ind`
    if isinstance(ind, dict) and "ind" in ind:
        index = ind["ind"][0]
    else:
        index = ind[0]
    
    data_point = df.iloc[index]
    
    # The x, y position of the scatter point is retrieved using scatter.get_offsets()
    pos = scatter.get_offsets()[index]
    annot.xy = pos
    
    info = (f"RT1: {data_point['RT1']}\n"
            f"RT2: {data_point['RT2']}\n"
            f"Compound: {data_point['Compound']}\n"
            f"Major: {data_point['Major']}\n"
            f"Qual: {data_point['Qual']}")

    annot.set_text(info)

# Define the event handler
def on_pick(event):
    vis = annot.get_visible()
    if event.artist != scatter:
        return
    update_annot(event.ind)
    annot.set_visible(True)
    fig.canvas.draw_idle()
    print("Point clicked!")

def on_key(event):
    if event.key == 'a':
        toolbar = fig.canvas.toolbar
        
        # If zoom is active, deactivate it.
        if toolbar.mode == 'zoom rect':
            toolbar.zoom()
            
        # If pan is active, deactivate it.
        if toolbar.mode == 'pan/zoom':
            toolbar.pan()
        
        print("Mode set to default. You can click on points now!")


# Connect the event handlers
fig.canvas.mpl_connect('pick_event', on_pick)
fig.canvas.mpl_connect('key_press_event', on_key)

plt.show()
