import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog, QLabel
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMessageBox


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Excel Comparator'
        self.left = 100
        self.top = 100
        self.width = 300
        self.height = 200
        self.file_path = None
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        layout = QVBoxLayout()

        self.label = QLabel('Select an Excel file.')
        layout.addWidget(self.label)
        
        btn_browse = QPushButton('Browse File', self)
        btn_browse.clicked.connect(self.browse_file)
        layout.addWidget(btn_browse)
        
        btn_ok = QPushButton('OK', self)
        btn_ok.clicked.connect(self.process_file)
        layout.addWidget(btn_ok)
        
        self.setLayout(layout)
        
        self.show()
    
    def browse_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Excel File", "", "Excel Files (*.xlsx);;All Files (*)", options=options)
        if file_name:
            self.file_path = file_name
            self.label.setText(f'Selected: {file_name}')
            
    def process_file(self):
        if not self.file_path:  # Check if a file has been selected
            QMessageBox.warning(self, "Error", "Please select an Excel file first.")
            return

        # Define your individual tolerances for the matching criteria
        tolerances = {
            'RT1': 25,
            'RT2': 25,
            'Major': 0.12
        }

        # Open the selected Excel file using pandas
        xls = pd.ExcelFile(self.file_path)
        first_sheet = xls.sheet_names[0]
        df_first = pd.read_excel(xls, first_sheet)

        # Rename its columns to be prefixed with its name
        df_first.columns = [f"{col}_{first_sheet}" for col in df_first.columns]

        # Create result DataFrames for each metric
        result_dfs = {
            "Area": df_first[["Compound_" + first_sheet, "RT1_" + first_sheet, "RT2_" + first_sheet, "Major_" + first_sheet]].copy(),
            "Height": df_first[["Compound_" + first_sheet, "RT1_" + first_sheet, "RT2_" + first_sheet, "Major_" + first_sheet]].copy(),
            "SignalToNoise": df_first[["Compound_" + first_sheet, "RT1_" + first_sheet, "RT2_" + first_sheet, "Major_" + first_sheet]].copy(),
            "Area%": df_first[["Compound_" + first_sheet, "RT1_" + first_sheet, "RT2_" + first_sheet, "Major_" + first_sheet]].copy()
        }

        # For each other sheet in the Excel file, compare it to the first sheet
        for sheet_name in xls.sheet_names[1:]:
            df = pd.read_excel(xls, sheet_name)
            df.columns = [f"{col}_{sheet_name}" for col in df.columns]

            for index, row in df_first.iterrows():
                base_conditions = (
                    df[f"RT1_{sheet_name}"].between(row[f"RT1_{first_sheet}"] - tolerances['RT1'], row[f"RT1_{first_sheet}"] + tolerances['RT1']) &
                    df[f"RT2_{sheet_name}"].between(row[f"RT2_{first_sheet}"] - tolerances['RT2'], row[f"RT2_{first_sheet}"] + tolerances['RT2']) &
                    df[f"Major_{sheet_name}"].between(row[f"Major_{first_sheet}"] - tolerances['Major'], row[f"Major_{first_sheet}"] + tolerances['Major'])
                )

                # Area
                matched_row_area = df[base_conditions]["Area_" + sheet_name].values
                if matched_row_area.any():
                    result_dfs["Area"].loc[index, f"Area_{sheet_name}"] = matched_row_area[0]
                else:
                    result_dfs["Area"].loc[index, f"Area_{sheet_name}"] = None
                    
                # Height
                matched_row_height = df[base_conditions]["Height_" + sheet_name].values
                if matched_row_height.any():
                    result_dfs["Height"].loc[index, f"Height_{sheet_name}"] = matched_row_height[0]
                else:
                    result_dfs["Height"].loc[index, f"Height_{sheet_name}"] = None

                # SignalToNoise
                matched_row_stn = df[base_conditions]["Signal to Noise_" + sheet_name].values
                if matched_row_stn.any():
                    result_dfs["SignalToNoise"].loc[index, f"Signal to Noise_{sheet_name}"] = matched_row_stn[0]
                else:
                    result_dfs["SignalToNoise"].loc[index, f"Signal to Noise_{sheet_name}"] = None

                # Area%
                matched_row_area_percent = df[base_conditions]["Area %_" + sheet_name].values
                if matched_row_area_percent.any():
                    result_dfs["Area%"].loc[index, f"Area %_{sheet_name}"] = matched_row_area_percent[0]
                else:
                    result_dfs["Area%"].loc[index, f"Area %_{sheet_name}"] = None

        # Prompt user for where to save the output
        options = QFileDialog.Options()
        output_file_name, _ = QFileDialog.getSaveFileName(self, "Save As", "", "Excel Files (*.xlsx);;All Files (*)", options=options)
        
        if not output_file_name:
            # If user cancels the save dialog
            return

        # Save the result DataFrames to the chosen file
        with pd.ExcelWriter(output_file_name) as writer:
            for metric, df in result_dfs.items():
                df.to_excel(writer, sheet_name=metric, index=False)

        QMessageBox.information(self, "Done", f"Processed file saved as {output_file_name}")


app = QApplication(sys.argv)
ex = App()
sys.exit(app.exec_())
