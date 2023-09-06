import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QTextEdit
from PyQt5.QtCore import pyqtSignal

class CombinedApp(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Excel RT Converter and Column Checker'
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.label = QLabel('No file selected yet', self)

        self.browse_button = QPushButton('Browse File', self)
        self.browse_button.clicked.connect(self.browse_file)

        self.ok_button = QPushButton('OK', self)
        self.ok_button.clicked.connect(self.process_file)

        self.output_area = QTextEdit(self)
        self.output_area.setReadOnly(True)

        layout.addWidget(self.label)
        layout.addWidget(self.browse_button)
        layout.addWidget(self.ok_button)
        layout.addWidget(self.output_area)

        self.setLayout(layout)
        self.setWindowTitle(self.title)
        self.show()

    def browse_file(self):
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(self, "Open Excel File", "", "Excel Files (*.xlsx *.xls);;All Files (*)", options=options)
        if filePath:
            self.label.setText(filePath)
            self.input_file = filePath

    def rename_columns(self):
        # Determine the new file name for the renamed columns
        output_file_path = self.input_file.replace('.xlsx', '_fixednames.xlsx')

        with pd.ExcelFile(self.input_file) as xlsx:
            with pd.ExcelWriter(output_file_path) as writer:  # Write to the new file
                for sheet_name in xlsx.sheet_names:
                    df = pd.read_excel(xlsx, sheet_name=sheet_name)

                    # Only rename columns if it's not the 'Alkanes' or 'PAH' sheet
                    if sheet_name not in ['Alkanes', 'PAH']:  
                        df.rename(columns={'<sup>1</sup>t<sub>R</sub>': 'RT1', '<sup>2</sup>t<sub>R</sub>': 'RT2'}, inplace=True)

                    # Always write the dataframe to the new file, whether columns were renamed or not
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

        print("File Column Names Fixed")
        # Update the input file attribute to point to the newly created file
        self.input_file = output_file_path


    def check_columns(self):
        expected_columns = [
            "Compound", "MF", "RMF", "RT1", "RT2", "Area", "Height", "Quant", "Group", 
            "Area %", "Signal to Noise", "Major", "Qual"
        ]

        issues_found = []

        with pd.ExcelFile(self.input_file) as xlsx:
            for sheet_name in xlsx.sheet_names:
                if sheet_name not in ['Alkanes', 'PAH']:  # skip Alkanes and PAH
                    df = pd.read_excel(xlsx, sheet_name)
                    missing_columns = [col for col in expected_columns if col not in df.columns]

                    if missing_columns:
                        issues_found.append(f"Sheet '{sheet_name}' is missing columns: {', '.join(missing_columns)}")

        print("Column Names Checked")
        # Display results
        if not issues_found:
            return True
        else:
            self.output_area.setText("\n".join(issues_found))
            return False
        
    def closest_values(self, differences):
        # Get closest positive and closest negative values to 0
        pos_values = [value for value in differences if value >= 0]
        neg_values = [value for value in differences if value < 0]
        
        closest_pos = min(pos_values, default=0, key=lambda x: abs(x))
        closest_neg = max(neg_values, default=0, key=lambda x: abs(x))
        
        return closest_pos, closest_neg

    def convert_rt1(self, rtx, alkanes_df):
        differences = rtx - alkanes_df["RT1"]
        
        # Getting closest negative and positive differences
        closest_neg = differences[differences < 0].max()
        closest_pos = differences[differences >= 0].min()

        # Check for out-of-bounds indexing
        subset_neg = alkanes_df[differences == closest_neg]["RT1"]
        rt12 = subset_neg.iloc[0] if not subset_neg.empty else None

        subset_pos = alkanes_df[differences == closest_pos]["RT1"]
        rt11 = subset_pos.iloc[0] if not subset_pos.empty else None

        # If either rt12 or rt11 is None, handle this case.
        if rt12 is None or rt11 is None:
            # For now, let's just return the original rtx value if this occurs.
            # You can handle this differently if required.
            return rtx

        nc1 = alkanes_df[differences == closest_pos]["NC"].iloc[0]

        # Your formula calculation remains the same
        return (100 * nc1) + (100 * ((rtx - rt11) / (rt12 - rt11)))


    def convert_rt2(self, rty, pah_df):
        differences = rty - pah_df["RT2"]
        
        # Getting closest negative and positive differences
        closest_neg = differences[differences < 0].max()
        closest_pos = differences[differences >= 0].min()

        # Check for out-of-bounds indexing
        subset_neg = pah_df[differences == closest_neg]["RT2"]
        rt22 = subset_neg.iloc[0] if not subset_neg.empty else None

        subset_pos = pah_df[differences == closest_pos]["RT2"]
        rt21 = subset_pos.iloc[0] if not subset_pos.empty else None

        # If either rt22 or rt21 is None, handle this case.
        if rt22 is None or rt21 is None:
            # For now, let's just return the original rty value if this occurs.
            # You can handle this differently if required.
            return rty

        nr1 = pah_df[differences == closest_pos]["NR"].iloc[0]

        # Your formula calculation remains the same
        return (100 * nr1) + (100 * ((rty - rt21) / (rt22 - rt21)))


    def process_file(self):
        # Rename columns first
        self.rename_columns()
        with pd.ExcelFile(self.input_file) as xlsx:
            # Check for required sheets
            if "Alkanes" not in xlsx.sheet_names or "PAH" not in xlsx.sheet_names:
                self.output_area.setText("Error: The file does not contain both 'Alkanes' and 'PAH' sheets!")
                return

            if not self.check_columns():
                return

            alkanes_df = pd.read_excel(xlsx, 'Alkanes')
            pah_df = pd.read_excel(xlsx, 'PAH')

            sheets_data = {}
            for sheet_name in xlsx.sheet_names:
                if sheet_name not in ['Alkanes', 'PAH']:
                    sample_df = pd.read_excel(xlsx, sheet_name)
                    
                    # Convert to RT indices
                    sample_df["RT1"] = sample_df["RT1"].apply(lambda x: self.convert_rt1(x, alkanes_df))
                    sample_df["RT2"] = sample_df["RT2"].apply(lambda x: self.convert_rt2(x, pah_df))
                    
                    # Data cleaning
                    
                    # REMOVING SOLVENT
                    sample_df = sample_df[sample_df['Compound'] != 'Carbon disulfide']


                    # REMOVING COLUMN BLEED
                    sample_df = sample_df[~((sample_df['Major'].between(31.88, 32.12, inclusive='both')) & (sample_df['Qual'].between(43.88, 44.12, inclusive='both')))]
                    sample_df = sample_df[sample_df['Compound'] != 'Cyclotrisiloxane, hexamethyl-']
                    sample_df = sample_df[~((sample_df['Major'].between(31.88, 32.12, inclusive='both')) & (sample_df['Qual'].between(34.88, 35.12, inclusive='both')))]
                    sample_df = sample_df[~((sample_df['Major'].between(31.88, 32.12, inclusive='both')) & (sample_df['Qual'].between(75.88, 76.12, inclusive='both')))]
                    sample_df = sample_df[~((sample_df['Major'].between(31.88, 32.12, inclusive='both')) & (sample_df['Qual'].between(39.88, 40.12, inclusive='both')))]
                    sample_df = sample_df[~((sample_df['Major'].between(75.88, 76.12, inclusive='both')) & (sample_df['Qual'].between(31.88, 32.12, inclusive='both')))]
                    sample_df = sample_df[~((sample_df['Major'].between(176.7, 176.8, inclusive='both')) & (sample_df['Qual'].between(177.8, 177.9, inclusive='both')))]


                    # REMVING SAMPLES WITH A SIGNAL TO NOISE LESS THAN 10
                    sample_df = sample_df[sample_df['Signal to Noise'] >= 10]


                    sheets_data[sheet_name] = sample_df

        # Save to a new location
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Excel File", "", "Excel Files (*.xlsx *.xlsx);;All Files (*)", options=options)
        if filePath:
            with pd.ExcelWriter(filePath, engine='xlsxwriter') as writer:
                for sheet_name, data in sheets_data.items():
                    data.to_excel(writer, sheet_name=sheet_name, index=False)
                writer._save()

            self.output_area.setText("Column check passed, RT conversion and data cleaning done successfully!")

app = QApplication(sys.argv)
ex = CombinedApp()
sys.exit(app.exec_())
