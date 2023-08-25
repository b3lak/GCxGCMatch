import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QTextEdit
import numpy as np

class DataCheckApp(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Sample Quality Checker'
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        self.label = QLabel('No file selected yet', self)
        
        self.browse_button = QPushButton('Browse File', self)
        self.browse_button.clicked.connect(self.browse_file)
        
        self.ok_button = QPushButton('OK', self)
        self.ok_button.clicked.connect(self.process_data)

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

    def process_data(self):
        with pd.ExcelFile(self.input_file) as xlsx:
            df = pd.read_excel(xlsx, sheet_name="Area") 

            # Identify all columns that start with 'Area_'
            area_columns = [col for col in df.columns if col.startswith('Area_')]

            # Calculate Standard Deviation and Number of Results
            df['StandardDeviation'] = df[area_columns].std(axis=1)
            df['NumberOfResults'] = df[area_columns].notna().sum(axis=1)
            
            # Calculate Coefficient of Variation
            df['CV'] = (df['StandardDeviation'] / df[area_columns].mean(axis=1)) * 100

            # Retain only required columns
            output_df = df[['Compound_Results', 'RT1_Results', 'RT2_Results', 'Major_Results', 'StandardDeviation', 'NumberOfResults', 'CV']]

            # Save the processed data
            options = QFileDialog.Options()
            savePath, _ = QFileDialog.getSaveFileName(self, "Save Excel File", "", "Excel Files (*.xlsx *.xls);;All Files (*)", options=options)
            if savePath:
                output_df.to_excel(savePath, index=False)
                self.output_area.setText("File processed and saved successfully!")
            else:
                self.output_area.setText("Error saving the file!")

app = QApplication(sys.argv)
ex = DataCheckApp()
sys.exit(app.exec_())
