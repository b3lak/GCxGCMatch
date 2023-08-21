import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel
import pandas as pd

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Excel RT Converter'
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        self.label = QLabel('No file selected yet', self)
        
        browse_button = QPushButton('Browse File', self)
        browse_button.clicked.connect(self.browse_file)
        
        ok_button = QPushButton('OK', self)
        ok_button.clicked.connect(self.process_file)

        layout.addWidget(self.label)
        layout.addWidget(browse_button)
        layout.addWidget(ok_button)
        
        self.setLayout(layout)
        self.setWindowTitle(self.title)
        self.show()

    def browse_file(self):
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(self, "Open Excel File", "", "Excel Files (*.xlsx *.xlsx);;All Files (*)", options=options)
        if filePath:
            self.label.setText(filePath)
            self.input_file = filePath

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
        # Read Excel file once and store all sheets in memory
        with pd.ExcelFile(self.input_file) as xlsx:
            alkanes_df = pd.read_excel(xlsx, 'Alkanes')
            pah_df = pd.read_excel(xlsx, 'PAH')
            
            sheets_data = {}
            for sheet_name in xlsx.sheet_names:
                if sheet_name not in ['Alkanes', 'PAH']:
                    sample_df = pd.read_excel(xlsx, sheet_name)
                    sample_df["RT1"] = sample_df["RT1"].apply(lambda x: self.convert_rt1(x, alkanes_df))
                    sample_df["RT2"] = sample_df["RT2"].apply(lambda x: self.convert_rt2(x, pah_df))
                    sheets_data[sheet_name] = sample_df

        # Save to a new location
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Excel File", "", "Excel Files (*.xlsx *.xlsx);;All Files (*)", options=options)
        if filePath:
            with pd.ExcelWriter(filePath, engine='xlsxwriter') as writer:
                for sheet_name, data in sheets_data.items():
                    data.to_excel(writer, sheet_name=sheet_name, index=False)
                writer._save()



app = QApplication(sys.argv)
ex = App()
sys.exit(app.exec_())
