import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QTextEdit


class ColumnCheckerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Excel Column Checker'
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        self.label = QLabel('No file selected yet', self)
        
        self.browse_button = QPushButton('Browse File', self)
        self.browse_button.clicked.connect(self.browse_file)
        
        self.ok_button = QPushButton('OK', self)
        self.ok_button.clicked.connect(self.check_columns)

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

    def check_columns(self):
        expected_columns = [
            "Compound", "MF", "RMF", "RT1", "RT2", "Area", "Height", "Quant", "Group", 
            "Area %", "Signal to Noise", "Major", "Qual"
        ]

        issues_found = []

        with pd.ExcelFile(self.input_file) as xlsx:
            for sheet_name in xlsx.sheet_names:
                df = pd.read_excel(xlsx, sheet_name)
                missing_columns = [col for col in expected_columns if col not in df.columns]
                
                if missing_columns:
                    issues_found.append(f"Sheet '{sheet_name}' is missing columns: {', '.join(missing_columns)}")

        # Display results
        if not issues_found:
            self.output_area.setText("All sheets have the expected columns.")
        else:
            self.output_area.setText("\n".join(issues_found))


app = QApplication(sys.argv)
ex = ColumnCheckerApp()
sys.exit(app.exec_())
