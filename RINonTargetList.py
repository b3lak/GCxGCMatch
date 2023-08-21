import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QPushButton, QVBoxLayout, QWidget
import numpy as np

class ExcelMatcher(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setGeometry(200, 200, 400, 300)

        self.label = QLabel(self)
        self.browse_button = QPushButton('Browse', self)
        self.ok_button = QPushButton('Ok', self)

        self.browse_button.clicked.connect(self.browse_file)
        self.ok_button.clicked.connect(self.match_rows)

        vbox = QVBoxLayout()
        vbox.addWidget(self.label)
        vbox.addWidget(self.browse_button)
        vbox.addWidget(self.ok_button)

        self.setLayout(vbox)

    def browse_file(self):
        options = QFileDialog.Options()
        self.fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "", "Excel Files (*.xlsx)", options=options)
        if self.fileName:
            self.label.setText('Chosen file: ' + self.fileName)

    def match_rows(self):
        excel_file = pd.ExcelFile(self.fileName)
        sheet_names = excel_file.sheet_names

        tolerances = {'RT1': 25.0, 'RT2': 25.0, 'Major': 0.12, 'Qual': 0.12}

        df_all = pd.read_excel(excel_file, sheet_name=sheet_names[0])

        for sheet in sheet_names[1:]:
            df = pd.read_excel(excel_file, sheet_name=sheet)

            masks = []
            for _, row in df.iterrows():
                mask = (
                    np.isclose(df_all['RT1'], row['RT1'], atol=tolerances['RT1']) &
                    np.isclose(df_all['RT2'], row['RT2'], atol=tolerances['RT2']) &
                    np.isclose(df_all['Major'], row['Major'], atol=tolerances['Major']) &
                    np.isclose(df_all['Qual'], row['Qual'], atol=tolerances['Qual'])
                )
                masks.append(not mask.any())

            df_new = df[masks]
            df_all = pd.concat([df_all, df_new], ignore_index=True)

        output_file_name, _ = QFileDialog.getSaveFileName(self, "Save As", "", "Excel Files (*.xlsx)", options=QFileDialog.Options())

        if output_file_name:
            with pd.ExcelWriter(output_file_name, engine='openpyxl') as writer:
                df_all.to_excel(writer, index=False, sheet_name="Results")
            self.label.setText('Saved results to: ' + output_file_name)



if __name__ == '__main__':
    app = QApplication([])
    ex = ExcelMatcher()
    ex.show()
    app.exec_()
