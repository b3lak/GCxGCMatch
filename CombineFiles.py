import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QLabel, QWidget, QFileDialog


class ExcelMerger(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Excel Merger")

        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.label = QLabel()
        layout.addWidget(self.label)

        self.button1 = QPushButton("Search list")
        self.button1.clicked.connect(self.load_search_list)
        layout.addWidget(self.button1)

        self.button2 = QPushButton("Sample Results")
        self.button2.clicked.connect(self.load_sample_results)
        layout.addWidget(self.button2)

        self.button3 = QPushButton("OK")
        self.button3.clicked.connect(self.merge_files)
        layout.addWidget(self.button3)

        self.search_list_file = None
        self.sample_results_file = None

    def load_search_list(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        self.search_list_file, _ = QFileDialog.getOpenFileName(self, "Open Search List", "", "Excel Files (*.xlsx *.xls)", options=options)
        if self.search_list_file:
            self.label.setText('Search List: ' + self.search_list_file)

    def load_sample_results(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        self.sample_results_file, _ = QFileDialog.getOpenFileName(self, "Open Sample Results", "", "Excel Files (*.xlsx *.xls)", options=options)
        if self.sample_results_file:
            self.label.setText('Sample Results: ' + self.sample_results_file)

    def merge_files(self):
        if self.search_list_file and self.sample_results_file:
            search_list_excel = pd.ExcelFile(self.search_list_file)
            sample_results_excel = pd.ExcelFile(self.sample_results_file)

            output_file_name, _ = QFileDialog.getSaveFileName(self, "Save As", "", "Excel Files (*.xlsx)", options=QFileDialog.Options())

            if output_file_name:
                with pd.ExcelWriter(output_file_name, engine='openpyxl') as writer:
                    for sheet in search_list_excel.sheet_names:
                        pd.read_excel(search_list_excel, sheet_name=sheet).to_excel(writer, sheet_name=sheet, index=False)

                    # Keep track of sheet names already written
                    written_sheets = set(search_list_excel.sheet_names)

                    for sheet in sample_results_excel.sheet_names:
                        output_sheet_name = sheet
                        counter = 2
                        # If sheet name exists, append " - 2", " - 3", and so on
                        while output_sheet_name in written_sheets:
                            output_sheet_name = f"{sheet} - {counter}"
                            counter += 1
                        
                        pd.read_excel(sample_results_excel, sheet_name=sheet).to_excel(writer, sheet_name=output_sheet_name, index=False)
                        written_sheets.add(output_sheet_name)

                self.label.setText('Merged files saved to: ' + output_file_name)



app = QApplication(sys.argv)

window = ExcelMerger()
window.show()

app.exec_()
