import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QTextEdit

class ExcelColumnNameChanger(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.browse_button = QPushButton('Browse', self)
        self.browse_button.clicked.connect(self.open_file_dialog)

        self.ok_button = QPushButton('Ok', self)
        self.ok_button.clicked.connect(self.change_column_names)

        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)  # Makes the text box read-only

        layout.addWidget(self.browse_button)
        layout.addWidget(self.ok_button)
        layout.addWidget(self.text_edit)

        self.setLayout(layout)
        self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle('Excel Column Name Changer')
        self.show()

    def open_file_dialog(self):
        options = QFileDialog.Options()
        self.file_name, _ = QFileDialog.getOpenFileName(self, "Select Excel file", "",
                                                        "Excel Files (*.xlsx)", options=options)
        if self.file_name:
            self.text_edit.append(f'File selected: {self.file_name}')

    def change_column_names(self):
        if self.file_name:
            # Load workbook
            xls = pd.ExcelFile(self.file_name)

            # Open save file dialog
            options = QFileDialog.Options()
            output_file_name, _ = QFileDialog.getSaveFileName(self, "Save As", "",
                                                          "Excel Files (*.xlsx)", options=options)

            if output_file_name:
                try:
                    with pd.ExcelWriter(output_file_name) as writer:
                        for sheet_name in xls.sheet_names:
                            df = pd.read_excel(self.file_name, sheet_name=sheet_name)
                            df.rename(columns={'<sup>1</sup>t<sub>R</sub>': 'RT1', '<sup>2</sup>t<sub>R</sub>': 'RT2'}, inplace=True)
                            df.rename(columns={'Max Ion': 'Major', 'Qual Ion': 'Qual'}, inplace=True)
                            df.to_excel(writer, sheet_name=sheet_name, index=False)

                    self.text_edit.append(f"Column names changed and saved as {output_file_name}")
                except ValueError:
                    self.text_edit.append("Invalid file type!")
            else:
                self.text_edit.append("Please enter a filename!")
        else:
            self.text_edit.append("No file selected!")

def main():
    app = QApplication([])
    ex = ExcelColumnNameChanger()
    app.exec_()

if __name__ == '__main__':
    main()
