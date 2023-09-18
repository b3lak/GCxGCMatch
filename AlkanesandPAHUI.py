import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit, QFileDialog, QLabel

class ExcelProcessor(QWidget):
    def __init__(self):
        super().__init__()
        self.alkanes_df = None
        self.pah_df = None
        self.selected_file = None

        self.initUI()

    def initUI(self):
        self.setGeometry(300, 300, 600, 550)
        self.setWindowTitle('Excel Copier')

        layout = QVBoxLayout()

        # Instruction label
        self.instruction_label = QLabel("Paste the Alkanes table and press OK, then paste the PAH table and press OK.", self)
        layout.addWidget(self.instruction_label)

        self.prompt_label = QLabel("Paste Alkanes Table here:", self)
        layout.addWidget(self.prompt_label)

        self.paste_box = QTextEdit(self)
        layout.addWidget(self.paste_box)

        # OK Button
        self.ok_button = QPushButton('OK', self)
        self.ok_button.clicked.connect(self.processInput)
        layout.addWidget(self.ok_button)

        # Browse File Button
        self.browse_button = QPushButton('Browse File', self)
        self.browse_button.clicked.connect(self.browseFile)
        layout.addWidget(self.browse_button)

        # Create File Button
        self.createfile_button = QPushButton('Create File', self)
        self.createfile_button.clicked.connect(self.createAndSaveNewFile)
        layout.addWidget(self.createfile_button)

        # Label to display selected file path
        self.selected_file_label = QLabel("No file selected", self)
        layout.addWidget(self.selected_file_label)

        self.setLayout(layout)

    def processInput(self):
        if self.alkanes_df is None or self.alkanes_df.empty:
            self.alkanes_df = pd.read_clipboard(sep='\t')
            self.paste_box.clear()
            self.prompt_label.setText("Paste PAH Table here:")
        elif self.pah_df is None or self.pah_df.empty:
            self.pah_df = pd.read_clipboard(sep='\t')

    def browseFile(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Excel File", "", "Excel Files (*.xlsx);;All Files (*)", options=options)
        if file_name:
            self.selected_file = file_name
            self.selected_file_label.setText(f"Selected File: {file_name}")

    def createAndSaveNewFile(self):
        if not self.alkanes_df.empty and not self.pah_df.empty and self.selected_file:
            # Prompt the user to save the file first
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(self, "Save New File", "", "Excel Files (*.xlsx);;All Files (*)", options=options)
            if file_name:
                if not file_name.endswith('.xlsx'):
                    file_name += '.xlsx'
                
                # Save Alkanes and PAH to the new file
                with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
                    self.alkanes_df.to_excel(writer, sheet_name='Alkanes', index=False)
                    self.pah_df.to_excel(writer, sheet_name='PAH', index=False)

                # Now, append the sheets from the selected file to the new file
                with pd.ExcelFile(self.selected_file) as xls:
                    with pd.ExcelWriter(file_name, engine='openpyxl', mode='a') as writer:
                        for sheet_name in xls.sheet_names:
                            df = pd.read_excel(xls, sheet_name=sheet_name)
                            df.to_excel(writer, sheet_name=sheet_name, index=False)

                self.paste_box.clear()
                self.prompt_label.setText(f"Data saved to {file_name}")
                self.ok_button.setEnabled(False)  # Disable the OK button after saving
                self.createfile_button.setEnabled(False)  # Disable the Create File button after saving

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ExcelProcessor()
    window.show()
    sys.exit(app.exec_())
