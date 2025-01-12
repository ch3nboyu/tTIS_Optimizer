import sys
import yaml
import subprocess
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit, QFileDialog, QScrollArea, QFrame, QComboBox, QGroupBox
from PyQt5.QtCore import QProcess

class ConfigEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Config Editor")
        self.setGeometry(100, 100, 800, 600)

        self.config_file = 'config_tTIS.yaml'
        self.config_data = self.load_config()

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        layout = QHBoxLayout(main_widget)

        left_frame = QFrame()
        left_layout = QVBoxLayout(left_frame)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(left_frame)

        right_layout = QVBoxLayout()

        self.create_parameter_widgets(left_layout)
        self.create_buttons(right_layout)
        self.command_output = QTextEdit()
        self.command_output.setReadOnly(True)
        right_layout.addWidget(self.command_output)

        layout.addWidget(scroll_area)
        layout.addLayout(right_layout)

        self.setCentralWidget(main_widget)

    def create_parameter_widgets(self, layout):
        # General parameters
        general_group = QGroupBox("General Parameters")
        general_layout = QVBoxLayout(general_group)
        for key, value in self.config_data.items():
            if isinstance(value, str) or isinstance(value, int) or isinstance(value, float):
                label = QLabel(key)
                line_edit = QLineEdit(str(value))
                line_edit.setObjectName(f"general_{key}")
                general_layout.addWidget(label)
                general_layout.addWidget(line_edit)
        layout.addWidget(general_group)

        # Method selection
        method_group = QGroupBox("Method Selection")
        method_layout = QVBoxLayout(method_group)
        self.method_combo = QComboBox()
        self.method_combo.addItems(['GeneticAlgorithm', 'ParticleSwarm'])
        self.method_combo.currentIndexChanged.connect(self.update_method_params)
        method_layout.addWidget(QLabel("Method"))
        method_layout.addWidget(self.method_combo)
        layout.addWidget(method_group)

        # Method parameters
        self.method_params_group = QGroupBox("Method Parameters")
        self.method_params_layout = QVBoxLayout(self.method_params_group)
        self.update_method_params()
        layout.addWidget(self.method_params_group)

        # Case selection
        case_group = QGroupBox("Case Selection")
        case_layout = QVBoxLayout(case_group)
        self.case_combo = QComboBox()
        self.case_combo.addItems(['case1', 'case2'])
        self.case_combo.currentIndexChanged.connect(self.update_case_params)
        case_layout.addWidget(QLabel("Case"))
        case_layout.addWidget(self.case_combo)
        layout.addWidget(case_group)

        # Case parameters
        self.case_params_group = QGroupBox("Case Parameters")
        self.case_params_layout = QVBoxLayout(self.case_params_group)
        self.update_case_params()
        layout.addWidget(self.case_params_group)

    def update_method_params(self):
        while self.method_params_layout.count():
            widget = self.method_params_layout.takeAt(0).widget()
            if widget is not None:
                widget.deleteLater()

        method = self.method_combo.currentText()
        method_params = self.config_data.get('GA_para' if method == 'GeneticAlgorithm' else 'PSO_para', {})

        for key, value in method_params.items():
            label = QLabel(key)
            line_edit = QLineEdit(str(value))
            line_edit.setObjectName(f"method_{key}")
            self.method_params_layout.addWidget(label)
            self.method_params_layout.addWidget(line_edit)

    def update_case_params(self):
        while self.case_params_layout.count():
            widget = self.case_params_layout.takeAt(0).widget()
            if widget is not None:
                widget.deleteLater()

        case_key = self.case_combo.currentText()
        case_params = self.config_data[case_key]

        for key, value in case_params.items():
            label = QLabel(key)
            line_edit = QLineEdit(str(value))
            line_edit.setObjectName(f"case_{case_key}_{key}")
            self.case_params_layout.addWidget(label)
            self.case_params_layout.addWidget(line_edit)

    def create_buttons(self, layout):
        button_layout = QHBoxLayout()
        save_button = QPushButton("Save CFG")
        prepare_button = QPushButton("Prepare")
        optimize_button = QPushButton("Optimize")
        simulate_button = QPushButton("Simulate")

        save_button.clicked.connect(self.save_config)
        prepare_button.clicked.connect(lambda: self.run_script('prepare_headandLF.py'))
        optimize_button.clicked.connect(lambda: self.run_script('optimization_tTIS.py'))
        simulate_button.clicked.connect(lambda: self.run_script('performones_tTIS.py'))

        button_layout.addWidget(save_button)
        button_layout.addWidget(prepare_button)
        button_layout.addWidget(optimize_button)
        button_layout.addWidget(simulate_button)

        layout.addLayout(button_layout)

    def load_config(self):
        with open(self.config_file, 'r') as file:
            return yaml.safe_load(file)

    def save_config(self):
        for obj in self.findChildren(QLineEdit):
            section, *keys = obj.objectName().split('_')
            value = obj.text()
            try:
                value = eval(value)  # Convert to appropriate type
            except (NameError, SyntaxError):
                pass

            if section == "general":
                self.config_data[keys[0]] = value
            elif section == "method":
                method_key = 'GA_para' if self.method_combo.currentText() == 'GeneticAlgorithm' else 'PSO_para'
                self.config_data[method_key][keys[0]] = value
            elif section == "case":
                case_key = keys[0]
                param_key = keys[1]
                self.config_data[case_key][param_key] = value

        with open(self.config_file, 'w') as file:
            yaml.dump(self.config_data, file, default_flow_style=False)

    def run_script(self, script_name):
        process = QProcess(self)
        process.readyReadStandardOutput.connect(self.update_command_output)
        process.start(f"python {script_name}")

    def update_command_output(self):
        process = self.sender()
        output = bytes(process.readAllStandardOutput()).decode("utf-8")
        self.command_output.append(output)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    editor = ConfigEditor()
    editor.show()
    sys.exit(app.exec_())



