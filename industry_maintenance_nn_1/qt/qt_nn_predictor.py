import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QPushButton, QLineEdit, QLabel, QFileDialog, QMessageBox,
                            QGroupBox, QProgressBar, QFormLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5 import QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from ctypes import CDLL, c_char_p, c_int, c_float, c_void_p, CFUNCTYPE, POINTER, Structure, byref

# Logging setup
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define Rust struct TrainingProgress
class TrainingProgress(Structure):
    _fields_ = [
        ("epoch", c_int),
        ("training_accuracy", c_float),
        ("validation_accuracy", c_float),
        ("training_loss", c_float),
        ("validation_loss", c_float),
    ]

class PredictionProgress(Structure):
    _fields_ = [
        ("input_data", c_float * 4),
        ("probabilities", c_float * 4),
        ("predicted_class", c_int)
    ]

# Define Rust struct PredictionResult
class PredictionResult(Structure):
    _fields_ = [
        ("class_id", c_int),
        ("probabilities", c_float * 4),
        ("class_name", c_char_p)  
    ]

# Load Rust library
try:
    lib_path = os.path.expanduser("~/industry_maintenance_nn_1/target/release/libindustry_maintenance_nn.so")
    rust_lib = CDLL(lib_path)
    logger.info(f"Rust library loaded from: {lib_path}")

    # Define Rust FFI signatures
    progress_callback_type = CFUNCTYPE(None, c_void_p, TrainingProgress)

    rust_lib.train_industry_model_with_progress.argtypes = [
        c_char_p, c_int, c_char_p, c_char_p,
        POINTER(c_float), progress_callback_type, c_void_p
    ]
    rust_lib.train_industry_model_with_progress.restype = c_int

    rust_lib.predict_failure_type_with_progress.argtypes = [
        c_float, c_float, c_float, c_float, 
        c_char_p, CFUNCTYPE(None, c_void_p, PredictionProgress), c_void_p
    ]
    rust_lib.predict_failure_type_with_progress.restype = POINTER(PredictionResult)

    rust_lib.free_prediction_result.argtypes = [POINTER(PredictionResult)]
    rust_lib.free_prediction_result.restype = None

except Exception as e:
    logger.error(f"Error loading Rust library: {e}")
    rust_lib = None


class TrainingThread(QThread):
    update_progress = pyqtSignal(dict)
    training_complete = pyqtSignal(float)
    training_failed = pyqtSignal(str)

    def __init__(self, csv_path, epochs, learning_rate, plot_path, model_path):
        super().__init__()
        self.csv_path = csv_path
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.plot_path = plot_path
        self.model_path = model_path
        self._is_running = True

    def run(self):
        if not rust_lib:
            self.training_failed.emit("Rust library not loaded")
            return

        # The callback function needs to be properly indented inside run()
        @CFUNCTYPE(None, c_void_p, TrainingProgress)
        def progress_callback(context, progress):
            if self._is_running:
                self.update_progress.emit({
                    'epoch': progress.epoch,
                    'train_acc': progress.training_accuracy,
                    'val_acc': progress.validation_accuracy,
                    'train_loss': progress.training_loss,
                    'val_loss': progress.validation_loss
                })

        accuracy = c_float(0.0)

        success = rust_lib.train_industry_model_with_progress(
            self.csv_path.encode('utf-8'),
            self.epochs,
            self.plot_path.encode('utf-8'),
            self.model_path.encode('utf-8'),
            byref(accuracy),
            progress_callback,
            None
        )

        if success:
            self.training_complete.emit(accuracy.value)
        else:
            self.training_failed.emit("Training failed")

    def stop(self):
        self._is_running = False
        self.quit()
        self._is_running = False
        self.quit()


class NeuralNetworkGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Industry Maintenance Neural Network")
        self.setGeometry(100, 100, 1200, 900)
        
        # Elegant light theme styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f7fa;
            }
            QGroupBox {
                background-color: white;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
                color: #374151;
                font-weight: 500;
                font-size: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
                color: #4b5563;
            }
            QLabel {
                color: #4b5563;
                font-size: 12px;
            }
            QLineEdit {
                background-color: white;
                border: 1px solid #d1d5db;
                border-radius: 4px;
                padding: 6px;
                color: #111827;
                selection-background-color: #3b82f6;
            }
            QPushButton {
                background-color: white;
                border: 1px solid #d1d5db;
                border-radius: 6px;
                color: #111827;
                padding: 8px;
                min-width: 80px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #f3f4f6;
                border-color: #9ca3af;
            }
            QPushButton:pressed {
                background-color: #e5e7eb;
            }
            QPushButton:disabled {
                background-color: #f9fafb;
                color: #9ca3af;
            }
            QProgressBar {
                border: 1px solid #d1d5db;
                border-radius: 4px;
                text-align: center;
                background-color: white;
                color: #374151;
                height: 14px;
            }
            QProgressBar::chunk {
                background-color: #3b82f6;
                border-radius: 3px;
            }
        """)
        
        # Initialize state variables
        self.training_active = False
        self.model_loaded = False
        self.model_path = ""
        
        # Main widget with subtle gradient
        self.main_widget = QWidget()
        self.main_widget.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #f9fafb, stop:1 #f3f4f6);
        """)
        self.setCentralWidget(self.main_widget)
        
        # Main layout with proper spacing
        self.main_layout = QHBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(20)
        
        # Left Panel (Controls) - 40% width
        self.left_panel = QWidget()
        self.left_panel.setStyleSheet("background: transparent;")
        self.left_layout = QVBoxLayout(self.left_panel)
        self.left_layout.setAlignment(Qt.AlignTop)
        self.left_layout.setSpacing(15)
        self.left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Right Panel (Plots) - 60% width with card-like appearance
        self.right_panel = QWidget()
        self.right_panel.setStyleSheet("""
            background: white;
            border-radius: 8px;
            border: 1px solid #e5e7eb;
        """)
        self.right_layout = QVBoxLayout(self.right_panel)
        self.right_layout.setContentsMargins(15, 15, 15, 15)
        
        self.main_layout.addWidget(self.left_panel, 4)
        self.main_layout.addWidget(self.right_panel, 6)
        
        # Initialize UI components with elegant styling
        self.init_file_selection()
        self.init_training_controls()
        self.init_progress_display()
        self.init_prediction_controls()
        self.init_plots()
        
        # Customize plot appearance to match theme
        self.customize_plot_style()

    def customize_plot_style(self):
        """Apply elegant light theme to matplotlib plots"""
        self.figure.patch.set_facecolor('white')
        
        for ax in [self.ax1, self.ax2]:
            ax.set_facecolor('white')
            ax.title.set_color('#111827')
            ax.xaxis.label.set_color('#4b5563')
            ax.yaxis.label.set_color('#4b5563')
            ax.tick_params(axis='x', colors='#6b7280')
            ax.tick_params(axis='y', colors='#6b7280')
            ax.spines['bottom'].set_color('#d1d5db')
            ax.spines['top'].set_color('#d1d5db') 
            ax.spines['right'].set_color('#d1d5db')
            ax.spines['left'].set_color('#d1d5db')
            ax.grid(color='#e5e7eb', linestyle='--')
        
        # Update line colors to match theme
        self.loss_train_line.set_color('#ef4444')  # Red
        self.loss_val_line.set_color('#3b82f6')    # Blue
        self.acc_train_line.set_color('#10b981')   # Green
        self.acc_val_line.set_color('#8b5cf6')     # Violet
        
        # Update legend
        for ax in [self.ax1, self.ax2]:
            legend = ax.legend()
            if legend:
                legend.get_frame().set_facecolor('white')
                legend.get_frame().set_edgecolor('#e5e7eb')
                for text in legend.get_texts():
                    text.set_color('#374151')
        
        self.canvas.draw()

    def init_file_selection(self):
        group = QGroupBox("Dataset Selection")
        layout = QVBoxLayout()
        layout.setSpacing(10)
        
        self.file_label = QLabel("No file selected")
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet("""
            QLabel {
                background-color: #f9fafb;
                border: 1px solid #e5e7eb;
                border-radius: 4px;
                padding: 8px;
                color: #4b5563;
            }
        """)
        
        browse_btn = QPushButton("Browse CSV")
        browse_btn.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border: none;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
            QPushButton:pressed {
                background-color: #1d4ed8;
            }
        """)
        browse_btn.setIcon(QtGui.QIcon.fromTheme("document-open"))
        browse_btn.clicked.connect(self.browse_csv)
        
        layout.addWidget(self.file_label)
        layout.addWidget(browse_btn)
        group.setLayout(layout)
        
        self.left_layout.addWidget(group)

    def init_training_controls(self):
        group = QGroupBox("Training Parameters")
        layout = QFormLayout()
        layout.setVerticalSpacing(12)
        layout.setHorizontalSpacing(15)
        
        # Style input fields
        input_style = """
            QLineEdit {
                background-color: white;
                border: 1px solid #d1d5db;
                border-radius: 4px;
                padding: 6px;
            }
        """
        
        self.epochs_input = QLineEdit("500")
        self.epochs_input.setValidator(QtGui.QIntValidator(1, 99999))
        self.epochs_input.setStyleSheet(input_style)
        
        self.lr_input = QLineEdit("0.0005")
        self.lr_input.setValidator(QtGui.QDoubleValidator(0.0000001, 1.0, 7))
        self.lr_input.setStyleSheet(input_style)
        
        # Style training button
        self.train_btn = QPushButton("Start Training")
        self.train_btn.setStyleSheet("""
            QPushButton {
                background-color: #10b981;
                color: white;
                border: none;
                padding: 10px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #024733;
            }
            QPushButton:pressed {
                background-color: #024733;
            }
        """)
        self.train_btn.clicked.connect(self.start_training)
        self.train_btn.setEnabled(False)
        
        layout.addRow("Epochs:", self.epochs_input)
        layout.addRow("Learning Rate:", self.lr_input)
        layout.addRow(self.train_btn)
        
        group.setLayout(layout)
        self.left_layout.addWidget(group)

    def init_progress_display(self):
        group = QGroupBox("Training Progress")
        layout = QVBoxLayout()
        layout.setSpacing(10)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        
        # Progress labels with subtle styling
        label_style = """
            QLabel {
                color: #4b5563;
                font-size: 12px;
            }
        """
        
        self.epoch_label = QLabel("Epoch: 0/0")
        self.epoch_label.setStyleSheet(label_style)
        self.train_acc_label = QLabel("Training Accuracy: 0.0%")
        self.train_acc_label.setStyleSheet(label_style)
        self.val_acc_label = QLabel("Validation Accuracy: 0.0%")
        self.val_acc_label.setStyleSheet(label_style)
        self.train_loss_label = QLabel("Training Loss: 0.0")
        self.train_loss_label.setStyleSheet(label_style)
        self.val_loss_label = QLabel("Validation Loss: 0.0")
        self.val_loss_label.setStyleSheet(label_style)
        
        layout.addWidget(self.progress_bar)
        layout.addSpacing(5)
        layout.addWidget(self.epoch_label)
        layout.addWidget(self.train_acc_label)
        layout.addWidget(self.val_acc_label)
        layout.addWidget(self.train_loss_label)
        layout.addWidget(self.val_loss_label)
        
        group.setLayout(layout)
        self.left_layout.addWidget(group)

    def init_plots(self):
        group = QGroupBox("Training Metrics")
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Create figure with light theme
        self.figure = Figure(figsize=(7, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        
        # Create subplots
        self.ax1 = self.figure.add_subplot(211)
        self.ax2 = self.figure.add_subplot(212)
        
        # Configure loss plot with elegant colors
        self.ax1.set_title('Training & Validation Loss')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.loss_train_line, = self.ax1.plot([], [], color='#ef4444', linewidth=2, label='Training Loss')
        self.loss_val_line, = self.ax1.plot([], [], color='#3b82f6', linewidth=2, label='Validation Loss')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
        # Configure accuracy plot with elegant colors
        self.ax2.set_title('Training & Validation Accuracy')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Accuracy')
        self.acc_train_line, = self.ax2.plot([], [], color='#10b981', linewidth=2, label='Training Accuracy')
        self.acc_val_line, = self.ax2.plot([], [], color='#8b5cf6', linewidth=2, label='Validation Accuracy')
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.3)
        
        self.figure.tight_layout()
        layout.addWidget(self.canvas)
        group.setLayout(layout)
        self.right_layout.addWidget(group)

    def init_prediction_controls(self):
        group = QGroupBox("Prediction")
        layout = QFormLayout()
        layout.setVerticalSpacing(12)
        layout.setHorizontalSpacing(15)
        
        # Style input fields
        input_style = """
            QLineEdit {
                background-color: white;
                border: 1px solid #d1d5db;
                border-radius: 4px;
                padding: 6px;
            }
        """
        
        self.air_temp_input = QLineEdit()
        self.air_temp_input.setStyleSheet(input_style)
        self.process_temp_input = QLineEdit()
        self.process_temp_input.setStyleSheet(input_style)
        self.rotational_speed_input = QLineEdit()
        self.rotational_speed_input.setStyleSheet(input_style)
        self.torque_input = QLineEdit()
        self.torque_input.setStyleSheet(input_style)
        
        # Style predict button
        self.predict_btn = QPushButton("Predict")
        self.predict_btn.setStyleSheet("""
            QPushButton {
                background-color: #f59e0b;
                color: white;
                border: none;
                padding: 10px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #d97706;
            }
            QPushButton:pressed {
                background-color: #b45309;
            }
        """)
        self.predict_btn.clicked.connect(self.predict)
        self.predict_btn.setEnabled(False)
        
        # Style result display
        self.result_label = QLabel("Result: ")
        self.result_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: 500;
                color: #3b82f6;
            }
        """)
        
        # Style probability labels with different colors
        colors = ['#ef4444', '#f59e0b', '#10b981', '#8b5cf6']
        self.probability_labels = []
        class_names = [
            "Power Failure",
            "Overstrain Failure",
            "No Failure",
            "Heat Dissipation Failure"
        ]
        
        for i, name in enumerate(class_names):
            label = QLabel(f"{name}: 0%")
            label.setStyleSheet(f"""
                QLabel {{
                    color: {colors[i]};
                    font-size: 12px;
                }}
            """)
            self.probability_labels.append(label)
        
        # Add widgets to layout
        layout.addRow("Air Temperature (K):", self.air_temp_input)
        layout.addRow("Process Temperature (K):", self.process_temp_input)
        layout.addRow("Rotational Speed (rpm):", self.rotational_speed_input)
        layout.addRow("Torque (Nm):", self.torque_input)
        layout.addRow(self.predict_btn)
        layout.addRow(self.result_label)
        
        for label in self.probability_labels:
            layout.addRow(label)
        
        group.setLayout(layout)
        self.left_layout.addWidget(group)

    def browse_csv(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select CSV File", "", "CSV Files (*.csv)", options=options)
        
        if file_name:
            self.file_label.setText(file_name)
            self.train_btn.setEnabled(True)

    def start_training(self):
        if not self.training_active:
            csv_path = self.file_label.text()
            if not csv_path or not os.path.exists(csv_path):
                QMessageBox.warning(self, "Error", "Please select a valid CSV file first.")
                return
            
            try:
                epochs = int(self.epochs_input.text())
                learning_rate = float(self.lr_input.text())
            except ValueError:
                QMessageBox.warning(self, "Error", "Please enter valid numbers for epochs and learning rate.")
                return
            
            # Reset UI
            self.progress_bar.setValue(0)
            self.epoch_label.setText(f"Epoch: 0/{epochs}")
            self.train_acc_label.setText("Training Accuracy: 0.0%")
            self.val_acc_label.setText("Validation Accuracy: 0.0%")
            self.train_loss_label.setText("Training Loss: 0.0")
            self.val_loss_label.setText("Validation Loss: 0.0")
            
            # Clear previous plot data and reset plots
            self.ax1.clear()
            self.ax2.clear()
            
            # Reinitialize plot settings
            self.ax1.set_title('Training & Validation Loss')
            self.ax1.set_xlabel('Epoch')
            self.ax1.set_ylabel('Loss')
            self.loss_train_line, = self.ax1.plot([], [], 'r-', label='Training Loss')
            self.loss_val_line, = self.ax1.plot([], [], 'b-', label='Validation Loss')
            self.ax1.legend()
            self.ax1.grid(True)
            
            self.ax2.set_title('Training & Validation Accuracy')
            self.ax2.set_xlabel('Epoch')
            self.ax2.set_ylabel('Accuracy')
            self.acc_train_line, = self.ax2.plot([], [], 'g-', label='Training Accuracy')
            self.acc_val_line, = self.ax2.plot([], [], 'm-', label='Validation Accuracy')
            self.ax2.legend()
            self.ax2.grid(True)
            
            self.figure.tight_layout()
            self.canvas.draw()
            
            # Set output paths
            plot_path = os.path.join(os.path.dirname(csv_path), "training_plot.png")
            model_path = os.path.join(os.path.dirname(csv_path), "model.bin")
            self.model_path = model_path

            try:
                # Ensure directories exist
                os.makedirs(os.path.dirname(plot_path), exist_ok=True)
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
            except OSError as e:
                QMessageBox.warning(self, "Error", f"Could not create output directories: {str(e)}")
                return
            
            # Create and start training thread
            self.training_thread = TrainingThread(
                csv_path, epochs, learning_rate, plot_path, model_path)
            
            self.training_thread.update_progress.connect(self.update_training_progress)
            self.training_thread.training_complete.connect(self.training_completed)
            self.training_thread.training_failed.connect(self.training_failed)
            
            self.training_active = True
            self.train_btn.setText("Training...")
            self.training_thread.start()
        else:
            if self.training_thread:
                self.training_thread.stop()
                self.training_thread.wait()
            
            self.training_active = False
            self.train_btn.setText("Start Training")

    def update_training_progress(self, progress_data):
        total_epochs = int(self.epochs_input.text())
        epoch = progress_data['epoch']
        
        # Initialize data structures if this is the first epoch
        if epoch == 0:
            self.loss_x_data = []
            self.loss_train_data = []
            self.loss_val_data = []
            self.acc_x_data = []
            self.acc_train_data = []
            self.acc_val_data = []
        
        progress = int((epoch + 1) / total_epochs * 100)
        train_acc = progress_data['train_acc']
        val_acc = progress_data['val_acc']
        train_loss = progress_data['train_loss']
        val_loss = progress_data['val_loss']
        
        # Update progress bar and labels
        self.progress_bar.setValue(progress)
        self.epoch_label.setText(f"Epoch: {epoch + 1}/{total_epochs}")
        self.train_acc_label.setText(f"Training Accuracy: {train_acc * 100:.2f}%")
        self.val_acc_label.setText(f"Validation Accuracy: {val_acc * 100:.2f}%")
        self.train_loss_label.setText(f"Training Loss: {train_loss:.4f}")
        self.val_loss_label.setText(f"Validation Loss: {val_loss:.4f}")
        
        # Update plot data
        self.loss_x_data.append(epoch)
        self.loss_train_data.append(train_loss)
        self.loss_val_data.append(val_loss)
        
        self.acc_x_data.append(epoch)
        self.acc_train_data.append(train_acc)
        self.acc_val_data.append(val_acc)
        
        # Update plots
        self.loss_train_line.set_data(self.loss_x_data, self.loss_train_data)
        self.loss_val_line.set_data(self.loss_x_data, self.loss_val_data)
        self.acc_train_line.set_data(self.acc_x_data, self.acc_train_data)
        self.acc_val_line.set_data(self.acc_x_data, self.acc_val_data)
        
        # Adjust axes limits
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        
        # Redraw canvas
        self.canvas.draw()

    def training_completed(self, final_accuracy):
        self.training_active = False
        self.train_btn.setText("Start Training")
        self.predict_btn.setEnabled(True)
        self.model_loaded = True

    def training_failed(self, message):
        self.training_active = False
        self.train_btn.setText("Start Training")
        QMessageBox.critical(self, "Training Failed", message)

    def predict(self):
        if not hasattr(self, 'model_path') or not os.path.exists(self.model_path):
            QMessageBox.warning(self, "Error", "No trained model available. Please train a model first.")
            return

        try:
            # Get input values
            air_temp = float(self.air_temp_input.text())
            process_temp = float(self.process_temp_input.text())
            rotational_speed = float(self.rotational_speed_input.text())
            torque = float(self.torque_input.text())
        except ValueError:
            QMessageBox.warning(self, "Error", "Please enter valid numbers for all input parameters.")
            return

        try:
            # Define callback type
            PREDICTION_CALLBACK = CFUNCTYPE(None, c_void_p, PredictionProgress)
            
            # Define callback function
            @PREDICTION_CALLBACK
            def prediction_callback(context, progress):
                logger.info(f"Prediction Progress - Class: {progress.predicted_class}, Probs: {list(progress.probabilities)}")

            # Call Rust prediction function
            result_ptr = rust_lib.predict_failure_type_with_progress(
                c_float(air_temp),
                c_float(process_temp),
                c_float(rotational_speed),
                c_float(torque),
                self.model_path.encode('utf-8'),
                prediction_callback,
                None
            )
            
            if not result_ptr:
                QMessageBox.warning(self, "Error", "Prediction returned null pointer")
                return

            result = result_ptr.contents
            
            # Get class name from Rust
            class_name = result.class_name.decode('utf-8') if result.class_name else "Unknown"
            self.result_label.setText(f"Result: {class_name}")

            # Update probability labels
            class_names = [
                "Power Failure",
                "Overstrain Failure",
                "No Failure",
                "Heat Dissipation Failure"
            ]
            
            for i, label in enumerate(self.probability_labels):
                prob = result.probabilities[i] * 100
                label.setText(f"{class_names[i]}: {prob:.2f}%")
            
            # Free memory allocated by Rust
            rust_lib.free_prediction_result(result_ptr)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = NeuralNetworkGUI()
    window.show()
    
    sys.exit(app.exec_())	