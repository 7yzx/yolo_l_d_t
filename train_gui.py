# version 0.2.0 (PyQt5)

import sys
import os
from queue import Queue, Empty
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QLineEdit, QTextEdit, QComboBox, QProgressBar, QFileDialog,
    QDockWidget, QStackedWidget, QRadioButton, QButtonGroup, QSizePolicy,
    QMessageBox, QSpacerItem
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon, QPalette, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
import shutil
import multiprocessing
import json
# Assuming these modules exist and have been adapted or are compatible
from src.train import create_yaml # Keep this, might need adjustments
# Attempt to import the training function
try:
    from src.train import train_yolo
except ImportError:
    # Define a placeholder if import fails, so the app can still load.
    # Training will fail later if the function is truly missing.
    print("Warning: Could not import 'train_yolo' from src.train. Training will not work.")
    def train_yolo(*args, **kwargs):
        raise NotImplementedError("train_yolo function not found in src.train")

# --- Application State Class ---
class AppState:
    def __init__(self, base_path=None): # Accept optional base_path
        self.project_name = ""
        self.train_data_path = ""
        self.base_model_save_path = "" # Store the user-selected base path
        self.model_save_path = ""      # Store the full path (base + project name)
        self.selected_model_size = ""
        self.input_size = ""
        self.epochs = ""
        self.batch_size = ""
        self.class_names = []

        # Use provided base_path if available, otherwise default
        if base_path and os.path.isdir(base_path):
            self.base_model_save_path = os.path.join(base_path, "output")
        else:
            # Fallback to original behavior if base_path is not provided or invalid
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.base_model_save_path = os.path.join(script_dir, "output") # Base path is now ./output/ relative to this script

        print(f"AppState Initialized with base_model_save_path: {self.base_model_save_path}") # Debug print
        os.makedirs(self.base_model_save_path, exist_ok=True) # Ensure it exists


# --- Helper Functions ---

def model_name_to_type(model_name):
    # Keep this mapping function
    model_map = {
        "YOLOv8-Nano": "yolov8n", "YOLOv8-Small": "yolov8s", "YOLOv8-Medium": "yolov8m", "YOLOv8-Large": "yolov8l", "YOLOv8-ExtraLarge": "yolov8x",
        "YOLOv9-Compact": "yolov9c", "YOLOv9-Enhanced": "yolov9e",
        "YOLOv10-Nano": "yolov10n", "YOLOv10-Small": "yolov10s", "YOLOv10-Medium": "yolov10m", "YOLOv10-Balanced": "yolov10b", "YOLOv10-Large": "yolov10l", "YOLOv10-ExtraLarge": "yolov10x",
        "YOLOv11-Nano": "yolo11n", "YOLOv11-Small": "yolo11s", "YOLOv11-Medium": "yolo11m", "YOLOv11-Large": "yolo11l", "YOLOv11-ExtraLarge": "yolo11x", # Corrected v11 names if needed
    }
    return model_map.get(model_name, "")

# --- PyQt Threads for Background Tasks ---

class TrainingThread(QThread):
    output_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(object) # New: Pass result or exception on finish

    def __init__(self, train_func, *args, **kwargs): # New init: accept function and args
        super().__init__()
        self.train_func = train_func
        self.args = args
        self.kwargs = kwargs
        self._is_running = True

    def run(self):
        result = None
        error = None
        self._is_running = True
        try:
            # Note: Standard output from train_func will go to the console,
            # not automatically to output_signal unless train_func is modified
            # or stdout is redirected (which is complex).
            self.output_signal.emit("--- Starting Training Function ---")
            # Call the actual training function passed during initialization
            result = self.train_func(*self.args, **self.kwargs) # Execute the function

            if not self._is_running: # Check flag after function returns
                 self.output_signal.emit("--- Training Cancelled (function finished) ---")
            else:
                 self.output_signal.emit("--- Training Function Finished ---")
                 if result: # If the function returns something, emit it
                     self.output_signal.emit(f"Result: {result}")
        except Exception as e:
            error = e
            self.output_signal.emit(f"Error during training function: {e}\n")
        finally:
            self._is_running = False
            # Emit the error object if one occurred, otherwise emit the result
            self.finished_signal.emit(error if error else result)


    def stop(self):
        if self._is_running:
            self.output_signal.emit("--- Stop requested (Note: May not interrupt the running function immediately) ---")
            self._is_running = False


class TrainWidget(QWidget):
    def __init__(self, app_state, parent=None): # Accept app_state
        super().__init__(parent)
        self.app_state = app_state # Store app_state
        self.training_thread = None
        self._init_ui()
        # Connect project name changes after UI is initialized
        self.project_name_entry.editingFinished.connect(self._update_project_name_and_path)


    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        # --- Left Side (Inputs) ---
        form_layout = QVBoxLayout() # Use QVBoxLayout for simpler stacking

        # Project Name
        form_layout.addWidget(QLabel("项目名称: 请在下方输入项目名称"))
        self.project_name_entry = QLineEdit(placeholderText="项目名称")
        # Connect signal in __init__ after this line exists
        form_layout.addWidget(self.project_name_entry)

        # Train Data Path
        form_layout.addWidget(QLabel("选择训练数据:"))
        self.train_data_button = QPushButton("选择训练数据文件夹")
        self.train_data_button.clicked.connect(self.select_train_data)
        form_layout.addWidget(self.train_data_button)
        self.train_data_label = QLabel("未选择") # To display selected path
        self.train_data_label.setStyleSheet("color: red;") # Initial color red
        form_layout.addWidget(self.train_data_label)
        # Add a label for classes.txt status
        self.classes_status_label = QLabel("类别文件 (classes.txt): 未加载")
        form_layout.addWidget(self.classes_status_label)


        # Model Save Path
        form_layout.addWidget(QLabel("选择保存文件夹:"))
        self.model_save_button = QPushButton("选择模型保存路径")
        self.model_save_button.clicked.connect(self.select_model_save_folder)
        form_layout.addWidget(self.model_save_button)
        self.model_save_label = QLabel("未选择") # To display selected path
        self.model_save_label.setStyleSheet("color: red;") # Initial color red
        form_layout.addWidget(self.model_save_label)

        # Model Selection
        form_layout.addWidget(QLabel("选择YOLO模型:"))
        self.model_menu = QComboBox()
        model_options = ["YOLOv8-Nano", "YOLOv8-Small", "YOLOv8-Medium", "YOLOv8-Large", "YOLOv8-ExtraLarge",
                         "YOLOv9-Compact", "YOLOv9-Enhanced",
                         "YOLOv10-Nano", "YOLOv10-Small", "YOLOv10-Medium", "YOLOv10-Balanced", "YOLOv10-Large", "YOLOv10-ExtraLarge",
                         "YOLOv11-Nano", "YOLOv11-Tiny", "YOLOv11-Medium","YOLOv11-Large","YOLOv11-ExtraLarge"]
        self.model_menu.addItems(model_options)
        form_layout.addWidget(self.model_menu)

        # Epochs
        form_layout.addWidget(QLabel("训练轮数: 【默认: 100】"))
        self.epochs_entry = QLineEdit("100") # Default value 100
        self.epochs_entry.setPlaceholderText("训练轮数")
        form_layout.addWidget(self.epochs_entry)

        # Batch Size
        form_layout.addWidget(QLabel("批量大小: 【默认: 4】"))
        self.batch_size_entry = QLineEdit("4") # Default value 4
        self.batch_size_entry.setPlaceholderText("批量大小")
        form_layout.addWidget(self.batch_size_entry)

        # --- Add Usage Guide to form_layout ---
        form_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Minimum)) # Add some space before guide
        guide_label = QLabel("操作指南：")
        guide_label.setStyleSheet("font-weight: bold;") # Make title bold
        form_layout.addWidget(guide_label)
        form_layout.addWidget(QLabel("1. 首先输入自定义项目名称"))
        form_layout.addWidget(QLabel("2. 选择标注好的文件目录 (包含images, labels, classes.txt)"))
        form_layout.addWidget(QLabel("3. 直接开始训练"))
        form_layout.addWidget(QLabel("(4.会自动选择保存在代码目录的output文件夹作为模型保存路径)"))
        # --- End Usage Guide ---


        left_layout.addLayout(form_layout)
        left_layout.addStretch(1) # Push controls to the top

        # --- Control Buttons Layout ---
        button_layout = QHBoxLayout() # Layout for Start and Cancel buttons

        # Start Training Button
        self.start_train_button = QPushButton("开始训练!")
        self.start_train_button.setStyleSheet("QPushButton { background-color: chocolate; color: white; font-size: 20px; font-weight: bold; padding: 10px; }")
        self.start_train_button.clicked.connect(self.start_training)
        button_layout.addWidget(self.start_train_button)

        # Cancel Training Button
        self.cancel_train_button = QPushButton("取消训练")
        self.cancel_train_button.setStyleSheet("QPushButton { background-color: gray; color: white; font-size: 20px; font-weight: bold; padding: 10px; }")
        self.cancel_train_button.clicked.connect(self.cancel_training)
        self.cancel_train_button.setEnabled(False) # Initially disabled
        self.cancel_train_button.setVisible(False) # Initially hidden
        button_layout.addWidget(self.cancel_train_button)

        left_layout.addLayout(button_layout) # Add the button layout

        # --- Right Side (Output) ---
        right_layout.addWidget(QLabel("训练输出:"))
        self.output_textbox = QTextEdit()
        self.output_textbox.setReadOnly(True)
        right_layout.addWidget(self.output_textbox)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0) # Indeterminate
        self.progress_bar.setVisible(False) # Initially hidden
        right_layout.addWidget(self.progress_bar)

        # --- Add layouts to main layout ---
        main_layout.addLayout(left_layout, 1) # Left takes 1 part
        main_layout.addLayout(right_layout, 2) # Right takes 2 parts

    def select_train_data(self):
        path = QFileDialog.getExistingDirectory(self, "选择训练数据文件夹 (包含images, labels, classes.txt)")
        if path:
            self.app_state.train_data_path = path # Use app_state
            self.train_data_label.setText(f"数据文件夹: {os.path.basename(path)}") # Show folder name
            self.train_data_label.setStyleSheet("color: green;") # Change color to green

            # --- Automatically set base model save path ---
            try:
                self._update_full_model_save_path()
            except Exception as e:
                print(f"Error setting suggested base model save path: {e}")
                # Optionally reset or handle the error, for now, just print it
                self.app_state.base_model_save_path = ""
                self._update_full_model_save_path() # Update label to reflect reset
                # Trigger project list refresh in other widgets even on error (to clear lists)



            # Attempt to read classes.txt
            classes_file_path = os.path.join(self.app_state.train_data_path, "classes.txt") # Use app_state
            self.app_state.class_names = [] # Reset class names in app_state
            if os.path.exists(classes_file_path):
                try:
                    with open(classes_file_path, 'r', encoding='utf-8') as f:
                        self.app_state.class_names = [line.strip() for line in f if line.strip()] # Use app_state
                    if self.app_state.class_names: # Use app_state
                        self.classes_status_label.setText(f"类别文件 (classes.txt): 已加载 ({len(self.app_state.class_names)} 类)") # Use app_state
                        self.classes_status_label.setStyleSheet("color: green;")
                        print(f"Loaded classes: {self.app_state.class_names}") # Use app_state
                    else:
                        self.classes_status_label.setText("类别文件 (classes.txt): 文件为空")
                        self.classes_status_label.setStyleSheet("color: orange;")
                except Exception as e:
                    self.classes_status_label.setText(f"类别文件 (classes.txt): 读取错误 - {e}")
                    self.classes_status_label.setStyleSheet("color: red;")
                    self.app_state.class_names = [] # Use app_state
            else:
                self.classes_status_label.setText("类别文件 (classes.txt): 未找到")
                self.classes_status_label.setStyleSheet("color: red;")
                self.app_state.class_names = [] # Use app_state

    def select_model_save_folder(self):
        # This button allows overriding the automatically suggested path
        path = QFileDialog.getExistingDirectory(self, "选择模型保存的基础路径 (将在此路径下创建项目文件夹)")
        if path:
            self.app_state.base_model_save_path = path # Store the selected base path
            self._update_full_model_save_path() # Update the full path and label


    def _update_project_name_and_path(self):
        """Handles updates when project name changes."""
        self.app_state.project_name = self.project_name_entry.text().strip()
        # Always update the display path based on the new project name and existing base path (if any)
        self._update_full_model_save_path()

    def _update_full_model_save_path(self):
        """Constructs the full model save path if possible and updates the UI label."""
        project_name = self.app_state.project_name
        base_path = self.app_state.base_model_save_path

        if base_path and project_name:
            # Both base path and project name are set
            self.app_state.model_save_path = os.path.join(base_path, project_name)
            display_path = self.app_state.model_save_path
            self.model_save_label.setText(f"保存路径: {display_path}")
            self.model_save_label.setStyleSheet("color: green;")
        elif base_path and not project_name:
            # Base path set, but project name is missing/empty
            self.app_state.model_save_path = "" # Clear the full path as it's incomplete
            display_path = f"{base_path} (请输入项目名称)"
            self.model_save_label.setText(f"保存路径: {display_path}")
            self.model_save_label.setStyleSheet("color: orange;") # Indicate pending action
        elif not base_path and project_name:
            # Project name set, but base path is missing
            self.app_state.model_save_path = "" # Clear the full path as it's incomplete
            display_path = f"(请选择基础路径)/{project_name}"
            self.model_save_label.setText(f"保存路径: {display_path}")
            self.model_save_label.setStyleSheet("color: orange;") # Indicate pending action
        else:
            # Neither base path nor project name is set
            self.app_state.model_save_path = "" # Clear the full path
            self.model_save_label.setText("未选择")
            self.model_save_label.setStyleSheet("color: red;") # Initial state

    def start_training(self):
        # Get values from UI elements and store/update app_state
        # Project name is already updated by _update_project_name_and_path
        # self.app_state.project_name = self.project_name_entry.text().strip() # No longer needed here

        # Ensure the full model save path is set correctly before proceeding
        # This handles the case where the user enters the project name *after*
        # selecting the folder but *before* clicking start.
        self._update_project_name_and_path() # Call again to be sure

        epochs_str = self.epochs_entry.text().strip()
        batch_size_str = self.batch_size_entry.text().strip()
        selected_model_name = self.model_menu.currentText()
        self.app_state.selected_model_size = model_name_to_type(selected_model_name) # Use app_state

        # --- Hardcode input size ---
        self.app_state.input_size = 640 # Use app_state

        # --- Validation ---
        if not self.app_state.project_name: # Use app_state
            QMessageBox.warning(self, "输入错误", "请输入项目名称。")
            return
        if not self.app_state.train_data_path: # Use app_state
            QMessageBox.warning(self, "输入错误", "请选择训练数据文件夹。")
            return
        if not self.app_state.class_names: # Use app_state
            QMessageBox.warning(self, "输入错误", "未能从 classes.txt 加载类别名称，请检查文件是否存在且包含内容。")
            return
        if not self.app_state.base_model_save_path: # Check if base path was selected
             QMessageBox.warning(self, "输入错误", "请选择模型保存的基础路径。")
             return
        # Check the derived model_save_path which depends on project name being set
        if not self.app_state.model_save_path:
             QMessageBox.warning(self, "输入错误", "模型保存路径无效 (项目名称可能未设置?)。")
             return
        if not self.app_state.selected_model_size: # Use app_state
             QMessageBox.warning(self, "输入错误", "请选择有效的YOLO模型。")
             return
        if not epochs_str.isdigit() or int(epochs_str) <= 0:
            QMessageBox.warning(self, "输入错误", "请输入有效的训练轮数 (正整数)。")
            return
        if not batch_size_str.isdigit() or int(batch_size_str) <= 0:
            QMessageBox.warning(self, "输入错误", "请输入有效的批量大小 (正整数)。")
            return

        self.app_state.epochs = int(epochs_str) # Use app_state
        self.app_state.batch_size = int(batch_size_str) # Use app_state

        # --- Create YAML ---
        try:
            # Pass the already loaded class_names list from app_state
            # Use the final model_save_path which includes the project name
            if not os.path.exists(self.app_state.model_save_path):
                os.makedirs(self.app_state.model_save_path, exist_ok=True) # Ensure the directory exists
            classes_path = os.path.join(self.app_state.train_data_path, "classes.txt") # Use app_state
            if not os.path.exists(classes_path):
                QMessageBox.critical(self, "错误", f"类别文件不存在: {classes_path}")
                return
            else:
                shutil.copy2(classes_path, self.app_state.model_save_path) # Copy classes.txt to model save path
            yaml_path = create_yaml(self.app_state.project_name, self.app_state.train_data_path, self.app_state.class_names, self.app_state.model_save_path) # Use final app_state path
            if not yaml_path or not os.path.exists(yaml_path):
                 QMessageBox.critical(self, "错误", f"创建或找到YAML文件失败: {yaml_path}")
                 return

            # --- Prepare arguments for train_yolo function ---
            # Ensure train_yolo is available (imported or placeholder defined)
            if not callable(train_yolo):
                 QMessageBox.critical(self, "错误", "'train_yolo' function is not available or callable.")
                 return

            # Define arguments based on the expected signature of train_yolo
            # Example signature: train_yolo(yaml_path, model_type, img_size, batch_size, epochs, save_dir, project_name)
            # Adjust these arguments to match your actual train_yolo function signature
            train_args = (
                yaml_path,
                self.app_state.selected_model_size, # model_type
                self.app_state.input_size,         # img_size
                self.app_state.batch_size,         # batch_size
                self.app_state.epochs,             # epochs
                self.app_state.model_save_path,    # save_dir (or base path where project will be created)
                self.app_state.project_name        # project_name
                # Add other necessary arguments like device, workers, etc. if needed
            )


            self.output_textbox.clear()
            # self.output_textbox.append(f"Starting training with command:\n{' '.join(cmd_args)}\n---") # Old message
            self.output_textbox.append(f"--- Preparing to call training function ---")
            self.output_textbox.append(f"Project: {self.app_state.project_name}")
            self.output_textbox.append(f"Model: {self.app_state.selected_model_size}")
            self.output_textbox.append(f"Epochs: {self.app_state.epochs}, Batch Size: {self.app_state.batch_size}")
            self.output_textbox.append(f"Save Path: {self.app_state.model_save_path}")
            self.output_textbox.append(f"YAML: {yaml_path}")
            self.output_textbox.append("---")

            self.progress_bar.setVisible(True)
            self.start_train_button.setEnabled(False)
            self.start_train_button.setText("训练中...")
            self.cancel_train_button.setEnabled(True) # Enable cancel button
            self.cancel_train_button.setVisible(True) # Show cancel button
            self.cancel_train_button.setStyleSheet("QPushButton { background-color: red; color: white; font-size: 20px; font-weight: bold; padding: 10px; }") # Make it red

            # Run in a separate thread using the modified TrainingThread
            # self.training_thread = TrainingThread(cmd_args) # Old thread creation
            self.training_thread = TrainingThread(train_yolo, *train_args) # New: Pass function and args
            self.training_thread.output_signal.connect(self.update_output)
            self.training_thread.finished_signal.connect(self.on_training_finished) # Connect new signal
            self.training_thread.start()

        except Exception as e:
            QMessageBox.critical(self, "启动错误", f"启动训练时出错: {e}")
            # ... (reset UI elements as before) ...
            self.progress_bar.setVisible(False)
            self.start_train_button.setEnabled(True)
            self.start_train_button.setText("开始训练!")
            self.cancel_train_button.setEnabled(False)
            self.cancel_train_button.setVisible(False)
            self.cancel_train_button.setStyleSheet("QPushButton { background-color: gray; color: white; font-size: 20px; font-weight: bold; padding: 10px; }")

    def cancel_training(self):
        if self.training_thread and self.training_thread.isRunning():
            self.output_textbox.append("\n--- Sending Cancel Request ---")
            self.training_thread.stop()
            self.cancel_train_button.setEnabled(False) # Disable cancel button immediately
            self.cancel_train_button.setText("正在取消...")
            # The rest of the UI reset will happen in on_training_finished when the thread actually stops

    def update_output(self, text):
        self.output_textbox.append(text.strip())
        self.output_textbox.verticalScrollBar().setValue(self.output_textbox.verticalScrollBar().maximum()) # Auto-scroll

    def on_training_finished(self, result_or_error): # Modify to accept argument from signal
        # Check if cancellation was requested
        # The flag self.training_thread._is_running might be False if stop() was called,
        # but the function might have completed normally anyway.
        # Relying on button state might be more indicative of user action.
        was_cancelled = not self.cancel_train_button.isEnabled() and "取消" in self.cancel_train_button.text()

        if isinstance(result_or_error, Exception):
            self.output_textbox.append(f"--- Training Failed: {result_or_error} ---")
        elif was_cancelled:
             # Check if stop() was called - thread's _is_running might be False
             is_thread_marked_stopped = hasattr(self.training_thread, '_is_running') and not self.training_thread._is_running
             if is_thread_marked_stopped:
                 self.output_textbox.append("--- Training Cancelled (Stop Requested) ---")
             else: # Should not happen if button state is correct, but as fallback
                 self.output_textbox.append("--- Training Finished (Cancellation state unclear) ---")
        else:
            self.output_textbox.append("--- Training Finished Successfully ---")
            if result_or_error: # Display result if any was returned and emitted
                self.output_textbox.append(f"Final Result: {result_or_error}")


        self.progress_bar.setVisible(False)
        self.start_train_button.setEnabled(True)
        self.start_train_button.setText("开始训练!")
        self.cancel_train_button.setEnabled(False)
        self.cancel_train_button.setVisible(False)
        self.cancel_train_button.setText("取消训练") # Reset text
        self.cancel_train_button.setStyleSheet("QPushButton { background-color: gray; color: white; font-size: 20px; font-weight: bold; padding: 10px; }") # Reset style
        self.training_thread = None # Clean up thread reference



# --- Main Application Window ---

class YoloApp(QMainWindow):
    # Modify __init__ to accept base_path
    def __init__(self, base_path=None):
        super().__init__()
        # Pass base_path to AppState
                # Determine the base path correctly for script or frozen executable
        if getattr(sys, 'frozen', False):
            # If the application is run as a bundle/frozen executable (e.g., PyInstaller)
            base_path = os.path.dirname(sys.executable)
        else:
            # If run as a normal script (.py file)
            base_path = os.path.dirname(os.path.abspath(__file__))

        print(f"Training GUI Base Path: {base_path}") # Debug print
        self.app_state = AppState(base_path=base_path) # Create the state instance with base_path
        self.theme_settings = self.load_theme_settings()  # Load theme settings
        self._init_ui()

    def load_theme_settings(self):
        """Load theme settings from a JSON file."""
        theme_file = os.path.join(os.path.dirname(__file__), "theme.json")
        try:
            with open(theme_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading theme settings: {e}")
            # Fallback to default theme settings
            return {
                "Light": {
                    "background": "#FFFFFF",
                    "text": "#000000",
                    "button_background": "#E0E0E0",
                    "button_text": "#000000"
                },
                "Dark": {
                    "background": "#3B3E40",
                    "text": "#FFFFFF",
                    "button_background": "#444747",
                    "button_text": "#FFFFFF"
                }
            }

    def _init_ui(self):
        self.setWindowTitle('目标检测训练')
        # Get screen size using QApplication
        screen_geometry = QApplication.primaryScreen().availableGeometry()
        self.setGeometry(screen_geometry) # Use available geometry

        # Central Widget and Layout (will hold the stacked widget)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget) # Main layout for the central area

        # Stacked Widget for Pages
        self.stacked_widget = QStackedWidget()
        self.main_layout.addWidget(self.stacked_widget)

        # Create Pages (Widgets) - Pass the app_state instance
        self.train_page = TrainWidget(self.app_state, self)

        # Add Pages to Stacked Widget
        self.stacked_widget.addWidget(self.train_page)


        # Set initial page
        self.stacked_widget.setCurrentWidget(self.train_page)

        # Apply initial style
        self.change_appearance_mode("Dark") # Default to Light


    def change_appearance_mode(self, mode):
        """Apply theme settings dynamically based on the mode."""
        theme = self.theme_settings.get(mode, {})
        if theme:
            palette = QPalette()
            palette.setColor(QPalette.Window, QColor(theme.get("background", "#FFFFFF")))
            palette.setColor(QPalette.WindowText, QColor(theme.get("text", "#000000")))
            palette.setColor(QPalette.Base, QColor(theme.get("background", "#FFFFFF")))
            palette.setColor(QPalette.Text, QColor(theme.get("text", "#000000")))
            palette.setColor(QPalette.Button, QColor(theme.get("button_background", "#E0E0E0")))
            palette.setColor(QPalette.ButtonText, QColor(theme.get("button_text", "#000000")))
            QApplication.setPalette(palette)

            # Apply button-specific styles
            button_style = (
                f"QPushButton {{ background-color: {theme.get('button_background', '#E0E0E0')}; "
                f"color: {theme.get('button_text', '#000000')}; font-size: 16px; font-weight: bold; padding: 8px; }}"
            )
            self.train_page.start_train_button.setStyleSheet(button_style)
            self.train_page.cancel_train_button.setStyleSheet(button_style)
            self.train_page.train_data_button.setStyleSheet(button_style)
            self.train_page.model_save_button.setStyleSheet(button_style)

            # Apply styles to input fields (QLineEdit)
            input_field_style = (
                f"QLineEdit {{ background-color: {theme.get('background', '#FFFFFF')}; "
                f"color: {theme.get('text', '#000000')}; border: 1px solid {theme.get('text', '#000000')}; "
                f"padding: 5px; }}"
            )
            self.train_page.project_name_entry.setStyleSheet(input_field_style)
            self.train_page.epochs_entry.setStyleSheet(input_field_style)
            self.train_page.batch_size_entry.setStyleSheet(input_field_style)

            # Apply styles to model menu (QComboBox)
            combo_box_style = (
                f"QComboBox {{ background-color: {theme.get('background', '#FFFFFF')}; "
                f"color: {theme.get('text', '#000000')}; border: 1px solid {theme.get('text', '#000000')}; "
                f"padding: 5px; }}"
            )
            self.train_page.model_menu.setStyleSheet(combo_box_style)
        else:
            print(f"Theme mode '{mode}' not found. Falling back to default palette.")
            QApplication.setPalette(QApplication.style().standardPalette())

    def closeEvent(self, event):
        # Ensure camera thread is stopped before closing
        # Check if camera_detect_page exists before accessing it
        if hasattr(self, 'camera_detect_page') and self.camera_detect_page.camera_thread and self.camera_detect_page.camera_thread.isRunning():
            print("Stopping camera thread before exit...")
            self.camera_detect_page.camera_thread.stop()
            self.camera_detect_page.camera_thread.wait(3000) # Wait up to 3 seconds

        # Add similar cleanup for other threads if necessary
        if self.train_page.training_thread and self.train_page.training_thread.isRunning():
             print("Stopping training thread before exit...")
             self.train_page.training_thread.stop() # Request stop (may not be immediate)
             # Don't wait indefinitely if the function doesn't support cancellation
             # self.train_page.training_thread.wait(3000) # Optional: wait a bit

        print("Exiting application.")
        event.accept()


# --- Main Execution ---
if __name__ == "__main__":
    multiprocessing.freeze_support() # <--- 在开头添加此行

    # Necessary for high-DPI displays
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    # When running train_gui.py directly, base_path will be None, using default behavior
    main_window = YoloApp()
    main_window.show()
    sys.exit(app.exec_())