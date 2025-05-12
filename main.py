# version 0.2.0 (PyQt5)

import sys
import os
import cv2
import subprocess
import threading
import datetime
from queue import Queue, Empty
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QLineEdit, QTextEdit, QComboBox, QProgressBar, QFileDialog,
    QDockWidget, QStackedWidget, QRadioButton, QButtonGroup, QSizePolicy,
    QMessageBox, QSpacerItem
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon, QPalette, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize

# Assuming these modules exist and have been adapted or are compatible
from src.train import create_yaml # Keep this, might need adjustments
from src.detect import detect_images # Re-added import
from src.camera import CameraDetection # Needs significant adaptation for PyQt signals

# --- Application State Class ---
class AppState:
    def __init__(self):
        self.project_name = ""
        self.train_data_path = ""
        self.base_model_save_path = "" # Store the user-selected base path
        self.model_save_path = ""      # Store the full path (base + project name)
        self.selected_model_size = ""
        self.input_size = ""
        self.epochs = ""
        self.batch_size = ""
        self.class_names = []
        # Image Detection State
        self.detection_images_folder_path = ""
        self.detection_model_path = ""
        self.detection_save_dir = "" # Also used by Camera
        # Note: image_paths and current_image_index are specific to ImageDetectionWidget's display logic
        # Note: camera_detection object is managed within CameraDetectionWidget

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
    finished_signal = pyqtSignal()

    def __init__(self, cmd_args):
        super().__init__()
        self.cmd_args = cmd_args
        self.process = None

    def run(self):
        try:
            self.process = subprocess.Popen(
                self.cmd_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1, # Line buffered
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0 # Hide console window on Windows
            )
            for line in iter(self.process.stdout.readline, ''):
                self.output_signal.emit(line)
            self.process.stdout.close()
            self.process.wait()
        except Exception as e:
            self.output_signal.emit(f"Error during training: {e}\n")
        finally:
            self.finished_signal.emit()

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process.wait()

# Placeholder for Image Detection Thread (Needs implementation)
class ImageDetectionThread(QThread):
    finished_signal = pyqtSignal(str) # Emits results directory path
    error_signal = pyqtSignal(str)    # Signal for errors
    progress_signal = pyqtSignal(int) # Optional: for progress updates

    def __init__(self, folder_path, model_path):
        super().__init__()
        self.folder_path = folder_path
        self.model_path = model_path

    def run(self):
        try:
            print(f"Calling detect_images for: {self.folder_path} with {self.model_path}")
            # Call the function from src.detect
            # Modify detect_images in src/detect.py if it needs to return error info
            # or raise specific exceptions.
            results_dir = detect_images(self.folder_path, self.model_path)

            if results_dir and os.path.isdir(results_dir):
                print(f"Detection finished. Results available at: {results_dir}")
                self.finished_signal.emit(results_dir)
            else:
                # Handle cases where detect_images might return None or an invalid path on error
                error_msg = f"Detection failed or results directory not found (returned: {results_dir}). Check console output from detect_images."
                print(error_msg)
                self.error_signal.emit(error_msg)
                self.finished_signal.emit("") # Emit empty path on failure

        except Exception as e:
            # Catch any exceptions raised by detect_images
            print(f"An error occurred during image detection call: {e}")
            self.error_signal.emit(f"检测时发生错误: {e}")
            self.finished_signal.emit("") # Emit empty path on error

# Placeholder for Camera Detection Thread (Needs significant implementation)
class CameraDetectionThread(QThread):
    frame_signal = pyqtSignal(QPixmap)
    status_signal = pyqtSignal(str)
    stopped_signal = pyqtSignal()

    def __init__(self, model_path, camera_id=0, save_dir=None):
        super().__init__()
        self.model_path = model_path
        self.camera_id = camera_id
        self.save_dir = save_dir if save_dir else "."
        os.makedirs(self.save_dir, exist_ok=True)
        self.running = False
        self.cap = None
        self.model = None # Load YOLO model here (e.g., using ultralytics)
        self.scene_id = 0
        self.conf_threshold = 0.5 # Example threshold

        # --- Load Model (Example using ultralytics) ---
        # try:
        #     from ultralytics import YOLO
        #     self.model = YOLO(self.model_path)
        # except ImportError:
        #     self.status_signal.emit("Error: ultralytics library not found.")
        #     self.model = None
        # except Exception as e:
        #     self.status_signal.emit(f"Error loading model: {e}")
        #     self.model = None

    def set_save_directory(self, directory):
        self.save_dir = directory
        os.makedirs(self.save_dir, exist_ok=True)

    def run(self):
        if not self.model:
             self.status_signal.emit("Model not loaded. Cannot start detection.")
             self.stopped_signal.emit()
             return

        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            self.status_signal.emit(f"Error: Could not open camera {self.camera_id}")
            self.running = False
            self.stopped_signal.emit()
            return

        self.running = True
        self.status_signal.emit("Camera started.")

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.status_signal.emit("Error: Failed to grab frame.")
                self.msleep(50) # Avoid busy-waiting
                continue

            # Perform detection (using self.model)
            # results = self.model(frame) # Example
            # Draw bounding boxes on the frame (implement _draw_bounding_boxes or similar)
            # self._draw_bounding_boxes(frame, results)

            # Convert frame to QPixmap
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.frame_signal.emit(pixmap)

            # self.msleep(30) # Control frame rate if needed

        if self.cap:
            self.cap.release()
        self.status_signal.emit("Camera stopped.")
        self.stopped_signal.emit()

    def stop(self):
        self.running = False

    def capture_frame(self):
        if not self.cap or not self.cap.isOpened() or not self.model:
            self.status_signal.emit("Cannot capture: Camera not running or model not loaded.")
            return

        ret, frame = self.cap.read()
        if not ret:
            self.status_signal.emit("Capture failed: Could not read frame.")
            return

        self.scene_id += 1
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        base_filename = f"{timestamp}_{self.scene_id:04d}"

        origin_image_path = os.path.join(self.save_dir, f"{base_filename}_origin.jpg")
        cv2.imwrite(origin_image_path, frame)
        self.status_signal.emit(f"Saved original: {os.path.basename(origin_image_path)}")

        # --- Perform detection and save results ---
        # results = self.model(frame)
        # detection_frame = frame.copy() # Draw on a copy
        # self._draw_bounding_boxes(detection_frame, results) # Implement this

        # detection_image_path = os.path.join(self.save_dir, f"{base_filename}_detection.jpg")
        # cv2.imwrite(detection_image_path, detection_frame)
        # self.status_signal.emit(f"Saved detection: {os.path.basename(detection_image_path)}")

        # txt_path = os.path.join(self.save_dir, f"{base_filename}_detection.txt")
        # try:
        #     with open(txt_path, 'w') as f:
        #         # Example for ultralytics results
        #         for result in results[0].boxes:
        #             if result.conf[0] >= self.conf_threshold:
        #                 x1, y1, x2, y2 = map(int, result.xyxy[0])
        #                 label = self.model.names[int(result.cls[0])]
        #                 confidence = result.conf[0]
        #                 f.write(f"{label} {confidence:.2f} {x1} {y1} {x2} {y2}\n")
        #     self.status_signal.emit(f"Saved labels: {os.path.basename(txt_path)}")
        # except Exception as e:
        #      self.status_signal.emit(f"Error saving labels: {e}")


    def _draw_bounding_boxes(self, frame, results):
         # Example implementation for ultralytics results
         if not self.model: return
         for result in results[0].boxes:
             if result.conf[0] >= self.conf_threshold:
                 x1, y1, x2, y2 = map(int, result.xyxy[0])
                 label = self.model.names[int(result.cls[0])]
                 confidence = result.conf[0]
                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                 cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


# --- Widgets for each page ---

class TrainWidget(QWidget):
    def __init__(self, app_state, parent=None): # Accept app_state
        super().__init__(parent)
        self.app_state = app_state # Store app_state
        self.parent_window = parent
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
                parent_dir = os.path.dirname(path)
                suggested_base_path = os.path.join(parent_dir, "output")
                self.app_state.base_model_save_path = suggested_base_path
                # Update the full path and label immediately
                self._update_full_model_save_path()
                print(f"Suggested base model save path set to: {suggested_base_path}")
                # Trigger project list refresh in other widgets
                if self.parent_window:
                    self.parent_window.refresh_project_lists()
            except Exception as e:
                print(f"Error setting suggested base model save path: {e}")
                # Optionally reset or handle the error, for now, just print it
                self.app_state.base_model_save_path = ""
                self._update_full_model_save_path() # Update label to reflect reset
                # Trigger project list refresh in other widgets even on error (to clear lists)
                if self.parent_window:
                    self.parent_window.refresh_project_lists()
            # --- End automatic path setting ---


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
            # Trigger project list refresh in other widgets
            if self.parent_window:
                self.parent_window.refresh_project_lists()

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

        # --- Create YAML and Start Training ---
        try:
            # Pass the already loaded class_names list from app_state
            # Use the final model_save_path which includes the project name
            yaml_path = create_yaml(self.app_state.project_name, self.app_state.train_data_path, self.app_state.class_names, self.app_state.model_save_path) # Use final app_state path
            if not yaml_path or not os.path.exists(yaml_path):
                 QMessageBox.critical(self, "错误", f"创建或找到YAML文件失败: {yaml_path}")
                 return

            cmd_args = [
                sys.executable, # Use the current Python interpreter
                os.path.join(os.path.dirname(__file__), 'src', 'train.py'), # Ensure correct path to train.py
                self.app_state.project_name, # Use app_state
                self.app_state.train_data_path, # Use app_state
                ','.join(self.app_state.class_names), # Use app_state
                self.app_state.model_save_path, # Use final app_state path
                self.app_state.selected_model_size, # Use app_state
                str(self.app_state.input_size), # Use app_state
                str(self.app_state.epochs), # Use app_state
                yaml_path,
                str(self.app_state.batch_size) # Use app_state
            ]

            self.output_textbox.clear()
            self.output_textbox.append(f"Starting training with command:\n{' '.join(cmd_args)}\n---")
            self.progress_bar.setVisible(True)
            self.start_train_button.setEnabled(False)
            self.start_train_button.setText("训练中...")
            self.cancel_train_button.setEnabled(True) # Enable cancel button
            self.cancel_train_button.setVisible(True) # Show cancel button
            self.cancel_train_button.setStyleSheet("QPushButton { background-color: red; color: white; font-size: 20px; font-weight: bold; padding: 10px; }") # Make it red

            # Run in a separate thread
            self.training_thread = TrainingThread(cmd_args)
            self.training_thread.output_signal.connect(self.update_output)
            self.training_thread.finished_signal.connect(self.on_training_finished)
            self.training_thread.start()

        except Exception as e:
            QMessageBox.critical(self, "启动错误", f"启动训练时出错: {e}")
            self.progress_bar.setVisible(False)
            self.start_train_button.setEnabled(True)
            self.start_train_button.setText("开始训练!")
            self.cancel_train_button.setEnabled(False) # Disable cancel button on error
            self.cancel_train_button.setVisible(False) # Hide cancel button on error
            self.cancel_train_button.setStyleSheet("QPushButton { background-color: gray; color: white; font-size: 20px; font-weight: bold; padding: 10px; }") # Reset style

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

    def on_training_finished(self):
        # Check if cancellation was requested (button might be disabled but text changed)
        was_cancelled = not self.cancel_train_button.isEnabled() and "取消" in self.cancel_train_button.text()

        if was_cancelled:
            self.output_textbox.append("--- Training Cancelled ---")
        else:
            self.output_textbox.append("--- Training Finished ---")

        self.progress_bar.setVisible(False)
        self.start_train_button.setEnabled(True)
        self.start_train_button.setText("开始训练!")
        self.cancel_train_button.setEnabled(False)
        self.cancel_train_button.setVisible(False)
        self.cancel_train_button.setText("取消训练") # Reset text
        self.cancel_train_button.setStyleSheet("QPushButton { background-color: gray; color: white; font-size: 20px; font-weight: bold; padding: 10px; }") # Reset style
        self.training_thread = None # Clean up thread reference


class ImageDetectionWidget(QWidget):
    def __init__(self, app_state, parent=None): # Accept app_state
        super().__init__(parent)
        self.app_state = app_state # Store app_state
        # Keep these as instance variables specific to this widget's display logic
        self.image_paths = []
        self.current_image_index = 0
        self.detection_thread = None
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        top_layout = QHBoxLayout() # For controls
        bottom_layout = QHBoxLayout() # For progress bar

        # Image Display Area
        self.image_label = QLabel("请选择图片文件夹并开始检测")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored) # Allow label to resize
        self.image_label.setMinimumSize(400, 300) # Set a minimum size
        main_layout.addWidget(self.image_label, 1) # Give it stretch factor 1

        # --- Controls Layout ---
        self.select_folder_button = QPushButton("选择图片文件夹")
        self.select_folder_button.clicked.connect(self.select_detection_images_folder)
        top_layout.addWidget(self.select_folder_button)

        # Project Selection Dropdown
        top_layout.addWidget(QLabel("选择项目:"))
        self.project_selector_combo = QComboBox()
        self.project_selector_combo.setMinimumWidth(150)
        self.project_selector_combo.currentIndexChanged.connect(self._on_project_selected)
        top_layout.addWidget(self.project_selector_combo)

        # Label to display selected model path
        self.selected_model_label = QLabel("模型: 未选择")
        self.selected_model_label.setStyleSheet("color: red;")
        top_layout.addWidget(self.selected_model_label)

        # Remove the old select_model_button
        # self.select_model_button = QPushButton("选择模型")
        # self.select_model_button.clicked.connect(self.select_detection_model)
        # top_layout.addWidget(self.select_model_button)

        self.start_detection_button = QPushButton("开始检测!")
        self.start_detection_button.setStyleSheet("QPushButton { background-color: chocolate; color: white; font-size: 16px; padding: 5px; }")
        self.start_detection_button.clicked.connect(self.start_image_detection)
        top_layout.addWidget(self.start_detection_button)

        top_layout.addStretch(1) # Push controls to the left

        self.prev_button = QPushButton("◀")
        self.prev_button.clicked.connect(self.show_prev_image)
        self.prev_button.setEnabled(False)
        top_layout.addWidget(self.prev_button)

        self.image_index_label = QLabel("0/0")
        top_layout.addWidget(self.image_index_label)

        self.next_button = QPushButton("▶")
        self.next_button.clicked.connect(self.show_next_image)
        self.next_button.setEnabled(False)
        top_layout.addWidget(self.next_button)

        main_layout.addLayout(top_layout)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0) # Indeterminate
        self.progress_bar.setVisible(False)
        bottom_layout.addWidget(self.progress_bar)
        main_layout.addLayout(bottom_layout)

        # Initial population of dropdown
        self.update_project_dropdown()

    def select_detection_images_folder(self):
        path = QFileDialog.getExistingDirectory(self, "选择包含图片的文件夹")
        if path:
            self.app_state.detection_images_folder_path = path # Use app_state
            print(f"Selected folder: {self.app_state.detection_images_folder_path}") # Use app_state
            # Optionally display the path or clear previous results
            self.image_label.setText(f"已选择文件夹: {os.path.basename(path)}\n请选择模型并开始检测。")
            self.image_paths = []
            self.current_image_index = 0
            self.update_navigation()


    def update_project_dropdown(self):
        """Populates the project dropdown based on folders in base_model_save_path."""
        self.project_selector_combo.blockSignals(True) # Block signals during update
        self.project_selector_combo.clear()
        self.project_selector_combo.addItem("选择项目...")

        base_path = self.app_state.base_model_save_path
        if base_path and os.path.isdir(base_path):
            try:
                projects = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
                if projects:
                    self.project_selector_combo.addItems(sorted(projects))
                else:
                    self.project_selector_combo.addItem("无项目")
                    self.project_selector_combo.model().item(1).setEnabled(False) # Disable "No projects" item
            except Exception as e:
                print(f"Error reading projects from {base_path}: {e}")
                self.project_selector_combo.addItem("读取错误")
                self.project_selector_combo.model().item(1).setEnabled(False)
        else:
            self.project_selector_combo.addItem("请先设置基础路径")
            self.project_selector_combo.model().item(1).setEnabled(False)

        self.project_selector_combo.setCurrentIndex(0) # Reset selection
        self.project_selector_combo.blockSignals(False) # Unblock signals
        # Manually trigger selection logic for index 0 (or handle initial state)
        self._on_project_selected(0)


    def _on_project_selected(self, index):
        """Handles project selection from the dropdown."""
        selected_project = self.project_selector_combo.currentText()
        base_path = self.app_state.base_model_save_path

        if index <= 0 or not base_path or not selected_project or selected_project in ["选择项目...", "无项目", "读取错误", "请先设置基础路径"]:
            self.app_state.detection_model_path = ""
            self.selected_model_label.setText("模型: 未选择")
            self.selected_model_label.setStyleSheet("color: red;")
            return

        # Construct the expected path to best.pt
        model_path = os.path.join(base_path, selected_project, 'weights', 'best.pt')

        if os.path.exists(model_path):
            self.app_state.detection_model_path = model_path
            self.selected_model_label.setText(f"模型: {selected_project}/weights/best.pt")
            self.selected_model_label.setStyleSheet("color: green;")
            print(f"Detection model set to: {model_path}")
        else:
            self.app_state.detection_model_path = ""
            self.selected_model_label.setText(f"模型: {selected_project}/weights/best.pt (未找到!)")
            self.selected_model_label.setStyleSheet("color: red;")
            print(f"Warning: best.pt not found at {model_path}")

    def start_image_detection(self):
        if not self.app_state.detection_images_folder_path: # Use app_state
            QMessageBox.warning(self, "错误", "请先选择图片文件夹。")
            return
        if not self.app_state.detection_model_path: # Use app_state
            QMessageBox.warning(self, "错误", "请先选择模型文件。")
            return
        if self.detection_thread and self.detection_thread.isRunning():
            QMessageBox.warning(self, "运行中", "检测已在进行中。")
            return

        self.progress_bar.setVisible(True)
        self.start_detection_button.setEnabled(False)
        self.start_detection_button.setText("检测中...")
        self.image_label.setText("检测进行中...")
        self.image_paths = [] # Clear previous results
        self.update_navigation()

        # Start detection in a thread - pass paths from app_state
        self.detection_thread = ImageDetectionThread(self.app_state.detection_images_folder_path, self.app_state.detection_model_path) # Use app_state
        self.detection_thread.finished_signal.connect(self.on_detection_finished)
        # Connect progress signal if implemented in the thread
        # self.detection_thread.progress_signal.connect(self.update_progress)
        self.detection_thread.start()

    def on_detection_error(self, error_message):
        """Handles errors reported by the detection thread."""
        self.image_label.setText(f"检测错误: {error_message}")
        self.image_paths = [] # Clear image paths on error
        self.update_navigation()

    def on_detection_finished(self, results_dir):
        self.progress_bar.setVisible(False)
        self.start_detection_button.setEnabled(True)
        self.start_detection_button.setText("开始检测!")
        self.detection_thread = None

        if not results_dir or not os.path.isdir(results_dir):
             QMessageBox.warning(self, "错误", f"检测完成，但未找到结果目录: {results_dir}")
             self.image_label.setText("检测完成，但未找到结果。")
             self.image_paths = []
        else:
            try:
                self.image_paths = [os.path.join(results_dir, f) for f in sorted(os.listdir(results_dir))
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if not self.image_paths:
                    self.image_label.setText("检测完成，但结果目录中没有图片。")
                else:
                    self.current_image_index = 0
                    self.update_image()
            except Exception as e:
                 QMessageBox.critical(self, "错误", f"加载结果图片时出错: {e}")
                 self.image_label.setText("加载结果图片时出错。")
                 self.image_paths = []

        self.update_navigation()


    def update_image(self):
        if not self.image_paths:
            self.image_label.setText("无图片显示")
            return

        path = self.image_paths[self.current_image_index]
        pixmap = QPixmap(path)

        if pixmap.isNull():
            self.image_label.setText(f"无法加载图片:\n{os.path.basename(path)}")
            return

        # Scale pixmap to fit label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)
        self.update_navigation()

    def show_next_image(self):
        if self.image_paths:
            self.current_image_index = (self.current_image_index + 1) % len(self.image_paths)
            self.update_image()

    def show_prev_image(self):
        if self.image_paths:
            self.current_image_index = (self.current_image_index - 1) % len(self.image_paths)
            self.update_image()

    def update_navigation(self):
        num_images = len(self.image_paths)
        if num_images > 0:
            self.image_index_label.setText(f"{self.current_image_index + 1}/{num_images}")
            self.prev_button.setEnabled(True)
            self.next_button.setEnabled(True)
        else:
            self.image_index_label.setText("0/0")
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)

    # Override resizeEvent to rescale image when window resizes
    def resizeEvent(self, event):
        if self.image_paths:
            self.update_image()
        super().resizeEvent(event)


class CameraDetectionWidget(QWidget):
    def __init__(self, app_state, parent=None): # Accept app_state
        super().__init__(parent)
        self.app_state = app_state # Store app_state
        self.camera_thread = None
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        controls_layout = QHBoxLayout()

        # Camera Feed Display
        self.image_label = QLabel("摄像头画面将显示在此处")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setMinimumSize(640, 480)
        # Set background color for visibility
        self.image_label.setStyleSheet("QLabel { background-color: black; color: white; }")
        main_layout.addWidget(self.image_label, 1) # Expandable

        # --- Controls ---
        # Project Selection Dropdown
        controls_layout.addWidget(QLabel("选择项目:"))
        self.project_selector_combo = QComboBox()
        self.project_selector_combo.setMinimumWidth(150)
        self.project_selector_combo.currentIndexChanged.connect(self._on_project_selected)
        controls_layout.addWidget(self.project_selector_combo)

        # Label to display selected model path
        self.selected_model_label = QLabel("模型: 未选择")
        self.selected_model_label.setStyleSheet("color: red;")
        controls_layout.addWidget(self.selected_model_label)

        # Remove the old select_model_button
        # self.select_model_button = QPushButton("选择模型")
        # self.select_model_button.clicked.connect(self.select_detection_model)
        # controls_layout.addWidget(self.select_model_button)

        self.select_save_folder_button = QPushButton("选择保存文件夹")
        self.select_save_folder_button.clicked.connect(self.select_camera_save_folder)
        controls_layout.addWidget(self.select_save_folder_button)

        self.camera_id_entry = QLineEdit(placeholderText="摄像头 ID (例如: 0)")
        self.camera_id_entry.setFixedWidth(150)
        controls_layout.addWidget(self.camera_id_entry)

        controls_layout.addStretch(1)

        self.start_stop_button = QPushButton("START")
        self.start_stop_button.setStyleSheet("QPushButton { background-color: green; color: white; font-size: 16px; padding: 5px; }")
        self.start_stop_button.setCheckable(True) # Make it toggle-like
        self.start_stop_button.toggled.connect(self.toggle_camera_detection)
        controls_layout.addWidget(self.start_stop_button)

        main_layout.addLayout(controls_layout)

        # Status Label
        self.status_label = QLabel("状态: 未连接 | 按 Enter 键保存当前帧")
        main_layout.addWidget(self.status_label)

        # Allow widget to receive key presses
        self.setFocusPolicy(Qt.StrongFocus)

        # Initial population of dropdown
        self.update_project_dropdown()

    def update_project_dropdown(self):
        """Populates the project dropdown based on folders in base_model_save_path."""
        self.project_selector_combo.blockSignals(True) # Block signals during update
        self.project_selector_combo.clear()
        self.project_selector_combo.addItem("选择项目...")

        base_path = self.app_state.base_model_save_path
        if base_path and os.path.isdir(base_path):
            try:
                projects = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
                if projects:
                    self.project_selector_combo.addItems(sorted(projects))
                else:
                    self.project_selector_combo.addItem("无项目")
                    self.project_selector_combo.model().item(1).setEnabled(False) # Disable "No projects" item
            except Exception as e:
                print(f"Error reading projects from {base_path}: {e}")
                self.project_selector_combo.addItem("读取错误")
                self.project_selector_combo.model().item(1).setEnabled(False)
        else:
            self.project_selector_combo.addItem("请先设置基础路径")
            self.project_selector_combo.model().item(1).setEnabled(False)

        self.project_selector_combo.setCurrentIndex(0) # Reset selection
        self.project_selector_combo.blockSignals(False) # Unblock signals
        # Manually trigger selection logic for index 0 (or handle initial state)
        self._on_project_selected(0)


    def _on_project_selected(self, index):
        """Handles project selection from the dropdown."""
        selected_project = self.project_selector_combo.currentText()
        base_path = self.app_state.base_model_save_path

        if index <= 0 or not base_path or not selected_project or selected_project in ["选择项目...", "无项目", "读取错误", "请先设置基础路径"]:
            self.app_state.detection_model_path = ""
            self.selected_model_label.setText("模型: 未选择")
            self.selected_model_label.setStyleSheet("color: red;")
            # If camera is running, stop it because the model is no longer valid
            if self.camera_thread and self.camera_thread.isRunning():
                self.toggle_camera_detection(False)
            return

        # Construct the expected path to best.pt
        model_path = os.path.join(base_path, selected_project, 'weights', 'best.pt')

        if os.path.exists(model_path):
            self.app_state.detection_model_path = model_path
            self.selected_model_label.setText(f"模型: {selected_project}/weights/best.pt")
            self.selected_model_label.setStyleSheet("color: green;")
            print(f"Detection model set to: {model_path}")
            # If camera is running, stop it so it can be restarted with the new model
            if self.camera_thread and self.camera_thread.isRunning():
                self.status_label.setText("状态: 模型已更改，请重启摄像头。")
                self.toggle_camera_detection(False)
        else:
            self.app_state.detection_model_path = ""
            self.selected_model_label.setText(f"模型: {selected_project}/weights/best.pt (未找到!)")
            self.selected_model_label.setStyleSheet("color: red;")
            print(f"Warning: best.pt not found at {model_path}")
            # If camera is running, stop it because the model is no longer valid
            if self.camera_thread and self.camera_thread.isRunning():
                self.toggle_camera_detection(False)

    def select_camera_save_folder(self):
        path = QFileDialog.getExistingDirectory(self, "选择截图和标签保存文件夹")
        if path:
            self.app_state.detection_save_dir = path # Use app_state
            print(f"Selected save folder: {self.app_state.detection_save_dir}") # Use app_state
            self.status_label.setText(f"保存文件夹: {os.path.basename(path)}")
            if self.camera_thread and self.camera_thread.isRunning():
                 # Pass the updated path from app_state to the running thread
                 self.camera_thread.set_save_directory(self.app_state.detection_save_dir) # Use app_state

    def toggle_camera_detection(self, checked):
        if checked: # Start detection
            if not self.app_state.detection_model_path: # Use app_state
                QMessageBox.warning(self, "错误", "请先选择模型文件。")
                self.start_stop_button.setChecked(False) # Uncheck the button
                return

            camera_id_str = self.camera_id_entry.text().strip()
            if not camera_id_str.isdigit():
                 QMessageBox.warning(self, "错误", "请输入有效的摄像头 ID (数字)。")
                 self.start_stop_button.setChecked(False)
                 return
            camera_id = int(camera_id_str)

            self.start_stop_button.setText("STOP")
            self.start_stop_button.setStyleSheet("QPushButton { background-color: red; color: white; font-size: 16px; padding: 5px; }")
            self.camera_id_entry.setEnabled(False)
            self.select_save_folder_button.setEnabled(False)
            self.project_selector_combo.setEnabled(False)

            # Pass model path and save dir from app_state
            self.camera_thread = CameraDetectionThread(
                self.app_state.detection_model_path, # Use app_state
                camera_id,
                self.app_state.detection_save_dir # Use app_state
            )
            self.camera_thread.frame_signal.connect(self.update_frame)
            self.camera_thread.status_signal.connect(self.update_status)
            self.camera_thread.stopped_signal.connect(self.on_camera_stopped) # Connect stopped signal
            self.camera_thread.start()
            self.status_label.setText("状态: 正在启动摄像头...")
            self.image_label.setText("正在启动摄像头...") # Placeholder text

        else: # Stop detection
            if self.camera_thread and self.camera_thread.isRunning():
                self.status_label.setText("状态: 正在停止...")
                self.camera_thread.stop()
            # Don't reset button text/style here, wait for stopped_signal


    def update_frame(self, pixmap):
        # Scale pixmap to fit label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    def update_status(self, message):
        self.status_label.setText(f"状态: {message}")
        # Handle specific error messages if needed
        if "Error: Could not open camera" in message or "Model not loaded" in message:
             self.image_label.setText(message) # Show error on image label too
             self.start_stop_button.setChecked(False) # Ensure button is reset if start failed

    def on_camera_stopped(self):
        self.start_stop_button.setText("START")
        self.start_stop_button.setStyleSheet("QPushButton { background-color: green; color: white; font-size: 16px; padding: 5px; }")
        self.start_stop_button.setChecked(False) # Ensure state is unchecked
        self.camera_id_entry.setEnabled(True)
        self.select_save_folder_button.setEnabled(True)
        self.project_selector_combo.setEnabled(True)
        if "Error" not in self.status_label.text(): # Don't overwrite error messages
             self.status_label.setText("状态: 已停止 | 按 Enter 键保存当前帧")
        # Optionally clear the image label or show a "stopped" message
        # self.image_label.setText("摄像头已停止")
        self.camera_thread = None # Clean up thread reference

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if self.camera_thread and self.camera_thread.isRunning():
                self.status_label.setText("状态: 正在保存帧...")
                self.camera_thread.capture_frame()
            else:
                self.status_label.setText("状态: 摄像头未运行，无法保存。")
        else:
            super().keyPressEvent(event) # Pass other key events up

    # Ensure thread is stopped when widget is closed/hidden
    def closeEvent(self, event):
         if self.camera_thread and self.camera_thread.isRunning():
             self.camera_thread.stop()
             self.camera_thread.wait() # Wait for thread to finish
         super().closeEvent(event)

    def hideEvent(self, event):
        # Stop camera if the widget is hidden (e.g., switching tabs)
        if self.camera_thread and self.camera_thread.isRunning():
             self.toggle_camera_detection(False) # Trigger stop sequence
        super().hideEvent(event)


# --- Main Application Window ---

class YoloApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.app_state = AppState() # Create the state instance
        self._init_ui()

    def _init_ui(self):
        self.setWindowTitle('YOLO Train and Detect App (PyQt5)')
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
        self.image_detect_page = ImageDetectionWidget(self.app_state, self)
        self.camera_detect_page = CameraDetectionWidget(self.app_state, self)

        # Add Pages to Stacked Widget
        self.stacked_widget.addWidget(self.train_page)
        self.stacked_widget.addWidget(self.image_detect_page)
        self.stacked_widget.addWidget(self.camera_detect_page)

        # Create Sidebar (Dock Widget)
        self.create_sidebar()

        # Set initial page
        self.stacked_widget.setCurrentWidget(self.train_page)

        # Apply initial style
        self.change_appearance_mode("Light") # Default to Light

    def refresh_project_lists(self):
        """Calls methods in detection widgets to update their project dropdowns."""
        print("Refreshing project lists...")
        self.image_detect_page.update_project_dropdown()
        self.camera_detect_page.update_project_dropdown()

    def create_sidebar(self):
        self.sidebar = QDockWidget("导航", self)
        self.sidebar.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.sidebar.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable) # Allow moving/floating

        sidebar_widget = QWidget() # Container widget for the dock contents
        sidebar_layout = QVBoxLayout(sidebar_widget)
        sidebar_layout.setAlignment(Qt.AlignTop) # Align items to the top

        # Navigation Buttons
        train_button = QPushButton("训练")
        train_button.clicked.connect(lambda: self.stacked_widget.setCurrentWidget(self.train_page))
        sidebar_layout.addWidget(train_button)

        image_video_button = QPushButton("图片检测")
        image_video_button.clicked.connect(lambda: self.stacked_widget.setCurrentWidget(self.image_detect_page))
        sidebar_layout.addWidget(image_video_button)

        camera_button = QPushButton("Camera")
        camera_button.clicked.connect(lambda: self.stacked_widget.setCurrentWidget(self.camera_detect_page))
        sidebar_layout.addWidget(camera_button)

        sidebar_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)) # Spacer

        # YOLO Version Labels
        sidebar_layout.addWidget(QLabel("操作指南："))
        sidebar_layout.addWidget(QLabel("1. 首先输出项目名称"))
        sidebar_layout.addWidget(QLabel("2. 选择标注好的文件目录"))
        sidebar_layout.addWidget(QLabel("3. 选择保存模型的目录"))
        sidebar_layout.addWidget(QLabel("4. 点击训练即可"))


        sidebar_layout.addStretch(1) # Pushes appearance mode to bottom

        # Appearance Mode
        sidebar_layout.addWidget(QLabel("选择外观模式:"))
        self.appearance_group = QButtonGroup(self)
        light_mode_radio = QRadioButton("Light")
        dark_mode_radio = QRadioButton("Dark")
        self.appearance_group.addButton(light_mode_radio, 0)
        self.appearance_group.addButton(dark_mode_radio, 1)
        light_mode_radio.setChecked(True) # Default
        light_mode_radio.toggled.connect(lambda checked: self.change_appearance_mode("Light") if checked else None)
        dark_mode_radio.toggled.connect(lambda checked: self.change_appearance_mode("Dark") if checked else None)
        sidebar_layout.addWidget(light_mode_radio)
        sidebar_layout.addWidget(dark_mode_radio)

        sidebar_widget.setLayout(sidebar_layout)
        self.sidebar.setWidget(sidebar_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.sidebar)


    def change_appearance_mode(self, mode):
        if mode == "Dark":
            # Basic Dark Theme using Palette
            dark_palette = QPalette()
            dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.WindowText, Qt.white)
            dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
            dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
            dark_palette.setColor(QPalette.ToolTipText, Qt.white)
            dark_palette.setColor(QPalette.Text, Qt.white)
            dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ButtonText, Qt.white)
            dark_palette.setColor(QPalette.BrightText, Qt.red)
            dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
            dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
            dark_palette.setColor(QPalette.HighlightedText, Qt.black)
            # Set disabled colors
            dark_palette.setColor(QPalette.Disabled, QPalette.Text, Qt.darkGray)
            dark_palette.setColor(QPalette.Disabled, QPalette.ButtonText, Qt.darkGray)

            QApplication.setPalette(dark_palette)
            # Reapply styles that might be overridden by palette changes
            self.train_page.train_data_label.setStyleSheet(self.train_page.train_data_label.styleSheet()) # Reapply existing style
            self.train_page.model_save_label.setStyleSheet(self.train_page.model_save_label.styleSheet()) # Reapply existing style
            self.train_page.classes_status_label.setStyleSheet(self.train_page.classes_status_label.styleSheet()) # Reapply existing style
        else: # Light Mode
            QApplication.setPalette(QApplication.style().standardPalette())
            # Reapply styles for light mode if needed, or ensure they reset correctly
            self.train_page.train_data_label.setStyleSheet(self.train_page.train_data_label.styleSheet()) # Reapply existing style
            self.train_page.model_save_label.setStyleSheet(self.train_page.model_save_label.styleSheet()) # Reapply existing style
            self.train_page.classes_status_label.setStyleSheet(self.train_page.classes_status_label.styleSheet()) # Reapply existing style

    def closeEvent(self, event):
        # Ensure camera thread is stopped before closing
        if self.camera_detect_page.camera_thread and self.camera_detect_page.camera_thread.isRunning():
            print("Stopping camera thread before exit...")
            self.camera_detect_page.camera_thread.stop()
            self.camera_detect_page.camera_thread.wait(3000) # Wait up to 3 seconds

        # Add similar cleanup for other threads if necessary
        if self.train_page.training_thread and self.train_page.training_thread.isRunning():
             print("Stopping training thread before exit...")
             self.train_page.training_thread.stop() # Assuming TrainingThread has a stop method
             self.train_page.training_thread.wait(3000)

        print("Exiting application.")
        event.accept()


# --- Main Execution ---
if __name__ == "__main__":
    # Necessary for high-DPI displays
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    main_window = YoloApp()
    main_window.show()
    sys.exit(app.exec_())