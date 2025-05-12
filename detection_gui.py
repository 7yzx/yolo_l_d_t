import os
import sys
import cv2
import datetime
import time # Import time for cooldown
from collections import Counter # Import Counter for counting classes

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QLineEdit, QTextEdit, QComboBox, QProgressBar, QFileDialog,
    QDockWidget, QStackedWidget, QRadioButton, QButtonGroup, QSizePolicy,
    QMessageBox, QSpacerItem,QGroupBox
)
from PyQt5.QtGui import QPixmap, QImage, QPalette, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import multiprocessing
try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False
    print("Warning: PyTorch not found. Camera detection will run on CPU.")
from ultralytics import YOLO

from src.camera import CameraDetection


# --- Custom ComboBox ---
class ProjectComboBox(QComboBox):
    """A custom QComboBox that emits a signal just before showing the popup."""
    aboutToShowPopup = pyqtSignal()

    def showPopup(self):
        self.aboutToShowPopup.emit() # Emit signal before showing
        super().showPopup()

class CameraDetectionThread(QThread):
    frame_signal = pyqtSignal(QPixmap)
    status_signal = pyqtSignal(str)
    stopped_signal = pyqtSignal()
    detection_results_signal = pyqtSignal(dict) # Signal for class counts

    # Add class_names parameter
    def __init__(self, model_path, class_names, camera_id=0, save_dir=None):
        super().__init__()
        self.model_path = model_path
        self.class_names = class_names if class_names else [] # Store class names
        self.camera_id = camera_id
        self.save_dir = save_dir if save_dir else "."
        os.makedirs(self.save_dir, exist_ok=True)
        self.running = False
        self.cap = None
        self.model = None
        self.scene_id = 0
        self.conf_threshold = 0.5

        # Stability Tracking Parameters
        self.last_boxes = [] # Store boxes from the previous frame [{ 'xyxy': [x1,y1,x2,y2], 'class_id': id, 'conf': conf }, ...]
        self.stable_frames_count = 0
        self.stability_threshold_px = 15 # Max center point movement in pixels
        self.stable_frames_required = 5 # Number of consecutive frames for stability
        self.capture_cooldown_seconds = 3 # Cooldown after automatic capture
        self.last_capture_time = 0
        self.min_box_area_for_stability = 1000 # Ignore very small boxes for stability check

        # --- Load Model ---
        try:
            self.model = YOLO(self.model_path)
            # Override model names if class_names were provided externally
            if self.class_names:
                print(f"Using provided class names: {self.class_names}")
            elif hasattr(self.model, 'names'):
                self.class_names = list(self.model.names.values()) # Use names from model if not provided
                print(f"Using class names from model: {self.class_names}")
            else:
                print("Warning: Could not determine class names from model or input.")
                self.class_names = []
            # Determine device
            if torch_available and torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
            print(f"CameraDetection using device: {self.device}")

            if torch_available:
                self.model.to(self.device)
        except ImportError:
             self.status_signal.emit("Ultralytics YOLO not found. Please install it.")
             self.model = None # Ensure model is None if import fails
        except Exception as e:
             self.status_signal.emit(f"Error loading model: {e}")
             self.model = None

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
        self.status_signal.emit("Camera started. Looking for stable objects...")
        self.last_boxes = [] # Reset stability tracking on start
        self.stable_frames_count = 0

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.status_signal.emit("Error: Failed to grab frame.")
                self.msleep(50)
                continue

            current_time = time.time()
            results = self.model(frame, verbose=False, device=self.device, conf=self.conf_threshold)

            detection_frame = frame.copy()
            current_boxes = []
            class_counts = Counter()

            # Process results for drawing, counting, and stability check
            if results and results[0].boxes:
                for box in results[0].boxes:
                    if box.conf[0] >= self.conf_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        class_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        current_boxes.append({'xyxy': [x1, y1, x2, y2], 'class_id': class_id, 'conf': conf})
                        if 0 <= class_id < len(self.class_names):
                            class_name = self.class_names[class_id]
                            class_counts[class_name] += 1
                        else:
                            class_counts[f"Class_{class_id}"] += 1 # Fallback name

            # Emit class counts
            self.detection_results_signal.emit(dict(class_counts))

            # Draw bounding boxes
            self._draw_bounding_boxes(detection_frame, current_boxes) # Pass processed boxes

            # --- Stability Check ---
            is_stable = False
            if current_boxes and self.last_boxes:
                # Simple stability: Check if the center of the largest box moved less than threshold
                # More robust checks (IoU, tracking) could be added here.
                largest_current_box = max(current_boxes, key=lambda b: (b['xyxy'][2]-b['xyxy'][0])*(b['xyxy'][3]-b['xyxy'][1]), default=None)
                largest_last_box = max(self.last_boxes, key=lambda b: (b['xyxy'][2]-b['xyxy'][0])*(b['xyxy'][3]-b['xyxy'][1]), default=None)

                if largest_current_box and largest_last_box:
                    # Check if area is large enough
                    c_x1, c_y1, c_x2, c_y2 = largest_current_box['xyxy']
                    c_area = (c_x2 - c_x1) * (c_y2 - c_y1)

                    if c_area >= self.min_box_area_for_stability:
                        c_center_x = (c_x1 + c_x2) / 2
                        c_center_y = (c_y1 + c_y2) / 2

                        l_x1, l_y1, l_x2, l_y2 = largest_last_box['xyxy']
                        l_center_x = (l_x1 + l_x2) / 2
                        l_center_y = (l_y1 + l_y2) / 2

                        distance = ((c_center_x - l_center_x)**2 + (c_center_y - l_center_y)**2)**0.5

                        if distance < self.stability_threshold_px:
                            self.stable_frames_count += 1
                        else:
                            self.stable_frames_count = 0 # Reset if moved too much
                    else:
                         self.stable_frames_count = 0 # Reset if too small
                else:
                    self.stable_frames_count = 0 # Reset if no boxes found in current or last frame
            else:
                self.stable_frames_count = 0 # Reset if no boxes currently or previously

            # Check if stable enough and cooldown passed
            if self.stable_frames_count >= self.stable_frames_required:
                is_stable = True
                if (current_time - self.last_capture_time) > self.capture_cooldown_seconds:
                    self.status_signal.emit(f"Object stable ({self.stable_frames_count} frames), capturing automatically...")
                    self.capture_frame(frame, results, automatic=True) # Pass original frame and results
                    self.last_capture_time = current_time # Update cooldown timer
                    self.stable_frames_count = 0 # Reset stability count after capture
                    self.last_boxes = [] # Reset last boxes to prevent immediate re-trigger if object remains
                # else: # Optional: Indicate cooldown active
                #     self.status_signal.emit(f"Object stable, waiting for cooldown ({current_time - self.last_capture_time:.1f}s / {self.capture_cooldown_seconds}s)")


            # Update last boxes for the next iteration
            self.last_boxes = current_boxes

            # --- Convert frame to QPixmap ---
            # ... (rest of the frame conversion and emitting logic remains the same) ...
            if detection_frame is None or detection_frame.size == 0:
                self.status_signal.emit("Warning: Empty frame received.")
                continue

            # Check frame dimensions and channels
            if detection_frame.ndim != 3 or detection_frame.shape[2] != 3:
                    # Attempt to convert if grayscale
                    if detection_frame.ndim == 2:
                        detection_frame = cv2.cvtColor(detection_frame, cv2.COLOR_GRAY2BGR)
                    else:
                        self.status_signal.emit(f"Warning: Unexpected frame shape {detection_frame.shape}")
                        continue # Skip this frame

            rgb_image = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.frame_signal.emit(pixmap)

        # ... (rest of the run method cleanup) ...
        if self.cap:
            self.cap.release()
        self.status_signal.emit("Camera stopped.")
        self.stopped_signal.emit()

    def stop(self):
        self.running = False

    # Modify capture_frame to accept frame and results, add 'automatic' flag
    def capture_frame(self, frame=None, results=None, automatic=False):
        capture_time = time.time() # Use consistent time
        # If called manually (e.g., by keypress), grab a fresh frame and run detection
        if frame is None or results is None:
             if not self.cap or not self.cap.isOpened() or not self.model:
                 self.status_signal.emit("Cannot capture: Camera not running or model not loaded.")
                 return
             ret, frame = self.cap.read()
             if not ret or frame is None:
                 self.status_signal.emit("Capture failed: Could not read frame.")
                 return
             # Run detection on the captured frame
             results = self.model(frame, verbose=False, device=self.device, conf=self.conf_threshold)

        # --- Save Original Image ---
        self.scene_id += 1
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        base_filename = f"{timestamp}_{self.scene_id:04d}"
        origin_image_path = os.path.join(self.save_dir, f"{base_filename}_origin.jpg")
        try:
            cv2.imwrite(origin_image_path, frame)
            status_prefix = "Auto-saved" if automatic else "Saved"
            self.status_signal.emit(f"{status_prefix} original: {os.path.basename(origin_image_path)}")
        except Exception as e:
            self.status_signal.emit(f"Error saving original image: {e}")
            return

        # --- Perform detection (if needed) and save results ---
        try:
            detection_frame = frame.copy() # Draw on a copy
            processed_boxes = []
            if results and results[0].boxes:
                 for box in results[0].boxes:
                     if box.conf[0] >= self.conf_threshold:
                         x1, y1, x2, y2 = map(int, box.xyxy[0])
                         class_id = int(box.cls[0])
                         conf = float(box.conf[0])
                         processed_boxes.append({'xyxy': [x1, y1, x2, y2], 'class_id': class_id, 'conf': conf})

            self._draw_bounding_boxes(detection_frame, processed_boxes) # Draw using processed boxes

            detection_image_path = os.path.join(self.save_dir, f"{base_filename}_detection.jpg")
            cv2.imwrite(detection_image_path, detection_frame)
            self.status_signal.emit(f"{status_prefix} detection: {os.path.basename(detection_image_path)}")

            # --- Save Labels ---
            # txt_path = os.path.join(self.save_dir, f"{base_filename}_detection.txt")
            # with open(txt_path, 'w') as f:
            #     for box_info in processed_boxes:
            #         x1, y1, x2, y2 = box_info['xyxy']
            #         class_id = box_info['class_id']
            #         confidence = box_info['conf']
            #         if 0 <= class_id < len(self.class_names):
            #             label = self.class_names[class_id]
            #         else:
            #             label = f"Class_{class_id}" # Fallback label
            #         f.write(f"{label} {confidence:.4f} {x1} {y1} {x2} {y2}\n") # Higher precision for confidence
            # self.status_signal.emit(f"{status_prefix} labels: {os.path.basename(txt_path)}")

            # Reset stability tracking if captured automatically
            if automatic:
                self.stable_frames_count = 0
                self.last_boxes = []
                self.last_capture_time = capture_time # Ensure cooldown is based on this capture

        except AttributeError as e:
             self.status_signal.emit(f"Error processing detection results (maybe model structure changed?): {e}")
        except Exception as e:
             self.status_signal.emit(f"Error saving detection results: {e}")


    # Modify to accept processed boxes and use self.class_names
    def _draw_bounding_boxes(self, frame, boxes_info):
         if not self.class_names: return # Cannot draw labels without names

         try:
             for box_info in boxes_info:
                 x1, y1, x2, y2 = box_info['xyxy']
                 class_index = box_info['class_id']
                 confidence = box_info['conf']

                 if 0 <= class_index < len(self.class_names):
                     label = self.class_names[class_index]
                 else:
                     label = f"Class_{class_index}" # Fallback label

                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                 cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
         except Exception as e:
             self.status_signal.emit(f"Error drawing boxes: {e}")


# --- Widgets ---

class DetectionWindow(QWidget): # Renamed from ImageDetectionWidget
    MODE_CAMERA = "摄像头检测"

    # Modify __init__ to accept base_path
    def __init__(self, parent=None, base_path=None): # Accept optional base_path
        super().__init__(parent)

        # Use provided base_path if available, otherwise default
        if base_path and os.path.isdir(base_path):
            self.base_model_save_path = os.path.join(base_path, "output")
        else:
            # Fallback to original behavior if base_path is not provided or invalid
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.base_model_save_path = os.path.join(script_dir, "output") # Base path is now ./output/ relative to this script

        print(f"DetectionWindow Initialized with base_model_save_path: {self.base_model_save_path}") # Debug print
        os.makedirs(self.base_model_save_path, exist_ok=True) # Ensure base output dir exists

        # Local state variables (previously in AppState)
        self.detection_images_folder_path = ""
        self.detection_model_path = ""
        self.detection_save_dir = "" # Used specifically for camera captures
        self.class_names = [] # Store loaded class names

        # Image detection state
        self.image_paths = []
        self.current_image_index = 0
        # Thread management
        self.active_thread = None # Unified thread variable
        self.current_mode = self.MODE_CAMERA # Default mode

        self._init_ui()
        self.update_project_dropdown() # Initial population
        self._update_ui_for_mode() # Set initial UI state

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        palette = QPalette()
        palette.setColor(QPalette.Text, QColor("#FFFFFF"))
        palette.setColor(QPalette.ButtonText, QColor("#FFFFFF"))

        QApplication.setPalette(palette)

        # --- Common Controls (Project Selection) ---
        project_layout = QHBoxLayout()
        project_layout.addWidget(QLabel("选择项目:"))
        # Use the custom ProjectComboBox
        self.project_selector_combo = ProjectComboBox()
        self.project_selector_combo.setMinimumWidth(150)
        # Connect the standard signal for selection change
        self.project_selector_combo.currentIndexChanged.connect(self._on_project_selected)
        # Connect the custom signal to refresh the list before showing popup
        self.project_selector_combo.aboutToShowPopup.connect(self.update_project_dropdown)
        project_layout.addWidget(self.project_selector_combo)
        model_path = os.path.join(self.base_model_save_path, "best.pt")
        if os.path.exists(model_path):
            self.detection_model_path = model_path
            self.selected_model_label = QLabel(f"模型: {model_path}")
            self.selected_model_label.setStyleSheet("color: green;")
        else:
            self.selected_model_label = QLabel("模型: 未选择")
            self.selected_model_label.setStyleSheet("color: red;")
        project_layout.addWidget(self.selected_model_label)
        project_layout.addStretch(1)
        main_layout.addLayout(project_layout)


        # --- Display Area ---
        self.image_label = QLabel("请选择模式、项目和源")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) # Allow label to resize and expand
        self.image_label.setMinimumSize(640, 480) # Set a minimum size
        self.image_label.setStyleSheet("QLabel { background-color: black; color: white; }") # Default style
        main_layout.addWidget(self.image_label, 1) # Give it stretch factor 1


        # --- Consolidated Controls Group ---
        self.controls_group = QGroupBox("检测控制")
        controls_layout = QVBoxLayout(self.controls_group)
        main_layout.addWidget(self.controls_group) # Add group box to main layout

        # --- Mode Selection (Inside Group) ---
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("摄像头检测"))
        self.mode_selector_combo = QComboBox()
        self.mode_selector_combo.addItems([self.MODE_CAMERA])
        self.mode_selector_combo.currentTextChanged.connect(self._change_mode)
        # mode_layout.addWidget(self.mode_selector_combo)
        # mode_layout.addStretch(1)
        mode_layout.addWidget(QLabel("管子类别:")) # Changed label
        self.count_class_result = QLineEdit()
        self.count_class_result.setReadOnly(True) # Make read-only
        self.count_class_result.setFixedWidth(100) # Wider for class names
        self.count_class_result.setPlaceholderText("N/A")
        mode_layout.addWidget(self.count_class_result)

        mode_layout.addWidget(QLabel("螺丝数量:")) # Changed label

        self.positive_result = QLineEdit()
        self.positive_result.setReadOnly(True) # Make read-only
        self.positive_result.setFixedWidth(50)
        self.positive_result.setPlaceholderText("0")
        mode_layout.addWidget(self.positive_result)

        # Add the pass/fail label
        self.pass_fail_label = QLabel("状态: N/A")
        self.pass_fail_label.setFixedWidth(100) # Adjust width as needed
        self.pass_fail_label.setStyleSheet("color: gray;") # Initial style
        mode_layout.addWidget(self.pass_fail_label)

        mode_layout.addStretch(1)

        controls_layout.addLayout(mode_layout) # Add to group layout

        # --- Image Folder Mode Controls (Inside Group) ---
        self.image_controls_widget = QWidget() # Container for image controls
        image_controls_layout = QHBoxLayout(self.image_controls_widget)
        image_controls_layout.setContentsMargins(0,0,0,0) # Remove margins for tighter packing

        self.select_folder_button = QPushButton("选择图片文件夹")
        self.select_folder_button.clicked.connect(self.select_detection_images_folder)
        image_controls_layout.addWidget(self.select_folder_button)

        # Add label to display selected folder path
        self.selected_folder_label = QLabel("未选择文件夹")
        self.selected_folder_label.setStyleSheet("color: gray;") # Initial style
        image_controls_layout.addWidget(self.selected_folder_label)

        self.start_image_detection_button = QPushButton("开始检测!")
        self.start_image_detection_button.setStyleSheet("QPushButton { background-color: chocolate; color: white; font-size: 16px; padding: 5px; }")
        image_controls_layout.addWidget(self.start_image_detection_button)

        image_controls_layout.addStretch(1) # Push controls to the left

        self.prev_button = QPushButton("◀ 上一张")
        self.prev_button.clicked.connect(self.show_prev_image)
        self.prev_button.setEnabled(False)
        image_controls_layout.addWidget(self.prev_button)

        self.image_index_label = QLabel("0/0")
        image_controls_layout.addWidget(self.image_index_label)

        self.next_button = QPushButton("下一张 ▶")
        self.next_button.clicked.connect(self.show_next_image)
        self.next_button.setEnabled(False)
        image_controls_layout.addWidget(self.next_button)

        controls_layout.addWidget(self.image_controls_widget) # Add image controls container

        # --- Camera Mode Controls (Inside Group) ---
        self.camera_controls_widget = QWidget() # Container for camera controls
        camera_controls_layout = QHBoxLayout(self.camera_controls_widget)
        camera_controls_layout.setContentsMargins(0,0,0,0) # Remove margins

        self.select_save_folder_button = QPushButton("选择保存文件夹")
        self.select_save_folder_button.clicked.connect(self.select_camera_save_folder)
        camera_controls_layout.addWidget(self.select_save_folder_button)

        self.save_folder_label = QLabel("保存至: 默认")
        camera_controls_layout.addWidget(self.save_folder_label)

        # Add a label for the camera ID entry
        self.camera_id_label = QLabel("摄像头ID:")
        camera_controls_layout.addWidget(self.camera_id_label)
        self.camera_id_entry = QLineEdit(placeholderText="例如: 0 或者 rtsp://...") # Updated placeholder
        # Set default text to "0"
        self.camera_id_entry.setText("0")
        self.camera_id_entry.setFixedWidth(150)
        camera_controls_layout.addWidget(self.camera_id_entry)

        # 显示历史记录
        self.history_button = QPushButton("历史记录")
        self.history_button.setStyleSheet("QPushButton { background-color: lightblue; color: black; font-size: 16px; padding: 5px; }")
        self.history_button.clicked.connect(self.show_history) # Connect to history method
        camera_controls_layout.addWidget(self.history_button)
        camera_controls_layout.addStretch(1)

        self.start_stop_button = QPushButton("START")
        self.start_stop_button.setStyleSheet("QPushButton { background-color: green; color: white; font-size: 16px; padding: 5px; }")
        self.start_stop_button.setCheckable(True) # Make it toggle-like
        self.start_stop_button.toggled.connect(self.toggle_camera_detection) # Connect to specific toggle method
        camera_controls_layout.addWidget(self.start_stop_button)

        controls_layout.addWidget(self.camera_controls_widget) # Add camera controls container

        # --- Usage Guide (Inside Group) ---
        controls_layout.addSpacerItem(QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Minimum)) # Add space
        guide_label = QLabel("使用说明:")
        guide_label.setStyleSheet("font-weight: bold;")
        controls_layout.addWidget(guide_label)
        # Image Mode Guide (Updated)
        self.image_guide_label1 = QLabel("  图片文件夹检测: 1. 点击'选择项目' (列表会自动刷新) -> 2. 选择图片文件夹 -> 3. 点击'开始检测' -> 4. 使用导航按钮查看结果")
        controls_layout.addWidget(self.image_guide_label1)
        # Camera Mode Guide (Updated for auto-capture)
        self.camera_guide_label1 = QLabel("  摄像头检测: 1. 点击'选择项目' -> 2. (可选)选择保存文件夹 -> 3. 点击'START'")
        self.camera_guide_label2 = QLabel("                -> 4. 系统将自动检测稳定目标并保存 | 或按 Enter 键手动保存 -> 5. 点击'STOP'停止")
        controls_layout.addWidget(self.camera_guide_label1)
        controls_layout.addWidget(self.camera_guide_label2)


        # --- Status Bar / Progress Bar (Outside Group) ---
        status_layout = QHBoxLayout()
        self.status_label = QLabel("状态: 就绪") # Unified status label
        status_layout.addWidget(self.status_label, 1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0) # Indeterminate
        self.progress_bar.setVisible(False) # Hidden by default
        self.progress_bar.setMaximumWidth(200) # Limit width
        status_layout.addWidget(self.progress_bar)
        main_layout.addLayout(status_layout) # Add status layout to main layout

        # Allow widget to receive key presses (for camera capture)
        self.setFocusPolicy(Qt.StrongFocus)
        # Initialize save folder label for camera mode
        self._update_save_folder_label()


    def _change_mode(self, mode_text):
        """Handles switching between detection modes."""
        print(f"Changing mode to: {mode_text}")
        # Stop any active thread before switching mode
        self._stop_active_thread()

        self.current_mode = mode_text
        self._update_ui_for_mode()
        # Reset state relevant to the new mode
        self._reset_state_for_mode()

    def show_history(self):
        """Handles showing the history of detections."""
        # Implement your history display logic here
        print("Showing detection history...")
        # For example, you could open a new window or dialog with the history

    def _update_ui_for_mode(self):
        """Shows/hides UI elements based on the current mode."""
        # is_image_mode = (self.current_mode == self.MODE_IMAGE_FOLDER)
        is_image_mode = False
        is_camera_mode = (self.current_mode == self.MODE_CAMERA)

        # Show/hide control containers
        self.image_controls_widget.setVisible(is_image_mode)
        self.camera_controls_widget.setVisible(is_camera_mode)

        # Show/hide usage guide parts
        self.image_guide_label1.setVisible(is_image_mode)
        self.camera_guide_label1.setVisible(is_camera_mode)
        self.camera_guide_label2.setVisible(is_camera_mode)

        # Update progress bar visibility
        self.progress_bar.setVisible(is_image_mode and self.active_thread is not None and self.active_thread.isRunning())

        # Update status label and image label text/style
        if is_image_mode:
            self.status_label.setText("状态: 图片文件夹模式就绪")
            self.image_label.setText("请选择图片文件夹并开始检测") # Reset display
            self.image_label.setStyleSheet("QLabel { background-color: lightgrey; color: black; }") # Image mode style
        elif is_camera_mode:
            self.status_label.setText("状态: 摄像头模式就绪 | 按 Enter 键保存当前帧")
            self.image_label.setText("摄像头画面将显示在此处") # Reset display
            self.image_label.setStyleSheet("QLabel { background-color: black; color: white; }") # Camera mode style
        else:
             self.status_label.setText("状态: 未知模式")

        # Reset button states that might persist across modes
        # Image detection button state is handled by its own logic (start/finish/error)
        # Camera button state is handled by its toggle logic and on_camera_stopped
        # Ensure camera button is unchecked if switching away from camera mode while it was checked
        if not is_camera_mode and self.start_stop_button.isChecked():
             self.start_stop_button.setChecked(False) # This will trigger stop logic if needed


    def _reset_state_for_mode(self):
        """Resets internal state when switching modes."""
        self.image_label.clear() # Clear display
        self.image_label.setPixmap(QPixmap()) # Ensure no pixmap is held

        if self.current_mode == self.MODE_IMAGE_FOLDER:
            # Reset image folder specific state
            self.image_paths = []
            self.current_image_index = 0
            self.update_navigation()
            self.image_label.setText("请选择图片文件夹并开始检测")
            # Reset selected folder label
            self.selected_folder_label.setText("未选择文件夹")
            self.selected_folder_label.setStyleSheet("color: gray;")
        elif self.current_mode == self.MODE_CAMERA:
            # Reset camera specific state (UI elements are reset in _update_ui_for_mode)
            self.image_label.setText("摄像头画面将显示在此处")
            # Ensure camera UI elements are reset if switching TO camera mode
            self.on_camera_stopped() # Call this to reset camera UI elements


    def _stop_active_thread(self):
        """Stops the currently running detection thread, if any."""
        if self.active_thread and self.active_thread.isRunning():
            print("Stopping active thread...")
            self.active_thread.stop()



    def select_detection_images_folder(self):
        # Ensure this is only called in image mode (UI should prevent otherwise)
        path = QFileDialog.getExistingDirectory(self, "选择包含图片的文件夹")
        if path:
            self.detection_images_folder_path = path # Use instance variable
            print(f"Selected folder: {self.detection_images_folder_path}") # Use instance variable
            # Update the new label
            self.selected_folder_label.setText(f"已选: ...{os.path.basename(path)}") # Show only the folder name for brevity
            self.selected_folder_label.setStyleSheet("color: green;") # Indicate selection
            self.selected_folder_label.setToolTip(path) # Show full path on hover

            # Update status label only if in the correct mode
            if self.current_mode == self.MODE_IMAGE_FOLDER:
                self.status_label.setText(f"已选择文件夹: {os.path.basename(path)}")
                self.image_label.setText(f"已选择文件夹: {os.path.basename(path)}\n请选择模型并开始检测。")
            self.image_paths = []
            self.current_image_index = 0
            self.update_navigation()


    def update_project_dropdown(self):
        """Populates the project dropdown based on folders in base_model_save_path. Called before showing popup."""
        print("Refreshing project dropdown (triggered by aboutToShowPopup)...") # Add print statement for debugging
        current_selection = self.project_selector_combo.currentText() # Store current selection
        # Block signals ONLY for the clear/add operations, not for the whole function
        # because we need currentIndexChanged to work after selection.
        self.project_selector_combo.blockSignals(True)
        self.project_selector_combo.clear()
        self.project_selector_combo.addItem("默认模型")

        base_path = self.base_model_save_path # Use instance variable
        found_projects = False # Flag to track if any valid projects were found
        if base_path and os.path.isdir(base_path):
            try:
                # List directories directly inside the base_model_save_path (./output/)
                projects = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
                if projects:
                    self.project_selector_combo.addItems(sorted(projects))
                    found_projects = True
            except Exception as e:
                print(f"Error reading projects from {base_path}: {e}")
                # Add error item, but make it unselectable
                self.project_selector_combo.addItem("读取错误")
                error_item_index = self.project_selector_combo.count() - 1
                if error_item_index > 0: # Ensure it's not the only item
                    self.project_selector_combo.model().item(error_item_index).setEnabled(False)

        # Try to restore previous selection if it still exists
        index_to_select = 0 # Default to "Select project..."
        if found_projects and current_selection != "选择项目...":
            found_index = self.project_selector_combo.findText(current_selection)
            if found_index != -1: # If the previously selected item still exists
                index_to_select = found_index

        self.project_selector_combo.setCurrentIndex(index_to_select)
        self.project_selector_combo.blockSignals(False) # Unblock signals

        # DO NOT manually trigger _on_project_selected here.
        # It will be triggered automatically by setCurrentIndex if the index actually changes,
        # or by the user actually selecting an item after the popup is shown.
        # Forcing it here can lead to duplicate calls or incorrect state updates.
        print(f"Project dropdown refreshed. Current index set to: {index_to_select}") # Add print statement

    def _on_project_selected(self, index):
        """Handles project selection from the dropdown and loads class.txt."""
        selected_project = self.project_selector_combo.currentText()
        base_path = self.base_model_save_path
        self.class_names = [] # Reset class names
        self.detection_model_path = "" # Reset model path

        is_valid_selection = index > 0 and base_path and selected_project and selected_project not in ["默认模型", "选择项目...", "无项目 (output/下无文件夹)", "读取错误", "output/ 目录未找到或无效"]

        model_path = ""
        class_file_path = ""
        project_dir = ""

        if is_valid_selection:
            project_dir = os.path.join(base_path, selected_project)
            model_path = os.path.join(project_dir, 'weights', 'best.pt')
            class_file_path = os.path.join(project_dir, 'class.txt') # Path to class.txt

        model_exists = os.path.exists(model_path)
        class_file_exists = os.path.exists(class_file_path)

        if is_valid_selection and model_exists:
            self.detection_model_path = model_path
            self.selected_model_label.setText(f"模型: {selected_project}/weights/best.pt")
            self.selected_model_label.setStyleSheet("color: green;")
            print(f"Detection model set to: {model_path}")

            # Try to load class.txt
            if class_file_exists:
                try:
                    with open(class_file_path, 'r', encoding='utf-8') as f:
                        # Read lines, strip whitespace, ignore empty lines
                        self.class_names = [line.strip() for line in f if line.strip()]
                    if self.class_names:
                        print(f"Loaded {len(self.class_names)} classes from {class_file_path}: {self.class_names}")
                        # Optionally update status or label
                    else:
                        print(f"Warning: {class_file_path} is empty.")
                        self.status_label.setText("状态: 警告 - class.txt 为空")
                except Exception as e:
                    print(f"Error reading {class_file_path}: {e}")
                    QMessageBox.warning(self, "类文件错误", f"无法读取类文件:\n{class_file_path}\n\n错误: {e}")
                    self.class_names = [] # Ensure it's empty on error
                    self.status_label.setText(f"状态: 错误 - 无法读取 class.txt")
            else:
                print(f"Warning: class.txt not found at {class_file_path}. Will try to use model's internal names.")
                self.status_label.setText("状态: 警告 - 未找到 class.txt")
                # Keep self.class_names empty, CameraDetectionThread will try model.names

            # If in camera mode and running, warn user to restart
            if self.current_mode == self.MODE_CAMERA and self.active_thread and self.active_thread.isRunning():
                self.status_label.setText("状态: 项目/模型已更改，请重启摄像头。")
                self.start_stop_button.setChecked(False) # Force stop
        else:
            # Handle cases where model or project selection is invalid
            old_model_path = self.detection_model_path
            self.detection_model_path = ""
            self.class_names = [] # Clear classes if model is invalid/not found
            if is_valid_selection: # Project selected, but model not found
                self.selected_model_label.setText(f"模型: {selected_project}/weights/best.pt (未找到!)")
                self.selected_model_label.setStyleSheet("color: red;")
                print(f"Warning: best.pt not found at {model_path}")
            else: # Invalid project selection (e.g., "Select project...")
                self.selected_model_label.setText("模型: 未选择")
                self.selected_model_label.setStyleSheet("color: red;")

            # If model became invalid while camera was running, stop it
            if self.current_mode == self.MODE_CAMERA and self.active_thread and self.active_thread.isRunning() and old_model_path:
                 self.status_label.setText("状态: 模型无效/未找到，请停止摄像头。")
                 self.start_stop_button.setChecked(False) # Force stop

        # Reset detection counts display when project changes
        self.count_class_result.setText("N/A")
        self.positive_result.setText("0")
        self.pass_fail_label.setText("状态: N/A") # Reset pass/fail label
        self.pass_fail_label.setStyleSheet("color: gray;")


    def update_image(self):
        """Updates the image display (for Image Folder mode)."""
        if self.current_mode != self.MODE_IMAGE_FOLDER: return

        if not self.image_paths:
            self.image_label.setText("无图片显示")
            self.image_label.setStyleSheet("QLabel { background-color: lightgrey; color: black; }")
            return

        if not 0 <= self.current_image_index < len(self.image_paths):
             print(f"Warning: current_image_index ({self.current_image_index}) out of bounds for image_paths (len={len(self.image_paths)}). Resetting.")
             self.current_image_index = 0
             if not self.image_paths: # Double check if list became empty somehow
                 self.image_label.setText("无图片显示")
                 self.update_navigation()
                 return

        path = self.image_paths[self.current_image_index]
        pixmap = QPixmap(path)

        if pixmap.isNull():
            self.image_label.setText(f"无法加载图片:\n{os.path.basename(path)}")
            self.image_label.setStyleSheet("QLabel { background-color: lightgrey; color: red; }")
            # Optionally remove the invalid path here (handle index carefully)
            return

        # Scale pixmap to fit label while maintaining aspect ratio
        self.image_label.setStyleSheet("") # Reset style before setting pixmap
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)
        self.update_navigation()

    def show_next_image(self):
        if self.current_mode == self.MODE_IMAGE_FOLDER and self.image_paths:
            self.current_image_index = (self.current_image_index + 1) % len(self.image_paths)
            self.update_image()

    def show_prev_image(self):
        if self.current_mode == self.MODE_IMAGE_FOLDER and self.image_paths:
            self.current_image_index = (self.current_image_index - 1) % len(self.image_paths)
            self.update_image()

    def update_navigation(self):
        """Updates image navigation controls (for Image Folder mode)."""
        # Visibility is handled by _update_ui_for_mode
        # Here we just update text and enabled state if visible
        if self.current_mode != self.MODE_IMAGE_FOLDER:
            # Ensure they are disabled even if somehow visible
            self.image_index_label.setText("")
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)
            return

        num_images = len(self.image_paths)
        if num_images > 0:
            self.image_index_label.setText(f"{self.current_image_index + 1}/{num_images}")
            self.prev_button.setEnabled(True)
            self.next_button.setEnabled(True)
        else:
            self.image_index_label.setText("0/0")
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)

    # --- Camera Mode Specific Methods ---

    def select_camera_save_folder(self):
        # Ensure this is only called in camera mode (UI should prevent otherwise)
        suggested_path = "." # Default to current directory
        selected_project = self.project_selector_combo.currentText()
        base_path = self.base_model_save_path # Use instance variable
        # Adjusted condition text
        if selected_project and selected_project not in ["选择项目...", "无项目 (output/下无文件夹)", "读取错误", "output/ 目录未找到或无效"] and base_path:
             project_output_path = os.path.join(base_path, selected_project, "camera_captures")
             suggested_path = project_output_path # Suggest a subfolder within the project output

        path = QFileDialog.getExistingDirectory(self, "选择截图和标签保存文件夹", suggested_path)
        if path:
            self.detection_save_dir = path # Use instance variable
            print(f"Selected save folder: {self.detection_save_dir}") # Use instance variable
            self._update_save_folder_label()
            if self.active_thread and self.active_thread.isRunning() and isinstance(self.active_thread, CameraDetectionThread):
                 # Pass the updated path to the running thread
                 self.active_thread.set_save_directory(self.detection_save_dir) # Use instance variable
                 self.update_status(f"保存文件夹已更新: {os.path.basename(path)}")


    def _update_save_folder_label(self):
        """Updates the label showing the selected save directory."""
        if self.detection_save_dir: # Use instance variable
            self.save_folder_label.setText(f"保存至: {os.path.basename(self.detection_save_dir)}")
        else:
            # Determine default save path (e.g., relative to project or just '.')
            default_path_display = "默认 (项目/camera_captures 或 .)" # Example text
            self.save_folder_label.setText(f"保存至: {default_path_display}")


    def toggle_camera_detection(self, checked):
        """Starts or stops camera detection."""
        if self.current_mode != self.MODE_CAMERA: return # Should not happen

        if checked: # Start detection
            if not self.detection_model_path or not os.path.exists(self.detection_model_path): # Use instance variable and check existence
                QMessageBox.warning(self, "错误", "请先选择一个有效的模型文件 (best.pt)。")
                self.start_stop_button.setChecked(False) # Uncheck the button
                return

            camera_id_str = self.camera_id_entry.text().strip()
            if not camera_id_str:
                QMessageBox.warning(self, "错误", "请输入摄像头 ID 或 URL。")
                self.start_stop_button.setChecked(False)
                return

            # Use the entered string directly for VideoCapture, handles digits, URLs, file paths
            camera_id = int(camera_id_str)

            # Determine save directory if not explicitly set
            save_dir = self.detection_save_dir # Use instance variable
            if not save_dir:
                 # Default save logic: try to save inside project folder, else use '.' 
                 selected_project = self.project_selector_combo.currentText()
                 base_path = self.base_model_save_path # Use instance variable
                 # Adjusted condition text
                 if selected_project and selected_project not in ["选择项目...", "无项目 (output/下无文件夹)", "读取错误", "output/ 目录未找到或无效"] and base_path:
                     save_dir = os.path.join(base_path, selected_project, "camera_captures")
                 else:
                     save_dir = "." # Fallback to current directory
                 self.detection_save_dir = save_dir # Update instance variable with default
                 self._update_save_folder_label() # Update UI label

            os.makedirs(save_dir, exist_ok=True) # Ensure directory exists

            self.start_stop_button.setText("STOP")
            self.start_stop_button.setStyleSheet("QPushButton { background-color: red; color: white; font-size: 16px; padding: 5px; }")
            self.camera_id_entry.setEnabled(False)
            self.select_save_folder_button.setEnabled(False)
            self.project_selector_combo.setEnabled(False) # Disable project selection while running
            self.mode_selector_combo.setEnabled(False) # Disable mode switching while running

            # Pass model path, loaded class names, camera ID, and save dir
            self.active_thread = CameraDetectionThread(
                self.detection_model_path,
                self.class_names, # Pass loaded class names
                camera_id,
                self.detection_save_dir
            )
            self.active_thread.frame_signal.connect(self.update_frame)
            self.active_thread.status_signal.connect(self.update_status)
            self.active_thread.stopped_signal.connect(self.on_camera_stopped)
            self.active_thread.detection_results_signal.connect(self._update_detection_counts) # Connect new signal
            self.active_thread.start()
            self.update_status("正在启动摄像头...")
            self.image_label.setText("正在启动摄像头...")
            # Clear previous counts on start
            self.count_class_result.setText("N/A")
            self.positive_result.setText("0")
            self.pass_fail_label.setText("状态: N/A") # Clear pass/fail label on start
            self.pass_fail_label.setStyleSheet("color: gray;")

        else: # Stop detection
            if self.active_thread and self.active_thread.isRunning() and isinstance(self.active_thread, CameraDetectionThread):
                self.update_status("正在停止...")
                self.active_thread.stop()
            else:
                self.on_camera_stopped()


    def update_frame(self, pixmap):
        """Updates the display with a new frame from the camera."""
        if self.current_mode != self.MODE_CAMERA: return

        # Scale pixmap to fit label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    def update_status(self, message):
        """Updates the status label (primarily for Camera mode)."""
        # Prepend timestamp for clarity
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.status_label.setText(f"状态 ({timestamp}): {message}")

        # Handle specific error messages if needed (relevant to camera mode)
        if self.current_mode == self.MODE_CAMERA:
            if "Error: Could not open camera" in message or "Model not loaded" in message or "Model path invalid" in message:
                 self.image_label.setText(message) # Show error on image label too
                 # Ensure button is reset if start failed - trigger the stop sequence UI updates
                 if self.start_stop_button.isChecked():
                     self.start_stop_button.setChecked(False) # This will trigger on_camera_stopped via the toggle signal
                 else: # If already unchecked, manually call on_camera_stopped to reset UI
                     self.on_camera_stopped()


    def on_camera_stopped(self):
        """Resets UI elements to the stopped state for Camera mode."""
        print("Camera stopped signal received or called.")
        # Ensure UI elements are visible if we are in camera mode
        if self.current_mode == self.MODE_CAMERA:
            self.start_stop_button.setText("START")
            self.start_stop_button.setStyleSheet("QPushButton { background-color: green; color: white; font-size: 16px; padding: 5px; }")
            # Ensure state is unchecked, block signals briefly to prevent re-triggering toggle_camera_detection
            self.start_stop_button.blockSignals(True)
            self.start_stop_button.setChecked(False)
            self.start_stop_button.blockSignals(False)

            self.camera_id_entry.setEnabled(True)
            self.select_save_folder_button.setEnabled(True)
            self.project_selector_combo.setEnabled(True) # Re-enable project selection
            self.mode_selector_combo.setEnabled(True) # Re-enable mode switching

            # Update status only if not already showing a critical error
            current_status = self.status_label.text()
            if "Error" not in current_status and "停止" not in current_status and "无效" not in current_status:
                 self.status_label.setText("状态: 已停止 | 按 Enter 键保存当前帧")
            # Optionally clear the image label or show a "stopped" message
            # self.image_label.setText("摄像头已停止")
        else:
            # If mode changed while stopping, ensure button is visually reset anyway
            self.start_stop_button.setText("START")
            self.start_stop_button.setStyleSheet("QPushButton { background-color: green; color: white; font-size: 16px; padding: 5px; }")
            self.start_stop_button.blockSignals(True)
            self.start_stop_button.setChecked(False)
            self.start_stop_button.blockSignals(False)


        self.active_thread = None # Clean up thread reference

    def _update_detection_counts(self, class_counts):
        """Slot to receive class counts and update UI specifically for pipe (A/B) and screws (positive)."""
        pipe_class = "N/A"
        positive_count = 0

        # Check for pipe classes 'A' or 'B'
        count_a = class_counts.get('A', 0)
        count_b = class_counts.get('B', 0)

        if count_a > 0 and count_b > 0:
            pipe_class = 'A' if count_a >= count_b else 'B'
        elif count_a > 0:
            pipe_class = 'A'
        elif count_b > 0:
            pipe_class = 'B'

        # Get the count for 'positive' class (screws)
        positive_count = class_counts.get('positive', 0)

        # Update the UI elements
        self.count_class_result.setText(pipe_class)
        self.positive_result.setText(str(positive_count))

        # Update pass/fail label based on positive_count
        if positive_count == 4:
            self.pass_fail_label.setText("检测合格")
            self.pass_fail_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.pass_fail_label.setText("检测不合格")
            self.pass_fail_label.setStyleSheet("color: red; font-weight: bold;")


    def keyPressEvent(self, event):
        if self.current_mode == self.MODE_CAMERA and (event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter):
            if self.active_thread and self.active_thread.isRunning() and isinstance(self.active_thread, CameraDetectionThread):
                self.update_status("手动保存帧...")
                # Call capture_frame without frame/results to trigger manual capture logic
                self.active_thread.capture_frame(automatic=False)
            else:
                self.update_status("摄像头未运行，无法保存。")
        else:
            super().keyPressEvent(event) # Pass other key events up

    # --- Common Event Handlers ---

    def resizeEvent(self, event):
        # Check if the label has a pixmap before trying to update
        # This is relevant for both modes (image display and camera feed)
        current_pixmap = self.image_label.pixmap()
        if current_pixmap and not current_pixmap.isNull():
             # Scale pixmap to fit label while maintaining aspect ratio
             scaled_pixmap = current_pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
             self.image_label.setPixmap(scaled_pixmap)
        super().resizeEvent(event)



# Add a main execution block for standalone testing if desired
if __name__ == '__main__':
    multiprocessing.freeze_support() # <--- 在开头添加此行

    # Necessary for high-DPI displays
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    # When running detection_gui.py directly, base_path will be None, using default behavior
    window = DetectionWindow()
    window.show()
    sys.exit(app.exec_())
