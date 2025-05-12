import os
import sys
import cv2
import datetime
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit,
    QComboBox, QProgressBar, QFileDialog, QSizePolicy, QMessageBox, QGroupBox,
    QSpacerItem
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# Assuming detect_images is in src.detect relative to the execution path
try:
    # Adjust path if necessary based on your project structure
    # This assumes train_gui.py is run from the project root
    from src.detect import detect_images
except ImportError:
    print("Warning: Could not import detect_images from src.detect.")
    # Define a dummy function if import fails
    def detect_images(folder_path, model_path):
        print(f"Dummy detect_images called with: {folder_path}, {model_path}")
        print("Please ensure src.detect.detect_images is correctly implemented and importable.")
        # Simulate creating a results directory for basic UI flow
        # Use folder_path's parent directory for dummy results to avoid cluttering 'output'
        results_parent_dir = os.path.dirname(folder_path) if folder_path else "."
        results_dir = os.path.join(results_parent_dir, "dummy_results_" + os.path.basename(model_path).replace('.pt',''))
        os.makedirs(results_dir, exist_ok=True)
        # Create a dummy image file inside
        dummy_file_path = os.path.join(results_dir, "dummy_result.txt")
        with open(dummy_file_path, 'w') as f:
            f.write("Dummy detection result.")
        print(f"Created dummy results directory: {results_dir}")
        return results_dir # Return the dummy path


# --- Custom ComboBox ---
class ProjectComboBox(QComboBox):
    """A custom QComboBox that emits a signal just before showing the popup."""
    aboutToShowPopup = pyqtSignal()

    def showPopup(self):
        self.aboutToShowPopup.emit() # Emit signal before showing
        super().showPopup()

# --- Threads ---

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
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
        except ImportError:
            self.status_signal.emit("Error: ultralytics library not found.")
            self.model = None
        except Exception as e:
            self.status_signal.emit(f"Error loading model: {e}")
            self.model = None

    def set_save_directory(self, directory):
        self.save_dir = directory
        os.makedirs(self.save_dir, exist_ok=True)

    def run(self):
        # --- Model Loading Check ---
        # This check needs to happen *after* the model is actually loaded.
        # For now, we simulate a successful load if a path exists.
        # Replace this with actual model loading logic.
        if not self.model_path or not os.path.exists(self.model_path):
             self.status_signal.emit("Error: Model path invalid or model not loaded.")
             self.stopped_signal.emit()
             return
        else:
            # Placeholder: Assume model loaded successfully if path is valid
            # In reality, load the model here (e.g., self.model = YOLO(self.model_path))
            # and handle loading errors.
            print(f"Simulating model load for: {self.model_path}")
            self.model = "Simulated Model Object" # Replace with actual model object

        if not self.model: # Check if model loading actually succeeded
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

            # --- Perform detection (using self.model) ---
            # Example: Replace with actual detection call
            # results = self.model(frame)
            # Draw bounding boxes on the frame (implement _draw_bounding_boxes or similar)
            self._draw_bounding_boxes(frame, results) # Pass actual results

            # --- Convert frame to QPixmap ---
            try:
                # Ensure frame is not empty
                if frame is None or frame.size == 0:
                    self.status_signal.emit("Warning: Empty frame received.")
                    continue

                # Check frame dimensions and channels
                if frame.ndim != 3 or frame.shape[2] != 3:
                     # Attempt to convert if grayscale
                     if frame.ndim == 2:
                         frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                     else:
                         self.status_signal.emit(f"Warning: Unexpected frame shape {frame.shape}")
                         continue # Skip this frame

                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                self.frame_signal.emit(pixmap)

            except cv2.error as e:
                self.status_signal.emit(f"OpenCV Error during frame conversion: {e}")
                # Optionally try to recover or just continue
                self.msleep(50)
                continue
            except Exception as e:
                 self.status_signal.emit(f"Error processing frame: {e}")
                 # Optionally try to recover or just continue
                 self.msleep(50)
                 continue


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
        if not ret or frame is None:
            self.status_signal.emit("Capture failed: Could not read frame.")
            return

        self.scene_id += 1
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        base_filename = f"{timestamp}_{self.scene_id:04d}"

        origin_image_path = os.path.join(self.save_dir, f"{base_filename}_origin.jpg")
        try:
            cv2.imwrite(origin_image_path, frame)
            self.status_signal.emit(f"Saved original: {os.path.basename(origin_image_path)}")
        except Exception as e:
            self.status_signal.emit(f"Error saving original image: {e}")
            return # Don't proceed if saving original failed

        # --- Perform detection and save results ---
        # Example: Replace with actual detection and drawing
        # try:
        #     results = self.model(frame)
        #     detection_frame = frame.copy() # Draw on a copy
        #     self._draw_bounding_boxes(detection_frame, results) # Implement this
        #
        #     detection_image_path = os.path.join(self.save_dir, f"{base_filename}_detection.jpg")
        #     cv2.imwrite(detection_image_path, detection_frame)
        #     self.status_signal.emit(f"Saved detection: {os.path.basename(detection_image_path)}")
        #
        #     txt_path = os.path.join(self.save_dir, f"{base_filename}_detection.txt")
        #     with open(txt_path, 'w') as f:
        #         # Example for ultralytics results
        #         for result in results[0].boxes:
        #             if result.conf[0] >= self.conf_threshold:
        #                 x1, y1, x2, y2 = map(int, result.xyxy[0])
        #                 label = self.model.names[int(result.cls[0])] # Assumes model has 'names' attribute
        #                 confidence = result.conf[0]
        #                 f.write(f"{label} {confidence:.2f} {x1} {y1} {x2} {y2}\n")
        #     self.status_signal.emit(f"Saved labels: {os.path.basename(txt_path)}")
        # except AttributeError as e:
        #      self.status_signal.emit(f"Error processing detection results (maybe model structure changed?): {e}")
        # except Exception as e:
        #      self.status_signal.emit(f"Error saving detection results: {e}")


    def _draw_bounding_boxes(self, frame, results):
         # Example implementation for ultralytics results
         # Ensure self.model is loaded and has a 'names' attribute
         if not self.model or not hasattr(self.model, 'names'): return
         try:
             for result in results[0].boxes: # Assuming results structure
                 if result.conf[0] >= self.conf_threshold:
                     x1, y1, x2, y2 = map(int, result.xyxy[0])
                     class_index = int(result.cls[0])
                     if 0 <= class_index < len(self.model.names):
                         label = self.model.names[class_index]
                     else:
                         label = f"Class_{class_index}" # Fallback label
                     confidence = result.conf[0]
                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                     cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
         except IndexError:
             self.status_signal.emit("Warning: Error accessing detection results (IndexError).")
         except Exception as e:
             self.status_signal.emit(f"Error drawing boxes: {e}")


# --- Widgets ---

class DetectionWindow(QWidget): # Renamed from ImageDetectionWidget
    MODE_IMAGE_FOLDER = "图片文件夹检测" # Changed to Chinese
    MODE_CAMERA = "摄像头检测" # Changed to Chinese

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

        # Image detection state
        self.image_paths = []
        self.current_image_index = 0
        # Thread management
        self.active_thread = None # Unified thread variable
        self.current_mode = self.MODE_IMAGE_FOLDER # Default mode

        self._init_ui()
        self.update_project_dropdown() # Initial population
        self._update_ui_for_mode() # Set initial UI state

    def _init_ui(self):
        main_layout = QVBoxLayout(self)

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
        mode_layout.addWidget(QLabel("选择检测模式:"))
        self.mode_selector_combo = QComboBox()
        self.mode_selector_combo.addItems([self.MODE_IMAGE_FOLDER, self.MODE_CAMERA])
        self.mode_selector_combo.currentTextChanged.connect(self._change_mode)
        mode_layout.addWidget(self.mode_selector_combo)
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
        self.start_image_detection_button.clicked.connect(self.start_detection) # Connect to unified start
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
        self.camera_id_label = QLabel("摄像头ID/URL:")
        camera_controls_layout.addWidget(self.camera_id_label)
        self.camera_id_entry = QLineEdit(placeholderText="例如: 0 或 rtsp://...")
        self.camera_id_entry.setFixedWidth(150)
        camera_controls_layout.addWidget(self.camera_id_entry)

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
        # Camera Mode Guide (Updated)
        self.camera_guide_label1 = QLabel("  摄像头检测: 1. 点击'选择项目' (列表会自动刷新) -> 2. (可选)选择保存文件夹 -> 3. 输入摄像头ID/URL -> 4. 点击'START'")
        self.camera_guide_label2 = QLabel("                -> 5. 按 Enter 键保存当前帧及检测结果 -> 6. 点击'STOP'停止")
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


    def _update_ui_for_mode(self):
        """Shows/hides UI elements based on the current mode."""
        is_image_mode = (self.current_mode == self.MODE_IMAGE_FOLDER)
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
            thread_instance_type = type(self.active_thread) # Get type before potential termination
            try:
                if isinstance(self.active_thread, CameraDetectionThread):
                    self.active_thread.stop()
                    # Camera thread stop sequence handles UI updates via on_camera_stopped signal
                elif isinstance(self.active_thread, ImageDetectionThread):
                    # Image thread doesn't have an explicit stop, rely on termination
                    # Consider adding a graceful stop flag in ImageDetectionThread if needed
                    self.active_thread.terminate() # Forceful stop, use with caution
                    self.active_thread.wait(2000) # Wait a bit for termination
            except Exception as e:
                print(f"Error stopping/terminating thread: {e}")
            finally:
                self.active_thread = None # Clear reference
                print("Active thread stopped/terminated.")

                # Manually reset UI if termination was used or stop signal didn't fully reset
                if thread_instance_type == ImageDetectionThread:
                    self.progress_bar.setVisible(False)
                    # Ensure button is visible if we are in image mode
                    if self.current_mode == self.MODE_IMAGE_FOLDER:
                        self.start_image_detection_button.setEnabled(True)
                        self.start_image_detection_button.setText("开始检测!")
                        self.status_label.setText("状态: 检测已停止")
                elif thread_instance_type == CameraDetectionThread:
                     # Ensure camera UI is fully reset even if stopped signal had issues
                     self.on_camera_stopped()


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
        self.project_selector_combo.addItem("选择项目...")

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
        """Handles project selection from the dropdown."""
        selected_project = self.project_selector_combo.currentText()
        base_path = self.base_model_save_path # Use instance variable
        # Adjusted condition text
        is_valid_selection = index > 0 and base_path and selected_project and selected_project not in ["选择项目...", "无项目 (output/下无文件夹)", "读取错误", "output/ 目录未找到或无效"]

        model_path = ""
        if is_valid_selection:
            # Construct the expected path to best.pt within the selected project folder in ./output/
            model_path = os.path.join(base_path, selected_project, 'weights', 'best.pt')

        if is_valid_selection and os.path.exists(model_path):
            self.detection_model_path = model_path # Use instance variable
            self.selected_model_label.setText(f"模型: {selected_project}/weights/best.pt")
            self.selected_model_label.setStyleSheet("color: green;")
            print(f"Detection model set to: {model_path}")
            # If in camera mode and running, warn user to restart
            if self.current_mode == self.MODE_CAMERA and self.active_thread and self.active_thread.isRunning():
                self.status_label.setText("状态: 模型已更改，请重启摄像头。")
                self.start_stop_button.setChecked(False) # Force stop
        else:
            old_model_path = self.detection_model_path # Use instance variable
            self.detection_model_path = "" # Use instance variable
            if is_valid_selection:
                self.selected_model_label.setText(f"模型: {selected_project}/weights/best.pt (未找到!)")
                self.selected_model_label.setStyleSheet("color: red;")
                print(f"Warning: best.pt not found at {model_path}")
            else:
                self.selected_model_label.setText("模型: 未选择")
                self.selected_model_label.setStyleSheet("color: red;")

            # If model became invalid while camera was running, stop it
            if self.current_mode == self.MODE_CAMERA and self.active_thread and self.active_thread.isRunning() and old_model_path:
                 self.status_label.setText("状态: 模型无效/未找到，请停止摄像头。")
                 self.start_stop_button.setChecked(False) # Force stop


    def start_detection(self):
        """Starts detection based on the current mode (Image Folder)."""
        if self.current_mode != self.MODE_IMAGE_FOLDER:
            return # Should not be called in camera mode

        if not self.detection_images_folder_path: # Use instance variable
            QMessageBox.warning(self, "错误", "请先选择图片文件夹。")
            return
        if not self.detection_model_path: # Use instance variable
            QMessageBox.warning(self, "错误", "请先选择有效的模型文件 (best.pt)。")
            return
        if self.active_thread and self.active_thread.isRunning():
            QMessageBox.warning(self, "运行中", "检测已在进行中。")
            return

        # Ensure UI elements are visible for this mode
        self._update_ui_for_mode() # Refresh UI state just in case

        self.progress_bar.setVisible(True)
        self.start_image_detection_button.setEnabled(False)
        self.start_image_detection_button.setText("检测中...")
        self.status_label.setText("状态: 检测进行中...")
        self.image_label.setText("检测进行中...")
        self.image_paths = [] # Clear previous results
        self.update_navigation()

        # Start image detection in a thread - use instance variables
        self.active_thread = ImageDetectionThread(self.detection_images_folder_path, self.detection_model_path)
        self.active_thread.finished_signal.connect(self.on_detection_finished)
        self.active_thread.error_signal.connect(self.on_detection_error) # Connect error signal
        # Connect progress signal if implemented in the thread
        # self.active_thread.progress_signal.connect(self.update_progress)
        self.active_thread.start()

    def on_detection_error(self, error_message):
        """Handles errors reported by the image detection thread."""
        # Check if the error is for the current mode
        if self.current_mode != self.MODE_IMAGE_FOLDER:
            return

        self.progress_bar.setVisible(False)
        self.start_image_detection_button.setEnabled(True)
        self.start_image_detection_button.setText("开始检测!")
        self.active_thread = None # Clean up thread reference
        QMessageBox.critical(self, "检测错误", error_message)
        self.status_label.setText(f"状态: 检测错误: {error_message}")
        self.image_label.setText(f"检测错误: {error_message}")
        self.image_paths = [] # Clear image paths on error
        self.update_navigation()

    def on_detection_finished(self, results_dir):
        """Handles completion of the image detection thread."""
         # Check if the signal is for the current mode
        if self.current_mode != self.MODE_IMAGE_FOLDER:
            return

        self.progress_bar.setVisible(False)
        # Ensure button is visible if we are still in image mode
        if self.current_mode == self.MODE_IMAGE_FOLDER:
            self.start_image_detection_button.setEnabled(True)
            self.start_image_detection_button.setText("开始检测!")
        self.active_thread = None

        if not results_dir or not os.path.isdir(results_dir):
             # Check if an error occurred (error signal might have already handled this)
             if "错误" not in self.status_label.text(): # Avoid double messaging if error signal was emitted
                 QMessageBox.warning(self, "错误", f"检测完成，但未找到结果目录: {results_dir}")
                 self.status_label.setText("状态: 检测完成，但未找到结果。")
                 self.image_label.setText("检测完成，但未找到结果。")
             self.image_paths = []
        else:
            try:
                self.image_paths = [os.path.join(results_dir, f) for f in sorted(os.listdir(results_dir))
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))] # Added more image types
                if not self.image_paths:
                    self.status_label.setText("状态: 检测完成，但结果目录中没有支持的图片文件。")
                    self.image_label.setText("检测完成，但结果目录中没有支持的图片文件。")
                else:
                    self.current_image_index = 0
                    self.update_image() # Display first image
                    self.status_label.setText(f"状态: 检测完成，找到 {len(self.image_paths)} 张结果图片。")
            except Exception as e:
                 QMessageBox.critical(self, "错误", f"加载结果图片时出错: {e}")
                 self.status_label.setText("状态: 加载结果图片时出错。")
                 self.image_label.setText("加载结果图片时出错。")
                 self.image_paths = []

        self.update_navigation()


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

            try:
                # Allow non-numeric IDs if VideoCapture supports them (e.g., RTSP URLs)
                if camera_id_str.isdigit():
                    camera_id = int(camera_id_str)
                elif "rtsp://" in camera_id_str or "http://" in camera_id_str or "https://" in camera_id_str or os.path.exists(camera_id_str): # Also allow video file paths
                     camera_id = camera_id_str # Use the string directly
                else:
                     raise ValueError("Invalid Camera ID/URL/File format")
            except ValueError:
                 QMessageBox.warning(self, "错误", "请输入有效的摄像头 ID (数字、URL 或视频文件路径)。")
                 self.start_stop_button.setChecked(False)
                 return

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

            # Pass model path and save dir using instance variables
            self.active_thread = CameraDetectionThread(
                self.detection_model_path,
                camera_id,
                self.detection_save_dir
            )
            self.active_thread.frame_signal.connect(self.update_frame)
            self.active_thread.status_signal.connect(self.update_status)
            self.active_thread.stopped_signal.connect(self.on_camera_stopped) # Connect stopped signal
            self.active_thread.start()
            self.update_status("正在启动摄像头...")
            self.image_label.setText("正在启动摄像头...") # Placeholder text

        else: # Stop detection
            if self.active_thread and self.active_thread.isRunning() and isinstance(self.active_thread, CameraDetectionThread):
                self.update_status("正在停止...")
                self.active_thread.stop()
                # UI elements are re-enabled in on_camera_stopped
            else:
                # If stop is triggered when not running (e.g., user clicks START then STOP quickly, or error occurred)
                # Ensure UI is in the stopped state
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

    def keyPressEvent(self, event):
        if self.current_mode == self.MODE_CAMERA and (event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter):
            if self.active_thread and self.active_thread.isRunning() and isinstance(self.active_thread, CameraDetectionThread):
                self.update_status("正在保存帧...")
                self.active_thread.capture_frame()
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

    # Ensure thread is stopped when widget is closed/hidden
    # def closeEvent(self, event):
    #      print("Close event triggered for DetectionWindow")
    #      self._stop_active_thread()
    #      super().closeEvent(event)

    # def hideEvent(self, event):
    #     print("Hide event triggered for DetectionWindow")
    #     self._stop_active_thread() # Stop any active thread (image or camera)
    #     super().hideEvent(event)

# Add a main execution block for standalone testing if desired
if __name__ == '__main__':
    app = QApplication(sys.argv)
    # When running detection_gui.py directly, base_path will be None, using default behavior
    window = DetectionWindow()
    window.show()
    sys.exit(app.exec_())
