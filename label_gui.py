# label_gui.py
import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QStatusBar, QLabel, QFileDialog,
    QGraphicsView, QGraphicsScene, QDockWidget, QListWidget,
    QVBoxLayout, QWidget, QPushButton, QLineEdit, QHBoxLayout,
    QMessageBox, QAction, QGraphicsRectItem, QListWidgetItem,
    QMenu, QInputDialog
)
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QGuiApplication, QTransform, QCursor, QIcon
from PyQt5.QtCore import Qt, QRectF, QPointF

# Predefined list of colors for classes
CLASS_COLORS = [
    QColor("red"), QColor("lime"), QColor("blue"), QColor("yellow"),
    QColor("magenta"), QColor("cyan"), QColor("orange"), QColor("purple"),
    QColor("brown"), QColor("pink"), QColor("olive"), QColor("teal"),
    QColor(128, 0, 0), QColor(0, 128, 0), QColor(0, 0, 128), QColor(128, 128, 0)
]

# --- Dataset Manager ---
class DatasetManager:
    """Manages dataset paths, files, and classes."""
    def __init__(self):
        self.dataset_root = ""
        self.images_dir = ""
        self.labels_dir = ""
        self.classes_path = ""
        self.image_files = []
        self.classes = []

    def set_root_directory(self, root_path):
        """Sets the dataset root and derives sub-directory paths."""
        self.dataset_root = root_path
        self.images_dir = os.path.join(self.dataset_root, "images")
        self.labels_dir = os.path.join(self.dataset_root, "labels")
        self.classes_path = os.path.join(self.dataset_root, "classes.txt")

        # Basic validation
        if not os.path.isdir(self.dataset_root):
            print(f"Warning: Dataset root directory does not exist: {self.dataset_root}")
            return False
        if not os.path.isdir(self.images_dir):
            print(f"Warning: Images directory does not exist: {self.images_dir}")
            # Optionally create it? For now, just warn.
            # os.makedirs(self.images_dir, exist_ok=True)
            return False # Require images dir to exist for loading

        # Ensure labels directory exists for saving
        try:
            os.makedirs(self.labels_dir, exist_ok=True)
        except OSError as e:
            QMessageBox.critical(None, "Error Creating Directory", f"Could not create labels directory:\n{self.labels_dir}\n\nError: {e}")
            return False

        return True

    def load_image_files(self):
        """Loads image filenames from the images directory."""
        self.image_files = []
        if not self.images_dir or not os.path.isdir(self.images_dir):
            print("Error: Images directory not set or does not exist.")
            return False
        try:
            self.image_files = sorted([f for f in os.listdir(self.images_dir)
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])
            print(f"Loaded {len(self.image_files)} images from {self.images_dir}")
            return True
        except Exception as e:
            QMessageBox.warning(None, "Load Images Error", f"Could not list images in:\n{self.images_dir}\n\nError: {e}")
            return False

    def get_image_path(self, index):
        """Gets the full path for an image file by index."""
        if 0 <= index < len(self.image_files):
            return os.path.join(self.images_dir, self.image_files[index])
        return None

    def get_image_filename(self, index):
        """Gets the filename for an image file by index."""
        if 0 <= index < len(self.image_files):
            return self.image_files[index]
        return None

    def get_image_count(self):
        """Returns the number of loaded image files."""
        return len(self.image_files)

    def get_label_path(self, index):
        """Gets the full path for the corresponding label file by index."""
        if 0 <= index < len(self.image_files):
            base_name = os.path.splitext(self.image_files[index])[0]
            return os.path.join(self.labels_dir, base_name + ".txt")
        return None

    def load_classes(self):
        """Loads class names from classes.txt."""
        self.classes = []
        if not self.classes_path:
            print("Warning: Classes path not set.")
            self.classes = ["default_class"] # Fallback
            return False

        try:
            if os.path.exists(self.classes_path):
                with open(self.classes_path, 'r', encoding='utf-8') as f: # Specify encoding
                    self.classes = [line.strip() for line in f if line.strip()]
                print(f"Loaded {len(self.classes)} classes from {self.classes_path}")
            else:
                # Create a default class if the file doesn't exist
                print(f"classes.txt not found at {self.classes_path}. Creating with default.")
                self.classes = ["default_class"]
                self.save_classes() # Save the default
            return True
        except Exception as e:
            QMessageBox.warning(None, "Load Classes Error", f"Could not load classes from:\n{self.classes_path}\n\nError: {e}")
            self.classes = ["error_loading"] # Fallback
            return False

    def save_classes(self):
        """Saves current class names to classes.txt."""
        if not self.classes_path:
            QMessageBox.critical(None, "Save Classes Error", "Classes file path is not set.")
            return False
        try:
            # Also save a copy in the images directory as requested
            classes_in_images_path = os.path.join(self.images_dir, "classes.txt")

            for path in [self.classes_path, classes_in_images_path]:
                with open(path, 'w', encoding='utf-8') as f: # Specify encoding
                    for class_name in self.classes:
                        f.write(f"{class_name}\n")
                print(f"Classes saved to: {path}")
            return True
        except Exception as e:
            QMessageBox.critical(None, "Save Classes Error", f"Could not save classes file(s).\n\nError: {e}")
            return False

    def add_class(self, class_name):
        """Adds a class name if it doesn't exist."""
        if class_name and class_name not in self.classes:
            self.classes.append(class_name)
            return True
        return False

    def delete_class(self, index):
        """Deletes a class name by index."""
        if 0 <= index < len(self.classes):
            del self.classes[index]
            return True
        return False

    def save_annotations(self, index, annotations):
        """Saves annotations for a specific image index to its label file."""
        label_filename = self.get_label_path(index)
        if not label_filename:
            QMessageBox.critical(None, "Error Saving", "Could not determine label file path.")
            return False

        try:
            with open(label_filename, 'w') as f:
                for ann_data, _ in annotations:
                    class_id, xc, yc, w, h = ann_data
                    f.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
            print(f"Annotations saved to: {label_filename}")
            return True
        except Exception as e:
            QMessageBox.critical(None, "Error Saving", f"Could not save annotation file:\n{label_filename}\n\nError: {e}")
            return False


class AnnotationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Annotation Tool")
        self.setGeometry(100, 100, 1200, 800)

        # Use DatasetManager
        self.dataset_manager = DatasetManager()
        self.current_image_index = -1
        self.current_class_index = -1
        self.current_image_annotations = []
        self.pixmap = None

        self.current_rect_item = None
        self.start_point = None

        self.coord_label = QLabel("")
        self.annotation_count_label = QLabel("Annotations: 0")

        self._create_actions()
        self._create_menu_bar()
        self._create_status_bar()
        self._create_central_widget()
        self._create_dock_widget()

        # Set focus policy to accept key events
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)


    def _create_actions(self):
        self.open_dir_action = QAction("&打开图片文件夹", self)
        self.open_dir_action.triggered.connect(self.open_directory)
        self.save_action = QAction("&保存标注", self)
        self.save_action.triggered.connect(self.save_annotations)
        self.save_action.setEnabled(False)
        self.prev_action = QAction("&上一张 (A)", self) # Add shortcut hint
        self.prev_action.triggered.connect(self.prev_image)
        self.prev_action.setEnabled(False)
        self.next_action = QAction("&下一张 (D)", self) # Add shortcut hint
        self.next_action.triggered.connect(self.next_image)
        self.next_action.setEnabled(False)
        self.exit_action = QAction("&退出", self)
        self.exit_action.triggered.connect(self.close)

        # Re-add Reset Zoom Action
        self.zoom_reset_action = QAction("复原缩放", self)
        self.zoom_reset_action.triggered.connect(self.reset_zoom)
        self.zoom_reset_action.setEnabled(False)


    def _create_menu_bar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&文件")
        file_menu.addAction(self.open_dir_action)
        file_menu.addAction(self.save_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

        navigate_menu = menu_bar.addMenu("&导航")
        navigate_menu.addAction(self.prev_action)
        navigate_menu.addAction(self.next_action)

        # Re-add View Menu for Reset Zoom
        view_menu = menu_bar.addMenu("&视图")
        view_menu.addAction(self.zoom_reset_action)


    def _create_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("请选择图片文件夹")
        self.status_bar.addWidget(self.status_label)
        # Add coordinate and annotation count labels to the status bar
        self.status_bar.addPermanentWidget(self.coord_label)
        self.status_bar.addPermanentWidget(self.annotation_count_label) # Add annotation count label


    def _create_central_widget(self):
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Default to NoDrag, rely on scrollbars for panning if needed
        self.view.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.view.setMouseTracking(True) # Enable mouse tracking for coordinates
        self.view.viewport().setCursor(Qt.CursorShape.ArrowCursor) # Set cursor to arrow always
        self.setCentralWidget(self.view)

        # Connect mouse events for drawing
        self.view.mousePressEvent = self.graphics_view_mouse_press
        self.view.mouseMoveEvent = self.graphics_view_mouse_move
        self.view.mouseReleaseEvent = self.graphics_view_mouse_release
        # Connect wheel event for image navigation
        self.view.wheelEvent = self.graphics_view_wheel_event
        # Connect context menu event for deletion
        self.view.contextMenuEvent = self.graphics_view_context_menu


    def _create_dock_widget(self):
        self.dock_widget = QDockWidget("控制台", self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock_widget)

        # --- Main Actions Section ---
        action_section_widget = QWidget() # Main container for action buttons
        action_section_layout = QVBoxLayout() # Vertical layout for rows
        action_section_widget.setLayout(action_section_layout)

        # --- Row 1 Layout (Only Row Now) ---
        action_row1_layout = QHBoxLayout()
        # Create buttons for actions (Row 1)
        open_dir_button = QPushButton("打开文件夹")
        open_dir_button.clicked.connect(self.open_directory)
        save_button = QPushButton("保存")
        save_button.clicked.connect(self.save_annotations)
        save_button.setEnabled(False) # Mirror action state
        self.save_button_ref = save_button # Keep reference to update state
        prev_button = QPushButton("上一张 (A)") # Add shortcut hint
        prev_button.clicked.connect(self.prev_image)
        prev_button.setEnabled(False) # Mirror action state
        self.prev_button_ref = prev_button # Keep reference
        next_button = QPushButton("下一张 (D)") # Add shortcut hint
        next_button.clicked.connect(self.next_image)
        next_button.setEnabled(False) # Mirror action state
        self.next_button_ref = next_button # Keep reference
        # Add buttons to Row 1 layout
        action_row1_layout.addWidget(open_dir_button)
        action_row1_layout.addWidget(save_button)
        action_row1_layout.addWidget(prev_button)
        action_row1_layout.addWidget(next_button)
        # Add Reset Zoom button to the first row
        zoom_reset_button = QPushButton("复原缩放")
        zoom_reset_button.clicked.connect(self.reset_zoom)
        zoom_reset_button.setEnabled(False) # Mirror action state
        self.zoom_reset_button_ref = zoom_reset_button # Keep reference
        action_row1_layout.addWidget(zoom_reset_button)
        action_row1_layout.addStretch()

        # --- Row 2 Layout Removed ---

        # Add row layout to the main action section layout
        action_section_layout.addLayout(action_row1_layout)
        # action_section_layout.addLayout(action_row2_layout) # Removed

        # --- Category Section ---
        category_widget = QWidget()
        category_layout = QVBoxLayout()
        category_widget.setLayout(category_layout)

        category_label = QLabel("分类:")
        self.class_list_widget = QListWidget()
        self.class_list_widget.itemClicked.connect(self.select_class) # Connect click signal
        self.class_list_widget.itemSelectionChanged.connect(self.update_class_delete_button_state) # Connect selection change

        new_category_layout = QHBoxLayout()
        self.new_class_input = QLineEdit()
        self.new_class_input.setPlaceholderText("新建分类名称")
        add_class_button = QPushButton("添加")
        add_class_button.clicked.connect(self.add_class) # Connect add button
        self.delete_class_button = QPushButton("删除") # Add delete button
        self.delete_class_button.clicked.connect(self.delete_selected_class) # Connect delete button
        self.delete_class_button.setEnabled(False) # Initially disabled

        new_category_layout.addWidget(self.new_class_input)
        new_category_layout.addWidget(add_class_button)
        new_category_layout.addWidget(self.delete_class_button) # Add delete button to layout

        category_layout.addWidget(category_label)
        category_layout.addWidget(self.class_list_widget)
        category_layout.addLayout(new_category_layout)
        # --- End Category Section ---

        # --- Annotation List Section ---
        annotation_label = QLabel("当前的标注:")
        self.annotation_list_widget = QListWidget()
        # Connect selection change to potentially enable delete button
        self.annotation_list_widget.itemSelectionChanged.connect(self.update_delete_button_state)

        self.delete_annotation_button = QPushButton("删除选中的标注")
        self.delete_annotation_button.clicked.connect(self.delete_selected_annotation)
        self.delete_annotation_button.setEnabled(False) # Initially disabled
        # --- End Annotation List Section ---

        # --- Main Dock Layout ---
        dock_layout = QVBoxLayout()
        dock_layout.addWidget(action_section_widget) # Add action buttons section at the top
        dock_layout.addWidget(category_widget) # Add category controls next
        dock_layout.addSpacing(20)
        dock_layout.addWidget(annotation_label)
        dock_layout.addWidget(self.annotation_list_widget)
        dock_layout.addWidget(self.delete_annotation_button) # Add the button
        dock_layout.addStretch()

        # --- Usage Instructions ---
        usage_label = QLabel(
            "<b>使用说明:</b><br>"
            "- 使用鼠标左键拖动画框<br>"
            "- 使用鼠标滚轮切换上一张/下一张图片<br>"
            "- 按住 Ctrl + 鼠标滚轮进行缩放<br>" # Added Ctrl+Wheel instruction
            "- 右键点击标注框可删除该标注<br>" # Added right-click delete instruction
            "- 按 'A' 键切换到上一张图片 (自动保存)<br>"
            "- 按 'D' 键切换到下一张图片 (自动保存)<br>"
            "- 在列表中选择标注后按 '删除选中的标注' 删除<br>"
            "- 在分类列表中选择分类后按 '删除' 删除分类"
        )
        usage_label.setWordWrap(True) # Allow text wrapping
        usage_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        dock_layout.addWidget(usage_label)
        # --- End Usage Instructions ---


        dock_container = QWidget()
        dock_container.setLayout(dock_layout)
        self.dock_widget.setWidget(dock_container)

    def open_directory(self):
        # Ask user to select the main 'dataset' folder
        dir_path = QFileDialog.getExistingDirectory(self, "打开 Dataset 文件夹", ".") # Start in current dir or last used
        if dir_path:
            # Set the root directory in the manager
            if not self.dataset_manager.set_root_directory(dir_path):
                QMessageBox.warning(self, "错误", f"无法设置数据集目录或找不到 'images' 子目录:\n{dir_path}")
                return

            # Load image files and classes using the manager
            images_loaded = self.dataset_manager.load_image_files()
            classes_loaded = self.dataset_manager.load_classes()

            if not images_loaded:
                QMessageBox.warning(self, "错误", f"无法从以下位置加载图片:\n{self.dataset_manager.images_dir}")
                # Reset state if images failed to load
                self.current_image_index = -1
                self.scene.clear()
                self.pixmap = None
                self.status_label.setText("请选择有效的数据集文件夹")
                self.update_navigation_buttons()
                self.save_action.setEnabled(False)
                self.save_button_ref.setEnabled(False)
                self.zoom_reset_action.setEnabled(False) # Disable reset zoom
                self.zoom_reset_button_ref.setEnabled(False)
                return

            if not classes_loaded:
                 QMessageBox.warning(self, "警告", f"无法加载或创建 classes.txt:\n{self.dataset_manager.classes_path}")
                 # Continue even if classes fail, using fallback

            # Update class list widget
            self.update_class_list_widget()

            # Load the first image if available
            if self.dataset_manager.get_image_count() > 0:
                self.current_image_index = 0
                self.load_image() # This will update status bar with count/index
                self.save_action.setEnabled(True)
                self.save_button_ref.setEnabled(True)
                self.zoom_reset_action.setEnabled(True) # Enable reset zoom
                self.zoom_reset_button_ref.setEnabled(True)
            else:
                # No images found
                self.current_image_index = -1
                self.scene.clear()
                self.pixmap = None
                self.status_label.setText(f"在 '{os.path.basename(self.dataset_manager.images_dir)}' 中未找到图片")
                self.save_action.setEnabled(False)
                self.save_button_ref.setEnabled(False)
                self.zoom_reset_action.setEnabled(False) # Disable reset zoom
                self.zoom_reset_button_ref.setEnabled(False)

            self.update_navigation_buttons()


    def load_image(self):
        image_path = self.dataset_manager.get_image_path(self.current_image_index)
        if image_path:
            self.pixmap = QPixmap(image_path)
            if self.pixmap.isNull():
                 QMessageBox.warning(self, "Error", f"Failed to load image: {image_path}")
                 self.pixmap = None
                 self.scene.clear()
                 self.current_image_annotations.clear()
                 self.update_annotation_list_widget()
                 self.status_label.setText(f"Error loading image ({self.current_image_index + 1}/{self.dataset_manager.get_image_count()})")
                 self.zoom_reset_action.setEnabled(False) # Disable reset zoom
                 self.zoom_reset_button_ref.setEnabled(False)
                 return

            self.scene.clear()
            self.scene.addPixmap(self.pixmap)
            self.view.setSceneRect(QRectF(self.pixmap.rect()))
            # self.view.fitInView(QRectF(self.pixmap.rect()), Qt.AspectRatioMode.KeepAspectRatio) # FitInView is now in reset_zoom
            self.reset_zoom() # Call reset_zoom to fit initially
            # Update status bar with filename and count/index
            filename = self.dataset_manager.get_image_filename(self.current_image_index)
            count = self.dataset_manager.get_image_count()
            self.status_label.setText(f"图片: {filename} ({self.current_image_index + 1}/{count})")
            self.load_annotations_for_current_image()
            self.update_annotation_list_widget()
            self.update_annotation_count()
            self.zoom_reset_action.setEnabled(True) # Enable reset zoom
            self.zoom_reset_button_ref.setEnabled(True)
        else:
             # Handle case where index is out of bounds (shouldn't happen with proper nav)
             self.scene.clear()
             self.pixmap = None
             self.status_label.setText("Invalid image index")
             self.zoom_reset_action.setEnabled(False) # Disable reset zoom
             self.zoom_reset_button_ref.setEnabled(False)


    def next_image(self):
        # Save annotations for the current image BEFORE loading the next one
        self.save_annotations()
        if self.current_image_index < self.dataset_manager.get_image_count() - 1:
            self.current_image_index += 1
            self.load_image()
            self.update_navigation_buttons()

    def prev_image(self):
        # Save annotations for the current image BEFORE loading the previous one
        self.save_annotations()
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_image()
            self.update_navigation_buttons()

    def update_navigation_buttons(self):
        count = self.dataset_manager.get_image_count()
        can_go_prev = self.current_image_index > 0
        can_go_next = self.current_image_index < count - 1

        self.prev_action.setEnabled(can_go_prev)
        self.next_action.setEnabled(can_go_next)
        self.prev_button_ref.setEnabled(can_go_prev)
        self.next_button_ref.setEnabled(can_go_next)


    def save_annotations(self):
        # Only save if there are annotations or if the file should be empty
        if self.current_image_index != -1:
            # Use DatasetManager to save
            if not self.dataset_manager.save_annotations(self.current_image_index, self.current_image_annotations):
                # Error message is handled within save_annotations
                pass
        # No warning if no image is loaded, as saving might not be expected.
        # else:
        #      QMessageBox.warning(self, "No Image", "No image is currently loaded.")
        self.update_annotation_count()


    def load_annotations_for_current_image(self):
        # Clear existing graphics items (rectangles)
        items_to_remove = [item for item in self.scene.items() if isinstance(item, QGraphicsRectItem)]
        for item in items_to_remove:
            self.scene.removeItem(item)
        # Clear internal annotation list
        self.current_image_annotations.clear()

        if self.current_image_index < 0 or self.pixmap is None or self.pixmap.isNull():
            return

        label_filename = self.dataset_manager.get_label_path(self.current_image_index)

        if label_filename and os.path.exists(label_filename):
            try:
                with open(label_filename, 'r') as f:
                    img_width = self.pixmap.width()
                    img_height = self.pixmap.height()
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            try:
                                class_id = int(parts[0])
                                xc, yc, w, h = map(float, parts[1:])

                                # Validate class_id against loaded classes
                                if not (0 <= class_id < len(self.dataset_manager.classes)):
                                    print(f"Warning: Invalid class ID {class_id} found in {label_filename}. Skipping.")
                                    continue

                                # Convert YOLO format back to pixel coordinates
                                box_w = w * img_width
                                box_h = h * img_height
                                box_x = (xc * img_width) - (box_w / 2)
                                box_y = (yc * img_height) - (box_h / 2)

                                # Draw the rectangle
                                rect_item = self.draw_existing_rect(box_x, box_y, box_w, box_h, class_id)
                                if rect_item:
                                    annotation_data = (class_id, xc, yc, w, h)
                                    self.current_image_annotations.append((annotation_data, rect_item))
                            except ValueError:
                                print(f"Skipping invalid number format in line: {line.strip()} in {label_filename}")
                        else:
                            print(f"Skipping invalid line format in {label_filename}: {line.strip()}")

            except Exception as e:
                QMessageBox.warning(self, "Load Annotations Error", f"Could not load or parse annotation file:\n{label_filename}\n\nError: {e}")
        # Update list widget and count after loading (done in load_image)


    def draw_existing_rect(self, x, y, w, h, class_id):
        color = self.get_color_for_class(class_id)
        pen = QPen(color, 2)
        try:
            rect_item = self.scene.addRect(x, y, w, h, pen)
            rect_item.setData(0, class_id) # Store class_id if needed later
            return rect_item
        except Exception as e:
            print(f"Error drawing rectangle: {e}")
            return None


    def update_annotation_list_widget(self):
        self.annotation_list_widget.clear()
        for i, (ann_data, _) in enumerate(self.current_image_annotations):
            class_id, _, _, _, _ = ann_data
            # Use classes from dataset_manager
            class_name = self.dataset_manager.classes[class_id] if 0 <= class_id < len(self.dataset_manager.classes) else f"ID:{class_id}"
            item = QListWidgetItem(f"{i}: {class_name}")
            item.setData(Qt.ItemDataRole.UserRole, i)
            self.annotation_list_widget.addItem(item)
        self.update_delete_button_state()


    def update_delete_button_state(self):
        # Enable delete button only if an item is selected in the annotation list
        self.delete_annotation_button.setEnabled(len(self.annotation_list_widget.selectedItems()) > 0)


    def delete_selected_annotation(self):
        """Deletes the annotation selected in the list widget."""
        selected_items = self.annotation_list_widget.selectedItems()
        if not selected_items:
            return

        # Get the original index stored in the item's data
        index_to_delete = selected_items[0].data(Qt.ItemDataRole.UserRole)
        self.delete_annotation_by_index(index_to_delete)


    def delete_annotation_by_index(self, index):
        """Deletes the annotation at the specified index."""
        if 0 <= index < len(self.current_image_annotations):
            try:
                # Get the graphics item to remove
                _, rect_item_to_remove = self.current_image_annotations[index]

                # Remove from the scene
                if rect_item_to_remove in self.scene.items():
                    self.scene.removeItem(rect_item_to_remove)
                else:
                    print(f"Warning: Rect item for index {index} not found in scene during deletion.")

                # Remove from our internal list
                del self.current_image_annotations[index]

                # Update the list widget display (needs re-indexing)
                self.update_annotation_list_widget()
                self.update_annotation_count() # Update count after deleting

                # Optional: Mark changes as unsaved

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete annotation at index {index}: {e}")
                print(f"Error deleting annotation at index {index}: {e}") # Log error
        else:
            QMessageBox.warning(self, "Error", f"Invalid index provided for deletion: {index}")
            print(f"Error: Attempted to delete annotation with invalid index {index}.")


    def add_class(self):
        """Adds a new class using the DatasetManager."""
        new_class_name = self.new_class_input.text().strip()
        if not new_class_name:
             QMessageBox.warning(self, "Input Error", "Class name cannot be empty.")
             return

        if self.dataset_manager.add_class(new_class_name):
            if self.dataset_manager.save_classes(): # Save after adding
                self.update_class_list_widget()
                self.new_class_input.clear()
                # Select the newly added class
                new_index = len(self.dataset_manager.classes) - 1
                self.class_list_widget.setCurrentRow(new_index)
                self.current_class_index = new_index
            else:
                # Revert if save failed? For now, just warn.
                 QMessageBox.critical(self, "Error", "Class added but failed to save classes.txt.")
        else:
             QMessageBox.warning(self, "Input Error", f"Class '{new_class_name}' already exists or is invalid.")


    def update_class_list_widget(self):
        """Updates the QListWidget using classes from DatasetManager."""
        self.class_list_widget.clear()
        for i, class_name in enumerate(self.dataset_manager.classes): # Use manager's classes
            item = QListWidgetItem(class_name)
            color = self.get_color_for_class(i)
            pixmap = QPixmap(16, 16)
            pixmap.fill(color)
            icon = QIcon(pixmap)
            item.setIcon(icon)
            self.class_list_widget.addItem(item)

        # Reselect current class
        if 0 <= self.current_class_index < len(self.dataset_manager.classes):
            self.class_list_widget.setCurrentRow(self.current_class_index)
        else:
            self.current_class_index = -1
        self.update_class_delete_button_state() # Update button state


    def select_class(self, item):
        """Sets the current class index based on list selection."""
        row = self.class_list_widget.row(item)
        if 0 <= row < len(self.dataset_manager.classes): # Check against manager's classes
            self.current_class_index = row
            print(f"Selected class: {self.dataset_manager.classes[self.current_class_index]} (Index: {self.current_class_index})")
        else:
            self.current_class_index = -1

    def update_class_delete_button_state(self):
        """Enables/disables the delete class button based on selection."""
        self.delete_class_button.setEnabled(len(self.class_list_widget.selectedItems()) > 0)

    def delete_selected_class(self):
        """Deletes the selected class using the DatasetManager."""
        selected_items = self.class_list_widget.selectedItems()
        if not selected_items: return

        item = selected_items[0]
        class_name = item.text()
        row = self.class_list_widget.row(item)

        reply = QMessageBox.warning(self, "Confirm Delete",
                                    f"Are you sure you want to delete the class '{class_name}'?\n\n" +
                                    "WARNING: This shifts indices. Existing annotations using this or higher indices " +
                                    "will become INVALID or point to the WRONG class. This cannot be undone easily.",
                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                    QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            original_index = self.current_class_index # Store before deletion

            if self.dataset_manager.delete_class(row):
                if self.dataset_manager.save_classes(): # Save after deleting
                    print(f"Deleted class '{class_name}' from index {row}")

                    # Adjust current_class_index if necessary
                    if original_index == row:
                        self.current_class_index = -1
                        print("Current class selection reset.")
                    elif original_index > row:
                        self.current_class_index -= 1
                        print(f"Current class index shifted to {self.current_class_index}")

                    # Refresh the list widget display
                    self.update_class_list_widget() # This handles reselection and button state
                else:
                     QMessageBox.critical(self, "Error", "Class deleted but failed to save classes.txt.")
                     # Consider adding the class back if save fails?
            else:
                 QMessageBox.warning(self, "Error", "Failed to delete class from manager.")


    def update_annotation_count(self):
        """Updates the annotation count label in the status bar."""
        count = len(self.current_image_annotations)
        self.annotation_count_label.setText(f"Annotations: {count}")


    # --- Mouse Events ---
    def graphics_view_mouse_press(self, event):
        # Only handle left button for drawing start
        if event.button() == Qt.MouseButton.LeftButton and self.current_image_index != -1 and self.pixmap:
            # Check if a class is selected BEFORE starting to draw
            if self.current_class_index == -1:
                QMessageBox.warning(self, "No Class Selected", "Please select a category before drawing.")
                return # Do not start drawing

            # No need to change drag mode, it's always NoDrag

            self.start_point = self.view.mapToScene(event.pos())
            pen = QPen(QColor("cyan"), 2, Qt.PenStyle.DashLine) # Keep temp rect distinct
            # Ensure start point is valid before creating rect item
            if self.start_point:
                 self.current_rect_item = self.scene.addRect(QRectF(self.start_point, self.start_point), pen)
            else:
                 self.current_rect_item = None # Reset if start point is invalid
        # Ignore other mouse buttons for press (no panning start)
        # else:
        #     super(QGraphicsView, self.view).mousePressEvent(event) # Don't pass to super


    def graphics_view_mouse_move(self, event):
        # Update status bar with mouse coordinates regardless of drawing state
        if self.view.underMouse(): # Check if mouse is over the view
            scene_pos = self.view.mapToScene(event.pos())
            self.coord_label.setText(f"X: {scene_pos.x():.1f}, Y: {scene_pos.y():.1f}")
        else:
            self.coord_label.setText("") # Clear coords if mouse leaves view

        # Handle rectangle drawing update only if drawing started correctly (left button down)
        if event.buttons() == Qt.MouseButton.LeftButton and self.start_point and self.current_rect_item:
            current_pos = self.view.mapToScene(event.pos())
            rect = QRectF(self.start_point, current_pos).normalized()
            self.current_rect_item.setRect(rect)
        # Ignore move events for other buttons (no panning)
        # else:
        #     super(QGraphicsView, self.view).mouseMoveEvent(event) # Don't pass to super


    def graphics_view_mouse_release(self, event):
        was_drawing = (event.button() == Qt.MouseButton.LeftButton and self.start_point and self.current_rect_item)

        # --- Annotation saving logic (only if finishing a valid draw) ---
        if was_drawing and self.current_class_index != -1 and self.pixmap:
            end_point = self.view.mapToScene(event.pos())

            # Remove the temporary drawing rectangle
            if self.current_rect_item in self.scene.items():
                 self.scene.removeItem(self.current_rect_item)
            # self.current_rect_item = None # Reset later

            # Calculate final rectangle
            final_rect = QRectF(self.start_point, end_point).normalized()

            # Clamp rectangle coordinates to image boundaries
            img_rect = self.pixmap.rect()
            clamped_rect = final_rect.intersected(QRectF(img_rect)) # Intersect with image bounds

            # Check if rectangle has valid size after clamping
            if clamped_rect.width() < 3 or clamped_rect.height() < 3:
                print("Rectangle too small or outside image after clamping, ignoring.")
                # self.start_point = None # Reset later
                # return # Don't return yet, need to reset state below
            else:
                # Get image dimensions
                img_width = self.pixmap.width()
                img_height = self.pixmap.height()

                # Absolute coordinates from the clamped rectangle
                x_min = clamped_rect.left()
                y_min = clamped_rect.top()
                abs_width = clamped_rect.width()
                abs_height = clamped_rect.height()

                # Convert absolute pixel coordinates to YOLO format
                x_center = x_min + abs_width / 2
                y_center = y_min + abs_height / 2
                x_center_norm = x_center / img_width
                y_center_norm = y_center / img_height
                width_norm = abs_width / img_width
                height_norm = abs_height / img_height

                # Validate class index before adding
                if not (0 <= self.current_class_index < len(self.dataset_manager.classes)):
                     print(f"Error: Invalid current_class_index {self.current_class_index} before adding annotation.")
                     # Reset state without adding annotation
                else:
                    # Store the annotation in YOLO format and the graphics item
                    annotation_data = (self.current_class_index, x_center_norm, y_center_norm, width_norm, height_norm)
                    drawn_rect_item = self.draw_existing_rect(x_min, y_min, abs_width, abs_height, self.current_class_index)

                    if drawn_rect_item: # Only add if drawing was successful
                        self.current_image_annotations.append((annotation_data, drawn_rect_item))
                        # Update the annotation list display
                        self.update_annotation_list_widget()
                        self.update_annotation_count() # Update count after adding
                    else:
                        print("Failed to draw final rectangle, annotation not added.")


        # --- State Reset Logic (always happens after release) ---
        # If finishing a left-click action (drawing or just a click)
        if event.button() == Qt.MouseButton.LeftButton:
            # No need to change drag mode

            # Clean up drawing state variables
            if self.current_rect_item and self.current_rect_item in self.scene.items():
                self.scene.removeItem(self.current_rect_item) # Ensure temp rect is gone
            self.current_rect_item = None
            self.start_point = None

        # Ignore release events for other buttons
        # super(QGraphicsView, self.view).mouseReleaseEvent(event) # Don't pass to super


    def graphics_view_wheel_event(self, event):
        """Handle wheel event for image navigation or zooming."""
        # Check if an image is loaded
        if self.current_image_index == -1 or not self.pixmap:
            super(QGraphicsView, self.view).wheelEvent(event) # Allow default scroll if no image
            return

        angle = event.angleDelta().y()
        modifiers = event.modifiers()

        if modifiers == Qt.KeyboardModifier.ControlModifier:
            # --- Zooming ---
            factor = 1.15 if angle > 0 else 1 / 1.15
            self.zoom(factor)
        elif modifiers == Qt.KeyboardModifier.NoModifier:
            # --- Image Navigation ---
            # Only navigate if more than one image exists
            if self.dataset_manager.get_image_count() > 1:
                if angle > 0: # Scroll Up
                    if self.prev_action.isEnabled():
                        self.prev_image()
                    else:
                        print("Already at the first image.") # Optional feedback
                elif angle < 0: # Scroll Down
                    if self.next_action.isEnabled():
                        self.next_image()
                    else:
                        print("Already at the last image.") # Optional feedback
            else:
                # If only one image, allow default scroll behavior (though likely no scrollbars)
                super(QGraphicsView, self.view).wheelEvent(event)
                return # Don't accept the event if default scroll is used
        else:
            # Allow default scroll for other modifier combinations (e.g., Shift+Wheel)
            super(QGraphicsView, self.view).wheelEvent(event)
            return # Don't accept the event

        event.accept() # Consume the event if we handled zoom or navigation


    def graphics_view_context_menu(self, event):
        """Handle right-click for context menu (delete annotation)."""
        # Check if an image is loaded
        if self.current_image_index == -1 or not self.pixmap:
            return

        # Map click position to scene coordinates
        scene_pos = self.view.mapToScene(event.pos())

        # Find annotation items at this position
        clicked_annotation_index = -1
        # Iterate in reverse order so topmost items are checked first
        for i in range(len(self.current_image_annotations) - 1, -1, -1):
            _, rect_item = self.current_image_annotations[i]
            # Check if the scene position is within the bounding rect of the item
            # Add a small tolerance if needed, but usually contains works well
            if rect_item.contains(scene_pos):
                clicked_annotation_index = i
                break # Found the topmost item

        if clicked_annotation_index != -1:
            # Create context menu
            menu = QMenu(self.view)

            # Get class name for context
            ann_data, _ = self.current_image_annotations[clicked_annotation_index]
            class_id = ann_data[0]
            class_name = self.dataset_manager.classes[class_id] if 0 <= class_id < len(self.dataset_manager.classes) else f"ID:{class_id}"

            # Add delete action
            delete_action = QAction(f"删除标注 '{class_name}' (Index: {clicked_annotation_index})", self)
            # Use lambda to pass the correct index to the slot
            delete_action.triggered.connect(lambda checked=False, index=clicked_annotation_index: self.delete_annotation_by_index(index))
            menu.addAction(delete_action)

            # Show menu at global cursor position
            menu.exec_(event.globalPos())
        # else: No annotation found at this position, do nothing


    def get_color_for_class(self, index):
        """Gets a color for a given class index, cycling through the predefined list."""
        if index < 0: # Handle cases like no selection (-1)
            return QColor("gray") # Default color for invalid index
        return CLASS_COLORS[index % len(CLASS_COLORS)]


    def keyPressEvent(self, event):
        """Handle key presses for navigation."""
        key = event.key()

        # Check if focus is on an input field, if so, ignore navigation keys
        focused_widget = QApplication.focusWidget()
        if isinstance(focused_widget, QLineEdit):
            super().keyPressEvent(event) # Allow default handling for input fields
            return

        if key == Qt.Key.Key_A:
            if self.prev_action.isEnabled():
                self.prev_image()
            else:
                print("Cannot go to previous image.") # Optional feedback
        elif key == Qt.Key.Key_D:
            if self.next_action.isEnabled():
                self.next_image()
            else:
                print("Cannot go to next image.") # Optional feedback
        else:
            super().keyPressEvent(event) # Handle other keys normally


    # --- Re-add Zoom Methods ---
    def zoom(self, factor):
        """Zooms the view by a given factor."""
        if self.pixmap and not self.pixmap.isNull():
            # Anchor zoom around the mouse cursor position
            anchor = self.view.mapToScene(self.view.viewport().mapFromGlobal(QCursor.pos()))
            self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
            self.view.scale(factor, factor)
            # Optionally reset anchor to default if needed after scaling
            # self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)

    def reset_zoom(self):
        """Resets zoom and fits the image in the view."""
        if self.pixmap and not self.pixmap.isNull():
            self.view.setTransform(QTransform()) # Reset transform (zoom/pan)
            self.view.setSceneRect(QRectF(self.pixmap.rect())) # Ensure scene rect is correct
            self.view.fitInView(QRectF(self.pixmap.rect()), Qt.AspectRatioMode.KeepAspectRatio) # Fit image in view


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Launch AnnotationWindow directly for testing this file
    main_win = AnnotationWindow()

    # Optional: Center the window
    try:
        screen_geometry = QGuiApplication.primaryScreen().availableGeometry()
        window_rect = main_win.frameGeometry()
        center_point = screen_geometry.center()
        window_rect.moveCenter(center_point)
        main_win.move(window_rect.topLeft())
    except Exception as e:
        print(f"Could not center AnnotationWindow: {e}")

    main_win.show()

    sys.exit(app.exec_())
