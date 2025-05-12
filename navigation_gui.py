### -*- coding: utf-8 -*-
# @Time    : 2025/4/27
# @Author  : YeZi xiao
# @File    : navigation_gui.py
# @Software: vscode
# @Description: This script creates a navigation GUI for YOLO tool with three main functionalities:
#              1. Annotation Tool
#              2. Model Training
#              3. Object Detection
#              Each functionality is represented by a button, and clicking a button will display the corresponding page.
#              The GUI is built using PyQt5 and is designed to be user-friendly and visually appealing.
#              The script also handles the import of necessary modules and manages the layout of the GUI components.
#              The script is designed to be run as a standalone application or as part of a larger package.


import sys
import os # Import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSizePolicy, QStackedWidget, QGridLayout, QFrame # 导入 QFrame
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor # QColor 可能不需要，但放在这里以备后用

try:
    from label_gui import AnnotationWindow
except ImportError:
    # Handle case where the script is run directly or module not found
    print("Warning: Could not import AnnotationWindow from label_gui.py")
    AnnotationWindow = None # Placeholder

try:
    from train_gui import YoloApp
except ImportError:
    # Handle case where the script is run directly or module not found
    print("Warning: Could not import YoloApp from main.py") # Corrected print message
    YoloApp = None # Placeholder

try:
    from detection_gui import DetectionWindow # Assuming this is the correct import
except ImportError:
    print("Warning: Could not import DetectionWindow from detection_gui.py")
    DetectionWindow = None # Placeholder
import multiprocessing # <--- 添加导入


# 重命名窗口类，使其更通用
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Determine the base path correctly for script or frozen executable
        if getattr(sys, 'frozen', False):
            # If the application is run as a bundle/frozen executable (e.g., PyInstaller)
            self.base_path = os.path.dirname(sys.executable)
        else:
            # If run as a normal script (.py file)
            self.base_path = os.path.dirname(os.path.abspath(__file__))

        print(f"Navigation GUI Base Path: {self.base_path}") # Debug print

        # Pass the base_path to the widgets that need it
        self.annotation_widget = AnnotationWindow() if AnnotationWindow else QLabel("Annotation Module Unavailable") # Assuming AnnotationWindow doesn't need base_path
        self.training_widget = YoloApp(base_path=self.base_path) if YoloApp else QLabel("Training Module Unavailable")
        self.detection_widget = DetectionWindow(base_path=self.base_path) if DetectionWindow else QLabel("Detection Module Unavailable")

        # 创建 QStackedWidget
        self.stacked_widget = QStackedWidget()

        self.initUI()

    def initUI(self):
        # 更改窗口标题和大小
        self.setWindowTitle('YOLO Tool - Integrated')
        self.setGeometry(100, 100, 1000, 700) # 调整初始大小 x, y, width, height

        # --- 创建导航按钮区域 --- 
        button_frame = QFrame() # 创建 QFrame
        button_frame.setFrameShape(QFrame.StyledPanel) # 设置边框样式
        button_frame.setFrameShadow(QFrame.Raised)    # 设置阴影样式

        button_layout = QVBoxLayout() # 按钮布局仍然使用 QVBoxLayout
        button_layout.setContentsMargins(10, 10, 10, 10) # 在 Frame 内添加边距
        button_layout.setSpacing(15) # 按钮间距

        title_label = QLabel('功能选择')
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        button_layout.addWidget(title_label)
        button_layout.addStretch(1) # 向上推按钮

        # --- 按钮样式定义 ---
        # Style for unselected buttons
        self.common_style = """
            QPushButton {
                border: 1px solid #AAAAAA; /* Add a border for definition */
                padding: 8px 0px;
                text-align: center;
                font-size: 11pt;
                margin: 2px 0px;
                border-radius: 5px;
                color: #333333; /* Default text color */
                background-color: #E0E0E0; /* Default background (light gray) */
            }
            QPushButton:hover {
                background-color: #D0D0D0;
            }
            QPushButton:pressed {
                background-color: #C0C0C0;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
                border: 1px solid #BBBBBB;
            }
        """

        # Define selected styles by adding specific background color and text color
        self.annotate_selected_style = self.common_style + """
            QPushButton { background-color: #3498DB; color: white; border: 1px solid #2980B9;} /* Blue */
            QPushButton:hover { background-color: #2980B9; }
            QPushButton:pressed { background-color: #2471A3; }
        """

        self.train_selected_style = self.common_style + """
            QPushButton { background-color: #2ECC71; color: white; border: 1px solid #27AE60;} /* Green */
            QPushButton:hover { background-color: #27AE60; }
            QPushButton:pressed { background-color: #1F8B4C; }
        """

        self.detect_selected_style = self.common_style + """
            QPushButton { background-color: #F39C12; color: white; border: 1px solid #E67E22;} /* Orange */
            QPushButton:hover { background-color: #E67E22; }
            QPushButton:pressed { background-color: #D35400; }
        """


        # Annotation Button
        self.annotate_button = QPushButton('标注工具')
        self.annotate_button.setFont(QFont('Arial', 11))
        self.annotate_button.setMinimumHeight(35)
        # self.annotate_button.setStyleSheet(annotate_style) # Apply style dynamically
        if AnnotationWindow:
            self.annotate_button.clicked.connect(self.show_annotation_page)
        else:
            self.annotate_button.setEnabled(False)
        button_layout.addWidget(self.annotate_button)

        # Training Button
        self.train_button = QPushButton('模型训练')
        self.train_button.setFont(QFont('Arial', 11))
        self.train_button.setMinimumHeight(35)
        # self.train_button.setStyleSheet(train_style) # Apply style dynamically
        if YoloApp:
            self.train_button.clicked.connect(self.show_training_page)
        else:
            self.train_button.setEnabled(False)
        button_layout.addWidget(self.train_button)

        # Detection Button
        self.detect_button = QPushButton('目标检测')
        self.detect_button.setFont(QFont('Arial', 11))
        self.detect_button.setMinimumHeight(35)
        # self.detect_button.setStyleSheet(detect_style) # Apply style dynamically
        if DetectionWindow:
            self.detect_button.clicked.connect(self.show_detection_page)
        else:
            self.detect_button.setEnabled(False)
        button_layout.addWidget(self.detect_button)

        button_layout.addStretch(5) # 向下推按钮

        button_frame.setLayout(button_layout) # 将布局设置给 QFrame

        # --- 将页面添加到 QStackedWidget ---
        # 确保添加的是 QWidget 或其子类
        if isinstance(self.annotation_widget, QWidget):
            self.stacked_widget.addWidget(self.annotation_widget)
        if isinstance(self.training_widget, QWidget):
            self.stacked_widget.addWidget(self.training_widget)
        if isinstance(self.detection_widget, QWidget):
            self.stacked_widget.addWidget(self.detection_widget)

        # --- 创建主布局 ---
        main_layout = QHBoxLayout()
        # 将 button_frame 添加到主布局，而不是 button_layout
        main_layout.addWidget(button_frame, 1) # 按钮 Frame 占1份
        main_layout.addWidget(self.stacked_widget, 5) # StackedWidget 占5份

        self.setLayout(main_layout)

        # 初始显示第一个有效的页面并设置按钮样式
        initial_button = None
        if AnnotationWindow:
            self.show_annotation_page() # This will now also update styles
            initial_button = self.annotate_button # Keep track for initial call if needed
        elif YoloApp:
            self.show_training_page()
            initial_button = self.train_button
        elif DetectionWindow:
            self.show_detection_page()
            initial_button = self.detect_button

        # Apply initial styles if no page was shown initially (or just to be safe)
        if initial_button is None:
             self._update_button_styles(None) # Set all to common style if no page is active
        # else: # The show_... methods already call _update_button_styles
        #     self._update_button_styles(initial_button)


    def _update_button_styles(self, active_button):
        """Sets the style for all navigation buttons based on the active one."""
        buttons_styles = {
            self.annotate_button: self.annotate_selected_style,
            self.train_button: self.train_selected_style,
            self.detect_button: self.detect_selected_style,
        }

        for button, selected_style in buttons_styles.items():
            if button.isEnabled(): # Only change style if button is enabled
                if button == active_button:
                    button.setStyleSheet(selected_style)
                else:
                    button.setStyleSheet(self.common_style) # Apply unselected style
            # else: # Keep disabled style if button is disabled
            #     button.setStyleSheet(self.common_style) # Ensure disabled style from common_style applies


    # --- 修改槽函数以切换页面 ---
    def show_annotation_page(self):
        if isinstance(self.annotation_widget, QWidget):
            self.stacked_widget.setCurrentWidget(self.annotation_widget)
            self._update_button_styles(self.annotate_button) # Update styles

    def show_training_page(self):
        if isinstance(self.training_widget, QWidget):
            self.stacked_widget.setCurrentWidget(self.training_widget)
            self._update_button_styles(self.train_button) # Update styles

    def show_detection_page(self):
        if isinstance(self.detection_widget, QWidget):
            self.stacked_widget.setCurrentWidget(self.detection_widget)
            self._update_button_styles(self.detect_button) # Update styles


if __name__ == '__main__':

    multiprocessing.freeze_support() # <--- 在开头添加此行

    app = QApplication(sys.argv)
    # 实例化新的主窗口类
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())