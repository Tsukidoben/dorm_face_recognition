import sys
from dorm_face_recognition_gui import FaceRecognitionSystem
from PyQt5.QtWidgets import QApplication
if __name__ == "__main__":
    # 设置中文编码支持
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("dorm.face.recognition")

    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # 使用Fusion风格

    # 设置应用样式
    app.setStyleSheet("""
        QMainWindow {
            background-color: #ecf0f1;
        }
        QGroupBox {
            border: 1px solid #bdc3c7;
            border-radius: 8px;
            margin-top: 20px;
            padding: 10px;
            font-weight: bold;
            background-color: #ffffff;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 5px;
        }
        QPushButton {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px 15px;
            font-size: 14px;
            margin: 5px;
            border-radius: 5px;
        }
        QPushButton:hover {
            background-color: #2980b9;
        }
        QPushButton:pressed {
            background-color: #1c6ea4;
        }
        QPushButton:disabled {
            background-color: #bdc3c7;
        }
        QLabel {
            font-size: 14px;
            padding: 3px;
        }
        QComboBox, QSlider {
            padding: 4px;
            background-color: #ffffff;
        }
        QTextEdit {
            font-family: "Microsoft YaHei";
            font-size: 12px;
        }
    """)

    window = FaceRecognitionSystem()
    window.show()
    sys.exit(app.exec_())