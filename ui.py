from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QComboBox, QSlider, QMessageBox,
                             QTextEdit, QGroupBox, QScrollArea, QDialog, QListWidget)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFont
from face_recognition import FaceRecognition


class FaceRecognitionSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        # ... 原有初始化代码 ...

        # 初始化人脸识别器
        self.face_recognition = FaceRecognition()

        # 添加反馈按钮
        self.add_feedback_button()

    def add_feedback_button(self):
        """添加反馈按钮到界面"""
        self.feedback_btn = QPushButton("反馈识别错误")
        self.feedback_btn.setIcon(QIcon.fromTheme("dialog-warning"))
        self.feedback_btn.setStyleSheet("background-color: #f39c12;")
        self.feedback_btn.clicked.connect(self.handle_feedback)

        # 找到识别功能组并添加按钮
        for i in range(self.control_layout.count()):
            widget = self.control_layout.itemAt(i).widget()
            if isinstance(widget, QGroupBox) and widget.title() == "识别功能":
                layout = widget.layout()
                layout.addWidget(self.feedback_btn)
                break

    def handle_feedback(self):
        """处理用户反馈"""
        if not hasattr(self, 'last_results') or not self.last_results:
            QMessageBox.warning(self, "警告", "没有可反馈的识别结果")
            return

        # 创建反馈对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("识别错误反馈")
        dialog.setFixedSize(400, 300)
        layout = QVBoxLayout(dialog)

        # 添加当前识别结果
        result_label = QLabel("当前识别结果:")
        layout.addWidget(result_label)

        self.feedback_list = QListWidget()
        for i, result in enumerate(self.last_results, 1):
            label = result["label"]
            confidence = result["confidence"]
            self.feedback_list.addItem(f"人脸 #{i}: {label} (置信度: {confidence:.2f})")
        layout.addWidget(self.feedback_list)

        # 添加正确身份选择
        correct_label = QLabel("正确身份:")
        layout.addWidget(correct_label)

        self.correct_combo = QComboBox()
        self.correct_combo.addItems(["选择正确身份"] + self.face_recognition.dorm_members + ["陌生人", "不在列表中"])
        layout.addWidget(self.correct_combo)

        # 添加按钮
        btn_layout = QHBoxLayout()
        submit_btn = QPushButton("提交反馈")
        submit_btn.clicked.connect(lambda: self.submit_feedback(dialog))
        btn_layout.addWidget(submit_btn)

        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(cancel_btn)

        layout.addLayout(btn_layout)
        dialog.exec_()

    def submit_feedback(self, dialog):
        """提交反馈并更新模型"""
        selected_index = self.feedback_list.currentRow()
        if selected_index < 0:
            QMessageBox.warning(self, "警告", "请选择一个识别结果")
            return

        result = self.last_results[selected_index]
        correct_identity = self.correct_combo.currentText()

        if correct_identity == "选择正确身份":
            QMessageBox.warning(self, "警告", "请选择正确身份")
            return

        # 保存反馈数据
        self.face_recognition.save_feedback(
            self.current_image.copy(),
            result["box"],
            result["label"],
            correct_identity
        )

        QMessageBox.information(self, "反馈提交", "感谢您的反馈！数据已保存用于改进模型")
        dialog.accept()

    def recognize_faces(self, image):
        """识别人脸并在图像上标注结果"""
        # 使用人脸识别器进行识别
        self.last_results, display_image = self.face_recognition.recognize(
            image,
            threshold=self.threshold_slider.value() / 100
        )

        # 更新结果文本
        self.update_results_text()

        # 显示图像
        self.display_image(display_image)

    def update_results_text(self):
        """更新结果文本区域"""
        if not self.last_results:
            self.results_text.setText("未识别到任何人脸")
            return

        # 构建结果文本
        result_text = "<h3>识别结果：</h3>"
        for i, result in enumerate(self.last_results, 1):
            x1, y1, x2, y2 = result["box"]
            label = result["label"]
            confidence = result["confidence"]

            # 处理中文显示问题
            if label in self.face_recognition.dorm_members:
                result_text += (
                    f"<p><b>人脸 #{i}:</b> "
                    f"<span style='color:green;'>寝室成员 - {label}</span><br>"
                    f"位置: ({x1}, {y1}), 置信度: {confidence:.2f}</p>"
                )
            else:
                result_text += (
                    f"<p><b>人脸 #{i}:</b> "
                    f"<span style='color:red;'>陌生人</span><br>"
                    f"位置: ({x1}, {y1}), 置信度: {confidence:.2f}</p>"
                )

        self.results_text.setHtml(result_text)

    # ... 其余原有方法 ...
