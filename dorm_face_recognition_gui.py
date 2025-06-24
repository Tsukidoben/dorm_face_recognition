import sys
import os
import cv2
import numpy as np
import torch
from PyQt5.QtWidgets import QListWidget, QProgressDialog
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QComboBox, QSlider, QMessageBox,
                             QTextEdit, QGroupBox, QScrollArea, QDialog, QDialogButtonBox,
                             QTableWidget, QTableWidgetItem, QHeaderView, QGridLayout)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFont, QColor
import joblib
import logging
import json
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeedbackDialog(QDialog):
    """反馈对话框"""

    def __init__(self, parent=None, last_results=None, dorm_members=None):
        super().__init__(parent)
        self.setWindowTitle("识别错误反馈")
        self.setFixedSize(500, 400)

        self.last_results = last_results or []
        self.dorm_members = dorm_members or []

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 添加当前识别结果
        result_label = QLabel("当前识别结果:")
        layout.addWidget(result_label)

        # 使用表格显示结果
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["ID", "识别结果", "置信度", "位置"])
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)

        # 填充表格数据
        self.results_table.setRowCount(len(self.last_results))
        for i, result in enumerate(self.last_results):
            self.results_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.results_table.setItem(i, 1, QTableWidgetItem(result["label"]))
            self.results_table.setItem(i, 2, QTableWidgetItem(f"{result['confidence']:.2f}"))
            x, y = result.get("position", (0, 0))
            self.results_table.setItem(i, 3, QTableWidgetItem(f"({x}, {y})"))

        # 设置表格样式
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.verticalHeader().setVisible(False)
        layout.addWidget(self.results_table)

        # 添加正确身份选择
        correct_layout = QGridLayout()
        correct_label = QLabel("正确身份:")
        correct_layout.addWidget(correct_label, 0, 0)

        self.correct_combo = QComboBox()
        self.correct_combo.addItem("选择正确身份", None)
        for member in self.dorm_members:
            self.correct_combo.addItem(member, member)
        self.correct_combo.addItem("陌生人", "stranger")
        self.correct_combo.addItem("不在列表中", "unknown")
        correct_layout.addWidget(self.correct_combo, 0, 1)

        # 添加备注
        note_label = QLabel("备注:")
        correct_layout.addWidget(note_label, 1, 0)

        self.note_text = QTextEdit()
        self.note_text.setPlaceholderText("可添加额外说明...")
        self.note_text.setMaximumHeight(60)
        correct_layout.addWidget(self.note_text, 1, 1)

        layout.addLayout(correct_layout)

        # 添加按钮
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_selected_result(self):
        """获取选择的识别结果"""
        selected_row = self.results_table.currentRow()
        if selected_row >= 0 and selected_row < len(self.last_results):
            return self.last_results[selected_row]
        return None

    def get_feedback_data(self):
        """获取反馈数据"""
        selected_result = self.get_selected_result()
        if not selected_result:
            return None

        return {
            "timestamp": datetime.now().isoformat(),
            "original_label": selected_result["label"],
            "correct_label": self.correct_combo.currentData(),
            "confidence": selected_result["confidence"],
            "position": selected_result.get("position", (0, 0)),
            "note": self.note_text.toPlainText().strip()
        }


class FaceRecognitionSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("寝室人脸识别系统")
        self.setGeometry(100, 100, 1200, 800)

        # 初始化变量
        self.model_loaded = False
        self.camera_active = False
        self.video_capture = None
        self.timer = QTimer()
        self.current_image = None
        self.last_results = []  # 存储上次识别结果
        self.dorm_members = []  # 寝室成员列表

        # 创建主界面
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QHBoxLayout(self.main_widget)

        # 左侧控制面板 - 占40%宽度
        self.control_panel = QWidget()
        self.control_layout = QVBoxLayout(self.control_panel)
        self.control_layout.setAlignment(Qt.AlignTop)
        self.control_panel.setMaximumWidth(400)
        self.layout.addWidget(self.control_panel, 40)  # 40%宽度

        # 右侧图像显示区域 - 占60%宽度
        self.image_panel = QWidget()
        self.image_layout = QVBoxLayout(self.image_panel)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet("background-color: #333; border: 1px solid #555;")
        self.image_layout.addWidget(self.image_label)
        self.layout.addWidget(self.image_panel, 60)  # 60%宽度

        # 状态栏
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("系统初始化中...")

        # 初始化UI组件
        self.init_ui()

        # 添加工具栏（必须在UI初始化后）
        self.toolbar = self.addToolBar('工具栏')

        # 添加反馈按钮
        self.add_feedback_button()

        # 初始化模型
        self.init_models()

    def init_ui(self):
        """初始化用户界面组件"""
        # 标题
        title_label = QLabel("寝室人脸识别系统")
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50; padding: 10px;")
        self.control_layout.addWidget(title_label)

        # 模型加载
        model_group = QGroupBox("模型设置")
        model_layout = QVBoxLayout(model_group)

        self.load_model_btn = QPushButton("加载模型")
        self.load_model_btn.setIcon(QIcon.fromTheme("document-open"))
        self.load_model_btn.setStyleSheet("background-color: #3498db;")
        self.load_model_btn.clicked.connect(self.load_model)
        model_layout.addWidget(self.load_model_btn)

        self.model_status = QLabel("模型状态: 未加载")
        model_layout.addWidget(self.model_status)

        self.control_layout.addWidget(model_group)

        # 在模型设置部分添加重新训练按钮
        self.retrain_btn = QPushButton("重新训练模型")
        self.retrain_btn.setIcon(QIcon.fromTheme("view-refresh"))
        self.retrain_btn.setStyleSheet("background-color: #f39c12;")
        self.retrain_btn.clicked.connect(self.retrain_model)
        self.retrain_btn.setEnabled(False)  # 初始不可用
        model_layout.addWidget(self.retrain_btn)

        # 识别设置
        settings_group = QGroupBox("识别设置")
        settings_layout = QVBoxLayout(settings_group)

        # 置信度阈值
        threshold_layout = QHBoxLayout()
        threshold_label = QLabel("置信度阈值:")
        threshold_layout.addWidget(threshold_label)

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(70)
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        threshold_layout.addWidget(self.threshold_slider)

        self.threshold_value = QLabel("0.70")
        threshold_layout.addWidget(self.threshold_value)
        settings_layout.addLayout(threshold_layout)

        # 显示选项
        display_layout = QHBoxLayout()
        display_label = QLabel("显示模式:")
        display_layout.addWidget(display_label)

        self.display_combo = QComboBox()
        self.display_combo.addItems(["原始图像", "检测框", "识别结果"])
        self.display_combo.setCurrentIndex(2)
        display_layout.addWidget(self.display_combo)
        settings_layout.addLayout(display_layout)

        self.control_layout.addWidget(settings_group)

        # 识别功能
        recognition_group = QGroupBox("识别功能")
        recognition_layout = QVBoxLayout(recognition_group)

        # 图片识别
        self.image_recognition_btn = QPushButton("图片识别")
        self.image_recognition_btn.setIcon(QIcon.fromTheme("image-x-generic"))
        self.image_recognition_btn.setStyleSheet("background-color: #9b59b6;")
        self.image_recognition_btn.clicked.connect(self.open_image)
        self.image_recognition_btn.setEnabled(False)
        recognition_layout.addWidget(self.image_recognition_btn)

        # 摄像头识别
        self.camera_recognition_btn = QPushButton("启动摄像头识别")
        self.camera_recognition_btn.setIcon(QIcon.fromTheme("camera-web"))
        self.camera_recognition_btn.setStyleSheet("background-color: #e74c3c;")
        self.camera_recognition_btn.clicked.connect(self.toggle_camera)
        self.camera_recognition_btn.setEnabled(False)
        recognition_layout.addWidget(self.camera_recognition_btn)

        self.control_layout.addWidget(recognition_group)

        # 结果展示区域 - 使用QTextEdit替代QLabel
        results_group = QGroupBox("识别结果")
        results_layout = QVBoxLayout(results_group)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont("Microsoft YaHei", 12))  # 使用支持中文的字体
        self.results_text.setStyleSheet("background-color: #f8f9fa; border: 1px solid #ddd; padding: 10px;")
        self.results_text.setPlaceholderText("识别结果将显示在这里")

        # 添加滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.results_text)
        results_layout.addWidget(scroll_area)

        self.control_layout.addWidget(results_group, 1)  # 占据剩余空间

        # 系统信息
        info_group = QGroupBox("系统信息")
        info_layout = QVBoxLayout(info_group)

        self.device_label = QLabel(f"计算设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        info_layout.addWidget(self.device_label)

        self.model_info = QLabel("加载模型以显示信息")
        info_layout.addWidget(self.model_info)

        self.control_layout.addWidget(info_group)

        # 退出按钮
        exit_btn = QPushButton("退出系统")
        exit_btn.setIcon(QIcon.fromTheme("application-exit"))
        exit_btn.clicked.connect(self.close)
        exit_btn.setStyleSheet("background-color: #ff6b6b; color: white;")
        self.control_layout.addWidget(exit_btn)

    def add_feedback_button(self):
        """添加反馈按钮到界面"""
        # 创建反馈按钮
        self.feedback_button = QPushButton("提供反馈", self)
        self.feedback_button.setFixedSize(120, 40)  # 设置固定大小
        self.feedback_button.setStyleSheet(
            "QPushButton {"
            "   background-color: #4CAF50;"
            "   color: white;"
            "   border-radius: 5px;"
            "   font-weight: bold;"
            "}"
            "QPushButton:hover {"
            "   background-color: #45a049;"
            "}"
        )

        # 连接按钮点击事件
        self.feedback_button.clicked.connect(self.open_feedback_dialog)

        # 添加到工具栏
        self.toolbar.addWidget(self.feedback_button)

    def open_feedback_dialog(self):
        """打开反馈对话框"""
        if not self.last_results:
            QMessageBox.warning(self, "无法反馈", "没有可反馈的识别结果")
            return

        dialog = FeedbackDialog(
            self,
            last_results=self.last_results,
            dorm_members=self.dorm_members
        )

        if dialog.exec_() == QDialog.Accepted:
            feedback_data = dialog.get_feedback_data()
            if feedback_data:
                self.save_feedback(feedback_data)
                QMessageBox.information(self, "反馈提交", "感谢您的反馈！数据已保存用于改进模型")


    def init_models(self):
        """初始化模型组件"""
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device_label.setText(f"计算设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")

        # 初始化人脸检测器
        try:
            self.detector = MTCNN(
                keep_all=True,
                post_process=False,
                device=self.device
            )
            self.status_bar.showMessage("MTCNN 检测器初始化完成")
            logger.info("MTCNN 检测器初始化完成")
        except Exception as e:
            self.status_bar.showMessage(f"MTCNN 初始化失败: {str(e)}")
            logger.error(f"MTCNN 初始化失败: {str(e)}")
            return

        # 初始化人脸特征提取器
        try:
            self.embedder = InceptionResnetV1(
                pretrained='vggface2',
                classify=False,
                device=self.device
            ).eval()
            self.status_bar.showMessage("FaceNet 特征提取器初始化完成")
            logger.info("FaceNet 特征提取器初始化完成")
        except Exception as e:
            self.status_bar.showMessage(f"FaceNet 初始化失败: {str(e)}")
            logger.error(f"FaceNet 初始化失败: {str(e)}")

    def load_model(self):
        """加载预训练的SVM分类器"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "模型文件 (*.pkl);;所有文件 (*)", options=options
        )

        if file_path:
            try:
                # 加载模型
                model_data = joblib.load(file_path)
                self.classifier = model_data['classifier']
                self.label_encoder = model_data['label_encoder']
                self.dorm_members = model_data['dorm_members']

                # 启用重新训练按钮
                self.retrain_btn.setEnabled(True)

                # 更新UI状态
                self.model_loaded = True
                self.model_status.setText("模型状态: 已加载")
                self.model_info.setText(f"寝室成员: {', '.join(self.dorm_members)}")
                self.image_recognition_btn.setEnabled(True)
                self.camera_recognition_btn.setEnabled(True)

                # 状态栏消息
                self.status_bar.showMessage(f"模型加载成功: {os.path.basename(file_path)}")

                # 显示成功消息
                QMessageBox.information(
                    self, "模型加载",
                    f"模型加载成功！\n识别成员: {len(self.dorm_members)}人\n置信度阈值: {self.threshold_slider.value() / 100:.2f}"
                )

            except Exception as e:
                QMessageBox.critical(self, "加载错误", f"模型加载失败: {str(e)}")
                self.status_bar.showMessage(f"模型加载失败: {str(e)}")

    def update_threshold(self, value):
        """更新置信度阈值"""
        threshold = value / 100
        self.threshold_value.setText(f"{threshold:.2f}")
        self.status_bar.showMessage(f"置信度阈值更新为: {threshold:.2f}")

    def open_image(self):
        """打开图片文件进行识别"""
        if not self.model_loaded:
            QMessageBox.warning(self, "警告", "请先加载模型！")
            return

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择识别图片", "",
            "图片文件 (*.jpg *.jpeg *.png);;所有文件 (*)", options=options
        )

        if file_path:
            # 读取图片
            image = cv2.imread(file_path)
            if image is None:
                QMessageBox.critical(self, "错误", "无法读取图片文件！")
                return

            # 保存当前图片
            self.current_image = image.copy()

            # 进行识别
            self.recognize_faces(image)

    def toggle_camera(self):
        """切换摄像头状态"""
        if not self.model_loaded:
            QMessageBox.warning(self, "警告", "请先加载模型！")
            return

        if not self.camera_active:
            # 尝试打开摄像头
            self.video_capture = cv2.VideoCapture(0)
            if not self.video_capture.isOpened():
                QMessageBox.critical(self, "错误", "无法打开摄像头！")
                return

            # 启动摄像头
            self.camera_active = True
            self.camera_recognition_btn.setText("停止摄像头识别")
            self.camera_recognition_btn.setIcon(QIcon.fromTheme("media-playback-stop"))
            self.timer.timeout.connect(self.process_camera_frame)
            self.timer.start(30)  # 约33 FPS
            self.status_bar.showMessage("摄像头已启动")
        else:
            # 停止摄像头
            self.camera_active = False
            self.camera_recognition_btn.setText("启动摄像头识别")
            self.camera_recognition_btn.setIcon(QIcon.fromTheme("camera-web"))
            self.timer.stop()
            if self.video_capture:
                self.video_capture.release()
            self.status_bar.showMessage("摄像头已停止")

    def process_camera_frame(self):
        """处理摄像头帧"""
        ret, frame = self.video_capture.read()
        if ret:
            # 保存当前帧
            self.current_image = frame.copy()

            # 进行识别
            self.recognize_faces(frame)

    def retrain_model(self):
        """使用反馈数据重新训练模型"""
        # 获取所有反馈数据
        feedback_dir = os.path.join(os.getcwd(), "data", "feedback_data")
        feedback_files = [f for f in os.listdir(feedback_dir)
                          if f.endswith('.json') and os.path.isfile(os.path.join(feedback_dir, f))]

        if not feedback_files:
            QMessageBox.information(self, "无反馈数据", "没有找到反馈数据，无法重新训练")
            return

        # 确认对话框
        reply = QMessageBox.question(
            self, '确认重新训练',
            f"将使用 {len(feedback_files)} 条反馈数据重新训练模型。此操作可能需要几分钟时间，确定继续吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        try:
            # 创建进度对话框
            progress = QProgressDialog("正在重新训练模型...", "取消", 0, len(feedback_files), self)
            progress.setWindowTitle("模型重新训练")
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)

            # 收集所有反馈数据
            feedback_data = []
            for i, filename in enumerate(feedback_files):
                filepath = os.path.join(feedback_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    feedback_data.append(data)

                progress.setValue(i + 1)
                QApplication.processEvents()  # 保持UI响应

                if progress.wasCanceled():
                    return

            progress.setValue(len(feedback_files))

            # 重新训练模型
            self.status_bar.showMessage("正在重新训练模型...")
            self.face_recognition.retrain_with_feedback(feedback_data)

            # 更新UI状态
            self.model_status.setText("模型状态: 已重新训练")
            QMessageBox.information(self, "训练完成", "模型已成功使用反馈数据重新训练！")

        except Exception as e:
            logger.error(f"重新训练失败: {str(e)}")
            QMessageBox.critical(self, "训练错误", f"重新训练模型时出错: {str(e)}")

    def recognize_faces(self, image):
        """识别人脸并在图像上标注结果"""
        # 清空上次结果
        self.last_results = []

        # 转换为 PIL 图像
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # 检测人脸
        boxes, probs, _ = self.detector.detect(pil_image, landmarks=True)

        # 获取显示选项
        display_mode = self.display_combo.currentIndex()

        # 准备显示图像
        display_image = image.copy()

        # 如果没有检测到人脸
        if boxes is None:
            if display_mode == 2:  # 识别结果模式
                cv2.putText(display_image, "未检测到人脸", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.results_text.setText("未检测到人脸")
        else:
            # 提取每个人脸
            faces = []
            for box in boxes:
                x1, y1, x2, y2 = box
                face = pil_image.crop((x1, y1, x2, y2))
                faces.append(face)

            # 提取特征
            embeddings = []
            if faces and self.model_loaded:
                # 批量处理所有人脸
                face_tensors = [self.preprocess_face(face) for face in faces]
                if face_tensors:
                    face_tensors = torch.stack(face_tensors).to(self.device)

                    with torch.no_grad():
                        embeddings = self.embedder(face_tensors).cpu().numpy()

            # 处理每个人脸
            for i, (box, prob) in enumerate(zip(boxes, probs)):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1

                # 在图像上绘制结果
                if display_mode == 0:  # 原始图像
                    # 不绘制任何内容
                    pass
                elif display_mode == 1:  # 检测框
                    # 绘制人脸框
                    cv2.rectangle(display_image, (int(x1), int(y1)),
                                  (int(x2), int(y2)), (0, 255, 0), 2)
                elif display_mode == 2:  # 识别结果
                    # 绘制人脸框
                    color = (0, 255, 0)  # 绿色

                    # 如果有嵌入向量，则进行识别
                    if i < len(embeddings):
                        # 预测
                        probabilities = self.classifier.predict_proba([embeddings[i]])[0]
                        max_prob = np.max(probabilities)
                        pred_class = self.classifier.predict([embeddings[i]])[0]
                        pred_label = self.label_encoder.inverse_transform([pred_class])[0]

                        # 获取置信度阈值
                        threshold = self.threshold_slider.value() / 100

                        # 判断是否为陌生人
                        if max_prob < threshold or pred_label == 'stranger':
                            label = "陌生人"
                            color = (0, 0, 255)  # 红色
                        else:
                            label = pred_label
                            color = (0, 255, 0)  # 绿色

                        # 保存结果用于文本显示
                        result = {
                            "position": (int(x1), int(y1)),
                            "label": label,
                            "confidence": max_prob
                        }
                        self.last_results.append(result)

                        # 绘制标签
                        cv2.rectangle(display_image, (int(x1), int(y1)),
                                      (int(x2), int(y2)), color, 2)
                        cv2.putText(display_image, f"{label} ({max_prob:.2f})",
                                    (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    else:
                        # 无法识别的处理
                        cv2.rectangle(display_image, (int(x1), int(y1)),
                                      (int(x2), int(y2)), (0, 165, 255), 2)
                        cv2.putText(display_image, "处理中...",
                                    (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            # 更新结果文本
            self.update_results_text()

        # 在图像上显示FPS（摄像头模式下）
        if self.camera_active:
            fps = self.timer.interval()
            if fps > 0:
                cv2.putText(display_image, f"FPS: {1000 / fps:.1f}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

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
            x, y = result["position"]
            label = result["label"]
            confidence = result["confidence"]

            # 处理中文显示问题
            if label in self.dorm_members:
                result_text += (
                    f"<p><b>人脸 #{i}:</b> "
                    f"<span style='color:green;'>寝室成员 - {label}</span><br>"
                    f"位置: ({x}, {y}), 置信度: {confidence:.2f}</p>"
                )
            else:
                result_text += (
                    f"<p><b>人脸 #{i}:</b> "
                    f"<span style='color:red;'>陌生人</span><br>"
                    f"位置: ({x}, {y}), 置信度: {confidence:.2f}</p>"
                )

        self.results_text.setHtml(result_text)

    def preprocess_face(self, face_img):
        """预处理人脸图像"""
        # 调整大小
        face_img = face_img.resize((160, 160))

        # 转换为张量并归一化
        face_img = np.array(face_img).astype(np.float32) / 255.0
        face_img = (face_img - 0.5) / 0.5  # 归一化到[-1, 1]
        face_img = torch.tensor(face_img).permute(2, 0, 1)  # HWC to CHW

        return face_img

    def display_image(self, image):
        """在QLabel中显示图像"""
        # 将OpenCV图像转换为Qt格式
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

        # 缩放图像以适应标签
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.width(),
            self.image_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))

    def closeEvent(self, event):
        """关闭事件处理"""
        if self.camera_active:
            self.timer.stop()
            if self.video_capture:
                self.video_capture.release()

        # 确认退出
        reply = QMessageBox.question(
            self, '确认退出',
            "确定要退出系统吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)


    # 设置全局异常处理
    def handle_exception(exc_type, exc_value, exc_traceback):
        """全局异常处理"""
        import traceback
        error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        print(f"未捕获的异常:\n{error_msg}")

        # 记录到文件
        with open("error.log", "a") as f:
            f.write(f"\n\n{datetime.now()}:\n{error_msg}")

        # 显示给用户
        QMessageBox.critical(None, "系统错误", f"发生未处理的异常:\n{str(exc_value)}")
        sys.exit(1)


    sys.excepthook = handle_exception

    window = FaceRecognitionSystem()
    window.show()
    sys.exit(app.exec_())
