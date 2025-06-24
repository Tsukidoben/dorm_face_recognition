import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 禁用 TensorFlow 日志（如果仍有依赖）
import cv2
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import logging
import sys
import glob
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import gc

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_gpu_environment():
    """检查 GPU 环境"""
    print("=" * 60)
    print("GPU 环境检查")
    print("=" * 60)

    # 检查 CUDA 是否可用
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  显存总量: {torch.cuda.get_device_properties(i).total_memory / 1024 ** 3:.2f} GB")

    print("=" * 60)


class FaceDataset(Dataset):
    """人脸数据集类"""

    def __init__(self, data_dir, min_samples=10, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.faces = []
        self.labels = []
        self.label_map = {}
        self.dorm_members = []

        self._load_dataset(min_samples)

    def _load_dataset(self, min_samples):
        """加载数据集"""
        # 遍历每个成员文件夹
        for member_dir in os.listdir(self.data_dir):
            member_path = os.path.join(self.data_dir, member_dir)

            if not os.path.isdir(member_path):
                continue

            # 记录寝室成员
            self.dorm_members.append(member_dir)
            self.label_map[member_dir] = len(self.label_map)

            # 遍历成员的所有照片
            member_faces = []
            for img_file in os.listdir(member_path):
                img_path = os.path.join(member_path, img_file)
                try:
                    # 使用 PIL 加载图像
                    img = Image.open(img_path).convert('RGB')
                    member_faces.append(img)
                except Exception as e:
                    logger.warning(f"无法加载图像 {img_path}: {str(e)}")

            # 确保每个成员有足够样本
            if len(member_faces) < min_samples:
                logger.warning(f"{member_dir} 只有 {len(member_faces)} 个有效样本，至少需要 {min_samples} 个")
                continue

            # 添加成员数据
            self.faces.extend(member_faces)
            self.labels.extend([self.label_map[member_dir]] * len(member_faces))

        # 添加陌生人样本
        stranger_faces = self._generate_stranger_samples(len(self.faces) // 4)
        self.faces.extend(stranger_faces)
        self.labels.extend([len(self.label_map)] * len(stranger_faces))
        self.label_map['stranger'] = len(self.label_map)

        logger.info(f"数据集加载完成: {len(self.faces)} 个样本, {len(self.dorm_members)} 个成员")

    def _generate_stranger_samples(self, num_samples):
        """生成陌生人样本"""
        stranger_faces = []

        # 使用公开数据集的人脸作为陌生人
        # 这里使用 LFW 数据集作为示例（实际项目中应使用真实数据）
        for _ in range(num_samples):
            # 生成随机噪声图像（实际应用中应使用真实陌生人照片）
            random_face = Image.fromarray(np.uint8(np.random.rand(160, 160, 3) * 255))
            stranger_faces.append(random_face)

        return stranger_faces

    def __len__(self):
        return len(self.faces)

    def __getitem__(self, idx):
        face = self.faces[idx]
        label = self.labels[idx]

        if self.transform:
            face = self.transform(face)

        return face, label


class DormFaceRecognizer:
    """寝室人脸识别系统 (PyTorch 实现)"""

    def __init__(self, threshold=0.7, device=None):
        # 设置设备
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")

        # 初始化人脸检测器
        self.detector = MTCNN(
            keep_all=True,
            post_process=False,
            device=self.device
        )
        logger.info("MTCNN 检测器初始化完成")

        # 初始化人脸特征提取器
        self.embedder = InceptionResnetV1(
            pretrained='vggface2',
            classify=False,
            device=self.device
        ).eval()  # 设置为评估模式
        logger.info("FaceNet 特征提取器初始化完成")

        # 初始化其他组件
        self.classifier = None
        self.label_encoder = None
        self.threshold = threshold
        self.dorm_members = []

        # 数据预处理
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def create_dataset(self, data_dir, min_samples=10, batch_size=32, num_workers=4):
        """创建数据集"""
        dataset = FaceDataset(
            data_dir,
            min_samples=min_samples,
            transform=self.transform
        )

        # 保存成员信息
        self.dorm_members = dataset.dorm_members
        self.label_encoder = LabelEncoder().fit(
            list(dataset.label_map.keys()) + ['stranger']
        )

        # 创建数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        return dataset, dataloader

    def extract_features(self, dataloader):
        """提取人脸特征向量"""
        embeddings = []
        labels = []

        logger.info("开始提取特征...")
        start_time = time.time()

        with torch.no_grad():
            for batch_idx, (faces, batch_labels) in enumerate(dataloader):
                # 移动到设备
                faces = faces.to(self.device)

                # 提取特征
                batch_embeddings = self.embedder(faces)

                # 保存结果
                embeddings.append(batch_embeddings.cpu().numpy())
                labels.append(batch_labels.numpy())

                # 每10个批次打印一次进度
                if (batch_idx + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    logger.info(f"已处理 {batch_idx + 1}/{len(dataloader)} 批次, 耗时: {elapsed:.2f}秒")

        # 合并结果
        embeddings = np.vstack(embeddings)
        labels = np.hstack(labels)

        logger.info(f"特征提取完成: {embeddings.shape[0]} 个样本, 耗时: {time.time() - start_time:.2f}秒")

        return embeddings, labels

    def train_classifier(self, embeddings, labels):
        """训练 SVM 分类器"""
        logger.info("开始训练分类器...")
        start_time = time.time()

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=0.2, random_state=42
        )

        # 创建并训练 SVM 分类器
        self.classifier = SVC(kernel='linear', probability=True, C=1.0)
        self.classifier.fit(X_train, y_train)

        # 评估模型
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        logger.info(f"分类器训练完成, 准确率: {accuracy:.4f}, 耗时: {time.time() - start_time:.2f}秒")

        return accuracy

    def recognize_face(self, image):
        """识别单张图像中的人脸"""
        # 转换为 PIL 图像
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # 检测人脸
        boxes, probs, landmarks = self.detector.detect(image, landmarks=True)

        recognitions = []
        if boxes is not None:
            # 提取每个人脸
            faces = []
            for box in boxes:
                x1, y1, x2, y2 = box
                face = image.crop((x1, y1, x2, y2))
                faces.append(face)

            # 预处理人脸
            face_tensors = torch.stack([self.transform(face) for face in faces]).to(self.device)

            # 提取特征
            with torch.no_grad():
                embeddings = self.embedder(face_tensors).cpu().numpy()

            # 预测
            probabilities = self.classifier.predict_proba(embeddings)
            pred_classes = self.classifier.predict(embeddings)

            for i, (box, prob) in enumerate(zip(boxes, probs)):
                max_prob = np.max(probabilities[i])
                pred_label = self.label_encoder.inverse_transform([pred_classes[i]])[0]

                # 判断是否为陌生人
                if max_prob < self.threshold or pred_label == 'stranger':
                    recognitions.append(("陌生人", max_prob, box))
                else:
                    recognitions.append((pred_label, max_prob, box))

        return recognitions

    def save_model(self, file_path):
        """保存模型"""
        model_data = {
            'classifier': self.classifier,
            'label_encoder': self.label_encoder,
            'threshold': self.threshold,
            'dorm_members': self.dorm_members
        }
        joblib.dump(model_data, file_path)
        logger.info(f"模型已保存至: {file_path}")

    def load_model(self, file_path):
        """加载模型"""
        model_data = joblib.load(file_path)
        self.classifier = model_data['classifier']
        self.label_encoder = model_data['label_encoder']
        self.threshold = model_data['threshold']
        self.dorm_members = model_data['dorm_members']
        logger.info(f"模型已加载，寝室成员: {', '.join(self.dorm_members)}")


def main():
    """主函数"""
    print(f"[{time.strftime('%H:%M:%S')}] 程序启动")

    # 检查 GPU 环境
    check_gpu_environment()

    # 检查并创建必要的目录
    os.makedirs('data/dorm_faces', exist_ok=True)

    # 初始化识别器
    try:
        recognizer = DormFaceRecognizer(threshold=0.6)
        logger.info("人脸识别器初始化成功")
    except Exception as e:
        logger.error(f"初始化失败: {str(e)}")
        print("程序将在10秒后退出...")
        time.sleep(10)
        return

    # 数据集路径
    data_dir = "data/dorm_faces"

    # 检查数据集是否存在
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        logger.warning(f"数据集目录 '{data_dir}' 不存在或为空")
        print("请创建以下结构的目录:")
        print("dorm_faces/")
        print("├── 成员1/")
        print("│   ├── 照片1.jpg")
        print("│   ├── 照片2.jpg")
        print("│   └── ...")
        print("├── 成员2/")
        print("│   └── ...")
        print("└── ...")
        print("\n程序将在10秒后退出...")
        time.sleep(10)
        return

    # 步骤1: 创建数据集
    try:
        dataset, dataloader = recognizer.create_dataset(
            data_dir,
            min_samples=10,
            batch_size=64,
            num_workers=4
        )
    except Exception as e:
        logger.error(f"数据集创建失败: {str(e)}")
        return

    # 步骤2: 提取特征
    try:
        embeddings, labels = recognizer.extract_features(dataloader)
    except Exception as e:
        logger.error(f"特征提取失败: {str(e)}")
        return

    # 步骤3: 训练分类器
    try:
        accuracy = recognizer.train_classifier(embeddings, labels)
    except Exception as e:
        logger.error(f"分类器训练失败: {str(e)}")
        return

    # 保存模型
    model_path = "models/dorm_face_model_pytorch.pkl"
    try:
        recognizer.save_model(model_path)
    except Exception as e:
        logger.error(f"模型保存失败: {str(e)}")

    # 测试识别
    test_image_path = "test_photo.jpg"
    if not os.path.exists(test_image_path):
        logger.warning(f"测试图片 '{test_image_path}' 不存在，跳过识别测试")
    else:
        logger.info(f"正在测试识别: {test_image_path}")
        try:
            test_image = cv2.imread(test_image_path)

            if test_image is None:
                logger.error(f"无法读取图片: {test_image_path}")
            else:
                recognitions = recognizer.recognize_face(test_image)

                if not recognitions:
                    logger.info("未检测到人脸")
                else:
                    # 在图像上绘制结果
                    for name, confidence, box in recognitions:
                        x1, y1, x2, y2 = box
                        label = f"{name} ({confidence:.2f})"
                        color = (0, 255, 0) if name != "陌生人" else (0, 0, 255)

                        # 绘制矩形框
                        cv2.rectangle(test_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                        # 绘制标签
                        cv2.putText(test_image, label, (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # 显示结果
                    cv2.imshow("人脸识别结果", test_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                    # 保存结果图像
                    result_path = "recognition_result_pytorch.jpg"
                    cv2.imwrite(result_path, test_image)
                    logger.info(f"识别结果已保存至: {result_path}")
        except Exception as e:
            logger.error(f"人脸识别失败: {str(e)}")

    logger.info("程序执行完成")


if __name__ == "__main__":
    main()
