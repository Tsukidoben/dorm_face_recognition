import cv2
import numpy as np
import torch
import insightface
from insightface.app import FaceAnalysis
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import joblib
import os
import pickle
from datetime import datetime
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class FaceRecognition:
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_loaded = False
        self.init_models()

    def init_models(self):
        """初始化人脸识别模型"""
        try:
            # 初始化ArcFace模型
            self.arcface_model = FaceAnalysis()
            self.arcface_model.prepare(ctx_id=0, det_size=(640, 640))
            self.arcface_model.load_model("./models/buffalo_l")

            # 初始化FaceNet模型作为备选
            self.facenet_model = InceptionResnetV1(
                pretrained='vggface2',
                classify=False,
                device=self.device
            ).eval()

            # 状态标记
            self.models_initialized = True
            print("模型初始化完成")
        except Exception as e:
            print(f"模型初始化失败: {str(e)}")
            self.models_initialized = False

    def load_classifier(self, model_path):
        """加载分类器模型"""
        try:
            model_data = joblib.load(model_path)
            self.classifier = model_data['classifier']
            self.label_encoder = model_data['label_encoder']
            self.dorm_members = model_data['dorm_members']
            self.model_loaded = True
            print(f"分类器加载成功，成员: {', '.join(self.dorm_members)}")
            return True
        except Exception as e:
            print(f"分类器加载失败: {str(e)}")
            self.model_loaded = False
            return False

    def extract_features(self, face_img):
        """使用ArcFace提取人脸特征"""
        try:
            # 将图像从BGR转换为RGB
            rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            faces = self.arcface_model.get(rgb_img)
            if faces:
                return faces[0].embedding
            return None
        except Exception as e:
            print(f"特征提取失败: {str(e)}")
            return None

    def extract_features_facenet(self, face_img):
        """使用FaceNet提取人脸特征（备选）"""
        try:
            # 转换为PIL图像并预处理
            face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            face_tensor = self.preprocess_face(face_pil).to(self.device)

            with torch.no_grad():
                features = self.facenet_model(face_tensor.unsqueeze(0)).cpu().numpy()[0]

            return features
        except Exception as e:
            print(f"FaceNet特征提取失败: {str(e)}")
            return None

    def preprocess_face(self, face_img):
        """预处理人脸图像"""
        # 调整大小
        face_img = face_img.resize((160, 160))

        # 转换为张量并归一化
        face_img = np.array(face_img).astype(np.float32) / 255.0
        face_img = (face_img - 0.5) / 0.5  # 归一化到[-1, 1]
        face_img = torch.tensor(face_img).permute(2, 0, 1)  # HWC to CHW

        return face_img

    def recognize(self, image, threshold=0.7):
        """识别人脸"""
        if not self.model_loaded or not self.models_initialized:
            return [], image.copy()

        # 使用ArcFace检测人脸
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = self.arcface_model.get(rgb_img)

        results = []
        display_img = image.copy()

        if faces:
            for face in faces:
                # 获取人脸框
                x1, y1, x2, y2 = face.bbox.astype(int)

                # 提取特征
                embedding = face.embedding

                # 预测
                probabilities = self.classifier.predict_proba([embedding])[0]
                max_prob = np.max(probabilities)
                pred_class = self.classifier.predict([embedding])[0]
                pred_label = self.label_encoder.inverse_transform([pred_class])[0]

                # 判断是否为陌生人
                if max_prob < threshold or pred_label == 'stranger':
                    label = "陌生人"
                    color = (0, 0, 255)  # 红色
                else:
                    label = pred_label
                    color = (0, 255, 0)  # 绿色

                # 保存结果
                results.append({
                    "box": [x1, y1, x2, y2],
                    "label": label,
                    "confidence": max_prob
                })

                # 在图像上绘制结果
                cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_img, f"{label} ({max_prob:.2f})",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return results, display_img

    def save_feedback(self, image, detected_box, incorrect_label, correct_label):
        """保存用户反馈数据"""
        feedback_dir = "data/feedback_data"
        os.makedirs(feedback_dir, exist_ok=True)

        # 创建唯一文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"feedback_{timestamp}.pkl"
        filepath = os.path.join(feedback_dir, filename)

        # 准备数据
        feedback_data = {
            "image": image,
            "detected_box": detected_box,
            "incorrect_label": incorrect_label,
            "correct_label": correct_label,
            "timestamp": timestamp
        }

        # 保存数据
        with open(filepath, "wb") as f:
            pickle.dump(feedback_data, f)

        print(f"反馈数据已保存: {filepath}")
        return True


class TripletFaceDataset(Dataset):
    """三元组人脸数据集"""

    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
        self.label_to_indices = {}

        # 创建标签到索引的映射
        for idx, label in enumerate(labels):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)

    def __getitem__(self, index):
        anchor_label = self.labels[index]

        # 随机选择正样本
        positive_idx = index
        while positive_idx == index:
            positive_idx = random.choice(self.label_to_indices[anchor_label])

        # 随机选择负样本
        negative_label = random.choice([l for l in set(self.labels) if l != anchor_label])
        negative_idx = random.choice(self.label_to_indices[negative_label])

        return (
            self.embeddings[index],
            self.embeddings[positive_idx],
            self.embeddings[negative_idx]
        )

    def __len__(self):
        return len(self.embeddings)


class TripletLoss(nn.Module):
    """三元组损失函数"""

    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


def train_triplet_model(embeddings, labels, epochs=100):
    """训练三元组模型"""
    dataset = TripletFaceDataset(embeddings, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = nn.Sequential(
        nn.Linear(embeddings.shape[1], 256),
        nn.ReLU(),
        nn.Linear(256, 128)
    )

    criterion = TripletLoss(margin=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        total_loss = 0.0
        for anchor, positive, negative in dataloader:
            optimizer.zero_grad()

            anchor_embed = model(anchor)
            positive_embed = model(positive)
            negative_embed = model(negative)

            loss = criterion(anchor_embed, positive_embed, negative_embed)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

    return model
