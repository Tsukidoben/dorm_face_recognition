import os

# 定义模型路径
model_dir = "./models/buffalo_l"

# 检查路径是否存在
if os.path.exists(model_dir) and os.path.isdir(model_dir):
    print(f"✅ 模型目录存在: {model_dir}")

    # 检查关键文件是否存在
    required_files = [
        '1k3d68.onnx',
        '2d106det.onnx',
        'det_10g.onnx',
        'genderage.onnx',
        'w600k_r50.onnx'
    ]

    missing_files = []
    for file in required_files:
        file_path = os.path.join(model_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)

    if missing_files:
        print(f"❌ 模型文件缺失: {', '.join(missing_files)}")
    else:
        print("✅ 所有必需模型文件都存在")
else:
    print(f"❌ 模型目录不存在: {model_dir}")
