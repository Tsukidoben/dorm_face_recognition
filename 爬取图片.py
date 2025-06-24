import requests
from bs4 import BeautifulSoup
import json
import os
import re
import time
import random
from PIL import Image
import io


def download_bing_images(search_url, save_dir='E:/zunlong/', max_images=50):
    """
    从 Bing 图片搜索下载图片到指定目录，并按数字顺序命名

    参数:
    search_url: Bing 图片搜索 URL
    save_dir: 图片保存目录 (默认: E:/wuyanzu/)
    max_images: 最大下载图片数量 (默认: 50)
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    print(f"图片将保存到: {save_dir}")

    # 设置请求头模拟浏览器
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Referer': 'https://cn.bing.com/'
    }

    try:
        # 获取搜索结果页
        response = requests.get(search_url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # 查找包含图片信息的元素
        image_elements = soup.find_all('a', class_='iusc')
        downloaded_count = 0

        print(f"找到 {len(image_elements)} 个图片元素，开始下载...")

        # 获取目录中已有的最大数字文件名
        existing_files = [f for f in os.listdir(save_dir) if f.split('.')[0].isdigit()]
        start_index = 1
        if existing_files:
            max_num = max(int(f.split('.')[0]) for f in existing_files)
            start_index = max_num + 1

        for idx, elem in enumerate(image_elements):
            if downloaded_count >= max_images:
                break

            try:
                # 提取 JSON 格式的图片数据
                m_data = elem.get('m')
                if not m_data:
                    continue

                img_data = json.loads(m_data)
                img_url = img_data.get('murl')  # 真实图片 URL

                # 下载图片
                img_response = requests.get(img_url, headers=headers, timeout=20)
                if img_response.status_code != 200 or not img_response.content:
                    continue

                # 使用PIL验证图片完整性
                try:
                    img = Image.open(io.BytesIO(img_response.content))
                    img.verify()  # 验证图片完整性
                except Exception as img_error:
                    print(f"图片验证失败: {str(img_error)}")
                    continue

                # 获取图片扩展名
                content_type = img_response.headers.get('Content-Type', '').lower()
                ext = '.jpg'  # 默认扩展名
                if 'jpeg' in content_type:
                    ext = '.jpg'
                elif 'png' in content_type:
                    ext = '.png'
                elif 'gif' in content_type:
                    ext = '.gif'
                elif 'webp' in content_type:
                    ext = '.webp'

                # 生成数字文件名
                file_index = start_index + downloaded_count
                filename = f"{file_index}{ext}"
                save_path = os.path.join(save_dir, filename)

                # 保存图片
                with open(save_path, 'wb') as f:
                    f.write(img_response.content)

                # 获取文件大小并转换格式
                file_size = os.path.getsize(save_path)
                size_str = f"{file_size / 1024:.1f}KB"
                if file_size > 1024 * 1024:
                    size_str = f"{file_size / (1024 * 1024):.1f}MB"

                print(f"下载成功 [{downloaded_count + 1}/{max_images}]: {filename} ({size_str})")
                downloaded_count += 1

                # 随机延时，避免请求过快
                time.sleep(random.uniform(0.5, 1.5))

            except (json.JSONDecodeError, KeyError, requests.RequestException, OSError) as e:
                print(f"处理图片时出错: {str(e)}")
                continue

        print(f"\n下载完成! 共下载 {downloaded_count} 张图片到 {save_dir}")

    except requests.RequestException as e:
        print(f"请求失败: {str(e)}")
    except Exception as e:
        print(f"发生未知错误: {str(e)}")


# 使用示例
if __name__ == "__main__":
    # 目标URL - 吴彦祖正面图片搜索
    search_url = "https://cn.bing.com/images/search?q=%E5%B0%8A%E9%BE%99%E6%AD%A3%E8%84%B8%E5%9B%BE%E7%89%87&form=IQFRML&first=1"

    # 指定保存目录
    save_directory = "E:/zunlong/"

    # 开始下载
    download_bing_images(search_url, save_dir=save_directory, max_images=50)
