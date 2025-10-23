import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import os
import torch
from mo import SpineCNN

def crop_each_label_png(image_path, segmentation_path, output_dir):
    # 读取原图像和分割图像
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)  # 彩色
    segmentation = cv2.imread(str(segmentation_path), cv2.IMREAD_UNCHANGED)  # 单通道标签图

    # 获取所有标签（忽略背景 0）
    unique_labels = np.unique(segmentation)
    unique_labels = unique_labels[unique_labels != 0]

    # 输出文件夹
    out_img_dir = Path(output_dir) / "images"
    out_seg_dir = Path(output_dir) / "segmentations"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_seg_dir.mkdir(parents=True, exist_ok=True)


    for label in unique_labels:
        # 找到该标签的坐标
        coords = np.where(segmentation == label)

        if coords[0].size > 0:
            # 计算边界框
            y_min, y_max = np.min(coords[0]), np.max(coords[0])
            x_min, x_max = np.min(coords[1]), np.max(coords[1])

            # 裁剪原图和分割图
            cropped_image = image[y_min:y_max + 1, x_min:x_max + 1]
            cropped_seg = segmentation[y_min:y_max + 1, x_min:x_max + 1]

            # 转换为二值掩码（该标签 = 1，其余 = 0）
            cropped_binary_seg = (cropped_seg == label).astype(np.uint8) * 255  # 保存为黑白图

            # 构建保存路径
            out_img_path = out_img_dir / f"{int(label)}.png"
            out_seg_path = out_seg_dir / f"{int(label)}.png"

            # 保存裁剪结果
            cv2.imwrite(str(out_img_path), cropped_image)
            cv2.imwrite(str(out_seg_path), cropped_binary_seg)

    return out_img_dir ,out_seg_dir

def intensity_postprocessing_xray(image: np.ndarray) -> np.ndarray:
    """
    Intensity postprocessing for 2D X-ray image after Gaussian smoothing.
    Normalize to [0, 1] using percentile clipping (1% - 99%).
    """
    p_low, p_high = np.percentile(image, (1, 99))
    image = np.clip(image, p_low, p_high)
    image = (image - p_low) / (p_high - p_low + 1e-8)
    image = np.clip(image, 0.0, 1.0)
    return image.astype(np.float32)


def process_data(image_folder, seg_folder, size=(224, 224)):
    """
    按数字顺序处理 X-ray 图像及对应分割图，返回 (N,2,224,224)
    """
    # 🔹 按文件名中的数字排序
    files = sorted(
        [f for f in os.listdir(image_folder) if f.lower().endswith(".png")],
        key=lambda x: int(os.path.splitext(x)[0])  # '12.png' -> 12
    )

    arrays = []

    for file in files:
        img_path = os.path.join(image_folder, file)
        seg_path = os.path.join(seg_folder, file)

        # --- 原图 ---
        img = Image.open(img_path).convert("L")
        img_np = np.array(img, dtype=np.float32)
        img_np = intensity_postprocessing_xray(img_np)
        img_resized = Image.fromarray((img_np * 255).astype(np.uint8))
        img_resized = img_resized.resize(size, resample=Image.BILINEAR)
        img_resized = np.array(img_resized, dtype=np.float32) / 255.0

        # --- 分割 ---
        seg = Image.open(seg_path).convert("L")
        seg_resized = seg.resize(size, resample=Image.NEAREST)
        seg_resized = np.array(seg_resized, dtype=np.uint8)
        seg_resized = (seg_resized > 127).astype(np.uint8)

        # --- 堆叠 ---
        arrays.append(np.stack([img_resized, seg_resized], axis=0))

    combined = np.stack(arrays, axis=0)  # (N,2,224,224)
    print(f"✅ 已生成数组，shape={combined.shape}")
    return combined


def build_x_and_mask(combined_array, mask_list):
    """构造输入张量 x 和 mask"""
    N = combined_array.shape[0]
    x = np.zeros((24, 2, 224, 224), dtype=np.float32)
    mask = np.zeros(24, dtype=np.float32)

    ptr = 0  # combined_array 的指针
    for i, m in enumerate(mask_list[:24]):
        if m == 1:
            img = combined_array[ptr]
            resized = np.stack([cv2.resize(img[c], (224, 224)) for c in range(2)], axis=0)
            x[i] = resized
            mask[i] = 1.0
            ptr += 1  # 移到 combined_array 的下一张图

    x = torch.from_numpy(x).unsqueeze(0)   # (1, V, C, H, W)
    mask = torch.from_numpy(mask).unsqueeze(0)  # (1, V)
    return x, mask


def inference_probs(model, combined_array, mask_list, device):
    """返回每个椎骨的预测概率 (probs)"""
    model.eval()
    x, mask = build_x_and_mask(combined_array, mask_list)
    x = x.to(device)

    with torch.no_grad():
        logits, feat = model(x)
        probs = torch.sigmoid(logits).cpu().numpy().squeeze(0)  # (24,)
    return probs


labelMapping = {}
for i in range(1, 8):
    labelMapping[i] = f"C{i}"
for i in range(8, 20):
    labelMapping[i] = f"T{i - 7}"
for i in range(20, 25):
    labelMapping[i] = f"L{i - 19}"


def fracture_img(id, prob, input_path, output_folder):
    """
    生成椎骨概率图：
    - 输出图片宽120px，高150px，透明背景
    - 上部120px放等比缩放的原图，居中
    - 下部30px放文本 '椎骨标签 概率'
    """
    # 打开原图
    img = Image.open(input_path).convert("RGBA")

    # 创建透明画布
    final_w, final_h = 180, 225
    canvas = Image.new("RGBA", (final_w, final_h), (0, 0, 0, 0))

    # 上部容器尺寸
    container_w, container_h = 180, 180

    # 等比缩放原图到容器
    img_w, img_h = img.size
    scale = min(container_w / img_w, container_h / img_h)
    new_w, new_h = int(img_w * scale), int(img_h * scale)
    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 计算居中位置
    x_offset = (container_w - new_w) // 2
    y_offset = (container_h - new_h) // 2
    canvas.paste(img_resized, (x_offset, y_offset), img_resized)  # 使用自身 alpha

    # 绘制文字
    draw = ImageDraw.Draw(canvas)
    text = f"{labelMapping.get(id, str(id))}   {prob:.2f}"

    # 字体设置
    font_size = 28
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    text_x = (final_w - text_w) // 2
    text_y = container_h + (30 - text_h) // 2  # 下部30px区域居中
    draw.text((text_x, text_y), text, fill=(0, 0, 0, 255), font=font)

    # 保存图片
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{id}.png")
    canvas.save(output_path)


def fracture(image_path, seg_path, output_dir, points):
    image_folder, seg_folder = crop_each_label_png(image_path, seg_path, output_dir)
    combined_array = process_data(image_folder, seg_folder)
    mask_list = [0] * 24
    for p in points:
        if p['x'] is not None:
            mask_list[p['id'] - 1] = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpineCNN().to(device)
    model_path = "./model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))

    probs = inference_probs(model, combined_array, mask_list, device)

    images_dir = os.path.join(output_dir, "images")
    output_folder = Path(output_dir) / "fracture_images"
    output_folder.mkdir(parents=True, exist_ok=True)
    result = []
    for i, m in enumerate(mask_list):
        if m == 1:
            prob = float(probs[i]) if np.isfinite(probs[i]) else None
            result.append({
                "vertebra_id": i + 1,
                "fracture_prob": prob
            })

            if prob > 0.5:
                input_path = os.path.join(images_dir, f"{i + 1}.png")
                if os.path.exists(input_path):
                    fracture_img(i+1, prob, input_path, output_folder)
                else:
                    print(f"⚠️ 图片不存在: {input_path}")
    return result


# image_path = r"D:\Myproject\pictures\374c6a8d-bd5c-4288-94ad-0504fc5ed177\374c6a8d-bd5c-4288-94ad-0504fc5ed177.png"
# seg_path = r"D:\Myproject\pictures\374c6a8d-bd5c-4288-94ad-0504fc5ed177\374c6a8d-bd5c-4288-94ad-0504fc5ed177_seg1.png"
# task_dir = r'D:\Myproject\pictures\374c6a8d-bd5c-4288-94ad-0504fc5ed177'
# points = [
#   {
#     "id": 1,
#     "x": None,
#     "y": None
#   },
#   {
#     "id": 2,
#     "x": None,
#     "y": None
#   },
#   {
#     "id": 3,
#     "x": None,
#     "y": None
#   },
#   {
#     "id": 4,
#     "x": None,
#     "y": None
#   },
#   {
#     "id": 5,
#     "x": None,
#     "y": None
#   },
#   {
#     "id": 6,
#     "x": None,
#     "y": None
#   },
#   {
#     "id": 7,
#     "x": None,
#     "y": None
#   },
#   {
#     "id": 8,
#     "x": None,
#     "y": None
#   },
#   {
#     "id": 9,
#     "x": 731.2161865234375,
#     "y": 1254.3233642578125
#   },
#   {
#     "id": 10,
#     "x": 753.333740234375,
#     "y": 1460.7532958984375
#   },
#   {
#     "id": 11,
#     "x": 793.8824462890625,
#     "y": 1659.810791015625
#   },
#   {
#     "id": 12,
#     "x": 838.117431640625,
#     "y": 1910.475830078125
#   },
#   {
#     "id": 13,
#     "x": None,
#     "y": None
#   },
#   {
#     "id": 14,
#     "x": None,
#     "y": None
#   },
#   {
#     "id": 15,
#     "x": None,
#     "y": None
#   },
#   {
#     "id": 16,
#     "x": None,
#     "y": None
#   },
#   {
#     "id": 17,
#     "x": None,
#     "y": None
#   },
#   {
#     "id": 18,
#     "x": None,
#     "y": None
#   },
#   {
#     "id": 19,
#     "x": None,
#     "y": None
#   },
#   {
#     "id": 20,
#     "x": None,
#     "y": None
#   },
#   {
#     "id": 21,
#     "x": None,
#     "y": None
#   },
#   {
#     "id": 22,
#     "x": None,
#     "y": None
#   },
#   {
#     "id": 23,
#     "x": None,
#     "y": None
#   },
#   {
#     "id": 24,
#     "x": None,
#     "y": None
#   }
# ]
# fracture(image_path, seg_path, task_dir, points)