import os
import SimpleITK as sitk
from PIL import Image
import numpy as np

def process_image(input_path: str, output_dir: str) -> str:
    """
    处理上传的图像文件：
    - 如果输入是 .mha：保存原始 .mha，并生成对应的 .png
    - 如果输入是 .png：保存原始 .png，并生成对应的 .mha
    - 两种格式都会保证 output_dir 下有一份 .mha 和一份 .png
    - 返回 png 文件的路径（供 main.py 返回前端使用）

    Args:
        input_path (str): 上传文件的完整路径
        output_dir (str): 保存输出文件的目录（如 pictures/{task_id}）

    Returns:
        str: 生成的 PNG 文件路径
    """
    ext = os.path.splitext(input_path)[-1].lower()
    basename = os.path.splitext(os.path.basename(input_path))[0]

    out_png = os.path.join(output_dir, f"{basename}.png")
    out_mha = os.path.join(output_dir, f"{basename}.mha")

    if ext == ".mha":
        # 读取 MHA 并转 PNG
        image = sitk.ReadImage(input_path)
        array = sitk.GetArrayFromImage(image)
        img_pil = Image.fromarray(array)
        img_pil.save(out_png)

        # 保存 MHA（就是上传的原始文件，也拷贝到 output_dir）
        if input_path != out_mha:
            sitk.WriteImage(image, out_mha)

    elif ext == ".png":
        # 保存 PNG（就是上传的原始文件，也拷贝到 output_dir）
        if input_path != out_png:
            Image.open(input_path).save(out_png)

        # 读取 PNG 并转 MHA
        img_pil = Image.open(input_path).convert("L")  # 转灰度
        array = sitk.GetImageFromArray(np.array(img_pil))
        sitk.WriteImage(array, out_mha)

    else:
        raise ValueError("Unsupported file format: must be .mha or .png")

    return out_png


def get_color():
# 生成固定的颜色表（24个颜色）
    color_table = ["#566eb5", "#b4c0e3", "#cbaeb3", "#dbcc8a", "#92c4dd", "#a8ae5e",
                   "#f0e5b6", "#d0cff6", "#CE99B3", "#a98175", "#8c4356", "#f47983",
                   "#e29c45", "#d9b611", "#789262", "#96ce54", "#177cb0", "#065279",
                   "#a1afc9", "#4a4266", "#815476", "#e4c6d0", "#75878a", "#4c8dae"]


    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


    rgb = [hex_to_rgb(c) for c in color_table]

    return rgb


def seg_png(sitk_image, task_dir, task_id):
    """
    将 segmentation sitk.Image 转为彩色 PNG，保存到 task_dir 并返回 PNG 路径
    """
    segmentation_array = sitk.GetArrayFromImage(sitk_image)  # shape: (H, W)
    color_table_rgb = get_color()

    height, width = segmentation_array.shape
    seg1_path = os.path.join(task_dir, f"{task_id}_seg1.png")
    seg1_img = Image.fromarray(segmentation_array.astype(np.uint8))
    seg1_img.save(seg1_path)

    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    overlay_pixels = np.array(overlay)

    for label in range(1, 25):
        mask = segmentation_array == label
        if np.any(mask):
            overlay_pixels[mask] = (*color_table_rgb[label - 1], 160)  # 半透明

    seg2_img = Image.fromarray(overlay_pixels, 'RGBA')

    seg2_path = os.path.join(task_dir, f"{task_id}_seg2.png")
    seg2_img.save(seg2_path)

    return