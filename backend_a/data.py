
import numpy as np
import SimpleITK as sitk

from MedicalDataAugmentationTool.utils.segmentation.segmentation_test import SegmentationTest
from MedicalDataAugmentationTool.datasources.image_datasource import ImageDataSource
from MedicalDataAugmentationTool.generators.image_generator import ImageGenerator
import MedicalDataAugmentationTool.utils.io.text
import MedicalDataAugmentationTool.utils.sitk_np


def intensity_preprocessing_xray(image):
    image = sitk.Cast(image, sitk.sitkFloat32)

    image = sitk.SmoothingRecursiveGaussian(image, 0.75)

    return image

def intensity_postprocessing_xray(image):
    """
    Intensity postprocessing for 2D X-ray image (.mha) after Gaussian smoothing.
    Normalize to [0, 1] using percentile clipping (1% - 99%).
    :param image: The np input image.
    :return: The processed image.
    """
    # print("Before:", np.min(image), np.max(image))
    # print("Shape:", image.shape)

    # 1. 计算百分位 (忽略极端亮暗值)
    p_low, p_high = np.percentile(image, (1, 99))
    image = np.clip(image, p_low, p_high)

    # 2. 映射到 [0, 1]
    image = (image - p_low) / (p_high - p_low)
    image = np.clip(image, 0.0, 1.0)  # 确保范围 [0,1]

    return image.astype(np.float32)

segmentation_test = SegmentationTest(list(range(25)),
                                     channel_axis=0,
                                     largest_connected_component=False,
                                     all_labels_are_connected=False)


def landmark_mask_preprocessing(image):
    """
    Creates a landmark mask of ones, but with 25 mm zeroes on the top and the bottom of the volumes.
    :param image: The sitk input image
    :return: A mask as an sitk image.
    """
    image_np = np.ones(list(reversed(image.GetSize())), np.uint8)
    spacing_y = image.GetSpacing()[1]
    # set 25 mm on top and bottom of image to 0
    size = 0
    image_np[:int(size / spacing_y), ...] = 0
    image_np[-int(size / spacing_y):, ...] = 0
    return MedicalDataAugmentationTool.utils.sitk_np.np_to_sitk(image_np)

def fit_transform(image, target_size, target_spacing):
    """
    构造一个仿射变换，使图像 resample 到目标尺寸+spacing。
    :param image: SimpleITK.Image (输入图像)
    :param target_size: [h, w] 目标尺寸
    :param target_spacing: [sy, sx] 目标 spacing
    :return: sitk.Transform
    """
    dim = image.GetDimension()
    assert dim == len(target_size) == len(target_spacing)

    # 原始信息
    original_size = np.array(image.GetSize(), dtype=float)
    original_spacing = np.array(image.GetSpacing(), dtype=float)
    original_physical = original_size * original_spacing

    # 目标物理范围
    target_size = np.array(target_size, dtype=float)
    target_spacing = np.array(target_spacing, dtype=float)
    target_physical = target_size * target_spacing

    # 缩放因子
    scale_factors = original_physical / target_physical

    # 构造仿射变换
    affine = sitk.AffineTransform(dim)
    affine.SetMatrix(np.diag(scale_factors).flatten().tolist())
    affine.SetTranslation([0.0] * dim)

    return affine

def segmentation(input_path):
    ds_image = ImageDataSource(
        root_location="",   # 文件所在文件夹
        file_prefix="",               # 没有前缀
        file_suffix="",               # 没有后缀
        file_ext="",              # 后缀名
        set_zero_origin=True,
        set_identity_direction=True,
        set_identity_spacing=True,
        sitk_pixel_type=sitk.sitkUInt8,
        preprocessing=intensity_preprocessing_xray
    )
    # 假设文件名是 my_image.mha → 那么 image_id 就是 "my_image"
    input_image = ds_image.load_and_preprocess(input_path)


    transformation = fit_transform(input_image, [512, 1024], [1,1])

    generator_image = ImageGenerator(
        2,
        [512, 1024],
        [1, 1],
        interpolator='linear',
        post_processing_np=intensity_postprocessing_xray,
        data_format='channels_first',
        resample_default_pixel_value=0,
        name='image',
        parents=[input_image, transformation]   # 直接把 image + transformation 放进去
    )

    image = generator_image.get(input_image, transformation)

    return input_image, transformation, image


def localization(input_path):
    ds_image = ImageDataSource(
        root_location="",   # 文件所在文件夹
        file_prefix="",               # 没有前缀
        file_suffix="",               # 没有后缀
        file_ext="",              # 后缀名
        set_zero_origin=True,
        set_identity_direction=True,
        set_identity_spacing=True,
        sitk_pixel_type=sitk.sitkUInt8,
        preprocessing=intensity_preprocessing_xray
    )
    # 假设文件名是 my_image.mha → 那么 image_id 就是 "my_image"
    input_image = ds_image.load_and_preprocess(input_path)

    transformation = fit_transform(input_image, [512, 1024], [1, 1])

    generator_image = ImageGenerator(
        2,
        [512, 1024],
        [1, 1],
        interpolator='linear',
        post_processing_np=intensity_postprocessing_xray,
        data_format='channels_first',
        resample_default_pixel_value=0,
        name='image',
        parents=[input_image, transformation]
    )

    image = generator_image.get(input_image, transformation)

    return input_image, transformation, image
