
import os
import numpy as np
import SimpleITK as sitk
import tensorflow as tf
from network import UnetClassicAvgLinear2d, network_u
from MedicalDataAugmentationTool.generators.landmark_generator import LandmarkGeneratorHeatmap
import MedicalDataAugmentationTool.utils.io.text
import MedicalDataAugmentationTool.utils.sitk_np
from MedicalDataAugmentationTool.utils.landmark.common import Landmark
import threading



from data import segmentation_test, segmentation


tf.compat.v1.disable_eager_execution()

def load_model():
    # ==== 1. 占位符 ====
    data_val = tf.compat.v1.placeholder(tf.float32, shape=[1, 1, 1024, 512], name="data_val")
    single_heatmap_val = tf.compat.v1.placeholder(tf.float32, shape=[1, 1, 1024, 512], name="single_heatmap_val")
    data_heatmap_concat_val = tf.concat([data_val, single_heatmap_val], axis=1)

    # ==== 2. 构建网络（保持和训练/测试一致的 scope） ====
    with tf.compat.v1.variable_scope("net_1"):  # 这里根据测试时实际使用的 scope
        logits = network_u(
            input=data_heatmap_concat_val,
            is_training=False,
            num_labels=1,
            data_format='channels_first',
            actual_network=UnetClassicAvgLinear2d,
            num_filters_base=64,
            double_features_per_level=False,
            num_levels=5,
            activation='relu',
            dropout_ratio=0.25
        )
    prediction_softmax_val = tf.nn.sigmoid(logits, name="prediction_softmax_val")

    # ==== 3. 使用 Saver 直接 restore ====
    saver = tf.compat.v1.train.Saver()  # 直接管理所有 graph 中的变量

    # ==== 4. Session ====
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(BASE_DIR, "models_seg", "model-100000")
    saver.restore(sess, ckpt_path)
    print("✅ Checkpoint restored.")

    return sess, data_val, single_heatmap_val, prediction_softmax_val

def test_full_image(sess, data_val, single_heatmap_val, prediction_softmax_val,
                    image, single_heatmap, transformation):
    """
    Perform inference on image + single_heatmap
    image, single_heatmap: numpy arrays shaped (1,H,W)
    """
    feed_dict = {
        data_val: np.expand_dims(image, axis=0),              # (1,1,H,W)
        single_heatmap_val: np.expand_dims(single_heatmap, axis=0),  # (1,1,H,W)
    }
    prediction = sess.run(prediction_softmax_val, feed_dict=feed_dict)
    prediction = np.squeeze(prediction, axis=0)  # → (H,W)
    return image, prediction, transformation

_graph_seg = tf.Graph()
_sess_seg = None
_model_seg = None
_seg_lock = threading.Lock()  # 确保对 sess.run 的并发访问安全

def init_seg_model():
    """
    初始化 segmentation 模型（只会执行一次）
    要求 load_model() 在其内部创建占位符、网络并 restore checkpoint，
    并返回 (sess, data_val, single_heatmap_val, prediction_softmax_val)
    """
    global _sess_seg, _model_seg
    if _sess_seg is not None:
        return

    with _graph_seg.as_default():
        # load_model() 必须在这个 graph 上构建并 restore
        sess, data_val, single_heatmap_val, prediction_softmax_val = load_model()
        # 保存到全局变量供后面复用
        _sess_seg = sess
        _model_seg = (data_val, single_heatmap_val, prediction_softmax_val)
        # print("✅ seg model initialized")

def pipeline_seg(input_path, points):
    """
    运行 segmentation 推理（使用全局缓存的 session 和 graph）
    input_path: mha 路径
    points: 前端传入的点数组（id,x,y）
    """
    # 确保模型已初始化（若未初始化则加载）
    init_seg_model()

    # 下面所有 sess.run 必须在 _graph_seg 上下文中执行
    with _graph_seg.as_default():
        sess = _sess_seg
        data_val, single_heatmap_val, prediction_softmax_val = _model_seg

        # segmentation(...) -> 返回 ip_image (sitk), transformation, image(np)
        ip_image, transformation, image = segmentation(input_path)

        # 构建 landmarks 列表
        landmarks = []
        for p in points:
            if p["x"] is None or p["y"] is None:
                landmarks.append(Landmark(None, False))
            else:
                coords = np.array([p["x"], p["y"]], np.float32)
                landmarks.append(Landmark(coords))

        valid_landmarks = [i for i, lm in enumerate(landmarks) if lm.is_valid]

        first = True
        prediction_resampled_np = None
        input_image = None

        for landmark_id in valid_landmarks:
            if first:
                input_image = ip_image
                prediction_resampled_np = np.zeros([25] + list(reversed(input_image.GetSize())),
                                                   dtype=np.float16)
                prediction_resampled_np[0, ...] = 0.5
                first = False

            single_landmark = [landmarks[landmark_id]]

            generators_singleheatmap = LandmarkGeneratorHeatmap(
                2, [512, 1024], [1, 1],
                sigma=12.0, scale_factor=1.0,
                normalize_center=True, data_format='channels_first',
                name='single_heatmap', parents=[single_landmark, transformation]
            )

            single_heatmap = generators_singleheatmap.get(single_landmark, transformation)

            # ====== 关键：在 graph 上下文中，用线程锁保护 sess.run 调用 ======
            with _seg_lock:
                image, prediction, transformation = test_full_image(
                    sess, data_val, single_heatmap_val, prediction_softmax_val,
                    image, single_heatmap, transformation
                )

            # 后处理（你现有逻辑）
            prediction_thresh_np = (prediction > 0.5).astype(np.uint8)
            largest_connected_component = MedicalDataAugmentationTool.utils.np_image.largest_connected_component(prediction_thresh_np[0])
            prediction_thresh_np[largest_connected_component[None, ...] == 1] = 0
            prediction[prediction_thresh_np == 1] = 0

            prediction_resampled_sitk = MedicalDataAugmentationTool.utils.sitk_image.transform_np_output_to_sitk_input(
                output_image=prediction,
                output_spacing=[1, 1],
                channel_axis=0,
                input_image_sitk=input_image,
                transform=transformation,
                interpolator='linear',
                output_pixel_type=sitk.sitkFloat32
            )

            prediction_resampled_np[landmark_id + 1, ...] = MedicalDataAugmentationTool.utils.sitk_np.sitk_to_np(prediction_resampled_sitk[0])

        # 最终生成标签图（在 graph 上下文中也可以运行）
        prediction_labels = segmentation_test.get_label_image(prediction_resampled_np, reference_sitk=input_image)

    return prediction_labels
