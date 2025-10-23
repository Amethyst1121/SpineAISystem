
import os
import numpy as np
import tensorflow as tf
from network import spatial_configuration_net, UnetClassicAvgLinear2d
from MedicalDataAugmentationTool.utils.landmark.heatmap_test import HeatmapTest
from MedicalDataAugmentationTool.utils.landmark.spine_postprocessing import SpinePostprocessing
from data import localization
import threading

# 关闭 TF2.x 的 eager execution
tf.compat.v1.disable_eager_execution()

# ===== 全局变量 =====
_graph_center = tf.Graph()
_sess_center = None
_model_center = None
_center_lock = threading.Lock()   # 🔒 并发锁

def load_center_model():
    global _sess_center, _model_center
    if _sess_center is not None:
        return _sess_center, _model_center

    with _graph_center.as_default():
        # ==== 1. 占位符 ====
        data_val = tf.compat.v1.placeholder(tf.float32, shape=[1, 1, 1024, 512], name="data_val")

        # ==== 2. 构建网络 ====
        with tf.compat.v1.variable_scope("net", reuse=tf.compat.v1.AUTO_REUSE):
            prediction_val, local_prediction_val, spatial_prediction_val = spatial_configuration_net(
                input=data_val,
                num_labels=24,
                is_training=False,
                data_format='channels_first',
                actual_network=UnetClassicAvgLinear2d,
                padding="same",
                spatial_downsample=4
            )

        prediction_softmax_val = tf.nn.sigmoid(prediction_val, name="prediction_softmax_val")

        # ==== 3. Saver ====
        saver = tf.compat.v1.train.Saver()

        # ==== 4. Session ====
        sess = tf.compat.v1.Session(graph=_graph_center)
        sess.run(tf.compat.v1.global_variables_initializer())
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        ckpt_path = os.path.join(BASE_DIR, "models_lo", "model-12000")
        saver.restore(sess, ckpt_path)
        print("✅ localization checkpoint restored.")

        _sess_center = sess
        _model_center = (data_val, prediction_softmax_val, local_prediction_val, spatial_prediction_val)

    return _sess_center, _model_center


def test_full_image(sess, data_val, prediction_softmax_val, local_prediction_val, spatial_prediction_val,
                    image, transformation):
    """Perform inference on a full image."""
    feed_dict = {
        data_val: np.expand_dims(image, axis=0),  # (1,1,H,W)
    }
    pred, local_pred, spatial_pred = sess.run(
        [prediction_softmax_val, local_prediction_val, spatial_prediction_val],
        feed_dict=feed_dict
    )
    pred = np.squeeze(pred, axis=0)  # (H,W)
    return image, pred, transformation


def pipeline_center(input_path, task_dir):
    # ✅ 确保模型已加载
    sess, (data_val, prediction_softmax_val, local_prediction_val, spatial_prediction_val) = load_center_model()

    input_image, transformation, image = localization(input_path)

    # 🔒 Session 运行必须加锁
    with _center_lock:
        image, prediction, transformation = test_full_image(
            sess, data_val, prediction_softmax_val, local_prediction_val, spatial_prediction_val,
            image, transformation
        )

    # ==== heatmap 阈值 ====
    q99 = np.percentile(prediction, 99)
    q99_9 = np.percentile(prediction, 99.9)
    q99_95 = np.percentile(prediction, 99.95)
    q99_5 = np.percentile(prediction, 99.5)

    heatmap_maxima = HeatmapTest(
        0, False, return_multiple_maxima=True, min_max_distance=60,
        min_max_value=q99, multiple_min_max_value_factor=0.2
    )
    spine_postprocessing = SpinePostprocessing(
        num_landmarks=24,
        image_spacing=[1, 1],
        min_landmark_value_for_longest_sequence=q99_9,
        min_landmark_value_for_best_border_index=q99_95,
        min_landmark_value_for_front_back_append=q99_5
    )

    local_maxima_landmarks = heatmap_maxima.get_landmarks(
        prediction, input_image, [1, 1], transformation
    )
    landmark_sequence = spine_postprocessing.postprocess_landmarks(
        local_maxima_landmarks, input_image.GetSize()[1]
    )

    # ==== 转成前端需要的 points ====
    points = []
    for i, lm in enumerate(landmark_sequence, start=1):
        x, y = lm.coords if lm is not None else (None, None)
        if x is not None and not (np.isnan(x) or np.isinf(x)):
            x = float(x)
        else:
            x = None
        if y is not None and not (np.isnan(y) or np.isinf(y)):
            y = float(y)
        else:
            y = None
        points.append({"id": i, "x": x, "y": y})

    return points

# points = pipeline_center(r"D:\部署\pictures\44181907-a0c3-4921-bdef-40624e5c7eef\44181907-a0c3-4921-bdef-40624e5c7eef.mha", r"D:\部署\pictures\44181907-a0c3-4921-bdef-40624e5c7eef")
