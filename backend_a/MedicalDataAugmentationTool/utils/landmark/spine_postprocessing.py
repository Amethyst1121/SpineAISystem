
from .common import Landmark
import numpy as np

class SpinePostprocessing(object):
    """
    Extract landmark sequence for spine images.
    """
    def __init__(self,
                 num_landmarks,
                 image_spacing,
                 min_landmark_value_for_longest_sequence, # 主序列
                 min_landmark_value_for_best_border_index, # 最上一块
                 min_landmark_value_for_front_back_append, # 补点
                 border_distance_for_best_index_front=(120, 650),
                 border_distance_for_best_index_back=(220, 830),
                 min_valid_distance=100,
                 max_valid_distance=350,
                 min_valid_distance_for_front_back_append=100,
                 max_valid_distance_for_front_back_append=350):
        """
        Initializer.
        :param num_landmarks: The number of landmarks.
        :param image_spacing: The image spacing of the prediction images.
        :param min_landmark_value_for_longest_sequence: Minimal landmark value for calculating the longest sequence.
        :param min_landmark_value_for_best_border_index: Minimal landmark value for searching the best landmark index within the border distance.
        :param border_distance_for_best_index: Border distance in mm for searching the best landmark index.
        :param min_valid_distance: Minimal distance for neighboring landmarks to be considered valid.
        :param max_valid_distance: Maximal distance for neighboring landmarks to be considered valid.
        :param min_landmark_value_for_front_back_append: Minimal landmark value for appending landmarks to front and back.
        :param min_valid_distance_for_front_back_append: Minimal distance for neighboring landmarks to be considered valid, when appending landmarks.
        :param max_valid_distance_for_front_back_append: Maximal distance for neighboring landmarks to be considered valid,  when appending landmarks.
        """
        self.num_landmarks = num_landmarks
        self.image_spacing = image_spacing
        self.min_landmark_value_for_longest_sequence = min_landmark_value_for_longest_sequence
        self.min_landmark_value_for_best_border_index = min_landmark_value_for_best_border_index
        self.min_landmark_value_for_front_back_append = min_landmark_value_for_front_back_append
        self.border_distance_for_best_index_front = border_distance_for_best_index_front
        self.border_distance_for_best_index_back = border_distance_for_best_index_back
        self.min_valid_distance = min_valid_distance
        self.max_valid_distance = max_valid_distance
        self.min_valid_distance_for_front_back_append = min_valid_distance_for_front_back_append
        self.max_valid_distance_for_front_back_append = max_valid_distance_for_front_back_append

    def longest_sequence(self, landmarks, min_max_value):
        """
        Returns the longest valid sequence of local maxima indizes that is found within landmarks.
        :param landmarks: Landmarks.
        :param min_max_value: The minimum value a landmark is considered to be valid.
        :return: List of landmark indizes of the longest sequence. 找到响应最长的一段
        """
        sequences = []
        previous_was_invalid = True
        for i, landmark in enumerate(landmarks):
            if landmark[0].value > min_max_value:
                if previous_was_invalid:
                    sequences.append([i])
                else:
                    sequences[-1].append(i)
                previous_was_invalid = False
            else:
                previous_was_invalid = True
        # return empty sequence if no valid sequence was found
        if len(sequences) == 0:
            return []
        # take the longest sequence
        sequences = list(reversed(sorted(sequences, key=len))) # 选最长序列
        longest_sequence = sequences[0]
        return longest_sequence

    def min_max_index_from_longest_sequence(self, predicted_landmarks):
        """
        Calculate start and end index from longest valid sequence. If no longest sequence has been found, return (-1, -1).
        :param predicted_landmarks: The predicted landmarks.
        :return: Start and end index from longest valid sequence.
        """
        longest_sequence = self.longest_sequence(predicted_landmarks, self.min_landmark_value_for_longest_sequence) # 返回最长主序列 [18,19,20,21,22,23]
        if len(longest_sequence) == 0:
            return -1, -1
        return longest_sequence[0], longest_sequence[-1]

    def best_top_index_within_range(self, predicted_landmarks, coord_range, min_value):
        """
        Return landmark index with largest value within z-coordinate range.
        :param predicted_landmarks: The predicted landmarks.
        :param coord_range: The coord range to search within.
        :param min_value: The min value of a landmark to be considered a maximum.
        :return: The landmark index with the highest value with the given z-coordinate range. If not found, return -1.
        """
        best_index = -1
        best_y = float("inf")  # 初始为正无穷，保证第一个符合条件的 landmark 会更新

        for i, landmarks in enumerate(predicted_landmarks):
            landmark = landmarks[0]
            if landmark.is_valid:
                if coord_range[0] <= landmark.coords[1] <= coord_range[1] and landmark.value >= min_value:
                    if landmark.coords[1] < best_y:  # y 越小越优先
                        best_index = i
                        best_y = landmark.coords[1]

        return best_index

    def best_bottom_index_within_range(self, predicted_landmarks, coord_range, min_value):
        """
        Return landmark index with largest value within z-coordinate range.
        :param predicted_landmarks: The predicted landmarks.
        :param coord_range: The coord range to search within.
        :param min_value: The min value of a landmark to be considered a maximum.
        :return: The landmark index with the highest value with the given z-coordinate range. If not found, return -1.
        """
        best_index = -1
        best_y = float("-inf")  # 初始为负无穷，保证第一个符合条件的 landmark 会更新

        for i, landmarks in enumerate(predicted_landmarks):
            landmark = landmarks[0]
            if landmark.is_valid:
                if coord_range[0] <= landmark.coords[1] <= coord_range[1] and landmark.value >= min_value:
                    if landmark.coords[1] > best_y:  # y 越大越优先
                        best_index = i
                        best_y = landmark.coords[1]

        return best_index

    def top_border_best_landmark_index(self, predicted_landmarks, border_distance, min_value):
        """
        Return landmark index with largest value on the top part of the image.
        :param predicted_landmarks: The predicted landmarks.
        :param size_z: The image size in z-dimension.
        :param border_distance: The maximum distance from the border to consider.
        :param min_value: The min value of a landmark to be considered a maximum.
        :return: The landmark index with the highest value on the top part of the image. If not found, return -1.
        """
        coord_range = border_distance
        return self.best_top_index_within_range(predicted_landmarks, coord_range, min_value)

    def bottom_border_best_landmark_index(self, predicted_landmarks, size_y, border_distance, min_value):
        """
        Return landmark index with largest value on the bottom part of the image.
        :param predicted_landmarks: The predicted landmarks.
        :param size_z: The image size in z-dimension.
        :param border_distance: The maximum distance from the border to consider.
        :param min_value: The min value of a landmark to be considered a maximum.
        :return: The landmark index with the highest value on the bottom part of the image. If not found, return -1.
        """
        coord_range = (size_y - border_distance[1], size_y - border_distance[0])
        return self.best_bottom_index_within_range(predicted_landmarks, coord_range, min_value)

    def min_max_index_from_border_range(self, predicted_landmarks, y_size):
        """
        Return landmark index with largest value on the bottom part of the image.
        :param predicted_landmarks: The predicted landmarks.
        :param prediction_size: The image size.
        :return: The landmark indizes with the highest value on the top and bottom part of the image. If not found, an index is set to -1.
        """
        min_idx = self.top_border_best_landmark_index(predicted_landmarks, self.border_distance_for_best_index_front * self.image_spacing[1], self.min_landmark_value_for_best_border_index)
        max_idx = self.bottom_border_best_landmark_index(predicted_landmarks, y_size, self.border_distance_for_best_index_back * self.image_spacing[1], self.min_landmark_value_for_best_border_index)
        return min_idx, max_idx

    def min_max_landmark_index(self, predicted_landmarks, y_size):
        """
        Calculates the start and end landmark index visible on the image.
        :param predicted_landmarks: The predicted landmarks.
        :param prediction_size: The size of the prediction volume.
        :return: A tuple of start and end index. If not found, an index may be -1.
        """
        min_idx, max_idx = self.min_max_index_from_longest_sequence(predicted_landmarks)  # 最长主序列的最大最小序列
        min_idx_from_border_range, max_idx_from_border_range = self.min_max_index_from_border_range(predicted_landmarks, y_size) # 确定主序列的最上最下,确定一个放大的作用
        if min_idx_from_border_range > -1 and min_idx_from_border_range < min_idx:
            min_idx = min_idx_from_border_range
        if max_idx_from_border_range > -1 and max_idx_from_border_range > max_idx:
            max_idx = max_idx_from_border_range
        return max_idx, min_idx

    def all_valid_sequences(self, predicted_landmarks, min_distance, max_distance, allow_skip=True):
        """
        Return all valid sequences with conditions:
        1. The distance of subsequent landmarks must be within [min_distance, max_distance].
        2. The z-coordinate must be increasing.
        If no sequence found and allow_skip=True, try skipping layers (front or back).

        Returns:
            (valid_sequences, skip_front, skip_back)
        """
        valid_sequences = []

        # ---------- 原始逻辑 ----------
        for other_landmark in predicted_landmarks[0]:
            recursive_valid_sequences = self.all_valid_sequences_recursive(
                other_landmark, predicted_landmarks[1:], min_distance, max_distance
            )
            if recursive_valid_sequences is not None:
                valid_sequences.extend(recursive_valid_sequences)

        if valid_sequences:
            return [(seq, 0, 0) for seq in valid_sequences]  # 没跳过

        # ---------- 扩展逻辑：跳过 ----------
        if not allow_skip or len(predicted_landmarks) <= 1:
            return []

        n_layers = len(predicted_landmarks)
        best_front, best_back = ([], 0), ([], 0)  # (sequences, skip_count)

        # 尝试从前面跳过
        for skip in range(1, n_layers):
            result = self.all_valid_sequences(predicted_landmarks[skip:], min_distance, max_distance,
                                                    allow_skip=False)
            if result:
                best_front = ( [seq for seq, _, _ in result], skip )
                break  # 尽量少跳

        # 尝试从后面跳过
        for skip in range(1, n_layers):
            result = self.all_valid_sequences(predicted_landmarks[:-skip], min_distance, max_distance, allow_skip=False)
            if result:
                best_back = ([seq for seq, _, _ in result], skip)
                break

        # ---------- 比较选择 ----------
        if not best_front[0] and not best_back[0]:
            return []

        if best_front[0] and best_back[0]:
            len_front = len(best_front[0][0])
            len_back = len(best_back[0][0])
            if len_front > len_back:
                return [(seq, best_front[1], 0) for seq in best_front[0]]
            elif len_front < len_back:
                return [(seq, 0, best_back[1]) for seq in best_back[0]]
            else:  # 长度相同，返回两边组合
                front_seqs = [(seq, best_front[1], 0) for seq in best_front[0]]
                back_seqs = [(seq, 0, best_back[1]) for seq in best_back[0]]
                return front_seqs + back_seqs

        if best_front[0]:
            return [(seq, best_front[1], 0) for seq in best_front[0]]
        if best_back[0]:
            return [(seq, 0, best_back[1]) for seq in best_back[0]]

        return []

    def all_valid_sequences_recursive(self, current_landmark, predicted_landmarks, min_distance, max_distance):

        if len(predicted_landmarks) == 0:
            return [[current_landmark]]
        valid_sequences = []
        found = False
        for other_landmark in predicted_landmarks[0]:
            distance = np.linalg.norm(current_landmark.coords - other_landmark.coords)
            if distance > min_distance and distance < max_distance and other_landmark.coords[1] > current_landmark.coords[1]:
                found = True
                recursive_valid_sequences = self.all_valid_sequences_recursive(other_landmark, predicted_landmarks[1:], min_distance, max_distance)
                if recursive_valid_sequences is not None:
                    valid_sequences.extend(recursive_valid_sequences)
        if not found:
            return None

        return [[current_landmark] + valid_sequence for valid_sequence in valid_sequences]

    def append_landmarks_to_front_and_back(self, landmarks, all_other_landmarks, min_distance, max_distance, min_index, max_index, min_max_value):
        """
        Appends additional landmarks with valid conditions to front and back of a sequence.
        :param landmarks: A landmark sequence.
        :param all_other_landmarks: All predicted landmarks.
        :param min_distance: The minimal allowed distance of subsequent landmarks.
        :param max_distance: The maximal allowed distance of subsequent landmarks.
        :param min_index: The start index of the landmark sequence.
        :param max_index: The end index of the landmark sequence.
        :param min_max_value: The minimal maximal value for a landmark to be appended.
        :return: The landmark sequence with possible appended landmarks, new min index, new max index
        """

        # append valid landmarks to front
        found = True
        while found and min_index > 0:
            found = False
            first_landmark = landmarks[0]
            previous_landmarks = all_other_landmarks[min_index - 1]
            for other_landmark in previous_landmarks:
                distance = np.linalg.norm(first_landmark.coords - other_landmark.coords)
                if (
                        min_distance < distance < max_distance
                        and first_landmark.coords[1] > other_landmark.coords[1]
                        and other_landmark.value > min_max_value
                ):
                    landmarks = [other_landmark] + landmarks
                    min_index -= 1
                    found = True  # ✅ 只在找到时才设置 True
                    break  # 找到后直接跳出 for，继续往前走

        # append valid landmarks to back
        found = True
        while found and max_index < len(all_other_landmarks) - 1:
            found = False
            last_landmark = landmarks[-1]
            next_landmarks = all_other_landmarks[max_index + 1]
            for other_landmark in next_landmarks:
                distance = np.linalg.norm(last_landmark.coords - other_landmark.coords)
                if (
                        min_distance < distance < max_distance
                        and last_landmark.coords[1] < other_landmark.coords[1]
                        and other_landmark.value > min_max_value
                ):
                    landmarks = landmarks + [other_landmark]
                    max_index += 1
                    found = True
                    break

        return landmarks, min_index, max_index

    def finalize_landmarks(self, final_landmarks, min_idx, max_idx):
        """
        Appends invalid landmarks to the final landmark sequence such that the list contains self.num_landmark entries, and the list index is the landmark index.
        :param final_landmarks: The final landmark sequence.
        :param min_idx: The first landmark index.
        :param max_idx: The last landmark index.
        :return: A landmark sequence with self.num_landmark, where not found landmarks are set to be invalid.
        """
        # fill landmarks with invalid coordinates
        return [Landmark(coords=[np.nan] * 2, is_valid=False)] * min_idx + final_landmarks + [Landmark(coords=[np.nan] * 2, is_valid=False)] * (self.num_landmarks - max_idx - 1)

    def sequence_of_largest_maxima(self, predicted_landmarks):
        """
        Returns the sequence of landmark global maxima. No conditions are tested!
        :param predicted_landmarks: The predicted landmarks.
        :return: Sequence of largest maxima.
        """
        return [l[0] for l in predicted_landmarks]

    def postprocess_landmarks(self, predicted_landmarks, y_size):
        """
        Postprocess predicted landmarks such that the spine conditions hold.
        :param predicted_landmarks: The predicted landmarks. List of non maximum suppressed landmark lists.
        :param prediction_size: The size of the predicted volume.
        :return: The final sequence of landmarks.
        """
        try:
            # get min and max landmark index
            max_idx, min_idx = self.min_max_landmark_index(predicted_landmarks, y_size) # 最大最小
            # return invalid landmarks, if no start and end indizes were found
            if min_idx == -1 or max_idx == -1:
                return self.sequence_of_largest_maxima(predicted_landmarks) # 返回每个通道的最大
            # generate all valid sequences
            all_valid_landmark_sequences = self.all_valid_sequences(predicted_landmarks[min_idx:max_idx + 1], self.min_valid_distance, self.max_valid_distance) # 所有的合法序列
            if all_valid_landmark_sequences:
                # 计算每条序列的 value 总和
                values = [np.nansum([l.value for l in seq]) for seq, _, _ in all_valid_landmark_sequences]
                # 找最大值的索引
                best_index = int(np.argmax(values))
                # 得到最终序列及对应跳过层数
                final_landmarks, front_skip, back_skip = all_valid_landmark_sequences[best_index]
                min_idx += front_skip
                max_idx -= back_skip
            else:
                # failsafe，如果没有合法序列
                final_landmarks = [l[0] for l in predicted_landmarks[min_idx:max_idx + 1]]

            # append landmarks to front and back that are valid within the conditions
            final_landmarks, min_idx, max_idx = self.append_landmarks_to_front_and_back(final_landmarks, predicted_landmarks,
                                                                                        self.min_valid_distance_for_front_back_append, self.max_valid_distance_for_front_back_append,
                                                                                        min_idx, max_idx,
                                                                                        self.min_landmark_value_for_front_back_append)
            return self.finalize_landmarks(final_landmarks, min_idx, max_idx)
        except:
            return self.sequence_of_largest_maxima(predicted_landmarks)
