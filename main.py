import math

import cv2
import torch
from loader import DataSet
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou
from torch import Tensor
from loader import ObjectLabel
import numpy as np
from matplotlib import pyplot as plt




def iou_bb_matcher(gt_bbs_tensor, detected_bbs_tensor) -> None | tuple[dict, list]:
    '''
    Returns None if there are either no ground truths or detections such that they cannot be matched OR
    returns a dictionary of matches int -> tuple (matching id, iou, bb tensor) and a list of unmatched detections.
    :param gt_bbs_tensor: ground truths bounding boxes tensor
    :param detected_bbs_tensor: detected bounding boxes tensor
    '''
    if gt_bbs_tensor.size(dim=0) == 0 or detected_bbs_tensor.size(dim=0) == 0:
        return None

    iou = box_iou(gt_bbs_tensor, detected_bbs_tensor)
    n_ground_truths, m_detections = iou.shape[0], iou.shape[1]

    row_max_values, row_max_indices = iou.max(dim=1)
    row_augmented_ious = torch.zeros_like(iou)
    row_augmented_ious[torch.arange(n_ground_truths), row_max_indices] = row_max_values

    col_max_values, col_max_indices = row_augmented_ious.max(dim=0)
    col_augmented_ious = torch.zeros_like(row_augmented_ious)
    col_augmented_ious[col_max_indices, torch.arange(m_detections)] = col_max_values

    matched_ious = (col_augmented_ious >= 0.5).int()
    detections = [d for d in range(0, m_detections)]
    matching_dictionary = {}

    for idx_gt in range(0, n_ground_truths):
        compared_to_ground_truth_row = matched_ious[idx_gt]
        largest_idx = torch.argmax(compared_to_ground_truth_row).item()
        corresponding_iou_value = col_augmented_ious[idx_gt][largest_idx].item()
        all_zeros = torch.all(compared_to_ground_truth_row == 0).item()
        if all_zeros:
            matching_dictionary[idx_gt] = None
        else:
            matching_dictionary[idx_gt] = largest_idx, corresponding_iou_value, detected_bbs_tensor[largest_idx]
            detections.remove(largest_idx)
    return matching_dictionary, detections


def precision_and_recall(iou_bb_matcher_result: None | tuple[dict, list]) -> tuple[float, float]:
    '''
    Returns tuple of (precision, recall).
    :param iou_bb_matcher_result: Result of iou_bb_matcher function.
    '''
    if iou_bb_matcher_result is None:
        return 0, 0
    matching_dictionary, unmatched_detections = iou_bb_matcher_result
    true_positives = 0
    false_negatives = 0
    false_positives = len(unmatched_detections)
    for idx_gt in matching_dictionary.keys():
        matched_detection = matching_dictionary.get(idx_gt)
        if matched_detection is None:
            false_negatives += 1
        else:
            true_positives += 1

    return true_positives / (true_positives + false_positives), true_positives / (true_positives + false_negatives)


def draw_and_label_bbs(bbs: list[ObjectLabel], image, color, thickness, id_upper_or_lower):
    for idx, gt_bb in enumerate(bbs):
        start_point = (int(gt_bb.x_min), int(gt_bb.y_min))
        end_point = (int(gt_bb.x_max), int(gt_bb.y_max))
        if id_upper_or_lower:
            id_point = (int(gt_bb.x_min + (gt_bb.x_max - gt_bb.x_min)/2), int(gt_bb.y_min) - 7)
        else:
            id_point = (int(gt_bb.x_min + (gt_bb.x_max - gt_bb.x_min)/2), int(gt_bb.y_max) - 7)
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
        image = cv2.putText(image, str(idx), id_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness, cv2.LINE_AA)
    return image


def bb_drawer(gt_bbs: list[ObjectLabel], detected_bbs_tensor, image):
    red = (0, 0, 255)
    green = (0, 255, 0)
    thickness = 1
    # draw ground truths
    image = draw_and_label_bbs(gt_bbs, image, green, thickness, False)
    # draw detections
    image = draw_and_label_bbs(ObjectLabel.parse_list_from_xyxy_tensor(detected_bbs_tensor),
                               image, red, thickness, True)
    return image


def intersect_calculator(intrinsic_matrix, bb_tensor: Tensor) -> float:
    t = np.array([0, -1.65, 0])
    K_inv = np.linalg.inv(intrinsic_matrix)
    n = np.array([0, 1, 0])
    x_min = bb_tensor[0].item()
    x_max = bb_tensor[2].item()
    y_max = bb_tensor[3].item()
    pixel_position = np.array([(x_max + x_min) / 2, y_max, 1])
    x = np.dot(n, t) / np.dot(n, np.matmul(K_inv, pixel_position))
    world_position = x * np.matmul(K_inv, pixel_position) - t
    return math.sqrt(world_position[0]**2 + world_position[2]**2)


data_set = DataSet.load_from_directory('KITTI_Selection')
model = YOLO("yolo11n.pt")

plot_calculated_distances = []
plot_ground_truth_distances = []

for labeled_image in data_set.labeled_images:
    predicted_result = (model.predict(labeled_image.image_data, classes=[2]))[0]  # filter for cars (class 2)
    matching_result = iou_bb_matcher(labeled_image.object_labels_as_tensor, predicted_result.boxes.xyxy.data)
    print(labeled_image.name)
    print(matching_result)
    if matching_result is not None:
        for idx in matching_result[0].keys():
            match = matching_result[0].get(idx)
            if match is not None:
                bb_tensor = match[2]
                distance = intersect_calculator(labeled_image.intrinsic_matrix, bb_tensor)
                print(f'gt {idx}, calculated {distance}, actual {labeled_image.object_labels[idx].distance}')
                plot_calculated_distances.append(distance)
                plot_ground_truth_distances.append(labeled_image.object_labels[idx].distance)

    precision, recall = precision_and_recall(matching_result)
    print(f'precision {precision}, recall {recall}')
    cv2.imwrite("output/o_" + labeled_image.name + ".png", bb_drawer(labeled_image.object_labels,
                                                                     predicted_result.boxes.xyxy.data,
                                                                     labeled_image.image_data))


plt.plot([0, 70], [0, 70])
plt.plot(plot_ground_truth_distances, plot_calculated_distances, 'bo')
plt.axis((0, 70, 0, 70))
plt.grid(True)
plt.xlabel('Distance calculated using camera information')
plt.ylabel('Distance provided in ground truth')
plt.savefig('output/plot.png')
