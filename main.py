import torch
from loader import DataSet
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou
from torch import Tensor
from loader import ObjectLabel


def iou_bb_matcher(gt_bbs_tensor, detected_bbs_tensor) -> None | tuple[dict, list]:
    '''
    Returns None if there are either no ground truths or detections such that they cannot be matched OR
    returns a dictionary of matches and a list of unmatched detections.
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
        all_zeros = torch.all(compared_to_ground_truth_row == 0).item()
        if all_zeros:
            matching_dictionary[idx_gt] = None
        else:
            matching_dictionary[idx_gt] = largest_idx
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



data_set = DataSet.load_from_directory('KITTI_Selection')
model = YOLO("yolo11n.pt")


for labeled_image in data_set.labeled_images:
    ground_truth_labels = labeled_image.object_labels
    ground_truth_bbs = []
    for gtl in ground_truth_labels:
        ground_truth_bbs.append([gtl.x_min, gtl.y_min, gtl.x_max, gtl.y_max])
    ground_truth_bbs_tensor = Tensor(ground_truth_bbs)
    print(labeled_image.name)
    predicted_result = (model.predict(labeled_image.image_data, classes=[2]))[0]  # filter for cars (class 2)

    matching_result = iou_bb_matcher(ground_truth_bbs_tensor, predicted_result.boxes.xyxy.data)
    print(matching_result)
    precision, recall = precision_and_recall(matching_result)
    print(f'precision {precision}, recall {recall}')

    #predicted_result.save("output/o_" + labeled_image.name + ".png")

#results = model(data_set.labeled_images[0].image_data)
#results = model.predict(data_set.labeled_images[0].image_data, classes=[2])
#print(data_set.labeled_images[0].name)

#for result in results:
#    boxes = result.boxes
#    masks = result.masks
#    keypoints = result.keypoints
#    probs = result.probs
#    result.show()
