from loader import DataSet
from ultralytics import YOLO
from torch import Tensor
from loader import ObjectLabel

def iou_bb_matcher(ground_truth_labels: list[ObjectLabel], boxes_yxyx_data):
    bbs = []
    for detected_bb in boxes_yxyx_data:
        bbs.append(detected_bb)
    print(bbs)

data_set = DataSet.load_from_directory('KITTI_Selection')
model = YOLO("yolo11n.pt")


for labeled_image in data_set.labeled_images:
    ground_truths_labels = labeled_image.object_labels
    predicted_result = (model.predict(labeled_image.image_data, classes=[2]))[0]  # filter for cars (class 2)
    iou_bb_matcher(ground_truths_labels, predicted_result.boxes.xyxy.data)
    predicted_result.save("output/o_" + labeled_image.name + ".png")

#results = model(data_set.labeled_images[0].image_data)
#results = model.predict(data_set.labeled_images[0].image_data, classes=[2])
#print(data_set.labeled_images[0].name)

#for result in results:
#    boxes = result.boxes
#    masks = result.masks
#    keypoints = result.keypoints
#    probs = result.probs
#    result.show()
