
#todo:
# - confidance is being used to check for objects which are unknown, but also as a measure of confidance in the detection in the baseline. this is not a problem but it sucks as a method.


import csv
import numpy as np
from matplotlib import pyplot as plt
import os

ground_truth_file = "/home/emcastro/Downloads/gt.csv"
pred_file = "/home/emcastro/Downloads/predictions_wrong.csv"
datase_dir = "/home/emcastro/test_daily/"

def read_file(path):
    out = []
    with open(path, "r") as file:
        csv_reader = csv.reader(file, delimiter=',')
        for line in csv_reader:
            out.append(line)
    return out

def get_iou(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)

	# return the intersection over union value
	return iou

def get_subset_gt(ground_truth, classes):
    obj_id = 0
    gt = dict()
    for key in ground_truth.keys():
        for obj in ground_truth[key]:
            if obj[2] in classes:
                if key not in gt:
                    gt[key] = []
                obj_ = list(obj)
                obj_[0] = obj_id
                gt[key].append(tuple(obj_))
                obj_id += 1
    return gt

def get_subset_detections(detections, classes):
    list_dets = []
    for det in detections:
        if det[2] in classes:
            list_dets.append(det)
    return list_dets

def build_curve(ground_truth, detections, IoU_th):

    num_of_objs = 0
    for key in ground_truth:
        for item in ground_truth[key]:
            num_of_objs += 1

    precision = [0.0]
    recall = [0.0]

    already_found_flags = np.ones(num_of_objs)
    TPs = 0
    number_of_dets = 0
    for det in detections:
        img_name = det[0]
        det_bbox = det[1]

        if img_name in ground_truth.keys():
            candidates = ground_truth[img_name]
            for cand in candidates:
                obj_id = cand[0]
                cand_bbox = cand[1]
                iou = get_iou(det_bbox, cand_bbox)
                if (iou >= IoU_th) and already_found_flags[obj_id]:
                    TPs += 1
                    already_found_flags[obj_id] = 0
                    break

        number_of_dets += 1

        precision.append(TPs / number_of_dets)
        recall.append(TPs / num_of_objs)

    precision.append(0.0)
    recall.append(1.0)
    return precision, recall

def process_curve(precision, recall):
    for i in range(len(precision)-2, -1, -1):
        precision[i] = max(precision[i], precision[i+1])

    i_list = []
    for i in range(1, len(recall)):
        if recall[i] != recall[i-1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        ap += ((recall[i]-recall[i-1])*precision[i])
    return ap


def load_gt_and_dets(ground_truth_file, pred_file):
    ground_truth = dict()
    obj_id = 0
    for line in read_file(ground_truth_file):

        img_name = line[0]
        bbox = np.array([float(x) for x in line[1:5]])
        class_ = int(float(line[5]))

        if img_name not in ground_truth:
            ground_truth[img_name] = []

        ground_truth[img_name].append((obj_id, bbox, class_))
        obj_id += 1

    detections = []
    for line in read_file(pred_file):
        img_name = line[0]
        bbox = np.array([float(x) for x in line[1:5]])
        class_ = int(float(line[5]))
        confidence = float(line[6])

        detections.append((img_name, bbox, class_, confidence))
        detections.sort(key=lambda x:-x[3])

    return ground_truth, detections

# Returns:
# - MAP - detection task
# - AP for unknown objects - open set task
# - AP for empty image
def metrics(ground_truth_file, pred_file):

    ground_truth, detections = load_gt_and_dets(ground_truth_file, pred_file)
    classes = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    maps = []
    for c in classes:
        gt = get_subset_gt(ground_truth, [c])
        dets = get_subset_detections(detections, [c])
        aps = []
        for TH in np.arange(.5, 1.0, 0.05):
            precision, recall = build_curve(gt, dets, TH)
            ap = process_curve(precision, recall)
            aps.append(ap)
        maps.append(np.mean(aps))
    MAP = np.mean(maps)

    gt = get_subset_gt(ground_truth, [-1])
    dets = get_subset_detections(detections, [-1])
    aps = []
    for TH in np.arange(.5, 1.0, 0.05):
        precision, recall = build_curve(gt, dets, TH)
        ap = process_curve(precision, recall)
        aps.append(ap)
    AP_unknown = np.mean(aps)

    num_of_objects = 0
    confidence = dict()
    files = [x for x in os.listdir(datase_dir) if x[-4::]==".jpg"]
    empty = dict()
    for file in files:
        if file in ground_truth.keys():
            empty[file] = 0
        else:
            empty[file] = 1
            num_of_objects +=1
        confidence[file] = 0

    for det in detections:
        confidence[det[0]] = max(confidence[det[0]], det[3])

    confidence = sorted(confidence.items(), key=lambda kv: (kv[1], kv[0]), reverse=False)

    TPs = 0
    precision = []
    recall = []
    number_of_dets = 0
    for entry in confidence:
        print(entry)
        if empty[entry[0]]:
            TPs+=1
        number_of_dets += 1

        precision.append(TPs / number_of_dets)
        recall.append(TPs / num_of_objects)

    AP_EMPTY = process_curve(precision, recall)

    return MAP, AP_unknown, AP_EMPTY

values = metrics(ground_truth_file, pred_file)
print(values)
"""
classes = [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9]
gt = get_subset_gt(ground_truth, classes)
dets = get_subset_detections(detections, classes)
precision, recall = build_curve(gt, dets, .5)
#plt.plot(recall, precision)
#plt.xlim(0, 1)
#plt.ylim(0, 1)
ap = process_curve(precision, recall)
#plt.plot(recall, precision)
print(ap)
"""

