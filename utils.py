import config
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import random
import torch

from collections import Counter
from torch.utils.data import DataLoader
from tqdm import tqdm


def iou_width_height(boxes1, boxes2):
    """
    Parameters:
        boxes1 (tensor): width and height of the first bounding boxes
        boxes2 (tensor): width and height of the second bounding boxes
    Returns:
        tensor: Intersection over union of the corresponding boxes
    """

    # ! 두 bbox의 normalized width, height만을 사용하여 IoU 계산
    # ! 두 bbox의 중심좌표가 동일하다고 가정한다면 (i.e., (0,0)), normalized width, height만으로 쉽게 IoU를 계산할 수 있음
    # ! boxes1 = (2,)  -> 선택된 gt_bbox의 normalized width, height
    # ! boxes2 = (9,2) -> 9개 anchor boxes의 normalized width, height

    # ! intersection 계산
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    ) # ! (9,) -> 선택된 gt_bbox와 9개의 anchor boxes와의 intersection이 계산됨

    # ! union 계산
    union = (
        boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    ) # ! (9,) -> 선택된 gt_bbox와 9개의 anchor boxes와의 union이 계산됨

    # ! intersection / union = (9,) -> 선택된 gt_bbox와 9개의 anchor boxes와의 IoU가 계산되어 출력됨

    return intersection / union


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Video explanation of this function:
    https://youtu.be/XXYG5ZWtjj0

    This function calculates intersection over union (iou) given pred boxes
    and target boxes.

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Video explanation of this function:
    https://youtu.be/YDkjWEN8jNA

    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    """
    Video explanation of this function:
    https://youtu.be/FppOzcDvaDI

    This function calculates mean average precision (mAP)

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """

    # ! 주어진 IoU threshold 하에서 mAP를 계산
    # ! all_pred_boxes = (Qp, 7) -> 모든 Q개의 images에 대한 예측값이 (train_idx, class label, confidence score, x, y, w, h)의 형태로 저장되어 있음
    # ! all_true_boxes = (Qt, 7) -> 모든 Q개의 images에 대한 gt label이 (train_idx, class label, confidence score, x, y, w, h)의 형태로 저장되어 있음
    # ! iou_threshold  = 0.5
    # ! box_format     = "midpoint"
    # ! num_classes    = 20

    # list storing all AP for respective classes
    # ! 모든 classes에 대한 AP를 저장하기 위한 리스트 선언 
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    # ! 모든 classes에 대해 각각 AP를 계산
    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        # ! 선택된 class에 속하는 예측값 식별
        # ! n개라고 가정
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection) # ! (n, 7)

        # ! 선택된 class에 속하는 gt label 식별
        # ! m개라고 가정
        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box) # ! (m, 7)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        # ! 각 이미지마다 존재하는 gt label의 수 계산
        # ! 즉, 각 이미지에 주어진 class에 해당하는 gt label이 몇 개 있는지 계산
        # ! {0:3, 1:5, ...} -> 0번 이미지 3개, 1번 이미지 5개, ...
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        # ! Counter dictionary의 형식 변경
        # ! 만약 i번째 이미지에 j개의 gt label이 존재하는 경우,
        # ! dictionary의 i번째 key의 values를 j차원의 zero tensor로 변경
        # ! {0:3, 1:5, ...} -> {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0], ...}
        # ! 이러한 변경을 해주는 이유는 하나의 gt bbox에 대해 여러 개의 predictions가 존재할 수 있기 때문임
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        # ! n개의 예측값을 confidence score를 기준으로 내림차순 정렬
        detections.sort(key=lambda x: x[2], reverse=True) # ! (n, 7)

        # ! TP, FP 저장을 위한 tensor 정의
        TP = torch.zeros((len(detections))) # ! (n,)
        FP = torch.zeros((len(detections))) # ! (n,)
        total_true_bboxes = len(ground_truths) # ! gt label의 개수 = m

        # If none exists for this class then we can safely skip
        # ! 만약 해당 class의 gt label이 하나도 존재하지 않는 경우 계산을 skip
        if total_true_bboxes == 0:
            continue
        
        # ! 모든 정렬된 예측값에 대해 반복
        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            # ! 선택된 예측 bbox와 같은 이미지에 속한 gt label만 선택
            # ! k개라고 가정
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ] # ! (k, 7)

            num_gts = len(ground_truth_img) # ! k
            best_iou = 0

            # ! 선택된 예측 bbox와 같은 이미지에 속한 gt label이 없는 경우 for문은 실행되지 않으며,
            # ! best_iou는 0이 되어 해당 예측은 FP로 분류됨
            for idx, gt in enumerate(ground_truth_img):
                # ! 모든 k개의 gt label과 IoU 계산
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                # ! 가장 IoU가 큰 gt label의 index 저장
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            # ! IoU threshold를 만족하는 경우
            if best_iou > iou_threshold:
                # only detect ground truth detection once
                # ! 만약 해당 gt label을 detect한 예측 bbox가 이전에 없었다면,
                # ! 현재 예측값은 TP로 분류됨
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1

                # ! 만약 해당 gt label을 detect한 예측 bbox가 이전에 있었다면,
                # ! 이는 현재 예측값보다 높은 confidence로 해당 gt label을 예측한 결과가 있었다는 뜻이므로,
                # ! 현재 예측값은 FP로 분류됨
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            # ! IoU threshold를 만족하지 못하는 경우 FP로 처리함
            else:
                FP[detection_idx] = 1

        # ! TP와 FP의 누적합(cumulative sum)을 계산함
        # ! 예측값은 confidence scores를 기준으로 내림차순 정렬되어 있으므로,
        # ! Confidence scores를 점차 낮춰갈 때 모든 confidence level에서의 TP와 FP를 한 번에 계산
        # ! ex. [1, 1, 0, 1, 0, ...] -> [1, 2, 2, 3, 3, ...]
        TP_cumsum = torch.cumsum(TP, dim=0) # ! (n,)
        FP_cumsum = torch.cumsum(FP, dim=0) # ! (n,)

        # ! recall 및 precision 계산
        recalls = TP_cumsum / (total_true_bboxes + epsilon) # ! (n,)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon) # ! (n,)

        # ! PR curve 아래 영역 넓이(AP)를 계산(적분)하고 저장
        precisions = torch.cat((torch.tensor([1]), precisions)) # ! (n+1,)
        recalls = torch.cat((torch.tensor([0]), recalls)) # ! (n+1,)
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    # ! mAP를 출력함
    # ! = 모든 classes의 AP 평균 = sum(average_precisions) / len(average_precisions)
    # ! 현재의 예시는 고정된 IoU threshold를 사용함 (i.e., 0.5)
    # ! 따라서 현재 계산된 mAP는 mAP@0.5임
    # ! 고정된 IoU threshold를 사용하는 대신 IoU threshold를 변경하면서 mAP를 구하고, 이를 평균한 mAP를 사용할 수도 있음
    # ! e.g., mAP@0.5:0.05:0.95 -> IoU threshold를 0.5에서 0.95까지 0.05씩 증가시키면서 각각 평가했을 때 평균적인 mAP 값

    return sum(average_precisions) / len(average_precisions)


def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    cmap = plt.get_cmap("tab20b")
    class_labels = config.COCO_LABELS if config.DATASET=='COCO' else config.PASCAL_CLASSES
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle patch
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.text(
            upper_left_x * width,
            upper_left_y * height,
            s=class_labels[int(class_pred)],
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )

    plt.show()


def get_evaluation_bboxes(
    loader,
    model,
    iou_threshold,
    anchors,
    threshold,
    box_format="midpoint",
    device="cuda",
):
    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []
    for batch_idx, (x, labels) in enumerate(tqdm(loader)):

        # ! x      = (B, 3, 416, 416) -> 입력 이미지
        # ! labels = ([B, 3, 13, 13, 6], [B, 3, 26, 26, 6], [B, 3, 52, 52, 6])
        # ! -> 3개의 서로 다른 scale의 feature maps와 각각의 세 anchor boxes에 대해, 
        # ! -> 모든 feature maps(grid cell) positions에서의 gt targets가
        # ! -> (confidence, x, y, w, h, class)의 형태로 저장되어 있음

        x = x.to(device)

        with torch.no_grad():
            predictions = model(x)
            # ! predictions = [(B, 3, 13, 13, 25), (B, 3, 26, 26, 25), (B, 3, 52, 52, 25)]
            # ! -> 3개의 서로 다른 scale의 feature maps(13x13, 26x26, 52x52)에서 모든 grid cell positions, 모든 3개의 anchor boxes에 대한 예측값이 담긴 리스트

        batch_size = x.shape[0] # ! B

        # ! 결과를 저장할 리스트 선언 
        bboxes = [[] for _ in range(batch_size)] # ! (B,)

        # ! 모든 scale에 대해 bbox 변환
        for i in range(3):

            # ! predictions[i] = (B, 3, S, S, 25) -> i번째 scale에 대한 예측값
            S = predictions[i].shape[2] # ! S

            # ! i번째 scale에 해당하는 anchor boxes를 추출하고 width와 height를 scaling해줌
            anchor = torch.tensor([*anchors[i]]).to(device) * S # ! (3, 2)

            # ! 예측 bbox를 변환함
            # ! 예측 bboxes는 grid 혹은 grid cell에 대해 상대적으로 표현되어 있음
            # ! 이를 원본 이미지 차원의 값으로 변경해줌
            # ! 또한 각 grid cell마다 최적의 예측값을 선정하는 과정을 수행
            # ! predictions[i] = (B, 3, S, S, 25)
            # ! anchor         = (3, 2)
            boxes_scale_i = cells_to_bboxes(
                predictions[i], anchor, S=S, is_preds=True
            )
            # ! boxes_scale_i = (B, 3*S*S, 6) -> 모든 SxS grid cell positions와 각각의 3개의 anchor boxes에 대한 예측 정보가
            # ! -> (class label, confidence score, x, y, w, h)의 형태로 저장되어 있음

            # ! 결과 저장
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        # we just want one bbox for each label, not one for each scale
        # ! bboxes = (B, 3*3*S*S, 6) -> 모든 이미지에 대해, 모든 scale의 예측값을 저장

        # ! gt_box를 변환함
        # ! 이때 gt_box는 52x52 resolution의 targets에 대해서만 변환함
        # ! predictions[i] = (B, 3, S, S, 6)
        # ! anchor         = (3, 2)        
        true_bboxes = cells_to_bboxes(
            labels[2], anchor, S=S, is_preds=False
        )
        # ! true_bboxes = (B, 3*S*S, 6) -> 모든 SxS grid cell positions와 각각의 3개의 anchor boxes에 gt label 정보가
        # ! -> (class label, confidence score, x, y, w, h)의 형태로 저장되어 있음

        # ! NMS 수행
        for idx in range(batch_size):
            
            # ! 각 이미지의 예측값에 NMS 적용
            # ! bboxes[idx]   = (3*3*S*S, 6)
            # ! iou_threshold = 0.45
            # ! threshold     = 0.05
            # ! box_format    = "midpoint"
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )
            # ! NMS 이후 p개의 예측값이 남았다고 가정
            # ! nms_boxes = (p, 6)

            # ! p개의 예측 bboxes에 train_idx를 패딩하고 저장함
            # ! train_idx는 이미지 index를 의미
            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            # ! gt bboxes에 train_idx를 패딩하고 저장함
            # ! train_idx는 이미지 index를 의미
            # ! 이때 gt confidence score가 특정 threshold(0.05)보다 큰 경우만 저장함
            # ! 즉, 할당된 object가 있는 grid cell의 gt label (class label, confidence score, x, y, w, h)만 저장함
            # ! t개라고 가정함
            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()

    # ! 모든 Q개의 이미지마다 p개의 예측값, t개의 gt label이 저장되었다고 가정함
    # ! 최종적으로 모든 Q개의 images에 대한 예측값, gt label이 (train_idx, class label, confidence score, x, y, w, h)의 형태로 출력됨
    # ! 이때 train_idx는 예측 혹은 gt label이 속하는 이미지의 index임
    # ! all_pred_boxes = (Qp, 7)
    # ! all_true_boxes = (Qt, 7)

    return all_pred_boxes, all_true_boxes


def cells_to_bboxes(predictions, anchors, S, is_preds=True):
    """
    Scales the predictions coming from the model to
    be relative to the entire image such that they for example later
    can be plotted or.
    INPUT:
    predictions: tensor of size (N, 3, S, S, num_classes+5)
    anchors: the anchors used for the predictions
    S: the number of cells the image is divided in on the width (and height)
    is_preds: whether the input is predictions or the true bounding boxes
    OUTPUT:
    converted_bboxes: the converted boxes of sizes (N, num_anchors, S, S, 1+5) with class index,
                      object score, bounding box coordinates
    """

    # ! 예측/gt bboxes는 grid 혹은 grid cell에 대해 상대적으로 표현되어 있음
    # ! 이를 원본 이미지 차원의 값으로 변경해줌
    # ! 또한 각 grid cell마다 최적의 예측값을 선정하거나, 할당된 gt label을 식별하는 연산 수행

    # ! predictions[i] = (B, 3, S, S, 25) or (B, 3, S, S, 6) -> i번째 scale에 대한 예측값 or gt targets
    # ! anchor         = (3, 2)                              -> i번째 scale에 대한 anchor boxes (S에 따라 scaling되어 있음)

    BATCH_SIZE = predictions.shape[0] # ! B
    num_anchors = len(anchors) # ! 3

    # ! bbox 좌표 추출
    box_predictions = predictions[..., 1:5] # ! (B, 3, S, S, 4)

    # ! 만약 현재 입력이 예측값이면
    if is_preds:
        
        # ! anchor boxes padding
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2) # ! (1, 3, 1, 1, 2)

        # ! 예측 bbox 좌표 변환
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        # ! box_predictions = (B, 3, S, S, 4)

        # ! 예측 confidence score 계산
        scores = torch.sigmoid(predictions[..., 0:1])
        # ! scores = (B, 3, S, S, 1)

        # ! 예측 class label 식별
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
        # ! best_class = (B, 3, S, S, 1)
    
    # ! 만약 현재 입력이 gt targets이면
    else:
        # ! bbox 좌표 변환 필요 없음

        # ! gt confidence score 저장
        scores = predictions[..., 0:1]
        # ! scores = (B, 3, S, S, 1)

        # ! gt class label 식별
        best_class = predictions[..., 5:6]
        # ! scores = (B, 3, S, S, 1)

    # ! 좌표 변환을 위한 cell index 정의
    cell_indices = (
        torch.arange(S)
        .repeat(predictions.shape[0], 3, S, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    ) # ! (B, 3, S, S, 1)

    # ! 현재 예측/gt bboxes를 구성하는 x, y는 해당 bbox가 속하는 grid cell 내에서의 상대적인 위치로 표현되어 있음 (0-1)
    # ! box_predictions[..., :2] = S * (x,y) - cell_indices
    # ! 이를 다시 원본 이미지에 대해 normalize된 좌표로 변경 (0-1)
    # ! (x,y) = (x,y + cell_indices) / S
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices) # !  (B, 3, S, S, 1)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4)) # ! (B, 3, S, S, 1)

    # ! 마찬가지로 현재 예측/gt bboxes를 구성하는 w, h는 SxS grid size를 기준으로 계산한 bbox의 상대적 width, height임
    # ! box_predictions[..., 2:4] = S * (w,h)
    # ! 이를 다시 원본 이미지에 대해 normalize된 w, h로 변경 (0-1)
    # ! (w,h) = (w,h) / S
    w_h = 1 / S * box_predictions[..., 2:4] # ! (B, 3, S, S, 2)

    # ! 변환 결과를 저장
    converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 6)
    # ! converted_bboxes = (B, 3*S*S, 6) -> 모든 SxS grid cell positions와 각각의 3개의 anchor boxes에 대한 예측/bbox 정보가
    # ! -> (class label, confidence score, x, y, w, h)의 형태로 저장되어 있음

    return converted_bboxes.tolist()

def check_class_accuracy(model, loader, threshold):
    model.eval()
    tot_class_preds, correct_class = 0, 0
    tot_noobj, correct_noobj = 0, 0
    tot_obj, correct_obj = 0, 0

    for idx, (x, y) in enumerate(tqdm(loader)):

        # ! x = (B, 3, 416, 416) -> 입력 이미지
        # ! y = ([B, 3, 13, 13, 6], [B, 3, 26, 26, 6], [B, 3, 52, 52, 6])
        # ! -> 3개의 서로 다른 scale의 feature maps와 각각의 세 anchor boxes에 대해, 
        # ! -> 모든 feature maps(grid cell) positions에서의 gt targets가
        # ! -> (confidence, x, y, w, h, class)의 형태로 저장되어 있음

        x = x.to(config.DEVICE)
        with torch.no_grad():
            out = model(x)
            # ! out = [(B, 3, 13, 13, 25), (B, 3, 26, 26, 25), (B, 3, 52, 52, 25)]
            # ! -> 3개의 서로 다른 scale의 feature maps(13x13, 26x26, 52x52)에서 모든 grid cell positions, 모든 3개의 anchor boxes에 대한 예측값이 담긴 리스트

        # ! 모든 scale의 예측값에 대해 accuracy 계산
        for i in range(3):
            
            # ! 선택된 scale의 gt targets
            y[i] = y[i].to(config.DEVICE) # ! (B, 3, S, S, 6)

            # ! 모든 grid cell positions, 모든 anchor boxes에 대해 object가 할당된 anchor boxes 식별
            # ! True면 object가 할당된 anchor box
            obj = y[i][..., 0] == 1 # in paper this is Iobj_i # ! (B, 3, S, S, 6)

            # ! 모든 grid cell positions, 모든 anchor boxes에 대해 object가 할당되지 않은 anchor boxes 식별
            # ! True면 object가 할당되지 않은 anchor box
            noobj = y[i][..., 0] == 0  # in paper this is Iobj_i # ! (B, 3, S, S, 6)

            # ! Class 예측 정확도 계산
            # ! torch.argmax(out[i][..., 5:][obj], dim=-1) -> object가 할당된 anchor boxes에서 예측된 class
            # ! y[i][..., 5][obj]                          -> object가 할당된 anchor boxes의 gt class label
            # ! correct_class                              -> 맞춘 개수
            correct_class += torch.sum(
                torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]
            )

            # ! object가 할당된 anchor boxes의 수 = 전체 예측 수
            tot_class_preds += torch.sum(obj)

            # ! Confidence score에 대한 예측 정확도 계산
            # ! 예측된 confidence score 식별
            # ! 만약 예측된 confidence score가 특정 threshold(0.05)보다 크면 object가 존재한다고 예측한 것으로 판별 (obj_preds = True)
            obj_preds = torch.sigmoid(out[i][..., 0]) > threshold

            # ! object가 할당된 anchor boxes에 대한 confidence 예측 정확도 측정
            # ! 예측 targets = 1
            correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
            tot_obj += torch.sum(obj)

            # ! object가 할당되지 않은 anchor boxes에 대한 confidence 예측 정확도 측정
            # ! 예측 targets = 0
            correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
            tot_noobj += torch.sum(noobj)

    # ! 결과 출력
    print(f"Class accuracy is: {(correct_class/(tot_class_preds+1e-16))*100:2f}%")
    print(f"No obj accuracy is: {(correct_noobj/(tot_noobj+1e-16))*100:2f}%")
    print(f"Obj accuracy is: {(correct_obj/(tot_obj+1e-16))*100:2f}%")
    model.train()


def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_loaders(train_csv_path, test_csv_path):
    from dataset import YOLODataset

    IMAGE_SIZE = config.IMAGE_SIZE
    train_dataset = YOLODataset(
        train_csv_path,
        transform=config.train_transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
    )
    test_dataset = YOLODataset(
        test_csv_path,
        transform=config.test_transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    train_eval_dataset = YOLODataset(
        train_csv_path,
        transform=config.test_transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
    )
    train_eval_loader = DataLoader(
        dataset=train_eval_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, test_loader, train_eval_loader

def plot_couple_examples(model, loader, thresh, iou_thresh, anchors):
    model.eval()
    x, y = next(iter(loader))
    x = x.to("cuda")
    with torch.no_grad():
        out = model(x)
        bboxes = [[] for _ in range(x.shape[0])]
        for i in range(3):
            batch_size, A, S, _, _ = out[i].shape
            anchor = anchors[i]
            boxes_scale_i = cells_to_bboxes(
                out[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        model.train()

    for i in range(batch_size):
        nms_boxes = non_max_suppression(
            bboxes[i], iou_threshold=iou_thresh, threshold=thresh, box_format="midpoint",
        )
        plot_image(x[i].permute(1,2,0).detach().cpu(), nms_boxes)



def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
