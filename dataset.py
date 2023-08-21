"""
Creates a Pytorch dataset to load the Pascal VOC & MS COCO datasets
"""

import config
import numpy as np
import os
import pandas as pd
import torch

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from utils import (
    cells_to_bboxes,
    iou_width_height as iou,
    non_max_suppression as nms,
    plot_image
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLODataset(Dataset):
    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        anchors,
        image_size=416,
        S=[13, 26, 52],
        C=20,
        transform=None,
    ):
        # ! 정의된 anchors
        # ! ANCHORS = [
        # ! [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
        # ! [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
        # ! [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
        # ! ]
        # ! anchor box의 width, height가 0-1로 normalize되어 있음

        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size # ! 416
        self.transform = transform
        self.S = S # ! [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8] = [416 // 32, 416 // 16, 416 // 8] = [13, 26, 52]
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # ! (9, 2)
        self.num_anchors = self.anchors.shape[0] # ! 9
        self.num_anchors_per_scale = self.num_anchors // 3 # ! 9 // 3 = 3
        self.C = C # ! 클래스 개수 = 20
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        # ! 주어진 이미지에 대한 예측 targets 생성
        # ! FPN을 사용하여 3개의 scale에서 예측을 수행하기 때문에 3개의 targets가 필요함
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]
        # ! targets = [[3, 13, 13, 6], [3, 26, 26, 6], [3, 52, 52, 6]] = (3, 3, 13, 13, 6)
        for box in bboxes:
            
            # ! 현재 선택된 gt bbox와 anchor bboxes와의 iou를 계산
            # ! gt bbox의 좌표는 0-1로 normalized되어 있음
            # ! width와 height만을 이용하여 iou 계산
            # ! utils.iou_width_height 참고
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            # ! iou_anchors = (9,) -> 선택된 gt bbox와 9개의 anchor bboxes와의 IoU값

            # ! IoU를 기준으로 anchor boxes를 내림차순을 정렬한 indices 생성
            anchor_indices = iou_anchors.argsort(descending=True, dim=0) # ! (9,)

            # ! gt bbox 정보 추출
            x, y, width, height, class_label = box
            has_anchor = [False] * 3  # each scale should have one anchor
            # 
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)  # which cell
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i  # both between [0,1]
                    width_cell, height_cell = (
                        width * S,
                        height * S,
                    )  # can be greater than 1 since it's relative to cell
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction

        return image, tuple(targets)


def test():
    anchors = config.ANCHORS

    transform = config.test_transforms

    dataset = YOLODataset(
        "COCO/train.csv",
        "COCO/images/images/",
        "COCO/labels/labels_new/",
        S=[13, 26, 52],
        anchors=anchors,
        transform=transform,
    )
    S = [13, 26, 52]
    scaled_anchors = torch.tensor(anchors) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    for x, y in loader:
        boxes = []

        for i in range(y[0].shape[1]):
            anchor = scaled_anchors[i]
            print(anchor.shape)
            print(y[i].shape)
            boxes += cells_to_bboxes(
                y[i], is_preds=False, S=y[i].shape[2], anchors=anchor
            )[0]
        boxes = nms(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")
        print(boxes)
        plot_image(x[0].permute(1, 2, 0).to("cpu"), boxes)


if __name__ == "__main__":
    test()
