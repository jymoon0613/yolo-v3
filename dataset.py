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
        # ! 정의된 anchor boxes
        # ! ANCHORS = [
        # ! [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
        # ! [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
        # ! [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
        # ! ]
        # ! anchor box의 width, height가 0-1로 normalize되어 있음
        # ! anchor boxes는 세 가지 scale로 구분되어 있음 (ANCHORS[0], ANCHORS[1], ANCHORS[2])

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
        # ! FPN을 사용하여 3개의 서로 다른 scale의 feature maps에서 각각 예측을 수행하기 때문에 하나의 이미지마다 총 3개의 targets가 필요함
        # ! 생성 결과를 저장할 list of tensors 선언
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]
        # ! targets = [[3, 13, 13, 6], [3, 26, 26, 6], [3, 52, 52, 6]] = (3, 3, 13, 13, 6)
        for box in bboxes: # ! 이미지 내의 모든 bbox에 대해 반복

            # ! 현재 선택된 gt bbox와 anchor bboxes와의 iou를 계산
            # ! gt bbox의 좌표는 0-1로 normalized되어 있음
            # ! width와 height만을 이용하여 iou 계산
            # ! utils.iou_width_height 참고
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            # ! iou_anchors = (9,) -> 선택된 gt bbox와 9개의 anchor bboxes와의 IoU값

            # ! IoU를 기준으로 anchor boxes를 내림차순으로 정렬
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            # ! anchor_indices = (9,) 내림차순 정렬했을 때의 anchor box indices

            # ! gt bbox 정보 추출
            x, y, width, height, class_label = box
            # ! gt bbox는 필수적으로 3개의 scale anchor boxes와 대응됨
            has_anchor = [False] * 3  # each scale should have one anchor
            
            # ! 모든 anchor boxes에 대해 반복하되, 선택된 gt box와의 IoU가 가장 큰 anchor box부터 선택
            for anchor_idx in anchor_indices:
                # ! 선택된 anchor box가 어떤 scale range에 속하는지 식별
                # ! 현재 총 9개의 anchor boxes가 존재하고, 이는 0-2, 3-5, 6-8의 세 가지 scale range로 구분되어 있음
                # ! 따라서, scale_idx는 0,1,2 중 하나의 값을 가지며, 선택된 anchor box가 어떤 scale에 속하는 anchor인지를 식별함
                scale_idx = anchor_idx // self.num_anchors_per_scale

                # ! anchor_on_scale은 선택된 scale range에서 정확히 몇 번째에 위치한 anchor box인지를 식별함
                # ! ex. scale_idx = 1, anchor_on_scale = 2 -> 선택된 anchor box는 첫 번째 scale range에 속하며 그 중에서도 3번째 임 -> anchor_idx = 5
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale

                # ! 대상 feature scale 선택
                # ! 큰 scale의 anchor boxes인 anchor_idx = 0-2(scale_idx=0)는 13x13의 feature maps에서 사용되고,
                # ! 중간 scale의 anchor boxes인 anchor_idx = 3-5(scale_idx=1)는 26x26의 feature maps에서 사용되고,
                # ! 가장 작은 scale의 anchor boxes인 anchor_idx = 3-5(scale_idx=2)는 52x52의 feature maps에서 사용됨
                S = self.S[scale_idx]
                # ! S = [3, S, S, 6] -> SxS resolution의 feature maps를 위해 정의된 target tensor 선택

                # ! gt bbox가 어떤 grid cell에 속하는지 식별
                # ! (i,j)는 SxS의 feature maps(grid cell)의 한 position을 의미함
                # ! 0 <= i,j <= S-1
                i, j = int(S * y), int(S * x)  # which cell

                # ! gt box의 position i,j에서 같은 scale, 종류의 anchor box에 대해 target이 할당되었는지를 식별
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                # ! 만약 gt box의 position i,j에서 같은 scale, 종류의 anchor box에 대해 target이 할당되지 않았고, 
                # ! 현재 gt box가 같은 scale에 대한 예측 target이 없는 경우 예측값을 할당
                if not anchor_taken and not has_anchor[scale_idx]:

                    # ! position (i,j)에 주어진 anchor box에 대한 object가 존재한다는 것을 지시
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1

                    # ! bbox regression 예측 target 생성
                    # ! x_cell, y_cell은 gt box가 속하는 grid cell에서의 상대적인 위치를 의미 (0-1)
                    x_cell, y_cell = S * x - j, S * y - i  # both between [0,1]

                    # ! width_cell, height_cell은 grid cell 기준의 gt box width, height를 의미
                    width_cell, height_cell = (
                        width * S,
                        height * S,
                    )  # can be greater than 1 since it's relative to cell
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )

                    # ! target 값 저장
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                # ! 만약 gt box의 position i,j에서 같은 scale, 종류의 anchor box에 대해 target이 할당되지 않았고,
                # ! gt bbox와의 iou가 
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction

        # ! image          = (3, 416, 416) -> 입력 이미지
        # ! tuple(targets) = ([3, 13, 13, 6], [3, 26, 26, 6], [3, 52, 52, 6]) = (3, 3, 13, 13, 6)
        # ! 3개의 서로 다른 scale의 feature maps와 각각의 세 anchor boxes에 대해,
        # ! 모든 feature maps(grid cell) positions에서의 예측 targets가
        # ! (confidence, x, y, w, h, class)의 형태로 저장되어 있음

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
