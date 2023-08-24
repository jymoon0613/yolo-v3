"""
Implementation of Yolo Loss Function similar to the one in Yolov3 paper,
the difference from what I can tell is I use CrossEntropy for the classes
instead of BinaryCrossEntropy.
"""
import random
import torch
import torch.nn as nn

from utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Constants signifying how much to pay for each respective part of the loss
        # ! Loss 가중치 설정
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, predictions, target, anchors):

        # ! predictions = (B, 3, S, S, 25) -> SxS feature maps에서 예측된 결과
        # ! target      = (B, 3, S, S, 6)     -> SxS feature maps에 대한 targets
        # ! anchors     = (3, 2)           -> SxS feature maps에 대한 scaled anchor boxes

        # Check where obj and noobj (we ignore if target == -1)
        # ! 모든 positions에서, target object가 할당된 anchor boxes 식별
        # ! obj가 True면 object가 할당되었음을 의미
        # ! noobj가 True면 object가 할당되지 않았음을 의미
        obj = target[..., 0] == 1  # in paper this is Iobj_i
        noobj = target[..., 0] == 0  # in paper this is Inoobj_i

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        # ! Object가 존재하지 않는 경우 confidence score에 대한 loss 계산
        # ! Binary cross entropy 사용
        no_object_loss = self.bce(
            # ! 예측값과 gt_label에서 noobj에 해당하는 confidence score 값만 추출함
            # ! predictions[..., 0:1][noobj] -> 모든 grid cell positions에서 object가 할당되지 않은 anchor boxes에 대한 모델의 confidence score 예측값
            # ! target[..., 0:1][noobj]      -> 모든 grid cell positions에서 object가 할당되지 않은 anchor boxes에 대한 gt confidence score 값 (= 0)
            (predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]),
        ) # ! -> scalar loss

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        # ! Object가 존재하는 경우 confidence score에 대한 loss 계산
        # ! Mean squared error 사용
        anchors = anchors.reshape(1, 3, 1, 1, 2)

        # ! IoU 계산을 위해 bbox 예측 좌표 변환
        # ! bbox 예측값 (tx, ty, tw, th)는 각각
        # ! x = sigmoid(tx),
        # ! y = sigmoid(ty),
        # ! w = anchor_width * exp(tw),
        # ! h = anchor_height * exp(th),
        # ! 로 변환되어 예측 bbox (x, y, w, h)로 변환됨
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1) # ! (B, 3, S, S, 4)

        # ! IoU를 계산함
        # ! box_preds[obj]        = (n, 4) -> 모든 grid cell positions에서 object가 할당된 anchors에 대한 예측 bbox 좌표
        # ! target[..., 1:5][obj] = (n, 4) -> 모든 grid cell positions에서 object가 할당된 anchors에 대한 gt_box 좌표
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        # ! ious = (n, 1) n개의 예측 bbox와 gt bbox 간의 IoU가 저장되어 있음

        # ! Confidence score에 대한 loss 계산
        # ! Object가 존재하는 경우, target confidence score는 gt bbox와의 IoU 값이 됨
        # ! self.sigmoid(predictions[..., 0:1][obj]) -> 모든 grid cell positions에서 object가 할당된 anchor boxes에 대한 모델의 confidence score 예측값
        # ! target[..., 0:1][obj]                    -> 모든 grid cell positions에서 object가 할당된 anchor boxes에 대한 gt confidence score 값 (= 1)
        # ! ious * target[..., 0:1][obj]             -> 모든 grid cell positions에서 object가 할당된 anchor boxes에 대해 gt_bbox와 예측 bbox의 IoU값
        object_loss = self.mse(self.sigmoid(predictions[..., 0:1][obj]), ious * target[..., 0:1][obj]) # ! -> scalar loss

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        # ! Bbox regression 수행
        # ! 예측값 변환
        # ! tx = sigmoid(tx),
        # ! ty = sigmoid(ty),
        # ! tw = log(w) / anchor_width,
        # ! th = log(h) / anchor_height,
        # ! 을 의미하므로, 이에 맞게 예측값, targets 변환
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])  # x,y coordinates
        target[..., 3:5] = torch.log(
            (1e-16 + target[..., 3:5] / anchors)
        )  # width, height coordinates

        # ! Object가 존재하는 경우에 대해서만 bbox loss 계산
        # ! Mean squared error 사용
        # ! predictions[..., 1:5][obj] -> 모든 grid cell positions에서 object가 할당된 anchors에 대한 예측 bbox label
        # ! target[..., 1:5][obj]      -> 모든 grid cell positions에서 object가 할당된 anchors에 대한 gt bbox label
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj]) # ! -> scalar loss

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        # ! Classification loss 계산
        # ! Object가 존재하는 경우에 대해서만 classification loss 계산
        # ! Cross entropy 사용
        # ! predictions[..., 5:][obj] -> 모든 grid cell positions에서 object가 할당된 anchors에 대한 class 예측값 (one-hot)
        # ! target[..., 5][obj]       -> 모든 grid cell positions에서 object가 할당된 anchors에 대한 gt class
        class_loss = self.entropy(
            (predictions[..., 5:][obj]), (target[..., 5][obj].long()),
        ) # ! -> scalar loss

        #print("__________________________________")
        #print(self.lambda_box * box_loss)
        #print(self.lambda_obj * object_loss)
        #print(self.lambda_noobj * no_object_loss)
        #print(self.lambda_class * class_loss)
        #print("\n")

        # ! loss 가중치 적용하고 return -> scalar loss

        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )
