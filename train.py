"""
Main file for training Yolo model on Pascal VOC and COCO dataset
"""

import config
import torch
import torch.optim as optim

from model import YOLOv3
from tqdm import tqdm
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples
)
from loss import YoloLoss
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True

def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    loop = tqdm(train_loader, leave=True)
    losses = []
    for batch_idx, (x, y) in enumerate(loop):

        # ! x = (B, 3, 416, 416) -> 입력 이미지
        # ! y = ([B, 3, 13, 13, 6], [B, 3, 26, 26, 6], [B, 3, 52, 52, 6])
        # ! -> 3개의 서로 다른 scale의 feature maps와 각각의 세 anchor boxes에 대해, 
        # ! -> 모든 feature maps(grid cell) positions에서의 gt targets가
        # ! -> (confidence, x, y, w, h, class)의 형태로 저장되어 있음

        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )
        # ! y0 = (B, 3, 13, 13, 6)
        # ! y1 = (B, 3, 26, 26, 6)
        # ! y2 = (B, 3, 52, 52, 6)

        with torch.cuda.amp.autocast():
            # ! 예측값 생성
            # ! model.YOLOv3 참고
            # ! x = (B, 3, 416, 416) -> 입력 이미지
            out = model(x)
            # ! out = [(B, 3, 13, 13, 25), (B, 3, 26, 26, 25), (B, 3, 52, 52, 25)]
            # ! -> 3개의 서로 다른 scale의 feature maps(13x13, 26x26, 52x52)에서 모든 grid cell positions, 모든 3개의 anchor boxes에 대한 예측값이 담긴 리스트

            # ! Loss 계산
            # ! loss.YoloLoss 참고
            # ! out[0] = (B, 3, 13, 13, 25), y0 = (B, 3, 13, 13, 6), scaled_anchors[0] = (3, 2)
            # ! out[1] = (B, 3, 26, 26, 25), y1 = (B, 3, 26, 26, 6), scaled_anchors[1] = (3, 2)
            # ! out[2] = (B, 3, 52, 52, 25), y2 = (B, 3, 52, 52, 6), scaled_anchors[2] = (3, 2)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            ) # ! -> scalar loss value

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)

def main():
    # TODO: 리뷰 시작
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()

    # ! utils.get_loaders 참고
    # ! dataset.YOLODataset 참고
    # ! PASCAL VOC를 사용한다고 가정함
    train_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path=config.DATASET + "/train.csv", test_csv_path=config.DATASET + "/test.csv"
    )

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
        )

    # ! 정의된 9개의 anchor boxes를 feature maps size에 따라 변환
    # ! 0-2의 anchor boxes는 13x13에 맞게 scaling되며, 3-5의 anchor boxes는 26x26에 맞게 scaling되고, 6-8의 anchor boxes는 52x52에 맞게 scaling됨
    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)
    # ! scaled_anchors =
    # ! [
    # !  [
    # !   [ 3.6400,  2.8600],
    # !   [ 4.9400,  6.2400],
    # !   [11.7000, 10.1400]
    # !  ],
    # ! 
    # !  [
    # !   [ 1.8200,  3.9000],
    # !   [ 3.9000,  2.8600],
    # !   [ 3.6400,  7.5400]
    # !  ],
    # ! 
    # !  [
    # !   [ 1.0400,  1.5600],
    # !   [ 2.0800,  3.6400],
    # !   [ 4.1600,  3.1200]
    # !  ]
    # ! ]

    for epoch in range(config.NUM_EPOCHS):
        #plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors)
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)

        #if config.SAVE_MODEL:
        #    save_checkpoint(model, optimizer, filename=f"checkpoint.pth.tar")

        #print(f"Currently epoch {epoch}")
        #print("On Train Eval loader:")
        #print("On Train loader:")
        #check_class_accuracy(model, train_loader, threshold=config.CONF_THRESHOLD)

        if epoch > 0 and epoch % 3 == 0:
            
            # ! Metric 계산
            # ! Class, confidence score 예측 정확도 계산
            # ! utils.check_class_accuracy 참고
            check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)

            # ! mAP 계산을 위해 bbox 예측/변환 수행
            pred_boxes, true_boxes = get_evaluation_bboxes(
                test_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
            )
            # ! pred_boxes = -> (Qp, 7) -> 모든 Q개의 images에 대한 예측값이 (train_idx, class label, confidence score, x, y, w, h)의 형태로 저장되어 있음
            # ! true_boxes = -> (Qt, 7) -> 모든 Q개의 images에 대한 gt label이 (train_idx, class label, confidence score, x, y, w, h)의 형태로 저장되어 있음

            # ! mAP 계산
            # ! utils.mean_average_precision 참고
            mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            ) # ! -> scalar mAP 값
            print(f"MAP: {mapval.item()}")
            model.train()


if __name__ == "__main__":
    main()
