"""
Fichier d'entraînement du YOLOv3 sur le dataset Pascal VOC
"""

import config
import torch
import torch.optim as optim
import time
from datetime import timedelta

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
from metrics_logger import MetricsLogger
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True


def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    loop = tqdm(train_loader, leave=True)
    losses = []
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # mise à jour de la barre de progression
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)
    
    return mean_loss



def main():
    # initialisation du logger de métriques
    logger = MetricsLogger(experiment_name=f"yolov3_pascal_voc_{int(time.time())}")
    
    # logging de la configuration
    config_dict = {
        "DATASET": config.DATASET,
        "DEVICE": config.DEVICE,
        "BATCH_SIZE": config.BATCH_SIZE,
        "NUM_EPOCHS": config.NUM_EPOCHS,
        "LEARNING_RATE": config.LEARNING_RATE,
        "WEIGHT_DECAY": config.WEIGHT_DECAY,
        "IMAGE_SIZE": config.IMAGE_SIZE,
        "NUM_CLASSES": config.NUM_CLASSES,
        "CONF_THRESHOLD": config.CONF_THRESHOLD,
        "NMS_IOU_THRESH": config.NMS_IOU_THRESH,
        "MAP_IOU_THRESH": config.MAP_IOU_THRESH
    }
    logger.log_config(config_dict)
    
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()

    train_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path=config.DATASET + "/train.csv", test_csv_path=config.DATASET + "/test.csv"
    )

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
        )

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    # tracking du meilleur mAP
    best_map = 0.0
    best_epoch = 0
    start_time = time.time()

    for epoch in range(config.NUM_EPOCHS):
        epoch_start = time.time()
        print(f"\n=== Epoch {epoch+1}/{config.NUM_EPOCHS} ===")
        
        # entraînement
        mean_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)
        
        epoch_time = time.time() - epoch_start
        print(f"⏱️  Temps d'epoch: {timedelta(seconds=int(epoch_time))}")

        # sauvegarde du checkpoint à chaque epoch
        if config.SAVE_MODEL:
            save_checkpoint(model, optimizer, filename=f"checkpoint.pth.tar")
            print(f"✅ Checkpoint saved")

        # évaluation à chaque 3 epochs (sur un sous-ensemble pour rapidité)
        test_metrics = {}
        if epoch > 0 and epoch % 3 == 0:
            print("\n--- Evaluation (20 premiers batches) ---")
            
            # évaluation rapide sur sous-ensemble
            model.eval()
            pred_boxes, true_boxes = [], []
            train_idx = 0
            
            with torch.no_grad():
                for batch_idx, (x, labels) in enumerate(test_loader):
                    if batch_idx >= 20:  # limiter à 20 batches (~640 images)
                        break
                    
                    x = x.to(config.DEVICE)
                    predictions = model(x)
                    
                    batch_size = x.shape[0]
                    bboxes = [[] for _ in range(batch_size)]
                    
                    # obtenir les bboxes de toutes les échelles
                    for i in range(3):
                        S = predictions[i].shape[2]
                        anchor = torch.tensor([*config.ANCHORS[i]]).to(config.DEVICE) * S
                        boxes_scale_i = cells_to_bboxes(
                            predictions[i], anchor, S=S, is_preds=True
                        )
                        for idx, box in enumerate(boxes_scale_i):
                            bboxes[idx] += box
                    
                    # bboxes vraies
                    true_bboxes = cells_to_bboxes(
                        labels[2], anchor, S=S, is_preds=False
                    )
                    
                    # NMS et collecte des bboxes
                    for idx in range(batch_size):
                        from utils import non_max_suppression
                        nms_boxes = non_max_suppression(
                            bboxes[idx],
                            iou_threshold=config.NMS_IOU_THRESH,
                            threshold=config.CONF_THRESHOLD,
                            box_format="midpoint",
                        )
                        
                        for nms_box in nms_boxes:
                            pred_boxes.append([train_idx] + nms_box)
                        
                        for box in true_bboxes[idx]:
                            if box[1] > config.CONF_THRESHOLD:
                                true_boxes.append([train_idx] + box)
                        
                        train_idx += 1
            
            # Calculer mAP
            mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            )
            
            map_value = mapval.item()
            print(f"mAP (sur {train_idx} images): {map_value:.4f}")
            
            # Tracking du meilleur mAP
            if map_value > best_map:
                best_map = map_value
                best_epoch = epoch + 1
                print(f"Nouveau meilleur mAP! Epoch {best_epoch}")
            
            test_metrics = {
                "mAP": map_value
            }
            
            model.train()
        
        # Logger les métriques de l'epoch
        logger.log_epoch(
            epoch=epoch + 1,
            train_loss=mean_loss,
            test_metrics=test_metrics if test_metrics else None,
            learning_rate=config.LEARNING_RATE
        )
    
    # Fin de l'entraînement
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"ENTRAÎNEMENT TERMINÉ!")
    print(f"{'='*60}")
    print(f"Temps total: {timedelta(seconds=int(total_time))}")
    print(f"Meilleur mAP: {best_map:.4f} (Epoch {best_epoch})")
    print(f"Fichier de métriques: {logger.log_file}")
    print(f"{'='*60}")
    
    # Logger les métriques finales
    final_metrics = {
        "best_mAP": best_map,
        "best_epoch": best_epoch,
        "total_training_time_seconds": int(total_time),
        "total_training_time_formatted": str(timedelta(seconds=int(total_time))),
        "final_loss": mean_loss
    }
    logger.log_final_metrics(final_metrics)
    
    print(f"\nPour générer les graphiques et le rapport, lancez:")
    print(f"   python generate_report.py {logger.log_file}")
    
    return logger.log_file


if __name__ == "__main__":
    main()
