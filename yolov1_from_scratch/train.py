import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolo_V1
from dataset import VOCDataset
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)
from loss import Loss_Yolo
import os

seed = 123
torch.manual_seed(seed)

# hyperparametres 
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 64 
WEIGHT_DECAY = 0
EPOCHS = 50
NUM_WORKERS = 4
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "checkpoint_epoch_50.pth.tar"

# config pour les data import de kaggle dans ma session colab
KAGGLE_DATA_PATH = os.environ.get("KAGGLE_DATA_PATH", None)

if KAGGLE_DATA_PATH:
    # colab
    IMG_DIR = os.path.join(KAGGLE_DATA_PATH, "images")
    LABEL_DIR = os.path.join(KAGGLE_DATA_PATH, "labels")
    TRAIN_CSV = os.path.join(KAGGLE_DATA_PATH, "train.csv")
    TEST_CSV = os.path.join(KAGGLE_DATA_PATH, "test.csv")
else:
    # local
    IMG_DIR = "data/images"
    LABEL_DIR = "data/labels"
    TRAIN_CSV = "data/train.csv"
    TEST_CSV = "data/test.csv"


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    print(f"loss moyenne: {sum(mean_loss)/len(mean_loss)}")


def main():
    model = Yolo_V1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = Loss_Yolo()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    train_dataset = VOCDataset(
        TRAIN_CSV,
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )

    test_dataset = VOCDataset(
        TEST_CSV, 
        transform=transform, 
        img_dir=IMG_DIR, 
        label_dir=LABEL_DIR,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    for epoch in range(EPOCHS):
        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4, device=DEVICE
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        # print(f"Train: {mean_avg_prec}")

        train_fn(train_loader, model, optimizer, loss_fn)

         # Sauvegarder tous les 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "mAP": mean_avg_prec,
            }
            save_checkpoint(checkpoint, filename=f"checkpoint_epoch_{epoch+1}.pth.tar")
            print(f"sauvegarde du checkpoint model : epoch {epoch+1}")


if __name__ == "__main__":
    main()