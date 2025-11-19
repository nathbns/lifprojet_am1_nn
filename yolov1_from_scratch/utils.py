"""
utils YoloV1
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calcul de l'intersection sur l'union entre deux boites
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
        box1_y2 = boxes_preds[..., 3:4]  # (N, 1)
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) est pour le cas où elles ne se croisent pas
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Fait le Non Max Suppression données les bboxes

    Paramètres:
        bboxes (list): liste de listes contenant toutes les bboxes avec chaque bbox
        spécifiée comme [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): seuil où les bboxes prédites sont correctes
        threshold (float): seuil pour supprimer les bboxes prédites (indépendant de IoU) 
        box_format (str): "midpoint" ou "corners" utilisé pour spécifier les bboxes

    Returns:
        list: bboxes après avoir effectué le NMS donné un seuil IoU spécifique
    """
    assert type(bboxes) == list

    # filtre les bboxes qui ont une probabilité inférieure au seuil
    bboxes = [box for box in bboxes if box[1] > threshold]
    # trie les bboxes par probabilité décroissante
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    # liste des bboxes après le NMS
    bboxes_after_nms = []

    # tant que les bboxes sont présentes
    while bboxes:
        # choisit la bbox avec la probabilité la plus élevée
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
    Calcul de la moyenne de la précision moyenne

    Parametres:
        pred_boxes (list): liste de listes contenant toutes les bboxes avec chaque bbox
        spécifiée comme [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similaire à pred_boxes excepté toutes les vraies
        iou_threshold (float): seuil où les bboxes prédites sont correctes
        box_format (str): "midpoint" ou "corners" utilisé pour spécifier les bboxes
        num_classes (int): nombre de classes

    Returns:
        float: valeur mAP sur toutes les classes donné un seuil IoU spécifique 
    """

    # liste stockant toutes les AP pour chaque classe
    average_precisions = []

    # utilisé pour la stabilité numérique plus tard
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # parcourt toutes les prédictions et les targets,
        # et ne les ajoute que si elles appartiennent à la
        # classe c
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # trouve le nombre de bboxes pour chaque exemple d'entraînement
        # Counter trouve combien de bboxes vraies nous obtenons pour chaque exemple d'entraînement, donc disons que l'image 0 a 3, l'image 1 a 5 alors nous obtiendrons un dictionnaire avec: amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        # parcourt chaque clé, val dans ce dictionnaire et converti en le suivant (w.r.t même exemple): ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # trie par probabilité de bbox qui est l'index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        
        # Si aucun n'existe pour cette classe alors nous pouvons sauter en toute sécurité
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Ne prend que les ground_truths qui ont le même idx d'entraînement que la détection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # ne détecte que la vraie détection une fois
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive et ajoute cette bbox à vu
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # si IoU est plus bas alors la détection est un false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz pour l'intégration numérique
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)


def plot_image(image, boxes):
    """Affiche les bboxes prédites sur l'image"""
    im = np.array(image)
    height, width, _ = im.shape

    # Crée la figure et les axes
    fig, ax = plt.subplots(1)
    # Affiche l'image
    ax.imshow(im)

    # box[0] est le x midpoint, box[2] est la largeur
    # box[1] est le y midpoint, box[3] est la hauteur

    # Crée un Rectangle potch
    for box in boxes:
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Ajoute le patch aux axes
        ax.add_patch(rect)

    plt.show()

def get_bboxes(
    loader,
    model,
    iou_threshold,
    threshold,
    pred_format="cells",
    box_format="midpoint",
    device="cuda",
):
    all_pred_boxes = []
    all_true_boxes = []

    # assurez-vous que le model est en eval avant de get bboxes
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )


            #if batch_idx == 0 and idx == 0:
            #    plot_image(x[idx].permute(1,2,0).to("cpu"), nms_boxes)
            #    print(nms_boxes)

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # beaucoup seront convertis en 0 pred
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes



def convert_cellboxes(predictions, S=7):
    """
    Convertit les bboxes de sortie de Yolo avec
    an image split size of S into entire image ratios
    plutôt que relative aux cellules. Essayé de faire cela
    vectorisé, mais cela a donné un code assez difficile à lire
    ... Utiliser comme une boîte noire? Ou implémenter un plus intuitif,
    en utilisant 2 boucles for iterant range(S) et converti les uns après les autres,
    ce qui donne un code plus lent mais plus lisible.
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 30)
    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    scores = torch.cat(
        (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(
        -1
    )
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds


def cellboxes_to_boxes(out, S=7):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes

def save_checkpoint(state, filename="mon_checkpoint.pth.tar"):
    print("=> Sauvegarde du checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Chargement du checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])