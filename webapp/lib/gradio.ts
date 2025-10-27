import { Client } from "@gradio/client";

export type YoloPrediction = unknown;
export type YoloModel = "yolov1" | "yolov3";
export type ChessFenResult = {
  fen: string;
  boardHtml: string;
};

export async function predictYoloFromSpace(
  dataUrlImage: string,
  options?: { 
    confidence_threshold?: number; 
    iou_threshold?: number; 
    show_confidence?: boolean;
    model?: YoloModel;
  }
) {
  const model = options?.model || "yolov1";
  
  // Extraire le mime/type et construire un File nommé pour conserver l'extension
  const matches = dataUrlImage.match(/^data:(.*?);base64,(.*)$/);
  if (!matches) {
    throw new Error("Image invalide: format data URL attendu");
  }
  const mimeType = matches[1];
  const fileExt = mimeType.split("/")[1] || "jpg";
  const base64 = matches[2];
  const buffer = Buffer.from(base64, "base64");
  const blob = new Blob([buffer], { type: mimeType });
  const file = new File([blob], `input.${fileExt}`, { type: mimeType });

  // Appeler le modèle sélectionné
  if (model === "yolov3") {
    return await predictYoloV3(file, options);
  } else {
    return await predictYoloV1(file, options);
  }
}

// YOLOv1 utilise confidence_threshold
async function predictYoloV1(
  file: File,
  options?: { confidence_threshold?: number; iou_threshold?: number; show_confidence?: boolean }
) {
  const client = await Client.connect("nathbns/yolo1_from_scratch");
  const result = await client.predict("/detect_objects", {
    image: file,
    confidence_threshold: options?.confidence_threshold ?? 0.4,
    iou_threshold: options?.iou_threshold ?? 0.5,
    show_confidence: options?.show_confidence ?? true,
  } as Record<string, unknown>);
  
  return result as YoloPrediction;
}

// YOLOv3 utilise conf_threshold (nom différent!)
async function predictYoloV3(
  file: File,
  options?: { confidence_threshold?: number; iou_threshold?: number; show_confidence?: boolean }
) {
  const client = await Client.connect("nathbns/yolo3_from_scratch");
  const result = await client.predict("/predict", {
    image: file,
    conf_threshold: options?.confidence_threshold ?? 0.5,
    iou_threshold: options?.iou_threshold ?? 0.45,
  } as Record<string, unknown>);
  
  return result as YoloPrediction;
}

/**
 * Analyse une image d'échiquier et retourne le FEN détecté
 * Utilise le modèle nathbns/yoco_first_version
 */
export async function analyzeChessImage(dataUrlImage: string): Promise<ChessFenResult> {
  // Extraire le mime/type et construire un File
  const matches = dataUrlImage.match(/^data:(.*?);base64,(.*)$/);
  if (!matches) {
    throw new Error("Image invalide: format data URL attendu");
  }
  const mimeType = matches[1];
  const fileExt = mimeType.split("/")[1] || "jpg";
  const base64 = matches[2];
  const buffer = Buffer.from(base64, "base64");
  const blob = new Blob([buffer], { type: mimeType });
  const file = new File([blob], `chessboard.${fileExt}`, { type: mimeType });

  const client = await Client.connect("nathbns/yoco_first_version");
  const result = await client.predict("/analyze_chess_image", {
    image_input: file,
  } as Record<string, unknown>);

  // Le résultat contient [0] = FEN string, [1] = HTML visualization
  const data = result.data as [string, string];
  
  return {
    fen: data[0],
    boardHtml: data[1],
  };
}


