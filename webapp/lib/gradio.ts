import { Client } from "@gradio/client";

export type YoloPrediction = unknown;

export async function predictYoloFromSpace(
  dataUrlImage: string,
  options?: { confidence_threshold?: number; iou_threshold?: number; show_confidence?: boolean }
) {
  const spaceId = "nathbns/yolo1_from_scratch";
  const client = await Client.connect(spaceId);

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

  // Appel direct de l'endpoint nommé du Space
  const path = "/detect_objects";
  const result = await client.predict(path, {
    image: file,
    confidence_threshold: options?.confidence_threshold ?? 0.4,
    iou_threshold: options?.iou_threshold ?? 0.5,
    show_confidence: options?.show_confidence ?? true,
  } as Record<string, unknown>);
  return result as YoloPrediction;
}


