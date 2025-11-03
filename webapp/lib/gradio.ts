import { Client } from "@gradio/client";

export type YoloPrediction = unknown;
export type YoloModel = "yolov1" | "yolov3";
export type ChessFenResult = {
  fen: string;
  boardHtml: string;
};

/**
 * Convertit une URL (data URL ou HTTP) en File
 * Compatible avec Node.js (server-side) et navigateur (client-side)
 */
async function urlToFile(url: string, filename: string = "image.jpg"): Promise<File> {
  // Vérifier que url est bien une chaîne
  if (typeof url !== 'string') {
    throw new Error(`URL invalide: attendu une chaîne, reçu ${typeof url}`);
  }
  
  // Si c'est un data URL
  const dataUrlMatch = url.match(/^data:(.*?);base64,(.*)$/);
  if (dataUrlMatch) {
    const mimeType = dataUrlMatch[1];
    const base64 = dataUrlMatch[2];
    
    // Vérifier si on est dans Node.js (server-side) ou navigateur (client-side)
    const isNode = typeof process !== 'undefined' && process.versions != null && process.versions.node != null;
    
    let buffer: Uint8Array;
    
    if (isNode) {
      // Node.js: utiliser Buffer puis convertir en Uint8Array pour compatibilité
      const nodeBuffer = Buffer.from(base64, 'base64');
      buffer = new Uint8Array(nodeBuffer);
    } else {
      // Navigateur: utiliser atob (API navigateur)
      if (typeof atob === 'undefined') {
        throw new Error('atob is not available. This code should only run in the browser.');
      }
      const binaryString = atob(base64);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      buffer = bytes;
    }
    
    // Créer un Blob puis un File (fonctionne dans Node.js 18+ et navigateur)
    // Utiliser une assertion de type pour éviter les problèmes de typage TypeScript
    const blob = new Blob([buffer as BlobPart], { type: mimeType });
    const fileExt = mimeType.split("/")[1] || "jpg";
    return new File([blob], `${filename}.${fileExt}`, { type: mimeType });
  }
  
  // Si c'est une URL HTTP, télécharger l'image
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Erreur lors du téléchargement de l'image: ${response.status} ${response.statusText}`);
  }
  const blob = await response.blob();
  const contentType = response.headers.get("content-type") || "image/jpeg";
  const fileExt = contentType.split("/")[1] || "jpg";
  return new File([blob], `${filename}.${fileExt}`, { type: contentType });
}

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
  const file = await urlToFile(dataUrlImage, "input");

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
 * Accepte un data URL ou une URL HTTP
 */
export async function analyzeChessImage(imageUrl: string): Promise<ChessFenResult> {
  const file = await urlToFile(imageUrl, "chessboard");

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

/**
 * Préprocesse une image d'échiquier
 * Utilise le modèle nathbns/preprocess_yoco
 */
export async function preprocessChessImage(dataUrlImage: string): Promise<string> {
  const file = await urlToFile(dataUrlImage, "chessboard");

  const client = await Client.connect("nathbns/preprocess_yoco");
  const result = await client.predict("/process_image", {
    image: file,
  } as Record<string, unknown>);

  // Le résultat Gradio a cette structure :
  // result.data = [ { url: "...", path: "...", orig_name: "...", meta: {...} } ]
  let preprocessedImageUrl: string | undefined;
  
  if (result.data && Array.isArray(result.data) && result.data.length > 0) {
    const firstItem = result.data[0] as { url?: string; path?: string; [key: string]: unknown };
    // L'URL est dans la propriété 'url' de l'objet
    preprocessedImageUrl = firstItem?.url || firstItem?.path;
  }
  
  if (!preprocessedImageUrl || typeof preprocessedImageUrl !== 'string') {
    console.error("Format de réponse inattendu:", result);
    throw new Error("Format de réponse invalide de l'API de preprocessing");
  }
  
  return preprocessedImageUrl;
}
