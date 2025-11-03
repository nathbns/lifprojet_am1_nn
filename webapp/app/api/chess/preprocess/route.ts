import { NextResponse } from "next/server";
import { preprocessChessImage } from "@/lib/gradio";

export const runtime = "nodejs";

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const { imageDataUrl } = body || {};
    
    if (!imageDataUrl || typeof imageDataUrl !== "string") {
      return NextResponse.json(
        { error: "Champ 'imageDataUrl' requis (data URL base64)" },
        { status: 400 }
      );
    }

    const preprocessedImageUrl = await preprocessChessImage(imageDataUrl);
    return NextResponse.json({ preprocessedImageUrl });
  } catch (error: unknown) {
    return NextResponse.json(
      { error: (error as Error)?.message || "Erreur lors du preprocessing de l'image" },
      { status: 500 }
    );
  }
}

