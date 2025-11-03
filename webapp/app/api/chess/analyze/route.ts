import { NextResponse } from "next/server";
import { analyzeChessImage } from "@/lib/gradio";

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

    const result = await analyzeChessImage(imageDataUrl);
    return NextResponse.json(result);
  } catch (error: unknown) {
    return NextResponse.json(
      { error: (error as Error)?.message || "Erreur lors de l'analyse de l'échiquier" },
      { status: 500 }
    );
  }
}

