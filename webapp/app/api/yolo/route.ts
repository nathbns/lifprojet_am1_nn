import { NextResponse } from "next/server";
import { predictYoloFromSpace } from "@/lib/gradio";
import { Client } from "@gradio/client";

export const runtime = "nodejs";

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const { imageDataUrl, confidence_threshold, iou_threshold, show_confidence } = body || {};
    if (!imageDataUrl || typeof imageDataUrl !== "string") {
      return NextResponse.json(
        { error: "Champ 'imageDataUrl' requis (data URL base64)" },
        { status: 400 }
      );
    }

    const result = await predictYoloFromSpace(imageDataUrl, {
      confidence_threshold,
      iou_threshold,
      show_confidence,
    });
    return NextResponse.json({ result });
  } catch (error: any) {
    // Mode debug: renvoyer aussi le schéma d'API Gradio pour inspection
    try {
      const client = await Client.connect("nathbns/yolo1_from_scratch");
      const api = await client.view_api();
      // Certaines versions exposent aussi 'config' ou 'endpoints'
      const config: any = (client as any).config || null;
      const namedEndpoints: any = (client as any).endpoints || null;
      return NextResponse.json(
        {
          error: error?.message || "Erreur interne",
          debug: true,
          endpoints: Array.isArray((api as any)?.apis)
            ? (api as any).apis
            : Object.values(((api as any)?.apis) || {}).map((a: any) => ({
                endpoint: a?.endpoint || a?.path,
                inputs: a?.inputs?.map((i: any) => ({
                  label: i?.label,
                  name: i?.name,
                  type: i?.type || i?.component,
                })),
              })),
          apiRaw: api,
          config,
          namedEndpoints,
        },
        { status: 500 }
      );
    } catch (_) {
      return NextResponse.json(
        { error: error?.message || "Erreur interne" },
        { status: 500 }
      );
    }
  }
}


