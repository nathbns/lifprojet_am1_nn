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
  } catch (error: unknown) {
    // Mode debug: renvoyer aussi le schéma d'API Gradio pour inspection
    try {
      const client = await Client.connect("nathbns/yolo1_from_scratch");
      const api = await client.view_api();
      // Certaines versions exposent aussi 'config' ou 'endpoints'
      const config: unknown = (client as unknown as Record<string, unknown>).config || null;
      const namedEndpoints: unknown = (client as unknown as Record<string, unknown>).endpoints || null;
      return NextResponse.json(
        {
          error: (error as Error)?.message || "Erreur interne",
          debug: true,
          endpoints: Array.isArray((api as unknown as Record<string, unknown>)?.apis)
            ? (api as unknown as Record<string, unknown>).apis
            : Object.values(((api as unknown as Record<string, unknown>)?.apis) || {}).map((a: unknown) => ({
                endpoint: (a as Record<string, unknown>)?.endpoint || (a as Record<string, unknown>)?.path,
                inputs: ((a as Record<string, unknown>)?.inputs as unknown[])?.map((i: unknown) => ({
                  label: (i as Record<string, unknown>)?.label,
                  name: (i as Record<string, unknown>)?.name,
                  type: (i as Record<string, unknown>)?.type || (i as Record<string, unknown>)?.component,
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
        { error: (error as Error)?.message || "Erreur interne" },
        { status: 500 }
      );
    }
  }
}


