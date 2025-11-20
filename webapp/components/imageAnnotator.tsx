"use client";
import { useState, useRef, useEffect } from "react";
import Image from "next/image";
import { Copy, Download } from "lucide-react";
import { FileUpload } from "@/components/ui/file-upload";

interface Box {
  x: number;
  y: number;
  width: number;
  height: number;
  label: string;
}

interface ImageAnnotatorProps {
  labels: string[];
  title?: string;
}

export default function ImageAnnotator({ labels, title }: ImageAnnotatorProps) {
  // Image chargée par l’utilisateur (DataURL)
  const [image, setImage] = useState<string | null>(null);

  // Liste des zones annotées
  const [boxes, setBoxes] = useState<Box[]>([]);

  // Indice du rectangle actuellement sélectionné
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);

  // États utilisés pour dessiner un nouveau rectangle
  const [isDrawing, setIsDrawing] = useState(false);
  const [start, setStart] = useState<{ x: number; y: number } | null>(null);
  const [currentBox, setCurrentBox] = useState<Box | null>(null);

  // Label attribué au prochain rectangle créé
  const [selectedLabel, setSelectedLabel] = useState(labels[0]);

  // Référence au canvas de dessin
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // États dédiés au redimensionnement d’un rectangle existant
  const [resizingIndex, setResizingIndex] = useState<number | null>(null);
  const [resizeCorner, setResizeCorner] = useState<"tl" | "tr" | "bl" | "br" | null>(null);

  // Chargement d’image via FileUpload
  // Stockée directement en DataURL pour un rendu immédiat
  function handleImageUpload(files: File[]) {
    if (files.length > 0) {
      const reader = new FileReader();
      reader.onload = () => setImage(reader.result as string);
      reader.readAsDataURL(files[0]);
    }
  }

  // Déclenché lorsqu’on commence à tirer sur un coin de rectangle
  function handleMouseDownResize(index: number, corner: "tl" | "tr" | "bl" | "br", e: React.MouseEvent) {
    e.stopPropagation(); // Évite d’activer le dessin en même temps
    setResizingIndex(index);
    setResizeCorner(corner);
  }

  // Gestion centralisée des mouvements et relâchements souris
  // Permet de continuer à suivre les interactions même si le curseur sort du canvas
  useEffect(() => {
    function handleWindowMouseMove(e: MouseEvent) {
      // Mode redimensionnement
      if (resizingIndex !== null && resizeCorner !== null) {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const rect = canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        // Mise à jour du rectangle ciblé selon le coin tiré
        setBoxes((prev) => {
          const newBoxes = [...prev];
          const box = { ...newBoxes[resizingIndex] };

          switch (resizeCorner) {
            case "tl":
              // Ajustement depuis le coin supérieur-gauche
              box.width += box.x - mouseX;
              box.height += box.y - mouseY;
              box.x = mouseX;
              box.y = mouseY;
              break;
            case "tr":
              // Ajustement depuis le coin supérieur-droit
              box.width = mouseX - box.x;
              box.height += box.y - mouseY;
              box.y = mouseY;
              break;
            case "bl":
              // Ajustement depuis le coin inférieur-gauche
              box.width += box.x - mouseX;
              box.x = mouseX;
              box.height = mouseY - box.y;
              break;
            case "br":
              // Ajustement depuis le coin inférieur-droit
              box.width = mouseX - box.x;
              box.height = mouseY - box.y;
              break;
          }

          // Empêche les rectangles trop petits d’être créés
          box.width = Math.max(5, box.width);
          box.height = Math.max(5, box.height);

          newBoxes[resizingIndex] = box;
          return newBoxes;
        });
        return;
      }

      // Mode dessin (création d’un nouveau rectangle)
      if (!isDrawing || !start) return;

      const canvas = canvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const endX = e.clientX - rect.left;
      const endY = e.clientY - rect.top;

      // Rectangle dynamique affiché en prévisualisation
      setCurrentBox({
        x: Math.min(start.x, endX),
        y: Math.min(start.y, endY),
        width: Math.abs(endX - start.x),
        height: Math.abs(endY - start.y),
        label: selectedLabel,
      });
    }

    function handleWindowMouseUp(e: MouseEvent) {
      // Fin du redimensionnement
      if (resizingIndex !== null) {
        setResizingIndex(null);
        setResizeCorner(null);
        return;
      }

      // Fin du dessin d’une nouvelle zone
      if (!isDrawing || !start) return;

      const canvas = canvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const endX = e.clientX - rect.left;
      const endY = e.clientY - rect.top;

      const x = Math.min(start.x, endX);
      const y = Math.min(start.y, endY);
      const width = Math.abs(endX - start.x);
      const height = Math.abs(endY - start.y);

      // N’enregistre que les rectangles d’une taille raisonnable
      if (width > 5 && height > 5) {
        setBoxes((prev) => [...prev, { x, y, width, height, label: selectedLabel }]);
      }

      setIsDrawing(false);
      setStart(null);
      setCurrentBox(null);
    }

    window.addEventListener("mousemove", handleWindowMouseMove);
    window.addEventListener("mouseup", handleWindowMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleWindowMouseMove);
      window.removeEventListener("mouseup", handleWindowMouseUp);
    };
  }, [isDrawing, start, resizingIndex, resizeCorner, selectedLabel]);

  // Début d’un nouveau rectangle
  function handleMouseDown(e: React.MouseEvent<HTMLCanvasElement>) {
    if (!image) return;
    const rect = e.currentTarget.getBoundingClientRect();
    setIsDrawing(true);
    setStart({ x: e.clientX - rect.left, y: e.clientY - rect.top });
    setSelectedIndex(null);
  }

  // Mise à jour du rectangle temporaire (mirroir de useEffect pour le canvas)
  function handleMouseMove(e: React.MouseEvent<HTMLCanvasElement>) {
    if (!isDrawing || !start) return;

    const rect = e.currentTarget.getBoundingClientRect();
    const endX = e.clientX - rect.left;
    const endY = e.clientY - rect.top;

    setCurrentBox({
      x: Math.min(start.x, endX),
      y: Math.min(start.y, endY),
      width: Math.abs(endX - start.x),
      height: Math.abs(endY - start.y),
      label: selectedLabel,
    });
  }

  // Sélection d’un rectangle existant pour édition
  function handleSelectBox(index: number) {
    setSelectedIndex(index);
  }

  // Mise à jour du label sur un rectangle existant
  function updateSelectedLabel(newLabel: string) {
    if (selectedIndex === null) return;

    setBoxes((prev) =>
      prev.map((b, i) => (i === selectedIndex ? { ...b, label: newLabel } : b))
    );
  }

  // Suppression d’un rectangle via Delete / Backspace
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if ((e.key === "Delete" || e.key === "Backspace") && selectedIndex !== null) {
        setBoxes((prev) => prev.filter((_, i) => i !== selectedIndex));
        setSelectedIndex(null);
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [selectedIndex]);

  // Rendu global du composant
  return (
    <div className="min-h-screen bg-background p-8 space-y-6 select-none text-foreground">
      {/* Titre optionnel */}
      {title && <h1 className="text-3xl font-bold">{title}</h1>}

      {/* Import d’image */}
      <FileUpload onChange={handleImageUpload} />

      {/* Choix du label pour les prochaines annotations */}
      <select
        value={selectedLabel}
        onChange={(e) => setSelectedLabel(e.target.value)}
        className="border border-border bg-background text-foreground px-2 py-1"
      >
        {labels.map((l) => (
          <option key={l}>{l}</option>
        ))}
      </select>

      {/* Zone d’annotation avec image + canvas + overlays */}
      <div className="flex justify-center">
        <div className="relative border border-border inline-block">
          {image && (
            <>
              {/* Image annotée */}
              <Image src={image} alt="To annotate" width={800} height={800} />

              {/* Canvas transparent utilisé pour dessiner les rectangles */}
              <canvas
                ref={canvasRef}
                width={800}
                height={800}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                className="absolute top-0 left-0 cursor-crosshair"
              />

              {/* Rectangles enregistrés */}
              {boxes.map((b, i) => (
                <div
                  key={i}
                  onClick={() => handleSelectBox(i)}
                  className={`absolute text-xs flex items-center justify-center
                    ${selectedIndex === i ? "border-2 border-red-500" : "border border-blue-500"}
                    bg-blue-500/20 text-white`}
                  style={{ left: b.x, top: b.y, width: b.width, height: b.height }}
                >
                  {b.label}

                  {/* Poignées de redimensionnement visibles uniquement si sélectionné */}
                  {selectedIndex === i && (
                    <>
                      {["tl", "tr", "bl", "br"].map((corner) => (
                        <div
                          key={corner}
                          onMouseDown={(e) =>
                            handleMouseDownResize(i, corner as "tl" | "tr" | "bl" | "br", e)
                          }
                          className="absolute bg-white border border-gray-700 w-3 h-3 z-10"
                          style={{
                            cursor:
                              corner === "tl"
                                ? "nwse-resize"
                                : corner === "tr"
                                ? "nesw-resize"
                                : corner === "bl"
                                ? "nesw-resize"
                                : "nwse-resize",
                            left: corner.includes("r") ? b.width : 0,
                            top: corner.includes("b") ? b.height : 0,
                            transform: "translate(-50%, -50%)",
                          }}
                        />
                      ))}
                    </>
                  )}
                </div>
              ))}

              {/* Rectangle temporaire lors du dessin */}
              {currentBox && (
                <div
                  className="absolute border border-green-500 bg-green-500/10"
                  style={{
                    left: currentBox.x,
                    top: currentBox.y,
                    width: currentBox.width,
                    height: currentBox.height,
                  }}
                />
              )}
            </>
          )}
        </div>
      </div>

      {/* Panneau d’édition du label d’un rectangle sélectionné */}
      {selectedIndex !== null && (
        <div className="flex items-center gap-2 bg-background border border-border px-3 py-2 shadow">
          <span className="text-sm font-medium">Label sélectionné :</span>
          <select
            value={boxes[selectedIndex].label}
            onChange={(e) => updateSelectedLabel(e.target.value)}
            className="border border-border bg-background text-foreground px-2 py-1 text-sm"
          >
            {labels.map((label) => (
              <option key={label} value={label}>
                {label}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Boutons d’export JSON */}
      <div className="flex gap-2 items-center mb-2">
        {/* Copie dans le presse-papiers */}
        <button
          onClick={() => navigator.clipboard.writeText(JSON.stringify(boxes, null, 2))}
          className="flex items-center gap-1 px-1 py-1 bg-white border border-gray-300 text-black rounded-lg shadow-sm hover:bg-gray-100 transition-all text-sm"
        >
          <Copy className="w-4 h-4" />
        </button>

        {/* Téléchargement du fichier JSON */}
        <button
          onClick={() => {
            const blob = new Blob([JSON.stringify(boxes, null, 2)], { type: "application/json" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "annotations.json";
            a.click();
            URL.revokeObjectURL(url);
          }}
          className="flex items-center gap-1 px-1 py-1 bg-white border border-gray-300 text-black rounded-lg shadow-sm hover:bg-gray-100 transition-all text-sm"
        >
          <Download className="w-4 h-4" />
        </button>
      </div>

      {/* JSON visible en bas de page (aperçu/debug) */}
      <pre className="bg-background p-4 rounded border border-gray-200 text-foreground text-sm w-[800px] overflow-auto">
        {JSON.stringify(boxes, null, 2)}
      </pre>
    </div>
  );
}
