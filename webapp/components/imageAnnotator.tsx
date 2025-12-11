"use client";
import { useState, useRef, useEffect } from "react";
import Image from "next/image";
import { Copy, Download, Tag, Trash2, MousePointer2, Square, Info, ImageIcon, X } from "lucide-react";
import { FileUpload } from "@/components/ui/file-upload";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Highlighter } from "@/components/ui/highlighter";

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
  const [image, setImage] = useState<string | null>(null);
  const [imageSize, setImageSize] = useState<{ width: number; height: number }>({ width: 800, height: 600 });
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [start, setStart] = useState<{ x: number; y: number } | null>(null);
  const [currentBox, setCurrentBox] = useState<Box | null>(null);
  const [selectedPieceType, setSelectedPieceType] = useState(labels[0]);
  const [selectedColor, setSelectedColor] = useState<"white" | "black">("white");
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [resizingIndex, setResizingIndex] = useState<number | null>(null);
  const [resizeCorner, setResizeCorner] = useState<"tl" | "tr" | "bl" | "br" | null>(null);
  const [displaySize, setDisplaySize] = useState<{ width: number; height: number }>({ width: 800, height: 600 });

  // Combiner couleur + type pour créer le label final
  const selectedLabel = `${selectedColor}_${selectedPieceType}`;

  // Charger l'image et calculer ses dimensions
  function handleImageUpload(files: File[]) {
    if (files.length > 0) {
      const reader = new FileReader();
      reader.onload = () => {
        const img = new window.Image();
        img.onload = () => {
          setImageSize({ width: img.width, height: img.height });
          setImage(reader.result as string);
        };
        img.src = reader.result as string;
      };
      reader.readAsDataURL(files[0]);
    }
  }

  // Calculer la taille d'affichage responsive
  useEffect(() => {
    function updateDisplaySize() {
      if (!containerRef.current || !image) return;
      
      const containerWidth = containerRef.current.clientWidth;
      const maxWidth = Math.min(containerWidth - 32, 900); // padding
      const aspectRatio = imageSize.height / imageSize.width;
      
      let width = Math.min(imageSize.width, maxWidth);
      let height = width * aspectRatio;
      
      // Limiter la hauteur sur mobile
      const maxHeight = window.innerHeight * 0.6;
      if (height > maxHeight) {
        height = maxHeight;
        width = height / aspectRatio;
      }
      
      setDisplaySize({ width, height });
    }

    updateDisplaySize();
    window.addEventListener("resize", updateDisplaySize);
    return () => window.removeEventListener("resize", updateDisplaySize);
  }, [image, imageSize]);

  // Calculer le ratio pour convertir les coordonnées
  const scaleRatio = displaySize.width / imageSize.width;

  function handleMouseDownResize(index: number, corner: "tl" | "tr" | "bl" | "br", e: React.MouseEvent) {
    e.stopPropagation();
    setResizingIndex(index);
    setResizeCorner(corner);
  }

  useEffect(() => {
    function handleWindowMouseMove(e: MouseEvent) {
      if (resizingIndex !== null && resizeCorner !== null) {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const rect = canvas.getBoundingClientRect();
        const mouseX = (e.clientX - rect.left) / scaleRatio;
        const mouseY = (e.clientY - rect.top) / scaleRatio;

        setBoxes((prev) => {
          const newBoxes = [...prev];
          const box = { ...newBoxes[resizingIndex] };

          switch (resizeCorner) {
            case "tl":
              box.width += box.x - mouseX;
              box.height += box.y - mouseY;
              box.x = mouseX;
              box.y = mouseY;
              break;
            case "tr":
              box.width = mouseX - box.x;
              box.height += box.y - mouseY;
              box.y = mouseY;
              break;
            case "bl":
              box.width += box.x - mouseX;
              box.x = mouseX;
              box.height = mouseY - box.y;
              break;
            case "br":
              box.width = mouseX - box.x;
              box.height = mouseY - box.y;
              break;
          }

          box.width = Math.max(5, box.width);
          box.height = Math.max(5, box.height);

          newBoxes[resizingIndex] = box;
          return newBoxes;
        });
        return;
      }

      if (!isDrawing || !start) return;

      const canvas = canvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const endX = (e.clientX - rect.left) / scaleRatio;
      const endY = (e.clientY - rect.top) / scaleRatio;

      setCurrentBox({
        x: Math.min(start.x, endX),
        y: Math.min(start.y, endY),
        width: Math.abs(endX - start.x),
        height: Math.abs(endY - start.y),
        label: selectedLabel,
      });
    }

    function handleWindowMouseUp(e: MouseEvent) {
      if (resizingIndex !== null) {
        setResizingIndex(null);
        setResizeCorner(null);
        return;
      }

      if (!isDrawing || !start) return;

      const canvas = canvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const endX = (e.clientX - rect.left) / scaleRatio;
      const endY = (e.clientY - rect.top) / scaleRatio;

      const x = Math.min(start.x, endX);
      const y = Math.min(start.y, endY);
      const width = Math.abs(endX - start.x);
      const height = Math.abs(endY - start.y);

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
  }, [isDrawing, start, resizingIndex, resizeCorner, selectedLabel, scaleRatio]);

  function handleMouseDown(e: React.MouseEvent<HTMLCanvasElement>) {
    if (!image) return;
    const rect = e.currentTarget.getBoundingClientRect();
    setIsDrawing(true);
    setStart({ 
      x: (e.clientX - rect.left) / scaleRatio, 
      y: (e.clientY - rect.top) / scaleRatio 
    });
    setSelectedIndex(null);
  }

  function handleMouseMove(e: React.MouseEvent<HTMLCanvasElement>) {
    if (!isDrawing || !start) return;

    const rect = e.currentTarget.getBoundingClientRect();
    const endX = (e.clientX - rect.left) / scaleRatio;
    const endY = (e.clientY - rect.top) / scaleRatio;

    setCurrentBox({
      x: Math.min(start.x, endX),
      y: Math.min(start.y, endY),
      width: Math.abs(endX - start.x),
      height: Math.abs(endY - start.y),
      label: selectedLabel,
    });
  }

  function handleSelectBox(index: number, e: React.MouseEvent) {
    e.stopPropagation();
    setSelectedIndex(index);
  }

  // Parser un label pour extraire couleur et type
  function parseLabel(label: string): { color: "white" | "black"; type: string } {
    if (label.startsWith("white_")) {
      return { color: "white", type: label.replace("white_", "") };
    } else if (label.startsWith("black_")) {
      return { color: "black", type: label.replace("black_", "") };
    }
    // Fallback pour les anciens labels sans couleur
    return { color: "white", type: label };
  }

  function updateSelectedBoxColor(newColor: "white" | "black") {
    if (selectedIndex === null) return;
    setBoxes((prev) =>
      prev.map((b, i) => {
        if (i !== selectedIndex) return b;
        const { type } = parseLabel(b.label);
        return { ...b, label: `${newColor}_${type}` };
      })
    );
  }

  function updateSelectedBoxType(newType: string) {
    if (selectedIndex === null) return;
    setBoxes((prev) =>
      prev.map((b, i) => {
        if (i !== selectedIndex) return b;
        const { color } = parseLabel(b.label);
        return { ...b, label: `${color}_${newType}` };
      })
    );
  }

  function deleteSelectedBox() {
    if (selectedIndex === null) return;
    setBoxes((prev) => prev.filter((_, i) => i !== selectedIndex));
    setSelectedIndex(null);
  }

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

  function clearImage() {
    setImage(null);
    setBoxes([]);
    setSelectedIndex(null);
  }

  return (
    <div className="min-h-screen py-4 sm:py-8">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="mb-6 sm:mb-8 text-center">
          <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold mb-2">
            {title ? (
              <>
                {title.split(" - ")[0]}
                {title.includes(" - ") && (
                  <>
                    {" - "}
                    <Highlighter action="highlight" color="#c9c9c9" padding={3}>
                      <span className="text-background">{title.split(" - ")[1]}</span>
                    </Highlighter>
                  </>
                )}
              </>
            ) : (
              <>
                Annotation d&apos;
                <Highlighter action="highlight" color="#c9c9c9" padding={3}>
                  <span className="text-background">Image</span>
                </Highlighter>
              </>
            )}
          </h1>
          <p className="text-muted-foreground text-sm sm:text-base">
            <Highlighter action="underline" color="#c9c9c9" padding={3}>
              Dessinez des rectangles
            </Highlighter>
            {" "}sur l&apos;image pour annoter les objets
          </p>
        </div>

        {/* Bento Grid Layout */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
          
          {/* Row 1: Image + Zone d'annotation */}
          {/* Image Card */}
          <Card className="md:col-span-1 lg:col-span-1">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-sm">
                <ImageIcon className="h-4 w-4" />
                Image
              </CardTitle>
            </CardHeader>
            <CardContent>
              {!image ? (
      <FileUpload onChange={handleImageUpload} />
              ) : (
                <div className="space-y-3">
                  <div className="relative aspect-square bg-muted/50 overflow-hidden">
                    <Image 
                      src={image} 
                      alt="Thumbnail" 
                      fill 
                      className="object-contain"
                    />
                  </div>
                  <Button 
                    variant="outline" 
                    size="sm" 
                    className="w-full text-xs"
                    onClick={clearImage}
                  >
                    <X className="h-3 w-3 mr-1" />
                    Changer
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Zone d'annotation Card - Plus large */}
          <Card className="md:col-span-1 lg:col-span-3 md:row-span-2">
            <CardHeader className="pb-3">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
                <div>
                  <CardTitle className="text-sm">Zone d&apos;annotation</CardTitle>
                  <CardDescription className="text-xs">
                    {boxes.length} annotation{boxes.length !== 1 ? "s" : ""}
                  </CardDescription>
                </div>
                {image && boxes.length > 0 && (
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-7 text-xs"
                      onClick={() => navigator.clipboard.writeText(JSON.stringify(boxes, null, 2))}
                    >
                      <Copy className="h-3 w-3 mr-1" />
                      Copier
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-7 text-xs"
                      onClick={() => {
                        const blob = new Blob([JSON.stringify(boxes, null, 2)], { type: "application/json" });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement("a");
                        a.href = url;
                        a.download = "annotations.json";
                        a.click();
                        URL.revokeObjectURL(url);
                      }}
                    >
                      <Download className="h-3 w-3 mr-1" />
                      JSON
                    </Button>
                  </div>
                )}
              </div>
            </CardHeader>
            <CardContent ref={containerRef}>
              {image ? (
                <div 
                  className="relative border border-border mx-auto select-none"
                  style={{ width: displaySize.width, height: displaySize.height }}
                >
                  <Image 
                    src={image} 
                    alt="To annotate" 
                    fill
                    className="object-contain pointer-events-none"
                    sizes={`${displaySize.width}px`}
                  />

              <canvas
                ref={canvasRef}
                    width={displaySize.width}
                    height={displaySize.height}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                className="absolute top-0 left-0 cursor-crosshair"
              />

              {/* Rectangles enregistrés */}
              {boxes.map((b, i) => (
                <div
                  key={i}
                      onClick={(e) => handleSelectBox(i, e)}
                      className={`absolute text-[10px] sm:text-xs flex items-center justify-center font-medium transition-all border-2 ${
                        selectedIndex === i 
                          ? "border-foreground bg-foreground/20 ring-2 ring-foreground ring-offset-1" 
                          : "border-muted-foreground bg-muted-foreground/10"
                      }`}
                      style={{ 
                        left: b.x * scaleRatio, 
                        top: b.y * scaleRatio, 
                        width: b.width * scaleRatio, 
                        height: b.height * scaleRatio,
                      }}
                    >
                      <span className="bg-background/90 px-1.5 py-0.5 flex items-center gap-1 text-foreground">
                        <span className={`w-2 h-2 rounded-full flex-shrink-0 ${
                          b.label.startsWith("white_") 
                            ? "bg-white border border-gray-400" 
                            : "bg-gray-900 border border-gray-600"
                        }`}></span>
                        <span className="capitalize">{parseLabel(b.label).type}</span>
                      </span>

                  {selectedIndex === i && (
                    <>
                          {(["tl", "tr", "bl", "br"] as const).map((corner) => (
                        <div
                          key={corner}
                              onMouseDown={(e) => handleMouseDownResize(i, corner, e)}
                              className="absolute w-3 h-3 sm:w-4 sm:h-4 border-2 z-10 bg-foreground border-background"
                          style={{
                            cursor:
                                  corner === "tl" || corner === "br"
                                ? "nwse-resize"
                                    : "nesw-resize",
                                left: corner.includes("r") ? "100%" : 0,
                                top: corner.includes("b") ? "100%" : 0,
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
                      className="absolute border-2 border-dashed border-foreground bg-foreground/10 pointer-events-none"
                  style={{
                        left: currentBox.x * scaleRatio,
                        top: currentBox.y * scaleRatio,
                        width: currentBox.width * scaleRatio,
                        height: currentBox.height * scaleRatio,
                  }}
                />
              )}
                </div>
              ) : (
                <div className="aspect-video flex items-center justify-center bg-muted/30 border border-dashed border-muted-foreground/30">
                  <div className="text-center text-muted-foreground">
                    <ImageIcon className="h-12 w-12 mx-auto mb-2 opacity-50" />
                    <p className="text-sm">Téléchargez une image pour commencer</p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Row 2: Label + Sélection/Instructions */}
          {/* Label Selection Card */}
          <Card className="md:col-span-1 lg:col-span-1">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-sm">
                <Tag className="h-4 w-4" />
                Label
              </CardTitle>
              <CardDescription className="text-xs font-mono">{selectedLabel}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {/* Sélecteur de couleur */}
              <div>
                <label className="text-[10px] font-medium mb-1.5 block uppercase tracking-wide text-muted-foreground">Couleur</label>
                <div className="grid grid-cols-2 gap-1.5">
                  <button
                    onClick={() => setSelectedColor("white")}
                    className={`px-2 py-2 text-xs font-medium transition-all border flex items-center justify-center gap-1.5 ${
                      selectedColor === "white"
                        ? "border-foreground bg-foreground text-background"
                        : "border-muted-foreground/50 hover:border-foreground text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    <span className="w-3 h-3 rounded-full bg-white border border-gray-300"></span>
                    Blanc
                  </button>
                  <button
                    onClick={() => setSelectedColor("black")}
                    className={`px-2 py-2 text-xs font-medium transition-all border flex items-center justify-center gap-1.5 ${
                      selectedColor === "black"
                        ? "border-foreground bg-foreground text-background"
                        : "border-muted-foreground/50 hover:border-foreground text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    <span className="w-3 h-3 rounded-full bg-gray-900 border border-gray-700"></span>
                    Noir
                  </button>
        </div>
      </div>

              {/* Sélecteur de type de pièce */}
              <div>
                <label className="text-[10px] font-medium mb-1.5 block uppercase tracking-wide text-muted-foreground">Pièce</label>
                <div className="grid grid-cols-2 gap-1.5">
                  {labels.map((label) => (
                    <button
                      key={label}
                      onClick={() => setSelectedPieceType(label)}
                      className={`px-2 py-1.5 text-xs font-medium transition-all border capitalize ${
                        selectedPieceType === label
                          ? "border-foreground bg-foreground text-background"
                          : "border-muted-foreground/50 hover:border-foreground text-muted-foreground hover:text-foreground"
                      }`}
                    >
                {label}
                    </button>
            ))}
                </div>
        </div>
            </CardContent>
          </Card>

          {/* Row 3: Sélection ou Instructions + JSON */}
          {/* Sélection / Instructions Card */}
          {selectedIndex !== null && boxes[selectedIndex] ? (() => {
            const { color: boxColor, type: boxType } = parseLabel(boxes[selectedIndex].label);
            return (
              <Card className="md:col-span-1 lg:col-span-2 border-primary/50">
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between">
                    <CardTitle className="flex items-center gap-2 text-sm">
                      <MousePointer2 className="h-4 w-4" />
                      Sélection
                    </CardTitle>
                    <Button
                      variant="destructive"
                      size="sm"
                      className="h-6 text-xs px-2"
                      onClick={deleteSelectedBox}
                    >
                      <Trash2 className="h-3 w-3 mr-1" />
                      Suppr.
                    </Button>
                  </div>
                  <CardDescription className="text-xs font-mono">{boxes[selectedIndex].label}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-3">
                    {/* Modifier la couleur */}
                    <div>
                      <label className="text-[10px] font-medium mb-1.5 block uppercase tracking-wide text-muted-foreground">Couleur</label>
                      <div className="grid grid-cols-2 gap-1">
                        <button
                          onClick={() => updateSelectedBoxColor("white")}
                          className={`px-1.5 py-1.5 text-[10px] font-medium transition-all border flex items-center justify-center gap-1 ${
                            boxColor === "white"
                              ? "border-foreground bg-foreground text-background"
                              : "border-muted-foreground/50 hover:border-foreground text-muted-foreground"
                          }`}
                        >
                          <span className="w-2 h-2 rounded-full bg-white border border-gray-300"></span>
                          Blanc
                        </button>
        <button
                          onClick={() => updateSelectedBoxColor("black")}
                          className={`px-1.5 py-1.5 text-[10px] font-medium transition-all border flex items-center justify-center gap-1 ${
                            boxColor === "black"
                              ? "border-foreground bg-foreground text-background"
                              : "border-muted-foreground/50 hover:border-foreground text-muted-foreground"
                          }`}
                        >
                          <span className="w-2 h-2 rounded-full bg-gray-900 border border-gray-700"></span>
                          Noir
        </button>
                      </div>
                    </div>

                    {/* Modifier le type */}
                    <div>
                      <label className="text-[10px] font-medium mb-1.5 block uppercase tracking-wide text-muted-foreground">Type</label>
                      <div className="grid grid-cols-3 gap-1">
                        {labels.map((label) => (
        <button
                            key={label}
                            onClick={() => updateSelectedBoxType(label)}
                            className={`px-1 py-1.5 text-[10px] font-medium transition-all border capitalize ${
                              boxType === label
                                ? "border-foreground bg-foreground text-background"
                                : "border-muted-foreground/50 hover:border-foreground text-muted-foreground"
                            }`}
                          >
                            {label}
        </button>
                        ))}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            );
          })() : (
            <Card className="md:col-span-1 lg:col-span-2">
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center gap-2 text-sm">
                  <Info className="h-4 w-4" />
                  Instructions
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-xs text-muted-foreground">
                  <div className="flex items-center gap-2">
                    <Square className="h-3 w-3 flex-shrink-0" />
                    <span>Dessinez pour annoter</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <MousePointer2 className="h-3 w-3 flex-shrink-0" />
                    <span>Cliquez pour sélectionner</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Tag className="h-3 w-3 flex-shrink-0" />
                    <span>Redimensionnez les coins</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* JSON Preview Card */}
          <Card className="md:col-span-2 lg:col-span-2">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Données JSON</CardTitle>
              <CardDescription className="text-xs">Format d&apos;export</CardDescription>
            </CardHeader>
            <CardContent>
              <pre className="bg-muted/50 p-2 sm:p-3 text-[9px] sm:text-[10px] font-mono overflow-x-auto max-h-32">
                {boxes.length > 0 
                  ? JSON.stringify(boxes, null, 2)
                  : "// Aucune annotation"}
              </pre>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
