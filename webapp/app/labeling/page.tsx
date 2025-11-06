"use client";
import { useState, useRef, useEffect } from "react";
import Image from "next/image";

interface Box {
  x: number;
  y: number;
  width: number;
  height: number;
  label: string;
}

export default function AnnotatePage() {
  const [image, setImage] = useState<string | null>(null);
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);

  const [isDrawing, setIsDrawing] = useState(false);
  const [start, setStart] = useState<{ x: number; y: number } | null>(null);
  const [currentBox, setCurrentBox] = useState<Box | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const labels = ["pawn", "rook", "knight", "bishop", "queen", "king"];
  const [selectedLabel, setSelectedLabel] = useState(labels[0]);


  const [resizingIndex, setResizingIndex] = useState<number | null>(null);
  const [resizeCorner, setResizeCorner] = useState<"tl" | "tr" | "bl" | "br" | null>(null);

function handleMouseDownResize(index: number, corner: "tl" | "tr" | "bl" | "br", e: React.MouseEvent) {
  e.stopPropagation(); 
  setResizingIndex(index);
  setResizeCorner(corner);
}

useEffect(() => {
  function handleWindowMouseMove(e: MouseEvent) {
    // === RESIZE MODE ===
    if (resizingIndex !== null && resizeCorner !== null) {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

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

    // === DRAW MODE ===
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

    setCurrentBox({ x, y, width, height, label: selectedLabel });
  }

  function handleWindowMouseUp(e: MouseEvent) {
    // End resize
    if (resizingIndex !== null) {
      setResizingIndex(null);
      setResizeCorner(null);
      return;
    }

    // End drawing
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





  function handleImageUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = () => setImage(reader.result as string);
      reader.readAsDataURL(file);
    }
  }

  function handleMouseDown(e: React.MouseEvent<HTMLCanvasElement>) {
    if (!image) return;
    const rect = e.currentTarget.getBoundingClientRect();
    setIsDrawing(true);
    setStart({ x: e.clientX - rect.left, y: e.clientY - rect.top });
    setSelectedIndex(null);
  }

  function handleMouseMove(e: React.MouseEvent<HTMLCanvasElement>) {
    if (!isDrawing || !start) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const endX = e.clientX - rect.left;
    const endY = e.clientY - rect.top;

    const x = Math.min(start.x, endX);
    const y = Math.min(start.y, endY);
    const width = Math.abs(endX - start.x);
    const height = Math.abs(endY - start.y);

    setCurrentBox({ x, y, width, height, label: selectedLabel });
  }

function handleMouseUp(e: React.MouseEvent<HTMLCanvasElement>) {
  if (!start) return;
  const rect = e.currentTarget.getBoundingClientRect();
  const endX = e.clientX - rect.left;
  const endY = e.clientY - rect.top;

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


  function handleSelectBox(index: number) {
    setSelectedIndex(index);
  }

  function updateSelectedLabel(newLabel: string) {
  if (selectedIndex === null) return;
  setBoxes(prev =>
    prev.map((b, i) => (i === selectedIndex ? { ...b, label: newLabel } : b))
  );
}

  // Delete with DEL or Backspace
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if ((e.key === "Delete" || e.key === "Backspace") && selectedIndex !== null) {
        setBoxes(prev => prev.filter((_, i) => i !== selectedIndex));
        setSelectedIndex(null);
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [selectedIndex]);

  return (
    <div className="min-h-screen bg-gray-50 p-8 space-y-6 select-none">
      <h1 className="text-3xl font-bold text-gray-800">Annotation d’images – Échecs</h1>

      <input type="file" accept="image/*" onChange={handleImageUpload} className="mb-4" />

      <select
        value={selectedLabel}
        onChange={(e) => setSelectedLabel(e.target.value)}
        className="border rounded px-2 py-1"
      >
        {labels.map((l) => (
          <option key={l}>{l}</option>
        ))}
      </select>

      <div className="relative border border-gray-300 inline-block">
        {image && (
          <>
            <Image src={image} alt="To annotate" width={800} height={800} />
            <canvas
              ref={canvasRef}
              width={800}
              height={800}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              
              className="absolute top-0 left-0 cursor-crosshair"
            />

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

    {selectedIndex === i && (
      <>
        {["tl", "tr", "bl", "br"].map((corner) => (
<div
  key={corner}
  onMouseDown={(e) => handleMouseDownResize(i, corner as "tl" | "tr" | "bl" | "br", e)}
  className="absolute bg-white border border-gray-700 w-3 h-3 rounded-sm z-10"
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

      {selectedIndex !== null && (
  <div className="flex items-center gap-2 bg-white border px-3 py-2 rounded shadow">
    <span className="text-sm text-gray-700 font-medium">Label sélectionné :</span>
    <select
      value={boxes[selectedIndex].label}
      onChange={(e) => updateSelectedLabel(e.target.value)}
      className="border rounded px-2 py-1 text-sm"
    >
      {labels.map((label) => (
        <option key={label} value={label}>
          {label}
        </option>
      ))}
    </select>
  </div>
)}


      <pre className="bg-gray-100 p-4 rounded text-sm w-[800px]">
        {JSON.stringify(boxes, null, 2)}
      </pre>
    </div>
  );
}
