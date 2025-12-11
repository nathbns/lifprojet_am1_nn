import ImageAnnotator from "@/components/imageAnnotator";

export default function Labelling() {
  return (
    <ImageAnnotator
      title="Annotation d'image - Échecs"
      labels={["pawn", "rook", "bishop", "knight", "queen", "king"]}
    />
  );
}