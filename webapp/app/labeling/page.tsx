import ImageAnnotator from "@/components/imageAnnotator";

export default function Labelling() {
  return (
    <ImageAnnotator
    title="Annontation d'image - Echecs"
    labels={["pawn" ,"rook", "bishop", "knight", "queen", "king"]}
    />
  )
}