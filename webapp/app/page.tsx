"use client";

import { Button } from "@/components/ui/button";
import Link from "next/link";
import { FaGithub } from "react-icons/fa";
import dynamic from "next/dynamic";
import { Highlighter } from "@/components/ui/highlighter";

// Charger le composant 3D uniquement côté client
const ChessPiece3D = dynamic(() => import("@/components/ChessPiece3D"), {
  ssr: false,
  loading: () => <div className="w-48 h-48 sm:w-56 sm:h-56 md:w-64 md:h-64" />,
});

export default function Home() {

  return (
    <div className="relative flex w-full min-h-screen justify-center items-center px-4 sm:px-6 md:px-8">
      {/* Background avec grille et mask radial */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] dark:bg-[linear-gradient(to_right,#ffffff08_1px,transparent_1px),linear-gradient(to_bottom,#ffffff08_1px,transparent_1px)] bg-[size:24px_24px] [mask-image:radial-gradient(ellipse_at_center,black_10%,transparent_80%)]" />
      
      {/* Conteneur pour YO, pièce d'échecs, et CO */}
      <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 -mt-[12vh] sm:-mt-[10vh] md:-mt-[8vh] z-10">
        <Highlighter               
          action="circle"
          color={
            (typeof window !== "undefined" &&
              (document.documentElement.classList.contains("dark") ||
              window.matchMedia?.("(prefers-color-scheme: dark)").matches))
              ? "#c9c9c9" : "black"
          }
          animationDuration={2000}
          padding={65}
        >
          <div className="flex items-center justify-center gap-4 sm:gap-6 md:gap-8">
            {/* YO à gauche */}
            <Highlighter               
              action="highlight"
              color={
                (typeof window !== "undefined" &&
                  (document.documentElement.classList.contains("dark") ||
                  window.matchMedia?.("(prefers-color-scheme: dark)").matches))
                  ? "#c9c9c9" : "black"
              }
              animationDuration={1000}
            >
              <h1 className="text-[12vw] sm:text-[10vw] md:text-[8vw] lg:text-[7vw] font-bold tracking-tight text-background">
                YO
              </h1>
            </Highlighter>
            
            {/* Pièce d'échecs au centre */}
            <div className="z-20 flex-shrink-0">
              <ChessPiece3D />
            </div>
            
            {/* CO à droite */}
            <Highlighter               
              action="highlight"
              color={
                (typeof window !== "undefined" &&
                  (document.documentElement.classList.contains("dark") ||
                  window.matchMedia?.("(prefers-color-scheme: dark)").matches))
                  ? "#c9c9c9" : "black"
              }
              animationDuration={1000}
            >
              <h1 className="text-[12vw] sm:text-[10vw] md:text-[8vw] lg:text-[7vw] font-bold tracking-tight text-background">
                CO
              </h1>
            </Highlighter>
          </div>
        </Highlighter>
      </div>
      
      <div className="absolute z-10 flex flex-col items-center gap-6 sm:gap-8 md:gap-10 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 mt-40 w-full px-4 sm:px-6 md:px-8">
        <p className="opacity-70 text-sm sm:text-base md:text-lg text-center max-w-2xl leading-relaxed px-4 mt-20">
          From building from scratch Yolo v1 to Yolo v3 to YOCO (You Only Chess Once)!
        </p>
        <div className="relative inline-flex flex-col sm:flex-row gap-4 sm:gap-6 md:gap-8 w-full sm:w-auto">
          <Link href="/yolo" className="w-full sm:w-auto">
            <Button
              variant="outline"
              className="relative rounded-none z-10 w-full sm:w-auto px-10 sm:px-14 md:px-18 lg:px-22 py-3 sm:py-4 bg-background/70 backdrop-blur border text-foreground hover:bg-background/80 transition-all duration-200 text-sm sm:text-base"
            >
              Yolo
            </Button>
          </Link>
          <Link href="/chess" className="w-full sm:w-auto">
            <Button
              variant="outline"
              className="relative rounded-none z-10 w-full sm:w-auto px-10 sm:px-14 md:px-18 lg:px-22 py-3 sm:py-4 bg-background/70 backdrop-blur border text-foreground hover:bg-background/80 transition-all duration-200 text-sm sm:text-base"
            >
              Yoco
            </Button>
          </Link>
        </div>
      </div>
      
      <footer className="absolute bottom-0 left-0 w-full h-2 bg-background/70 backdrop-blur">
        <div className="container mx-auto px-4 py-2 text-center text-sm">
          <p className="flex items-center justify-center gap-2">
            <span>&copy; {new Date().getFullYear()} YOCO. All rights reserved.</span>
            <a href="https://github.com/nathbns/yoco"><FaGithub className="inline-block w-4 h-4" /></a>
          </p>
        </div>
      </footer>
    </div>
  );
}
