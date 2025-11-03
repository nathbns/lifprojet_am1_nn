"use client";

import { FlickeringGrid } from "@/components/ui/flickering-grid-hero";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import Link from "next/link";


// SVG du texte YOCO encodé en base64
const YOCO_LOGO_BASE64 = "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAwIiBoZWlnaHQ9IjIwMCIgdmlld0JveD0iMCAwIDYwMCAyMDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGRvbWluYW50LWJhc2VsaW5lPSJtaWRkbGUiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJzYW5zLXNlcmlmIiBmb250LXdlaWdodD0iYm9sZCIgZm9udC1zaXplPSIxMjAiIGZpbGw9IndoaXRlIj5ZT0NPPC90ZXh0Pjwvc3ZnPg==";

// Style du masque pour l'effet sur le texte
const maskStyle = {
  WebkitMaskImage: `url('${YOCO_LOGO_BASE64}')`,
  WebkitMaskSize: '80vw',
  WebkitMaskPosition: 'center',
  WebkitMaskRepeat: 'no-repeat',
  maskImage: `url('${YOCO_LOGO_BASE64}')`,
  maskSize: '80vw',
  maskPosition: 'center',
  maskRepeat: 'no-repeat',
} as const;

// Fonction pour convertir oklch/hsl en RGB
function getThemeColor(cssVariable: string): string {
  if (typeof window === 'undefined') return 'rgb(0, 0, 0)';
  
  const rootStyles = getComputedStyle(document.documentElement);
  const colorValue = rootStyles.getPropertyValue(cssVariable).trim();
  
  // Créer un élément temporaire pour obtenir la couleur RGB
  const temp = document.createElement('div');
  temp.style.color = colorValue;
  document.body.appendChild(temp);
  const rgb = getComputedStyle(temp).color;
  document.body.removeChild(temp);
  
  return rgb;
}

export default function Home() {
  const [themeColors, setThemeColors] = useState({
    primary: 'rgb(161, 117, 83)',
    background: 'rgb(161, 117, 83)',
  });
  const [isDark, setIsDark] = useState(false);

  useEffect(() => {
    // Fonction pour mettre à jour les couleurs du thème
    const updateThemeColors = () => {
      const isDarkMode = document.documentElement.classList.contains('dark');
      setIsDark(isDarkMode);
      
      setThemeColors({
        primary: getThemeColor('--primary'),
        // En mode clair, utiliser primary (plus foncé), en mode sombre, utiliser secondary
        background: isDarkMode ? getThemeColor('--secondary') : getThemeColor('--primary'),
      });
    };

    // Mettre à jour au chargement
    updateThemeColors();

    // Observer les changements de thème
    const observer = new MutationObserver(updateThemeColors);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class'],
    });

    return () => observer.disconnect();
  }, []);

  const GRID_CONFIG = {
    background: {
      color: themeColors.background, // Primary en clair, Secondary en sombre
      maxOpacity: isDark ? 0.4 : 0.1,
      flickerChance: 0.12,
      squareSize: 4,
      gridGap: 4,
    },
    logo: {
      color: themeColors.primary, // Utilise --primary du thème
      maxOpacity: isDark ? 0.7 : 1,
      flickerChance: 0.18,
      squareSize: 4,
      gridGap: 6,
    },
  };

  return (
    <div className="flex w-full min-h-screen justify-center items-center px-4 sm:px-6 md:px-8">
      <FlickeringGrid
        className="absolute inset-0 z-0 h-full w-full [mask-image:radial-gradient(400px_circle_at_center,white,transparent)] sm:[mask-image:radial-gradient(600px_circle_at_center,white,transparent)] md:[mask-image:radial-gradient(800px_circle_at_center,white,transparent)] lg:[mask-image:radial-gradient(1000px_circle_at_center,white,transparent)]"
        {...GRID_CONFIG.background}
      />
      <div 
        className="absolute inset-0 z-0 -translate-y-[8vh] sm:-translate-y-[4vh] md:translate-y-[2vh]" 
        style={{
          ...maskStyle,
          animation: 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        }}
      >
        <FlickeringGrid {...GRID_CONFIG.logo} />
      </div>
      <div className="absolute z-10 flex flex-col items-center gap-4 sm:gap-6 md:gap-8 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 mt-20 sm:mt-32 md:mt-40 w-full px-4 sm:px-6 md:px-8">
        <p className="opacity-70 text-sm sm:text-base md:text-lg text-center max-w-2xl leading-relaxed pt-4 sm:pt-6 md:pt-10 px-4">
          From building from scratch Yolo v1 to Yolo v3 to YOCO (You Only Chess Once)!
        </p>
        <div className="relative inline-flex flex-col sm:flex-row gap-3 sm:gap-4 md:gap-6 w-full sm:w-auto">
          <Link href="/yolo" className="w-full sm:w-auto">
            <Button
              variant="outline"
              className="relative rounded-none z-10 w-full sm:w-auto px-8 sm:px-12 md:px-16 lg:px-20 py-3 sm:py-4 bg-background/70 backdrop-blur border text-foreground hover:bg-background/80 transition-all duration-200 text-sm sm:text-base"
            >
              Yolo
            </Button>
          </Link>
          <Link href="/chess" className="w-full sm:w-auto">
            <Button
              variant="outline"
              className="relative rounded-none z-10 w-full sm:w-auto px-8 sm:px-12 md:px-16 lg:px-20 py-3 sm:py-4 bg-background/70 backdrop-blur border text-foreground hover:bg-background/80 transition-all duration-200 text-sm sm:text-base"
            >
              Yoco
            </Button>
          </Link>
        </div>
      </div>
    </div>
  );
}
