"use client"

import * as React from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { ThemeToggle } from "@/components/theme-toggle"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { useTheme } from "@/components/theme-provider"
import { Sheet, SheetContent, SheetTitle, SheetTrigger } from "@/components/ui/sheet"
import { Menu } from "lucide-react"
import { Button } from "@/components/ui/button"
import Image from "next/image"

const iconSize = 30

function usePixelPieces() {
  const { theme } = useTheme()
  const [isDark, setIsDark] = React.useState(false)

  React.useEffect(() => {
    const getIsDark = () => {
      if (theme === "system") {
        return window.matchMedia("(prefers-color-scheme: dark)").matches
      }
      return theme === "dark"
    }

    setIsDark(getIsDark())

    // Écouter les changements du thème système
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)")
    const handler = () => {
      if (theme === "system") {
        setIsDark(mediaQuery.matches)
      }
    }

    mediaQuery.addEventListener("change", handler)
    return () => mediaQuery.removeEventListener("change", handler)
  }, [theme])

  const px = (name: string) => `https://lichess1.org/assets/piece/pixel/${name}.svg`
  return {
    P: px(isDark ? "wP" : "bP"),
    R: px(isDark ? "wR" : "bR"),
    N: px(isDark ? "wN" : "bN"),
    B: px(isDark ? "wB" : "bB"),
    Q: px(isDark ? "wQ" : "bQ"),
    K: px(isDark ? "wK" : "bK"),
  }
}

const HomeIcon = ({ size = iconSize }: { size?: number }) => {
  const { P } = usePixelPieces()
  const imageSize = size || iconSize
  return <Image src={P} alt="pawn" width={imageSize} height={imageSize} style={{pointerEvents:"none"}} unoptimized />
}
const YoloIcon = ({ size = iconSize }: { size?: number }) => {
  const { R } = usePixelPieces()
  const imageSize = size || iconSize
  return <Image src={R} alt="rook" width={imageSize} height={imageSize} style={{pointerEvents:"none"}} unoptimized />
}
const StatsIcon = ({ size = iconSize }: { size?: number }) => {
  const { N } = usePixelPieces()
  const imageSize = size || iconSize
  return <Image src={N} alt="knight" width={imageSize} height={imageSize} style={{pointerEvents:"none"}} unoptimized />
}
const ChessIcon = ({ size = iconSize }: { size?: number }) => {
  const { B } = usePixelPieces()
  const imageSize = size || iconSize
  return <Image src={B} alt="bishop" width={imageSize} height={imageSize} style={{pointerEvents:"none"}} unoptimized />
}
const ConfigIcon = ({ size = iconSize }: { size?: number }) => {
  const { Q } = usePixelPieces()
  const imageSize = size || iconSize
  return <Image src={Q} alt="queen" width={imageSize} height={imageSize} style={{pointerEvents:"none"}} unoptimized />
}


const navigation = [
  {
    name: "Dashboard",
    href: "/",
    icon: HomeIcon,
  },
  {
    name: "YOLO",
    href: "/yolo",
    icon: YoloIcon,
  },
  {
    name: "Chess",
    href: "/chess",
    icon: ChessIcon,
  },
  {
    name: "Stats",
    href: "/docs",
    icon: StatsIcon,
  },
  {
    name: "Labels",
    href: "",
    icon: ConfigIcon,
  }
]

// Composant pour le contenu de la sidebar (réutilisable)
function SidebarContent({ isMobile = false }: { isMobile?: boolean }) {
  const pathname = usePathname()
  const { K } = usePixelPieces()

  return (
    <div className="flex flex-col h-full border-r border-border w-full">
      {/* Header */}
      <div className={`${isMobile ? 'h-14 sm:h-16 px-3 sm:px-4' : 'h-14 sm:h-16'} flex items-center ${isMobile ? 'justify-start' : 'justify-center'} border-b border-border`}>
        {isMobile ? (
          <>
            <Link href="/" className="hover:opacity-80 transition-opacity duration-150 flex items-center gap-2">
              <Image src={K} alt="king" width={28} height={28} style={{pointerEvents:"none"}} unoptimized />
            </Link>
          </>
        ) : (
          <Tooltip>
            <TooltipTrigger asChild>
              <Link href="/" className="hover:opacity-80 transition-opacity duration-150">
                <Image src={K} alt="king" width={28} height={28} style={{pointerEvents:"none"}} unoptimized />
              </Link>
            </TooltipTrigger>
            <TooltipContent side="right">
              <span className="font-medium">Accueil</span>
            </TooltipContent>
          </Tooltip>
        )}
      </div>

      {/* Navigation */}
      <nav className={`flex-1 py-4 sm:py-6 ${isMobile ? 'px-3 sm:px-4' : 'px-2'} space-y-1 sm:space-y-2`}>
        {navigation.map((item) => {
          const Icon = item.icon
          const isActive = pathname === item.href

          if (isMobile) {
            return (
              <Link key={item.name} href={item.href} className="block">
                <div
                    className={`flex items-center gap-3 h-10 sm:h-12 px-3 rounded-lg transition-colors duration-150 ${
                      isActive
                        ? 'bg-primary/10 text-primary'
                        : 'text-foreground hover:bg-muted/50'
                    }`}
                >
                  <Icon size={24} />
                  <span className="font-medium text-sm">{item.name}</span>
                </div>
              </Link>
            )
          }

          return (
            <Tooltip key={item.name}>
              <TooltipTrigger asChild>
                <Link href={item.href} className="block">
                  <div
                    className={`flex items-center justify-center h-10 sm:h-12 transition-colors duration-150 ${
                      isActive
                        ? 'text-[#15803d]'
                        : 'text-foreground hover:text-foreground'
                    }`}
                  >
                    <Icon size={24} />
                  </div>
                </Link>
              </TooltipTrigger>
              <TooltipContent side="right">
                <span className="font-medium">{item.name}</span>
              </TooltipContent>
            </Tooltip>
          )
        })}
      </nav>

      {/* Footer */}
      <div className={`border-t border-border ${isMobile ? 'p-4' : 'p-3'}`}>
        <div className={`flex ${isMobile ? 'justify-start' : 'justify-center'}`}>
          <div className={`p-2 rounded-lg hover:bg-muted/30 transition-colors duration-150 ${isMobile ? 'flex items-center gap-2' : ''}`}>
            <ThemeToggle />
            {isMobile}
          </div>
        </div>
      </div>
    </div>
  )
}

export function Sidebar() {
  return (
    <>
      {/* Sidebar Desktop - Cachée sur mobile */}
      <TooltipProvider delayDuration={0}>
        <div className="hidden md:flex fixed left-0 top-0 h-screen w-16 bg-background/95 backdrop-blur-sm z-50">
          <SidebarContent isMobile={false} />
        </div>
      </TooltipProvider>

      {/* Menu Hamburger Mobile */}
      <div className="md:hidden fixed top-2 left-2 z-50">
        <Sheet>
          <SheetTrigger asChild>
            <Button variant="outline" size="icon" className="h-9 w-9 sm:h-10 sm:w-10">
              <Menu className="h-4 w-4 sm:h-5 sm:w-5" />
              <span className="sr-only">Ouvrir le menu</span>
            </Button>
          </SheetTrigger>
          <SheetContent side="left" className="w-[280px] sm:w-64 p-0">
            <SheetTitle className="sr-only">Menu de navigation</SheetTitle>
            <SidebarContent isMobile={true} />
          </SheetContent>
        </Sheet>
      </div>
    </>
  )
}

