"use client"

import * as React from "react"
import Link from "next/link"
import Image from "next/image"
import { usePathname } from "next/navigation"
import { ThemeToggle } from "@/components/theme-toggle"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { FaChess, FaChessPawn, FaChessRook, FaChessKnight, FaChessBishop } from "react-icons/fa"

// Icônes de pièces d'échecs pour la navigation
const HomeIcon = () => <FaChessPawn className="w-5 h-5" />
const YoloIcon = () => <FaChessRook className="w-5 h-5" />
const DocsIcon = () => <FaChessKnight className="w-5 h-5" />
const ChessIcon = () => <FaChessBishop className="w-5 h-5" />


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
    name: "Docs",
    href: "/docs",
    icon: DocsIcon,
  },
  {
    name: "Chess",
    href: "/chess",
    icon: ChessIcon,
  },
]

export function Sidebar() {
  const pathname = usePathname()

  return (
    <TooltipProvider delayDuration={0}>
      <div className="fixed left-0 top-0 h-screen w-16 bg-background/95 backdrop-blur-sm border-r border-border z-50">
        <div className="flex flex-col h-full">
        {/* Header */}
        <div className="h-16 flex items-center justify-center border-b border-border">
          <Tooltip>
            <TooltipTrigger asChild>
              <Link href="/" className="hover:opacity-80 transition-opacity duration-150">
                <FaChess className="w-6 h-6" />
              </Link>
            </TooltipTrigger>
            <TooltipContent side="right">
              <span className="font-medium">Accueil</span>
            </TooltipContent>
          </Tooltip>
        </div>

        {/* Navigation */}
        <nav className="flex-1 py-6 px-2 space-y-2">
          {navigation.map((item) => {
            const Icon = item.icon
            const isActive = pathname === item.href

            return (
              <Tooltip key={item.name}>
                <TooltipTrigger asChild>
                  <Link href={item.href} className="block">
                    <div
                      className={`flex items-center justify-center h-12 transition-colors duration-150 ${
                        isActive
                          ? 'text-primary'
                          : 'text-muted-foreground hover:text-foreground'
                      }`}
                    >
                      <Icon />
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
        <div className="border-t border-border p-3">
          <div className="flex justify-center">
            <div className="p-2 rounded-lg hover:bg-muted/30 transition-colors duration-150">
              <ThemeToggle />
            </div>
          </div>
        </div>
        </div>
      </div>
    </TooltipProvider>
  )
}

