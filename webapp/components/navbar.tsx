"use client"

import * as React from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { Brain } from "lucide-react"
import { ThemeToggle } from "@/components/theme-toggle"

const navigation = [
  {
    name: "Dashboard",
    href: "/",
  },
  {
    name: "YOLO",
    href: "/yolo",
  },
  {
    name: "Documentation",
    href: "/docs",
  },
  {
    name: "Chess",
    href: "/chess",
  },
]

export function Navbar() {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = React.useState(false)
  const pathname = usePathname()

  // Gestion des raccourcis clavier
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Vérifier si cmd/ctrl est pressé
      if (e.metaKey || e.ctrlKey) {
        // Cmd+1 pour Dashboard
        if (e.key === "1") {
          e.preventDefault()
          window.location.href = "/"
        }
        // Cmd+2 pour Test
        else if (e.key === "2") {
          e.preventDefault()
          window.location.href = "/test"
        }
        // Cmd+3 pour Docs
        else if (e.key === "3") {
          e.preventDefault()
          window.location.href = "/docs"
        }
        // Cmd+4 pour Chess
        else if (e.key === "4") {
          e.preventDefault()
          window.location.href = "/chess"
        }
        // Échapper pour fermer le menu mobile
        else if (e.key === "Escape" && isMobileMenuOpen) {
          e.preventDefault()
          setIsMobileMenuOpen(false)
        }
      }
    }

    window.addEventListener("keydown", handleKeyDown)

    return () => {
      window.removeEventListener("keydown", handleKeyDown)
    }
  }, [isMobileMenuOpen])

  return (
    <>
      <div className="fixed top-3 left-0 w-full z-50 flex justify-center transition-opacity duration-300">
        <nav className="max-w-5xl h-12 w-full filter backdrop-blur-sm bg-background/60 rounded-lg border-1 border-border border-dashed mx-4">
          <div className="px-4 h-full">
            <div className="flex items-center justify-between h-full">
              <Link
                href="/"
                className="flex items-center space-x-2 text-lg font-bold hover:opacity-80 transition-opacity"
              >
                <Brain className="h-5 w-5" />
                <span>LIF Project</span>
              </Link>
              
              <div className="hidden md:flex items-center space-x-6">
                {navigation.map((item) => (
                  <Link
                    key={item.name}
                    href={item.href}
                    className="relative cursor-pointer group"
                  >
                    <span className={`relative z-10 transition-colors ${
                      pathname === item.href 
                        ? 'text-foreground font-medium' 
                        : 'text-foreground/70 hover:text-foreground'
                    }`}>
                      {item.name}
                    </span>
                    <span
                      className={`absolute left-0 -bottom-0.5 h-[1px] transition-all duration-300 ${
                        pathname === item.href 
                          ? 'w-full' 
                          : 'w-0 group-hover:w-full'
                      }`}
                      style={{
                        backgroundSize: "6px 1px",
                        backgroundImage:
                          "linear-gradient(to right, var(--border) 50%, transparent 50%)",
                        backgroundRepeat: "repeat-x",
                      }}
                      aria-hidden="true"
                    />
                  </Link>
                ))}
              </div>

              <div className="flex items-center gap-2">
                <div className="hidden md:flex items-center space-x-2">
                  <ThemeToggle />
                </div>
                
                <div className="md:hidden">
                  <button
                    onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
                    className="p-2 rounded-md text-foreground hover:bg-muted transition-colors"
                    aria-label="Open main menu"
                  >
                    {isMobileMenuOpen ? (
                      <svg
                        className="h-5 w-5"
                        xmlns="http://www.w3.org/2000/svg"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                        aria-hidden="true"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth="2"
                          d="M6 18L18 6M6 6l12 12"
                        />
                      </svg>
                    ) : (
                      <svg
                        className="h-5 w-5"
                        xmlns="http://www.w3.org/2000/svg"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                        aria-hidden="true"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth="2"
                          d="M4 6h16M4 12h16m-7 6h7"
                        />
                      </svg>
                    )}
                  </button>
                </div>
              </div>
            </div>
          </div>

          {isMobileMenuOpen && (
            <div className="md:hidden absolute right-0 mt-2 w-48 bg-background/90 backdrop-blur-sm rounded-lg border border-border py-2 shadow-lg">
              {navigation.map((item) => (
                <Link
                  key={item.name}
                  href={item.href}
                  className={`block px-4 py-2 text-sm transition-colors hover:bg-muted ${
                    pathname === item.href 
                      ? 'text-foreground font-medium bg-muted/50' 
                      : 'text-foreground/70'
                  }`}
                  onClick={() => setIsMobileMenuOpen(false)}
                >
                  {item.name}
                </Link>
              ))}
              <hr className="border-border my-2" />
              <div className="px-4 py-2">
                <ThemeToggle />
              </div>
            </div>
          )}
        </nav>
      </div>
      
      {/* Spacer to push content down */}
      <div className="h-16" />
    </>
  )
}
