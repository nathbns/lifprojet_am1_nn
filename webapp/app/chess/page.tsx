"use client"

import { useState, useCallback, useMemo } from "react"
import { Chess } from "chess.js"
import { Chessboard } from "react-chessboard"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { RotateCcw, Trophy, AlertCircle, Palette, Camera, Bot, X } from "lucide-react"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { FileUpload } from "@/components/ui/file-upload"
import { GlowingEffect } from "@/components/ui/glowing-effect"
import { Highlighter } from "@/components/ui/highlighter"
import type { Square } from "chess.js"
import Image from "next/image"

// Thèmes d'échiquier disponibles
const boardThemes = {
  green: {
    name: "Vert",
    lightSquare: "#ffffdd",
    darkSquare: "#86a666"
  },
  brown: {
    name: "Bois Classique",
    lightSquare: "#f0d9b5",
    darkSquare: "#b58863"
  },
  blue: {
    name: "Bleu",
    lightSquare: "#dee3e6",
    darkSquare: "#8ca2ad"
  },
  purple: {
    name: "Violet",
    lightSquare: "#e8e4f3",
    darkSquare: "#9f90b0"
  },
  gray: {
    name: "Gris Moderne",
    lightSquare: "#e8e8e8",
    darkSquare: "#6c6c6c"
  },
  maroon: {
    name: "Marron",
    lightSquare: "#efdfbb",
    darkSquare: "#c4856c"
  },
  coral: {
    name: "Corail",
    lightSquare: "#ffd6c9",
    darkSquare: "#ea8565"
  },
  dark: {
    name: "Sombre",
    lightSquare: "#4a4a4a",
    darkSquare: "#2b2b2b"
  }
}

const pieceSymbols: { [key: string]: string } = {
  p: "♟", n: "♞", b: "♝", r: "♜", q: "♛", k: "♚",
  P: "♙", N: "♘", B: "♗", R: "♖", Q: "♕", K: "♔"
}

export default function ChessPage() {
  const [game, setGame] = useState(new Chess())
  const [gameStatus, setGameStatus] = useState<string>("")
  const [moveHistory, setMoveHistory] = useState<string[]>([])
  const [capturedPieces, setCapturedPieces] = useState<{
    white: string[]
    black: string[]
  }>({ white: [], black: [] })
  const [boardTheme, setBoardTheme] = useState<keyof typeof boardThemes>("green")
  const [gameMode, setGameMode] = useState<"image" | "vs-computer">("image")
  const [imageDataUrl, setImageDataUrl] = useState<string>("")

  const currentTheme = useMemo(() => boardThemes[boardTheme], [boardTheme])
  const isGameOver = useMemo(() => game.isGameOver(), [game])

  const updateGameStatus = useCallback((currentGame: Chess) => {
    if (currentGame.isCheckmate()) {
      const winner = currentGame.turn() === "w" ? "Noirs" : "Blancs"
      setGameStatus(`Échec et mat! Les ${winner} gagnent !`)
    } else if (currentGame.isStalemate()) {
      setGameStatus("Pat ! Match nul.")
    } else if (currentGame.isDraw()) {
      setGameStatus("Match nul !")
    } else if (currentGame.isThreefoldRepetition()) {
      setGameStatus("Triple répétition ! Match nul.")
    } else if (currentGame.isInsufficientMaterial()) {
      setGameStatus("Matériel insuffisant ! Match nul.")
    } else if (currentGame.isCheck()) {
      setGameStatus("Échec !")
    } else {
      setGameStatus("")
    }
  }, [])

  const onDrop = useCallback(
    (sourceSquare: Square, targetSquare: Square) => {
      try {
        const gameCopy = new Chess(game.fen())
        
        const move = gameCopy.move({
          from: sourceSquare,
          to: targetSquare,
          promotion: "q"
        })

        if (move === null) return false

        // Capturer une pièce
        if (move.captured) {
          const capturedSymbol = move.color === "w" 
            ? pieceSymbols[move.captured.toLowerCase()] 
            : pieceSymbols[move.captured.toUpperCase()]
          
          setCapturedPieces(prev => ({
            ...prev,
            [move.color === "w" ? "white" : "black"]: [
              ...prev[move.color === "w" ? "white" : "black"],
              capturedSymbol
            ]
          }))
        }

        setGame(gameCopy)
        setMoveHistory(prev => [...prev, move.san])
        updateGameStatus(gameCopy)
        
        return true
      } catch {
        return false
      }
    },
    [game, updateGameStatus]
  )

  const resetGame = useCallback(() => {
    const newGame = new Chess()
    setGame(newGame)
    setGameStatus("")
    setMoveHistory([])
    setCapturedPieces({ white: [], black: [] })
  }, [])

  const undoMove = useCallback(() => {
    if (moveHistory.length === 0) return
    
    const gameCopy = new Chess(game.fen())
    gameCopy.undo()
    setGame(gameCopy)
    setMoveHistory(prev => prev.slice(0, -1))
    updateGameStatus(gameCopy)
  }, [game, moveHistory.length, updateGameStatus])

  const handleFileSelect = (file: File) => {
    if (!file.type.startsWith('image/')) return
    const reader = new FileReader()
    reader.onload = () => setImageDataUrl(reader.result as string)
    reader.readAsDataURL(file)
  }

  const handleFileUpload = (files: File[]) => {
    if (files.length > 0) {
      handleFileSelect(files[0])
    }
  }

  return (
    <div className="min-h-screen py-8">
      <div className="max-w-6xl mx-auto px-8">
        <div className="mb-8 text-center">
          <h1 className="text-4xl font-bold mb-2">
            Jeu d&apos;{" "}<Highlighter action="highlight" color="#15803d" padding={3}>
              Échecs
            </Highlighter>
          </h1>
          <p className="text-muted-foreground mb-4">
            <Highlighter action="underline" color="#15803d" padding={3}>
              Choisissez votre mode
            </Highlighter>
            {" "}de jeu et commencez à jouer
          </p>
          
          {/* Mode Switch - Discret */}
          <div className="inline-flex items-center gap-2 bg-muted/30 rounded-lg p-1">
            <Button
              variant={gameMode === "image" ? "default" : "ghost"}
              size="sm"
              onClick={() => setGameMode("image")}
              className="h-8 px-3"
            >
              <Camera className="h-4 w-4 mr-1" />
              Image
            </Button>
            <Button
              variant={gameMode === "vs-computer" ? "default" : "ghost"}
              size="sm"
              onClick={() => setGameMode("vs-computer")}
              className="h-8 px-3"
            >
              <Bot className="h-4 w-4 mr-1" />
              IA
            </Button>
          </div>
        </div>


        {/* Content based on mode */}
        {gameMode === "image" ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Upload Section */}
            <div className="space-y-6">
              <div className="relative">
                <Card>
                  <CardContent className="p-6">
                    {!imageDataUrl ? (
                      <FileUpload onChange={handleFileUpload} />
                    ) : (
                      <div className="space-y-4">
                        <div className="relative inline-block">
                          <Image 
                            src={imageDataUrl} 
                            alt="preview" 
                            width={320}
                            height={240}
                            className="max-h-80 mx-auto rounded-lg" 
                          />
                          <Button
                            variant="destructive"
                            size="sm"
                            className="absolute -top-2 -right-2 rounded-full h-8 w-8 p-0"
                            onClick={() => setImageDataUrl("")}
                          >
                            <X className="h-4 w-4" />
                          </Button>
                        </div>
                        <p className="text-sm text-muted-foreground text-center">Image prête pour l&apos;analyse</p>
                      </div>
                    )}
                  </CardContent>
                </Card>
                <GlowingEffect className="rounded-xl" borderWidth={1} />
              </div>
            </div>

            {/* Settings */}
            <div className="space-y-6">
              <div className="relative">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Palette className="h-5 w-5" />
                      Paramètres
                    </CardTitle>
                    <CardDescription>Personnalisez votre échiquier</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Thème Échiquier</label>
                      <Select 
                        value={boardTheme} 
                        onValueChange={(value) => setBoardTheme(value as keyof typeof boardThemes)}
                      >
                        <SelectTrigger className="w-full">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {Object.entries(boardThemes).map(([key, theme]) => (
                            <SelectItem key={key} value={key}>
                              {theme.name}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>

                    {/* Aperçu du thème */}
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Aperçu</label>
                      <div className="grid grid-cols-4 gap-0 h-20 rounded-md overflow-hidden border-2 border-border">
                        {[0, 1, 2, 3, 0, 1, 2, 3].map((_, idx) => (
                          <div
                            key={idx}
                            style={{
                              backgroundColor: idx % 2 === Math.floor(idx / 4) % 2 
                                ? currentTheme.lightSquare 
                                : currentTheme.darkSquare
                            }}
                          />
                        ))}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-8 gap-6">
        {/* Panneau de gauche */}
        <div className="lg:col-span-2 space-y-4">
          {/* Personnalisation */}
          <div className="relative">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Palette className="h-5 w-5" />
                  Personnalisation
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Thème Échiquier</label>
                  <Select 
                    value={boardTheme} 
                    onValueChange={(value) => setBoardTheme(value as keyof typeof boardThemes)}
                  >
                    <SelectTrigger className="w-full">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {Object.entries(boardThemes).map(([key, theme]) => (
                        <SelectItem key={key} value={key}>
                          {theme.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                      
                {/* Aperçu du thème */}
                <div className="space-y-2">
                  <label className="text-sm font-medium">Aperçu</label>
                  <div className="grid grid-cols-4 gap-0 h-20 rounded-md overflow-hidden border-2 border-border">
                    {[0, 1, 2, 3, 0, 1, 2, 3].map((_, idx) => (
                      <div
                        key={idx}
                        style={{
                          backgroundColor: idx % 2 === Math.floor(idx / 4) % 2 
                            ? currentTheme.lightSquare 
                            : currentTheme.darkSquare
                        }}
                      />
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
            <GlowingEffect className="rounded-xl" borderWidth={1} />
          </div>
          

          {/* Statut de la partie */}
          <div className="relative">
            <Card className={isGameOver ? "border-2 border-primary" : ""}>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  {isGameOver ? <Trophy className="h-5 w-5" /> : <AlertCircle className="h-5 w-5" />}
                  Partie en cours
                </CardTitle>
                <CardDescription>
                  {!isGameOver && (
                    <span className="font-bold">
                      Tour : {game.turn() === "w" ? "Blancs" : "Noirs"}
                    </span>
                  )}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                {gameStatus && (
                  <div className={`p-3 rounded-lg font-bold text-center ${
                    gameStatus.includes("Échec et mat") 
                      ? "bg-green-500/20 text-green-700 dark:text-green-300" 
                      : gameStatus.includes("Échec")
                      ? "bg-red-500/20 text-red-700 dark:text-red-300"
                      : "bg-blue-500/20 text-blue-700 dark:text-blue-300"
                  }`}>
                    {gameStatus}
                  </div>
                )}
                
                <div className="flex gap-2">
                  <Button onClick={resetGame} className="flex-1" variant="outline">
                    <RotateCcw className="mr-2 h-4 w-4" />
                    Nouveau
                  </Button>
                  <Button 
                    onClick={undoMove} 
                    className="flex-1" 
                    variant="outline"
                    disabled={moveHistory.length === 0}
                  >
                    Annuler
                  </Button>
                </div>
              </CardContent>
            </Card>
            <GlowingEffect className="rounded-xl" borderWidth={1} />
          </div>

          {/* Pièces capturées */}
          <div className="relative">
            <Card>
              <CardHeader>
                <CardTitle>Pièces Capturées</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <h3 className="text-sm font-semibold mb-2 text-muted-foreground">
                    Blancs ({capturedPieces.white.length})
                  </h3>
                  <div className="flex flex-wrap gap-1 min-h-[40px] p-2 bg-muted/30 rounded-md border border-border">
                    {capturedPieces.white.map((piece, idx) => (
                      <span key={idx} className="text-2xl">
                        {piece}
                      </span>
                    ))}
                  </div>
                </div>
                <div>
                  <h3 className="text-sm font-semibold mb-2 text-muted-foreground">
                    Noirs ({capturedPieces.black.length})
                  </h3>
                  <div className="flex flex-wrap gap-1 min-h-[40px] p-2 bg-muted/30 rounded-md border border-border">
                    {capturedPieces.black.map((piece, idx) => (
                      <span key={idx} className="text-2xl">
                        {piece}
                      </span>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
            <GlowingEffect className="rounded-xl" borderWidth={1} />
          </div>
        </div>

        {/* Panneau central - Échiquier */}
        <div className="lg:col-span-6 flex flex-col items-center justify-center">
          <div className="relative">
            <Card className="w-fit mx-auto max-w-[500px] sm:max-w-[700px] lg:max-w-[900px] shadow-2xl overflow-hidden">
              <div className="p-2 lg:p-4 bg-card/80">
                <Chessboard
                  position={game.fen()}
                  onPieceDrop={onDrop}
                  boardWidth={600}
                  customBoardStyle={{
                    borderRadius: "8px"
                  }}
                  customLightSquareStyle={{ 
                    backgroundColor: currentTheme.lightSquare
                  }}
                  customDarkSquareStyle={{ 
                    backgroundColor: currentTheme.darkSquare
                  }}
                />
              </div>
            </Card>
            <GlowingEffect className="rounded-xl" borderWidth={1} />
          </div>
          <p className="mt-4 text-sm text-center text-muted-foreground">
            Glissez-déposez les pièces pour jouer
          </p>
        </div>

      </div>
        )}
      </div>
    </div>
  )
}
