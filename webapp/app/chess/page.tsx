"use client"

import { useState, useCallback, useMemo, useEffect, useRef } from "react"
import * as React from "react"
import { Chess } from "chess.js"
import { Chessboard } from "react-chessboard"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { AlertCircle, Camera, Bot, X, Scan, CheckCircle2, Loader2, Copy, ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight, Flag, Trash2 } from "lucide-react"
import { FileUpload } from "@/components/ui/file-upload"
import { Highlighter } from "@/components/ui/highlighter"
import type { Square } from "chess.js"
import Image from "next/image"
import { User, Zap } from "lucide-react"
import { analyzeChessImage, preprocessChessImage } from "@/lib/gradio"
import { MinimaxAI } from "@/lib/minimax"

// Fonction helper pour obtenir l'élément JSX d'une pièce capturée
const getPieceSymbol = (piece: string, size: number = 20): React.ReactElement => {
  const pieceUrls: { [key: string]: string } = {
    p: "https://lichess1.org/assets/piece/pixel/bP.svg",
    n: "https://lichess1.org/assets/piece/pixel/bN.svg",
    b: "https://lichess1.org/assets/piece/pixel/bB.svg",
    r: "https://lichess1.org/assets/piece/pixel/bR.svg",
    q: "https://lichess1.org/assets/piece/pixel/bQ.svg",
    k: "https://lichess1.org/assets/piece/pixel/bK.svg",
    P: "https://lichess1.org/assets/piece/pixel/wP.svg",
    N: "https://lichess1.org/assets/piece/pixel/wN.svg",
    B: "https://lichess1.org/assets/piece/pixel/wB.svg",
    R: "https://lichess1.org/assets/piece/pixel/wR.svg",
    Q: "https://lichess1.org/assets/piece/pixel/wQ.svg",
    K: "https://lichess1.org/assets/piece/pixel/wK.svg",
  }
  
  return (
    <Image 
      src={pieceUrls[piece]} 
      alt={piece} 
      width={size}
      height={size}
      style={{ width: size, height: size, pointerEvents: "none" }}
      unoptimized
    />
  )
}

export default function ChessPage() {
  const [game, setGame] = useState(new Chess())
  const [gameStatus, setGameStatus] = useState<string>("")
  const [moveHistory, setMoveHistory] = useState<string[]>([])
  const [capturedPieces, setCapturedPieces] = useState<{
    white: React.ReactElement[]
    black: React.ReactElement[]
  }>({ white: [], black: [] })
  const [gameMode, setGameMode] = useState<"image" | "config" | "playing">("image")
  const [playerColor, setPlayerColor] = useState<"white" | "black">("white")
  const [timeControl, setTimeControl] = useState<number>(10) // minutes
  const [playerTime, setPlayerTime] = useState<number>(600) // secondes
  const [aiTime, setAiTime] = useState<number>(600) // secondes
  const [isTimerActive, setIsTimerActive] = useState<boolean>(false)
  const [imageDataUrl, setImageDataUrl] = useState<string>("")
  const [isDetecting, setIsDetecting] = useState<boolean>(false)
  const [detectedFen, setDetectedFen] = useState<string>("")
  const [detectionError, setDetectionError] = useState<string>("")
  const [manualFen, setManualFen] = useState<string>("")
  const [preprocessedImageUrl, setPreprocessedImageUrl] = useState<string>("")
  const [preprocessingStep, setPreprocessingStep] = useState<string>("")
  const [blurAmount, setBlurAmount] = useState<number>(20)
  const [correctedFen, setCorrectedFen] = useState<string>("")
  const [accuracyScore, setAccuracyScore] = useState<number | null>(null)
  const [correctionGame, setCorrectionGame] = useState<Chess | null>(null)
  const [selectedSquare, setSelectedSquare] = useState<Square | null>(null)
  const [selectedPieceToPlace, setSelectedPieceToPlace] = useState<string | null>(null) // Format: "wP", "bK", etc.
  const [currentMoveIndex, setCurrentMoveIndex] = useState<number>(-1)
  const [boardSize, setBoardSize] = useState<number>(600)
  const [containerHeight, setContainerHeight] = useState<number>(0)
  const boardContainerRef = useRef<HTMLDivElement>(null)
  const leftPanelRef = useRef<HTMLDivElement>(null)
  const rightPanelRef = useRef<HTMLDivElement>(null)
  const isGameOver = useMemo(() => game.isGameOver(), [game])
  const [aiDepth, setAiDepth] = useState<number>(3)
  const [isAiThinking, setIsAiThinking] = useState<boolean>(false)
  const aiRef = useRef<MinimaxAI>(new MinimaxAI(3))

  // Calculer la taille de l'échiquier basée sur la largeur disponible
  useEffect(() => {
    const updateBoardSize = () => {
      if (boardContainerRef.current) {
        const containerWidth = boardContainerRef.current.clientWidth
        // Utiliser la largeur disponible (moins un peu de padding pour le gap)
        // Mobile: utiliser presque toute la largeur, Desktop: max 600px
        const isMobile = window.innerWidth < 768
        const padding = isMobile ? 16 : 32
        const calculatedSize = Math.min(600, containerWidth - padding)
        const newBoardSize = Math.max(isMobile ? 300 : 400, calculatedSize) // Minimum 300px sur mobile, 400px sur desktop
        setBoardSize(newBoardSize)
      }
    }
    
    // Utiliser ResizeObserver pour détecter les changements de taille du conteneur
    if (boardContainerRef.current) {
      const resizeObserver = new ResizeObserver(updateBoardSize)
      resizeObserver.observe(boardContainerRef.current)
      
      // Délai pour s'assurer que le DOM est rendu
      const timeoutId = setTimeout(updateBoardSize, 100)
      
      return () => {
        resizeObserver.disconnect()
        clearTimeout(timeoutId)
      }
    }
  }, [gameMode])

  // Calculer la hauteur nécessaire basée sur la taille de l'échiquier (carré + padding)
  useEffect(() => {
    // La hauteur du conteneur = taille de l'échiquier + padding (16px)
    const calculatedHeight = boardSize + 16
    setContainerHeight(calculatedHeight)
  }, [boardSize])

  const updateGameStatus = useCallback((currentGame: Chess) => {
    if (currentGame.isCheckmate()) {
      const winner = currentGame.turn() === "w" ? "Noirs" : "Blancs"
      setGameStatus(`Échec et mat! Les ${winner} gagnent !`)
      setIsTimerActive(false)
    } else if (currentGame.isStalemate()) {
      setGameStatus("Pat ! Match nul.")
      setIsTimerActive(false)
    } else if (currentGame.isDraw()) {
      setGameStatus("Match nul !")
      setIsTimerActive(false)
    } else if (currentGame.isThreefoldRepetition()) {
      setGameStatus("Triple répétition ! Match nul.")
      setIsTimerActive(false)
    } else if (currentGame.isInsufficientMaterial()) {
      setGameStatus("Matériel insuffisant ! Match nul.")
      setIsTimerActive(false)
    } else if (currentGame.isCheck()) {
      setGameStatus("Échec !")
    } else {
      setGameStatus("")
    }
  }, [])

  // Mettre à jour la profondeur de l'IA
  useEffect(() => {
    aiRef.current.setDepth(aiDepth)
  }, [aiDepth])

  // Timer pour le temps de jeu
  useEffect(() => {
    // Ne pas faire tourner le timer si on n'est pas sur le dernier coup
    const isOnLatestMove = currentMoveIndex === moveHistory.length - 1 || (currentMoveIndex === -1 && moveHistory.length === 0)
    
    if (!isTimerActive || isGameOver || gameMode !== "playing" || !isOnLatestMove) return

    const interval = setInterval(() => {
      const isPlayerTurn = (game.turn() === 'w' && playerColor === 'white') || 
                           (game.turn() === 'b' && playerColor === 'black')
      
      if (isPlayerTurn) {
        setPlayerTime(prev => {
          if (prev <= 0) {
            setGameStatus("Temps écoulé ! Vous avez perdu.")
            setIsTimerActive(false)
            return 0
          }
          return prev - 1
        })
      } else {
        setAiTime(prev => {
          if (prev <= 0) {
            setGameStatus("Temps écoulé ! L'IA a perdu. Vous gagnez !")
            setIsTimerActive(false)
            return 0
          }
          return prev - 1
        })
      }
    }, 1000)

    return () => clearInterval(interval)
  }, [isTimerActive, game, gameMode, isGameOver, playerColor, currentMoveIndex, moveHistory.length])

  // Faire jouer l'IA automatiquement quand c'est son tour
  useEffect(() => {
    const isOnLatestMove = currentMoveIndex === moveHistory.length - 1 || (currentMoveIndex === -1 && moveHistory.length === 0)
    const aiColor = playerColor === 'white' ? 'b' : 'w'
    
    if (
      gameMode === "playing" &&
      game.turn() === aiColor &&
      !isGameOver &&
      !isAiThinking &&
      isOnLatestMove
    ) {
      const makeAiMove = async () => {
        setIsAiThinking(true)
        
        // Ajouter un petit délai pour que l'utilisateur voit que l'IA réfléchit
        await new Promise(resolve => setTimeout(resolve, 300))
        
        try {
          // Obtenir le meilleur coup de l'IA
          const result = aiRef.current.getBestMove(game)
          
          if (result.move) {
            const gameCopy = new Chess(game.fen())
            const move = gameCopy.move(result.move)
            
            if (move) {
              // Capturer une pièce si nécessaire
              if (move.captured) {
                const capturedPiece = move.color === "w" 
                  ? move.captured.toLowerCase()
                  : move.captured.toUpperCase()
                const capturedSymbol = getPieceSymbol(capturedPiece)
                
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
              setCurrentMoveIndex(moveHistory.length)
              updateGameStatus(gameCopy)
              
              console.log(`IA: ${move.san} (Score: ${result.score}, Nœuds: ${result.nodesEvaluated})`)
            }
          }
        } catch (error) {
          console.error("Erreur lors du calcul du coup de l'IA:", error)
        } finally {
          setIsAiThinking(false)
        }
      }
      
      makeAiMove()
    }
  }, [game, gameMode, isGameOver, isAiThinking, currentMoveIndex, moveHistory.length, updateGameStatus, playerColor])

  const onDrop = useCallback(
    (sourceSquare: Square, targetSquare: Square) => {
      // Empêcher le joueur de jouer si l'IA réfléchit
      if (isAiThinking) return false
      
      // En mode playing, vérifier que c'est bien le tour du joueur
      if (gameMode === "playing") {
        const playerTurn = playerColor === 'white' ? 'w' : 'b'
        if (game.turn() !== playerTurn) return false
      }
      
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
          // move.captured retourne toujours le type en minuscule (ex: "p", "n", etc.)
          // La couleur de la pièce capturée est l'inverse de move.color :
          // - Si les blancs capturent (move.color === "w"), la pièce capturée est noire → utiliser minuscule
          // - Si les noirs capturent (move.color === "b"), la pièce capturée est blanche → utiliser majuscule
          const capturedPiece = move.color === "w" 
            ? move.captured.toLowerCase()  // Pièce noire capturée par les blancs
            : move.captured.toUpperCase() // Pièce blanche capturée par les noirs
          const capturedSymbol = getPieceSymbol(capturedPiece)
          
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
        setCurrentMoveIndex(moveHistory.length) // Suivre le dernier coup
        updateGameStatus(gameCopy)
        
        return true
      } catch {
        return false
      }
    },
    [game, updateGameStatus, moveHistory.length, isAiThinking, gameMode, playerColor]
  )

  const goToMove = useCallback((index: number) => {
    if (index < -1 || index >= moveHistory.length) return
    setCurrentMoveIndex(index)
    
    // Recalculer les pièces capturées jusqu'à l'index spécifié
    const newCapturedPieces: { white: React.ReactElement[], black: React.ReactElement[] } = { white: [], black: [] }
    
    if (index === -1) {
      // Retour au début (position initiale ou FEN chargé)
      const initialFen = manualFen.trim() || "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
      try {
        const tempGame = new Chess(initialFen)
        setGame(tempGame)
        updateGameStatus(tempGame)
      } catch {
        const tempGame = new Chess()
        setGame(tempGame)
        updateGameStatus(tempGame)
      }
      // Réinitialiser les pièces capturées
      setCapturedPieces({ white: [], black: [] })
    } else {
      // Aller à un coup spécifique et recalculer les pièces capturées
      const initialFen = manualFen.trim() || "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
      try {
        const tempGame = new Chess(initialFen)
        for (let i = 0; i <= index; i++) {
          const move = tempGame.move(moveHistory[i])
          if (move && move.captured) {
            // move.captured retourne toujours le type en minuscule
            // Convertir selon la couleur du joueur qui capture
            const capturedPiece = move.color === "w" 
              ? move.captured.toLowerCase()  // Pièce noire capturée par les blancs
              : move.captured.toUpperCase() // Pièce blanche capturée par les noirs
            const capturedSymbol = getPieceSymbol(capturedPiece)
            if (move.color === "w") {
              newCapturedPieces.white.push(capturedSymbol)
            } else {
              newCapturedPieces.black.push(capturedSymbol)
            }
          }
        }
        setGame(tempGame)
        updateGameStatus(tempGame)
        setCapturedPieces(newCapturedPieces)
      } catch {
        // Si le FEN initial ne fonctionne pas, utiliser la position de départ standard
        const tempGame = new Chess()
        for (let i = 0; i <= index; i++) {
          const move = tempGame.move(moveHistory[i])
          if (move && move.captured) {
            // move.captured retourne toujours le type en minuscule
            // Convertir selon la couleur du joueur qui capture
            const capturedPiece = move.color === "w" 
              ? move.captured.toLowerCase()  // Pièce noire capturée par les blancs
              : move.captured.toUpperCase() // Pièce blanche capturée par les noirs
            const capturedSymbol = getPieceSymbol(capturedPiece)
            if (move.color === "w") {
              newCapturedPieces.white.push(capturedSymbol)
            } else {
              newCapturedPieces.black.push(capturedSymbol)
            }
          }
        }
        setGame(tempGame)
        updateGameStatus(tempGame)
        setCapturedPieces(newCapturedPieces)
      }
    }
  }, [moveHistory, manualFen, updateGameStatus])

  const handleFileSelect = (file: File) => {
    if (!file.type.startsWith('image/')) return
    const reader = new FileReader()
    reader.onload = () => setImageDataUrl(reader.result as string)
    reader.readAsDataURL(file)
  }

  const handleFileUpload = (files: File[]) => {
    if (files.length > 0) {
      handleFileSelect(files[0])
      setDetectedFen("")
      setDetectionError("")
      setBlurAmount(0) // Réinitialiser le blur pour la nouvelle image
    }
  }

  const handleDetectFen = async () => {
    if (!imageDataUrl) return
    
    setIsDetecting(true)
    setDetectionError("")
    setPreprocessedImageUrl("")
    setPreprocessingStep("")
    setBlurAmount(20) // Commencer avec un blur fort
    
    try {
      // Étape 1: Préprocessing avec animation de blur synchronisée
      setPreprocessingStep("Préprocessing de l'image...")
      
      // Variables partagées pour l'animation (utiliser un objet pour mutabilité)
      const startTime = Date.now()
      const initialBlur = 20
      const animationState = {
        actualDuration: null as number | null,
        animationId: null as number | null,
      }
      
      const animateBlur = () => {
        const elapsed = Date.now() - startTime
        
        if (animationState.actualDuration !== null) {
          // Le preprocessing est terminé, utiliser la durée réelle
          const progress = Math.min(elapsed / animationState.actualDuration, 1)
          const currentBlur = initialBlur * (1 - progress)
          setBlurAmount(Math.max(0, currentBlur))
          
          // Si le blur n'est pas encore à 0, continuer l'animation
          if (currentBlur > 0) {
            animationState.animationId = requestAnimationFrame(animateBlur)
          }
        } else {
          // Le preprocessing est en cours, animer progressivement
          // On utilise une estimation très généreuse pour éviter que le blur atteigne 0 trop tôt
          const estimatedDuration = 30000 // Estimation de 30 secondes (très généreuse)
          const progress = Math.min(elapsed / estimatedDuration, 0.95) // Limiter à 95% pour garder un peu de blur
          const currentBlur = initialBlur * (1 - progress)
          
          setBlurAmount(Math.max(0, currentBlur))
          
          // Continuer l'animation jusqu'à ce que le preprocessing soit terminé
          animationState.animationId = requestAnimationFrame(animateBlur)
        }
      }
      
      // Démarrer l'animation
      animationState.animationId = requestAnimationFrame(animateBlur)
      
      // Lancer le preprocessing en parallèle (appel direct Gradio côté client)
      const preprocessedImageUrl = await preprocessChessImage(imageDataUrl)
      
      // Le preprocessing est terminé, enregistrer la durée réelle
      animationState.actualDuration = Date.now() - startTime
      
      // L'animation continuera automatiquement avec la durée réelle
      // grâce à la vérification dans animateBlur
      
      // Attendre un peu pour que l'animation finisse si nécessaire
      await new Promise(resolve => setTimeout(resolve, 100))
      
      // S'assurer que le blur est complètement enlevé
      setBlurAmount(0)
      
      setPreprocessedImageUrl(preprocessedImageUrl)
      setPreprocessingStep("Image préprocessée ✓")
      
      // Petite pause pour voir l'image préprocessée
      await new Promise(resolve => setTimeout(resolve, 500))
      
      // Étape 2: Analyse avec le modèle pour détecter le FEN
      // IMPORTANT: Utiliser l'image originale, pas l'image préprocessée
      setPreprocessingStep("Analyse avec le modèle...")
      const result = await analyzeChessImage(imageDataUrl)
      
      if (result.fen) {
        
        // Mettre le FEN nettoyé directement dans l'input
        setDetectedFen(result.fen)
        setManualFen(result.fen)
        setPreprocessingStep("FEN détecté!")
        // Réinitialiser le score et la correction
        setCorrectedFen("")
        setAccuracyScore(null)
        // Créer un échiquier de correction avec le FEN détecté
        try {
          const correctionBoard = new Chess(result.fen)
          setCorrectionGame(correctionBoard)
          // Pour l'instant, on ne calcule pas le score initial car on compare avec detectedFen
          // Le score sera calculé automatiquement dès que l'utilisateur modifie l'échiquier
        } catch (err) {
          console.error("Erreur création échiquier de correction:", err)
          // Si le FEN détecté est invalide (ex: rois manquants ou multiples), créer un échiquier de correction
          // en sanitant uniquement pour l'affichage/correction.
          try {
            const parts = result.fen.split(' ')
            const boardStr = parts[0]
            const board: (string | null)[][] = Array(8).fill(null).map(() => Array(8).fill(null))
            let row = 0
            let col = 0
            for (const ch of boardStr) {
              if (ch === '/') {
                row++
                col = 0
              } else if (ch >= '1' && ch <= '8') {
                col += parseInt(ch)
              } else {
                board[row][col] = ch
                col++
              }
            }
            const boardCopy = sanitizeBoardKings(board)
            const tempFenValid = buildFenFromBoard(
              boardCopy,
              parts[1] || 'w',
              parts[2] || '-',
              parts[3] || '-',
              parseInt(parts[4]) || 0,
              parseInt(parts[5]) || 1
            )
            const correctionBoard = new Chess(tempFenValid)
            setCorrectionGame(correctionBoard)
          } catch (e) {
            console.error("Fallback correction board creation failed:", e)
          }
        }
      } else {
        setDetectionError("Aucun FEN n'a pu être détecté")
        setPreprocessingStep("")
      }
    } catch (error) {
      console.error("Erreur détection:", error)
      setDetectionError(
        error instanceof Error 
          ? error.message 
          : "Erreur lors de la détection de l'échiquier"
      )
      setPreprocessingStep("")
      setBlurAmount(0)
    } finally {
      setIsDetecting(false)
    }
  }

  const handleContinueGame = () => {
    // Utiliser le FEN corrigé s'il existe, sinon le FEN de l'échiquier de correction, sinon le FEN manuel
    const fenToUse = correctedFen || correctionGame?.fen() || manualFen.trim()
    if (!fenToUse) return
    
    // Charger le FEN et basculer vers le mode configuration
    try {
      const newGame = new Chess(fenToUse)
      setGame(newGame)
      updateGameStatus(newGame)
      setMoveHistory([])
      setCapturedPieces({ white: [], black: [] })
      setDetectionError("")
      setCurrentMoveIndex(-1)
      setGameMode("config")
    } catch (_err) {
      // Ne pas bloquer: sanitiser uniquement pour l'affichage/jeu avec chess.js
      try {
        const parts = fenToUse.split(' ')
        const boardStr = parts[0]
        const board: (string | null)[][] = Array(8).fill(null).map(() => Array(8).fill(null))
        let row = 0
        let col = 0
        for (const ch of boardStr) {
          if (ch === '/') { row++; col = 0 }
          else if (ch >= '1' && ch <= '8') { col += parseInt(ch) }
          else { board[row][col] = ch; col++ }
        }
        const sanitized = sanitizeBoardKings(board)
        const sanitizedFen = buildFenFromBoard(
          sanitized,
          parts[1] || 'w',
          parts[2] || '-',
          parts[3] || '-',
          parseInt(parts[4]) || 0,
          parseInt(parts[5]) || 1
        )
        const newGame = new Chess(sanitizedFen)
        setGame(newGame)
        updateGameStatus(newGame)
        setMoveHistory([])
        setCapturedPieces({ white: [], black: [] })
        // Ne pas afficher d'erreur: on a accepté le FEN (sanitisation interne)
        setDetectionError("")
        setCurrentMoveIndex(-1)
        setGameMode("config")
      } catch (err2) {
        console.error("Impossible d'initialiser même après sanitisation:", err2)
        // Dernier recours: ne pas bloquer l'UI; rester sur l'état actuel sans erreur bloquante
      }
    }
  }

  const startGame = () => {
    // Réinitialiser les timers
    const timeInSeconds = timeControl * 60
    setPlayerTime(timeInSeconds)
    setAiTime(timeInSeconds)
    setIsTimerActive(true)
    setGameMode("playing")
    setCurrentMoveIndex(-1)
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
  }

  // Fonction pour comparer deux FEN et calculer le pourcentage de réussite
  const calculateAccuracy = (detectedFen: string, correctedFen: string): number => {
    try {
      // Parser les deux FEN pour obtenir les positions des pièces
      const parseFenBoard = (fen: string): (string | null)[][] => {
        const board: (string | null)[][] = Array(8).fill(null).map(() => Array(8).fill(null))
        const parts = fen.split(' ')
        const boardStr = parts[0]
        
        let row = 0
        let col = 0
        
        for (const char of boardStr) {
          if (char === '/') {
            row++
            col = 0
          } else if (char >= '1' && char <= '8') {
            col += parseInt(char)
  } else {
            board[row][col] = char
            col++
          }
        }
        
        return board
      }
      
      const detectedBoard = parseFenBoard(detectedFen)
      const correctedBoard = parseFenBoard(correctedFen)
      
      let correctPieces = 0
      let totalPieces = 0
      
      // Comparer case par case
      for (let row = 0; row < 8; row++) {
        for (let col = 0; col < 8; col++) {
          const detected = detectedBoard[row][col]
          const corrected = correctedBoard[row][col]
          
          // Compter uniquement les cases qui ont une pièce dans au moins un des deux FEN
          if (detected || corrected) {
            totalPieces++
            // Comparer la pièce (exact match)
            if (detected === corrected) {
              correctPieces++
            }
          }
        }
      }
      
      // Calculer le pourcentage
      return totalPieces > 0 ? Math.round((correctPieces / totalPieces) * 100) : 100
    } catch (error) {
      console.error("Erreur lors du calcul de la précision:", error)
      return 0
    }
  }

  // Handler pour l'échiquier de correction - permet le déplacement libre des pièces
  const onCorrectionDrop = useCallback(
    (sourceSquare: Square, targetSquare: Square, _piece: string) => {
      if (!correctionGame) return false
      
      try {
        const gameCopy = new Chess(correctionGame.fen())
        
        // Récupérer la pièce de la case source
        const pieceOnSource = gameCopy.get(sourceSquare)
        
        if (!pieceOnSource) return false
        
        // Supprimer la pièce de la source
        gameCopy.remove(sourceSquare)
        
        // Supprimer la pièce de la destination si elle existe
        gameCopy.remove(targetSquare)
        
        // Placer la pièce à la destination
        gameCopy.put(pieceOnSource, targetSquare)
        
        setCorrectionGame(gameCopy)
        const newFen = gameCopy.fen()
        setManualFen(newFen)
        setCorrectedFen(newFen)
        
        // Calculer automatiquement le score de précision
        if (detectedFen) {
          const score = calculateAccuracy(detectedFen, newFen)
          setAccuracyScore(score)
        }
        
        return true
      } catch (err) {
        console.error("Erreur déplacement pièce:", err)
        return false
      }
    },
    [correctionGame, detectedFen]
  )

  // Ref pour suivre les clics et détecter le double-clic
  const lastClickRef = useRef<{ square: Square | null; time: number }>({ square: null, time: 0 })

  // Fonction helper pour construire un FEN à partir d'un board
  const buildFenFromBoard = useCallback((board: (string | null)[][], activeColor: string = 'w', castling: string = '-', enPassant: string = '-', halfmove: number = 0, fullmove: number = 1): string => {
    let fen = ''
    
    // Construire la partie board du FEN
    for (let row = 0; row < 8; row++) {
      let emptyCount = 0
      for (let col = 0; col < 8; col++) {
        const piece = board[row][col]
        if (!piece) {
          emptyCount++
    } else {
          if (emptyCount > 0) {
            fen += emptyCount
            emptyCount = 0
          }
          fen += piece
        }
      }
      if (emptyCount > 0) {
        fen += emptyCount
      }
      if (row < 7) {
        fen += '/'
      }
    }
    
    // Ajouter le reste du FEN
    fen += ` ${activeColor} ${castling} ${enPassant} ${halfmove} ${fullmove}`
    
    return fen
  }, [])

  // Sanitize: garantir exactement un roi blanc et un roi noir pour compatibilité chess.js (affichage uniquement)
  const sanitizeBoardKings = useCallback((board: (string | null)[][]): (string | null)[][] => {
    const copy = board.map(r => [...r])
    // Compter positions des rois
    const whiteKingPositions: Array<{ r: number; c: number }> = []
    const blackKingPositions: Array<{ r: number; c: number }> = []
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        if (copy[r][c] === 'K') whiteKingPositions.push({ r, c })
        if (copy[r][c] === 'k') blackKingPositions.push({ r, c })
      }
    }
    // Trop de rois: conserver le premier, supprimer les autres
    if (whiteKingPositions.length > 1) {
      for (let i = 1; i < whiteKingPositions.length; i++) {
        const { r, c } = whiteKingPositions[i]
        copy[r][c] = null
      }
    }
    if (blackKingPositions.length > 1) {
      for (let i = 1; i < blackKingPositions.length; i++) {
        const { r, c } = blackKingPositions[i]
        copy[r][c] = null
      }
    }
    // Manquants: ajouter sur la première case libre
    const ensureKing = (isWhite: boolean) => {
      const hasKing = isWhite ? whiteKingPositions.length >= 1 : blackKingPositions.length >= 1
      if (hasKing) return
      for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
          if (!copy[r][c]) {
            copy[r][c] = isWhite ? 'K' : 'k'
            return
          }
        }
      }
    }
    ensureKing(true)
    ensureKing(false)
    return copy
  }, [])

  // Fonction helper pour supprimer une pièce d'une case spécifique
  const deletePieceFromSquare = useCallback((square: Square) => {
    if (!correctionGame) return
    
    try {
      // Parser le FEN actuel pour obtenir le board
      const currentFen = correctionGame.fen()
      const fenParts = currentFen.split(' ')
      const boardStr = fenParts[0]
      
      // Parser le board en tableau 2D
      const board: (string | null)[][] = Array(8).fill(null).map(() => Array(8).fill(null))
      let fenRow = 0
      let fenCol = 0
      
      for (const char of boardStr) {
        if (char === '/') {
          fenRow++
          fenCol = 0
        } else if (char >= '1' && char <= '8') {
          fenCol += parseInt(char)
        } else {
          board[fenRow][fenCol] = char
          fenCol++
        }
      }
      
      // Supprimer la pièce de la case sélectionnée
      const squareToIndex = (sq: Square) => {
        const file = sq.charCodeAt(0) - 97 // a=0, b=1, etc.
        const rank = parseInt(sq.charAt(1)) - 1 // 1=0, 2=1, etc.
        return { boardRow: 7 - rank, boardCol: file } // Inverser la ligne car FEN commence en haut
      }
      
      const { boardRow, boardCol } = squareToIndex(square)
      board[boardRow][boardCol] = null
      
      // Construire le nouveau FEN manuellement
      const newFen = buildFenFromBoard(
        board,
        fenParts[1] || 'w',
        fenParts[2] || '-',
        fenParts[3] || '-',
        parseInt(fenParts[4]) || 0,
        parseInt(fenParts[5]) || 1
      )
      
      // Essayer de créer un Chess avec le nouveau FEN
      // Si ça échoue (FEN invalide), on ajoute temporairement les rois manquants pour l'affichage
      let newGame: Chess
      try {
        newGame = new Chess(newFen)
      } catch {
        // Sanitize rois (manquants ou multiples) uniquement pour affichage
        const boardCopy = sanitizeBoardKings(board)
        const tempFenValid = buildFenFromBoard(
          boardCopy,
          fenParts[1] || 'w',
          fenParts[2] || '-',
          fenParts[3] || '-',
          parseInt(fenParts[4]) || 0,
          parseInt(fenParts[5]) || 1
        )
        newGame = new Chess(tempFenValid)
      }
      
      setCorrectionGame(newGame)
      setManualFen(newFen)
      setCorrectedFen(newFen)
      
      // Calculer automatiquement le score de précision
      if (detectedFen) {
        const score = calculateAccuracy(detectedFen, newFen)
        setAccuracyScore(score)
      }
    } catch (err) {
      console.error("Erreur suppression pièce:", err)
    }
  }, [correctionGame, detectedFen, buildFenFromBoard])

  // Fonction pour placer une pièce sur une case
  const placePieceOnSquare = useCallback((square: Square, pieceString: string) => {
    if (!correctionGame) return
    
    try {
      // Parser le FEN actuel pour obtenir le board
      const currentFen = correctionGame.fen()
      const fenParts = currentFen.split(' ')
      const boardStr = fenParts[0]
      
      // Parser le board en tableau 2D
      const board: (string | null)[][] = Array(8).fill(null).map(() => Array(8).fill(null))
      let fenRow = 0
      let fenCol = 0
      
      for (const char of boardStr) {
        if (char === '/') {
          fenRow++
          fenCol = 0
        } else if (char >= '1' && char <= '8') {
          fenCol += parseInt(char)
        } else {
          board[fenRow][fenCol] = char
          fenCol++
        }
      }
      
      // Convertir la pièce sélectionnée (ex: "wP") en notation FEN (ex: "P")
      const pieceColor = pieceString.charAt(0) // 'w' ou 'b'
      const pieceType = pieceString.charAt(1).toLowerCase() // 'p', 'n', 'b', 'r', 'q', 'k'
      const fenPiece = pieceColor === 'w' ? pieceType.toUpperCase() : pieceType
      
      // Placer la pièce sur la case
      const squareToIndex = (sq: Square) => {
        const file = sq.charCodeAt(0) - 97 // a=0, b=1, etc.
        const rank = parseInt(sq.charAt(1)) - 1 // 1=0, 2=1, etc.
        return { boardRow: 7 - rank, boardCol: file } // Inverser la ligne car FEN commence en haut
      }
      
      const { boardRow, boardCol } = squareToIndex(square)
      board[boardRow][boardCol] = fenPiece
      
      // Construire le nouveau FEN manuellement
      const newFen = buildFenFromBoard(
        board,
        fenParts[1] || 'w',
        fenParts[2] || '-',
        fenParts[3] || '-',
        parseInt(fenParts[4]) || 0,
        parseInt(fenParts[5]) || 1
      )
      
      // Essayer de créer un Chess avec le nouveau FEN
      let newGame: Chess
      try {
        newGame = new Chess(newFen)
      } catch {
        // Sanitize rois (manquants ou multiples) uniquement pour affichage
        const boardCopy = sanitizeBoardKings(board)
        const tempFenValid = buildFenFromBoard(
          boardCopy,
          fenParts[1] || 'w',
          fenParts[2] || '-',
          fenParts[3] || '-',
          parseInt(fenParts[4]) || 0,
          parseInt(fenParts[5]) || 1
        )
        newGame = new Chess(tempFenValid)
      }
      
      setCorrectionGame(newGame)
      setManualFen(newFen)
      setCorrectedFen(newFen)
      
      // Calculer automatiquement le score de précision
      if (detectedFen) {
        const score = calculateAccuracy(detectedFen, newFen)
        setAccuracyScore(score)
      }
      
      // Réinitialiser la sélection de pièce après placement
      setSelectedPieceToPlace(null)
    } catch (err) {
      console.error("Erreur placement pièce:", err)
    }
  }, [correctionGame, detectedFen, buildFenFromBoard])

  // Handler pour sélectionner une case (pour suppression ou placement)
  const onCorrectionSquareClick = useCallback(
    (square: Square) => {
      if (!correctionGame) return
      
      // Si une pièce est sélectionnée dans la palette, la placer
      if (selectedPieceToPlace) {
        placePieceOnSquare(square, selectedPieceToPlace)
        return
      }
      
      // Vérifier si la case contient une pièce
      const piece = correctionGame.get(square)
      if (!piece) {
        setSelectedSquare(null)
        return
      }
      
      const now = Date.now()
      const lastClick = lastClickRef.current
      
      // Détecter le double-clic (dans les 300ms et sur la même case)
      if (lastClick.square === square && now - lastClick.time < 300) {
        // Supprimer la pièce directement
        deletePieceFromSquare(square)
        // Réinitialiser
        lastClickRef.current = { square: null, time: 0 }
        setSelectedSquare(null)
      } else {
        // Sélectionner la case pour afficher le bouton de suppression
        setSelectedSquare(square)
        lastClickRef.current = { square, time: now }
      }
    },
    [correctionGame, deletePieceFromSquare, selectedPieceToPlace, placePieceOnSquare]
  )

  // Fonction pour supprimer la pièce sélectionnée
  const handleDeleteSelectedPiece = useCallback(() => {
    if (!correctionGame || !selectedSquare) return
    deletePieceFromSquare(selectedSquare)
    setSelectedSquare(null)
  }, [correctionGame, selectedSquare, deletePieceFromSquare])

  const getPgn = () => {
    return `[Event "Partie d'échecs"]
[Site "Échecs App"]
[White "Vous"]
[Black "IA"]
[Result "*"]
${moveHistory.map((move, idx) => {
  const moveNum = Math.floor(idx / 2) + 1
  if (idx % 2 === 0) {
    return `${moveNum}. ${move}`
  } else {
    return ` ${move}`
  }
}).join(' ')}`
  }

  const resign = () => {
    setGameStatus("Vous avez abandonné. L'IA gagne !")
  }

  const resetGame = () => {
    const newGame = new Chess()
    setGame(newGame)
    setMoveHistory([])
    setCapturedPieces({ white: [], black: [] })
    setGameStatus("")
    setCurrentMoveIndex(-1)
    setIsAiThinking(false)
    setIsTimerActive(false)
    const timeInSeconds = timeControl * 60
    setPlayerTime(timeInSeconds)
    setAiTime(timeInSeconds)
  }

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <div className="min-h-screen py-2 sm:py-4 md:py-4 pb-4">
      <div className="max-w-[1400px] mx-auto px-2 sm:px-4 md:px-4">
        <div className="mb-3 sm:mb-4 md:mb-6 text-center pt-12 md:pt-0">
          <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold mb-1">
            Jeu d&apos;
            {" "}<Highlighter
              action="highlight"
              color={
                  (typeof window !== "undefined" &&
                    (document.documentElement.classList.contains("dark") ||
                      window.matchMedia?.("(prefers-color-scheme: dark)").matches))
                  ? "#c9c9c9"
                  : "black"
              }
            >
              <p className="text-background">Échecs</p>
            </Highlighter>
          </h1>
          <p className="text-muted-foreground mb-15">
            <Highlighter action="underline" color="#c9c9c9" padding={3}>
              Choisissez votre mode
            </Highlighter>
            {" "}de jeu et commencez à jouer
          </p>
          
          {/* Mode Switch - Discret */}
          {(gameMode === "image" || gameMode === "config") && (
            <div className="inline-flex items-center gap-2 bg-muted/30 p-1">
              <Button
                variant={gameMode === "image" ? "default" : "ghost"}
                size="sm"
                onClick={() => setGameMode("image")}
                className="h-8 px-2 sm:px-3 text-xs sm:text-sm"
              >
                <Camera className="h-3 w-3 sm:h-4 sm:w-4 sm:mr-1" />
                <span className="hidden sm:inline">Détection</span>
              </Button>
            </div>
          )}
        </div>


        {/* Content based on mode */}
        {gameMode === "config" ? (
          /* Configuration de la partie */
          <div className="w-full max-w-4xl mx-auto px-4">
            <Card className="rounded-none">
              <CardHeader className="pb-4">
                <CardTitle>Configuration de la partie</CardTitle>
                <CardDescription>Choisissez vos paramètres avant de commencer</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4 pb-4">
                {/* Aperçu de l'échiquier */}
                <div className="flex justify-center">
                  <div className="border-2 border-muted inline-block">
                    <Chessboard
                      position={game.fen()}
                      boardWidth={Math.min(400, typeof window !== 'undefined' ? window.innerWidth - 64 : 400)}
                      arePiecesDraggable={false}
                      customBoardStyle={{
                        backgroundImage: "url('/newspaper.svg')",
                        backgroundSize: "100% 100%",
                        backgroundRepeat: "no-repeat",
                        border: "none"
                      }}
                      customLightSquareStyle={{ backgroundColor: "transparent" }}
                      customDarkSquareStyle={{ backgroundColor: "transparent" }}
                      customPieces={{
                        wP: ({ squareWidth }) => <Image src="https://lichess1.org/assets/piece/pixel/wP.svg" alt="P" width={squareWidth} height={squareWidth} style={{ width: squareWidth, height: squareWidth, pointerEvents: "none" }} unoptimized />,
                        wN: ({ squareWidth }) => <Image src="https://lichess1.org/assets/piece/pixel/wN.svg" alt="N" width={squareWidth} height={squareWidth} style={{ width: squareWidth, height: squareWidth, pointerEvents: "none" }} unoptimized />,
                        wB: ({ squareWidth }) => <Image src="https://lichess1.org/assets/piece/pixel/wB.svg" alt="B" width={squareWidth} height={squareWidth} style={{ width: squareWidth, height: squareWidth, pointerEvents: "none" }} unoptimized />,
                        wR: ({ squareWidth }) => <Image src="https://lichess1.org/assets/piece/pixel/wR.svg" alt="R" width={squareWidth} height={squareWidth} style={{ width: squareWidth, height: squareWidth, pointerEvents: "none" }} unoptimized />,
                        wQ: ({ squareWidth }) => <Image src="https://lichess1.org/assets/piece/pixel/wQ.svg" alt="Q" width={squareWidth} height={squareWidth} style={{ width: squareWidth, height: squareWidth, pointerEvents: "none" }} unoptimized />,
                        wK: ({ squareWidth }) => <Image src="https://lichess1.org/assets/piece/pixel/wK.svg" alt="K" width={squareWidth} height={squareWidth} style={{ width: squareWidth, height: squareWidth, pointerEvents: "none" }} unoptimized />,
                        bP: ({ squareWidth }) => <Image src="https://lichess1.org/assets/piece/pixel/bP.svg" alt="p" width={squareWidth} height={squareWidth} style={{ width: squareWidth, height: squareWidth, pointerEvents: "none" }} unoptimized />,
                        bN: ({ squareWidth }) => <Image src="https://lichess1.org/assets/piece/pixel/bN.svg" alt="n" width={squareWidth} height={squareWidth} style={{ width: squareWidth, height: squareWidth, pointerEvents: "none" }} unoptimized />,
                        bB: ({ squareWidth }) => <Image src="https://lichess1.org/assets/piece/pixel/bB.svg" alt="b" width={squareWidth} height={squareWidth} style={{ width: squareWidth, height: squareWidth, pointerEvents: "none" }} unoptimized />,
                        bR: ({ squareWidth }) => <Image src="https://lichess1.org/assets/piece/pixel/bR.svg" alt="r" width={squareWidth} height={squareWidth} style={{ width: squareWidth, height: squareWidth, pointerEvents: "none" }} unoptimized />,
                        bQ: ({ squareWidth }) => <Image src="https://lichess1.org/assets/piece/pixel/bQ.svg" alt="q" width={squareWidth} height={squareWidth} style={{ width: squareWidth, height: squareWidth, pointerEvents: "none" }} unoptimized />,
                        bK: ({ squareWidth }) => <Image src="https://lichess1.org/assets/piece/pixel/bK.svg" alt="k" width={squareWidth} height={squareWidth} style={{ width: squareWidth, height: squareWidth, pointerEvents: "none" }} unoptimized />
                      }}
                    />
                  </div>
                </div>

                {/* Choix de la couleur */}
                <div className="space-y-2">
                  <label className="text-sm font-semibold uppercase">Jouer en tant que</label>
                  <div className="grid grid-cols-2 gap-2">
                    <Button
                      variant={playerColor === "white" ? "default" : "outline"}
                      onClick={() => setPlayerColor("white")}
                      className="h-12 rounded-none"
                    >
                      Blancs
                    </Button>
                    <Button
                      variant={playerColor === "black" ? "default" : "outline"}
                      onClick={() => setPlayerColor("black")}
                      className="h-12 rounded-none"
                    >
                      Noirs
                    </Button>
                  </div>
                </div>

                {/* Temps de jeu */}
                <div className="space-y-2">
                  <label className="text-sm font-semibold uppercase">Temps de jeu</label>
                  <div className="grid grid-cols-3 gap-2">
                    <Button
                      variant={timeControl === 5 ? "default" : "outline"}
                      onClick={() => setTimeControl(5)}
                      className="h-12 rounded-none"
                    >
                      5 min
                    </Button>
                    <Button
                      variant={timeControl === 10 ? "default" : "outline"}
                      onClick={() => setTimeControl(10)}
                      className="h-12 rounded-none"
                    >
                      10 min
                    </Button>
                    <Button
                      variant={timeControl === 15 ? "default" : "outline"}
                      onClick={() => setTimeControl(15)}
                      className="h-12 rounded-none"
                    >
                      15 min
                    </Button>
                  </div>
                </div>

                {/* Niveau de l'IA */}
                <div className="space-y-2">
                  <label className="text-sm font-semibold uppercase">Niveau de l&apos;IA</label>
                  <div className="grid grid-cols-3 gap-2">
                    <Button
                      variant={aiDepth === 2 ? "default" : "outline"}
                      onClick={() => setAiDepth(2)}
                      className="h-12 rounded-none"
                    >
                      Facile<br/><span className="text-xs opacity-70">(profondeur 2)</span>
                    </Button>
                    <Button
                      variant={aiDepth === 3 ? "default" : "outline"}
                      onClick={() => setAiDepth(3)}
                      className="h-12 rounded-none"
                    >
                      Moyen<br/><span className="text-xs opacity-70">(profondeur 3)</span>
                    </Button>
                    <Button
                      variant={aiDepth === 4 ? "default" : "outline"}
                      onClick={() => setAiDepth(4)}
                      className="h-12 rounded-none"
                    >
                      Difficile<br/><span className="text-xs opacity-70">(profondeur 4)</span>
                    </Button>
                  </div>
                </div>

                {/* Bouton démarrer */}
                <Button
                  onClick={startGame}
                  size="lg"
                  className="w-full h-14 text-lg font-semibold rounded-none"
                >
                  Commencer la partie
                </Button>
              </CardContent>
            </Card>
          </div>
        ) : gameMode === "image" ? (
          <div className="w-full max-w-2xl mx-auto px-4 sm:px-6">
            {/* Upload & Detection Section */}
            <div className="space-y-4">
              <div className="relative">
                <Card className="rounded-none">
                  <CardHeader className="pb-4">
                    <CardTitle className="flex items-center gap-2">
                      <Camera className="h-5 w-5" />
                      Photo d&apos;Échiquier
                    </CardTitle>
                    <CardDescription>Téléchargez une photo réelle et détectez la position FEN</CardDescription>
                  </CardHeader>
                  <CardContent className="p-4 sm:p-6 space-y-4 pb-4">
                    {!imageDataUrl ? (
                      <FileUpload onChange={handleFileUpload} />
                    ) : (
                      <>
                        <div className="flex flex-col items-center w-full">
                          <div className="relative inline-block">
                            {(preprocessedImageUrl || imageDataUrl) && (
                          <Image 
                                src={preprocessedImageUrl || imageDataUrl} 
                            alt="preview" 
                            width={400}
                            height={300}
                                className={`max-h-60 sm:max-h-80 w-full object-contain transition-all duration-500 ${
                                  preprocessedImageUrl ? 'opacity-100 scale-[1.02]' : 'opacity-100 scale-100'
                                }`}
                                style={{
                                  filter: `blur(${blurAmount}px)`,
                                  transition: 'filter 0.1s ease-out',
                                }}
                              />
                            )}
                          <Button
                            variant="destructive"
                            size="sm"
                              className="absolute -top-2 -right-2 h-8 w-8 p-0 z-10"
                            onClick={() => {
                              setImageDataUrl("")
                              setDetectedFen("")
                              setDetectionError("")
                              setManualFen("")
                              setPreprocessedImageUrl("")
                              setPreprocessingStep("")
                              setBlurAmount(20)
                              setCorrectedFen("")
                              setAccuracyScore(null)
                              setCorrectionGame(null)
                              setSelectedSquare(null)
                              setSelectedPieceToPlace(null)
                            }}
                          >
                            <X className="h-4 w-4" />
                          </Button>
                          </div>
                          {preprocessedImageUrl && (
                            <div className="mt-2 text-xs text-muted-foreground text-center animate-fade-in">
                              <span className="inline-flex items-center gap-1">
                                <CheckCircle2 className="h-3 w-3 text-foreground-muted" />
                                Image préprocessée
                              </span>
                            </div>
                          )}
                          {preprocessingStep && (
                            <div className="mt-2 text-xs text-center font-medium text-foreground-muted">
                              {preprocessingStep}
                            </div>
                          )}
                        </div>
                        
                        <Button 
                          onClick={handleDetectFen}
                          disabled={isDetecting}
                          className="w-full"
                          size="lg"
                        >
                          {isDetecting ? (
                            <>
                              <Loader2 className="mr-2 h-4 w-4 animate-spin text-foreground-muted" />
                              Analyse en cours...
                            </>
                          ) : (
                            <>
                              <Scan className="mr-2 h-4 w-4 text-foreground-muted" />
                              Détecter l&apos;échiquier
                            </>
                          )}
                        </Button>
                        
                        {detectionError && (
                          <div className="p-3 bg-red-500/20 text-red-700 dark:text-red-300 text-sm">
                            <AlertCircle className="inline h-4 w-4 mr-2" />
                            {detectionError}
                          </div>
                        )}
                        
                        {detectedFen && (
                          <div className="p-3 bg-foreground-muted text-foreground-muted">
                            <div className="flex items-center gap-2">
                              <CheckCircle2 className="h-4 w-4" />
                              <span className="font-semibold text-sm">FEN détecté !</span>
                            </div>
                          </div>
                        )}
                        
                        {/* Échiquier de correction */}
                        {detectedFen && correctionGame ? (
                          <div className="space-y-3 pt-3">
                            <div>
                              <label className="text-sm font-medium mb-2 block">
                                Corrigez les erreurs sur l&apos;échiquier :
                              </label>
                              <div className="flex flex-col items-center gap-4">
                                {/* Échiquier interactif */}
                                <div className="inline-block border-2 border-muted">
                                  <Chessboard
                                    position={correctionGame.fen()}
                                    onPieceDrop={onCorrectionDrop}
                                    onSquareClick={onCorrectionSquareClick}
                                    boardWidth={Math.min(350, typeof window !== 'undefined' ? window.innerWidth - 64 : 350)}
                                    customBoardStyle={{
                                      backgroundImage: "url('/newspaper.svg')",
                                      backgroundSize: "100% 100%",
                                      backgroundRepeat: "no-repeat",
                                      border: "none"
                                    }}
                                    customLightSquareStyle={{ 
                                      backgroundColor: "transparent"
                                    }}
                                    customDarkSquareStyle={{ 
                                      backgroundColor: "transparent"
                                    }}
                                    customPieces={{
                                      wP: ({ squareWidth }) => (
                                        <Image src="https://lichess1.org/assets/piece/pixel/wP.svg" alt="P" width={squareWidth} height={squareWidth} style={{ width: squareWidth, height: squareWidth, pointerEvents: "none" }} unoptimized />
                                      ),
                                      wN: ({ squareWidth }) => (
                                        <Image src="https://lichess1.org/assets/piece/pixel/wN.svg" alt="N" width={squareWidth} height={squareWidth} style={{ width: squareWidth, height: squareWidth, pointerEvents: "none" }} unoptimized />
                                      ),
                                      wB: ({ squareWidth }) => (
                                        <Image src="https://lichess1.org/assets/piece/pixel/wB.svg" alt="B" width={squareWidth} height={squareWidth} style={{ width: squareWidth, height: squareWidth, pointerEvents: "none" }} unoptimized />
                                      ),
                                      wR: ({ squareWidth }) => (
                                        <Image src="https://lichess1.org/assets/piece/pixel/wR.svg" alt="R" width={squareWidth} height={squareWidth} style={{ width: squareWidth, height: squareWidth, pointerEvents: "none" }} unoptimized />
                                      ),
                                      wQ: ({ squareWidth }) => (
                                        <Image src="https://lichess1.org/assets/piece/pixel/wQ.svg" alt="Q" width={squareWidth} height={squareWidth} style={{ width: squareWidth, height: squareWidth, pointerEvents: "none" }} unoptimized />
                                      ),
                                      wK: ({ squareWidth }) => (
                                        <Image src="https://lichess1.org/assets/piece/pixel/wK.svg" alt="K" width={squareWidth} height={squareWidth} style={{ width: squareWidth, height: squareWidth, pointerEvents: "none" }} unoptimized />
                                      ),
                                      bP: ({ squareWidth }) => (
                                        <Image src="https://lichess1.org/assets/piece/pixel/bP.svg" alt="p" width={squareWidth} height={squareWidth} style={{ width: squareWidth, height: squareWidth, pointerEvents: "none" }} unoptimized />
                                      ),
                                      bN: ({ squareWidth }) => (
                                        <Image src="https://lichess1.org/assets/piece/pixel/bN.svg" alt="n" width={squareWidth} height={squareWidth} style={{ width: squareWidth, height: squareWidth, pointerEvents: "none" }} unoptimized />
                                      ),
                                      bB: ({ squareWidth }) => (
                                        <Image src="https://lichess1.org/assets/piece/pixel/bB.svg" alt="b" width={squareWidth} height={squareWidth} style={{ width: squareWidth, height: squareWidth, pointerEvents: "none" }} unoptimized />
                                      ),
                                      bR: ({ squareWidth }) => (
                                        <Image src="https://lichess1.org/assets/piece/pixel/bR.svg" alt="r" width={squareWidth} height={squareWidth} style={{ width: squareWidth, height: squareWidth, pointerEvents: "none" }} unoptimized />
                                      ),
                                      bQ: ({ squareWidth }) => (
                                        <Image src="https://lichess1.org/assets/piece/pixel/bQ.svg" alt="q" width={squareWidth} height={squareWidth} style={{ width: squareWidth, height: squareWidth, pointerEvents: "none" }} unoptimized />
                                      ),
                                      bK: ({ squareWidth }) => (
                                        <Image src="https://lichess1.org/assets/piece/pixel/bK.svg" alt="k" width={squareWidth} height={squareWidth} style={{ width: squareWidth, height: squareWidth, pointerEvents: "none" }} unoptimized />
                                      )
                                    }}
                                  />
                                </div>
                                
                                {/* Palette de pièces */}
                                <div className="w-full space-y-2">
                                  <label className="text-xs font-medium block text-center">
                                    Sélectionnez une pièce à placer :
                                  </label>
                                  <div className="grid grid-cols-6 gap-1 sm:gap-2 p-1 sm:p-2 bg-muted/30 rounded">
                                    {/* Pièces blanches */}
                                    {['wP', 'wR', 'wN', 'wB', 'wQ', 'wK'].map((piece) => {
                                      const pieceType = piece.charAt(1)
                                      const pieceUrls: { [key: string]: string } = {
                                        P: "https://lichess1.org/assets/piece/pixel/wP.svg",
                                        N: "https://lichess1.org/assets/piece/pixel/wN.svg",
                                        B: "https://lichess1.org/assets/piece/pixel/wB.svg",
                                        R: "https://lichess1.org/assets/piece/pixel/wR.svg",
                                        Q: "https://lichess1.org/assets/piece/pixel/wQ.svg",
                                        K: "https://lichess1.org/assets/piece/pixel/wK.svg",
                                      }
                                      return (
                                        <button
                                          key={piece}
                                          onClick={() => setSelectedPieceToPlace(selectedPieceToPlace === piece ? null : piece)}
                                          className={`p-1 sm:p-2 border-2 transition-all ${
                                            selectedPieceToPlace === piece
                                              ? 'border-primary bg-primary/20 scale-110'
                                              : 'border-muted hover:border-foreground/50 hover:bg-muted/50'
                                          }`}
                                          title={`Pièce blanche: ${pieceType}`}
                                        >
                                          <Image
                                            src={pieceUrls[pieceType]}
                                            alt={piece}
                                            width={32}
                                            height={32}
                                            className="w-6 h-6 sm:w-8 sm:h-8 mx-auto pointer-events-none"
                                            unoptimized
                                          />
                                        </button>
                                      )
                                    })}
                                    {/* Pièces noires */}
                                    {['bP', 'bR', 'bN', 'bB', 'bQ', 'bK'].map((piece) => {
                                      const pieceType = piece.charAt(1)
                                      const pieceUrls: { [key: string]: string } = {
                                        P: "https://lichess1.org/assets/piece/pixel/bP.svg",
                                        N: "https://lichess1.org/assets/piece/pixel/bN.svg",
                                        B: "https://lichess1.org/assets/piece/pixel/bB.svg",
                                        R: "https://lichess1.org/assets/piece/pixel/bR.svg",
                                        Q: "https://lichess1.org/assets/piece/pixel/bQ.svg",
                                        K: "https://lichess1.org/assets/piece/pixel/bK.svg",
                                      }
                                      return (
                                        <button
                                          key={piece}
                                          onClick={() => setSelectedPieceToPlace(selectedPieceToPlace === piece ? null : piece)}
                                          className={`p-1 sm:p-2 border-2 transition-all ${
                                            selectedPieceToPlace === piece
                                              ? 'border-primary bg-primary/20 scale-110'
                                              : 'border-muted hover:border-foreground/50 hover:bg-muted/50'
                                          }`}
                                          title={`Pièce noire: ${pieceType}`}
                                        >
                                          <Image
                                            src={pieceUrls[pieceType]}
                                            alt={piece}
                                            width={32}
                                            height={32}
                                            className="w-6 h-6 sm:w-8 sm:h-8 mx-auto pointer-events-none"
                                            unoptimized
                                          />
                                        </button>
                                      )
                                    })}
                                  </div>
                                  {selectedPieceToPlace && (
                                    <p className="text-xs text-center text-muted-foreground">
                                      Cliquez sur une case pour placer la pièce sélectionnée
                                    </p>
                                  )}
                                </div>
                                
                                {/* Instructions et bouton de suppression */}
                                <div className="w-full space-y-2">
                                  <div className="text-xs text-muted-foreground text-center space-y-1">
                                    <p>• Glissez-déposez pour déplacer les pièces</p>
                                    <p>• Double-cliquez sur une pièce pour la supprimer</p>
                                    <p>• Ou sélectionnez une pièce puis cliquez sur le bouton pour supprimer</p>
                                  </div>
                                  {selectedSquare && correctionGame?.get(selectedSquare) && !selectedPieceToPlace && (
                                    <div className="flex items-center justify-center gap-2">
                                      <span className="text-xs text-muted-foreground">
                                        Pièce sélectionnée: {selectedSquare}
                                      </span>
                                      <Button
                                        variant="destructive"
                                        size="sm"
                                        onClick={handleDeleteSelectedPiece}
                                        className="h-7"
                                      >
                                        <Trash2 className="h-3 w-3 mr-1" />
                                        Supprimer
                                      </Button>
                                    </div>
                                  )}
                                </div>
                              </div>
                            </div>
                            
                            {/* Score de précision */}
                            {accuracyScore !== null && (
                              <div className="p-3 bg-muted/50 border">
                                <div className="flex items-center justify-between">
                                  <span className="text-sm font-medium">Précision :</span>
                                  <span className={`text-lg font-bold ${
                                    accuracyScore >= 90 ? 'text-green-600 dark:text-green-400' :
                                    accuracyScore >= 70 ? 'text-yellow-600 dark:text-yellow-400' :
                                    'text-red-600 dark:text-red-400'
                                  }`}>
                                    {accuracyScore}%
                                  </span>
                                </div>
                              </div>
                            )}
                            
                            <Button 
                              onClick={handleContinueGame}
                              disabled={!correctionGame}
                              className="w-full"
                              size="lg"
                            >
                              <Bot className="mr-2 h-4 w-4" />
                              Continuer la partie avec l&apos;IA
                            </Button>
                          </div>
                        ) : (
                          /* Input manuel pour FEN (si pas de détection) */
                        <div className="space-y-2 pt-2">
                          <label className="text-sm font-medium">
                              Ou entrez un FEN manuellement :
                          </label>
                          <input
                            type="text"
                            value={manualFen}
                            onChange={(e) => setManualFen(e.target.value)}
                            placeholder="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
                            className="w-full px-3 py-2 text-sm border bg-background font-mono"
                          />
                          
                          <Button 
                            onClick={handleContinueGame}
                            disabled={!manualFen.trim()}
                            className="w-full"
                            size="lg"
                          >
                            <Bot className="mr-2 h-4 w-4" />
                            Continuer la partie avec l&apos;IA
                          </Button>
                        </div>
                        )}
                      </>
                    )}
                  </CardContent>
                </Card>
              </div>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-12 gap-2 items-stretch">
        
        {/* Panneau de gauche - Info joueur et contrôles */}
        <div className="col-span-1 md:col-span-3 grid grid-rows-[auto_1fr] gap-2" ref={leftPanelRef} style={containerHeight > 0 ? { height: `${containerHeight}px` } : undefined}>
          {/* Titre et info adversaire (IA) */}
          <Card className="rounded-none">
            <CardContent className="pt-0 p-2 sm:p-3">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <div className="w-7 h-7 bg-muted flex items-center justify-center">
                    <Bot className="h-3.5 w-3.5" />
                  </div>
                  <div>
                    <p className="font-semibold text-xs">
                      IA ({playerColor === "white" ? "Noirs" : "Blancs"})
                    </p>
                  </div>
                </div>
                {/* Timer IA */}
                <div className={`text-lg font-mono font-bold ${
                  ((game.turn() === 'b' && playerColor === 'white') || (game.turn() === 'w' && playerColor === 'black')) && isTimerActive
                    ? 'text-foreground' 
                    : 'text-muted-foreground'
                }`}>
                  {formatTime(aiTime)}
                </div>
              </div>
              {/* Pièces capturées par l'IA */}
              <div className="flex flex-wrap gap-1 min-h-[20px]">
                {playerColor === "white" ? (
                  capturedPieces.black.map((piece, idx) => (
                    <span key={idx} className="text-base">{piece}</span>
                  ))
                ) : (
                  capturedPieces.white.map((piece, idx) => (
                    <span key={idx} className="text-base">{piece}</span>
                  ))
                )}
              </div>
            </CardContent>
          </Card>

          {/* FEN et PGN - Scrollable */}
          <Card className="rounded-none overflow-hidden flex flex-col">
            <CardContent className="p-2 sm:p-3 space-y-2 flex-1 overflow-y-auto">
              {/* FEN */}
              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="text-[10px] font-semibold uppercase">FEN</label>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-5 w-5 p-0"
                    onClick={() => copyToClipboard(game.fen())}
                  >
                    <Copy className="h-2.5 w-2.5" />
                  </Button>
                </div>
                <div className="p-1.5 bg-muted/50 text-[10px] font-mono break-all leading-tight">
                  {game.fen()}
                </div>
              </div>

              {/* PGN */}
              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="text-[10px] font-semibold uppercase">PGN</label>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-5 w-5 p-0"
                    onClick={() => copyToClipboard(getPgn())}
                  >
                    <Copy className="h-2.5 w-2.5" />
                  </Button>
                </div>
                <div className="p-1.5 bg-muted/50 text-[10px] font-mono max-h-32 overflow-y-auto leading-tight">
                  {moveHistory.length === 0 ? (
                    <span className="text-muted-foreground">Aucun coup</span>
                  ) : (
                    <pre className="whitespace-pre-wrap">{getPgn()}</pre>
                  )}
                </div>
              </div>

              {/* Image préprocessée */}
              {preprocessedImageUrl && (
                <div className="space-y-2">
                  <label className="text-[10px] font-semibold uppercase mb-1 block">Image préprocessée</label>
                  <div className="relative w-full aspect-square bg-muted/50 overflow-hidden">
                    <Image
                      src={preprocessedImageUrl}
                      alt="Image préprocessée"
                      fill
                      className="object-contain"
                      sizes="(max-width: 768px) 100vw, 200px"
                    />
                  </div>
                  {/* Score de précision */}
                  {accuracyScore !== null && (
                    <div className="p-2 bg-muted/50 border">
                      <div className="flex items-center justify-between">
                        <span className="text-[10px] font-semibold uppercase">Précision :</span>
                        <span className={`text-sm font-bold ${
                          accuracyScore >= 90 ? 'text-green-600 dark:text-green-400' :
                          accuracyScore >= 70 ? 'text-yellow-600 dark:text-yellow-400' :
                          'text-red-600 dark:text-red-400'
                        }`}>
                          {accuracyScore}%
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Panneau central - Échiquier */}
        <div className="col-span-1 md:col-span-6 flex items-center justify-center" ref={boardContainerRef} style={containerHeight > 0 ? { height: `${containerHeight}px` } : undefined}>
          <div className="flex items-center justify-center w-full max-w-full overflow-hidden">
            <div className="border-2 sm:border-4 border-foreground-muted inline-block">
            <Chessboard
              position={game.fen()}
              onPieceDrop={onDrop}
              boardWidth={boardSize}
              boardOrientation={playerColor === "white" ? "white" : "black"}
              customBoardStyle={{
                backgroundImage: "url('/newspaper.svg')",
                backgroundSize: "100% 100%",
                  backgroundRepeat: "no-repeat",
                  border: "none"
              }}
              customLightSquareStyle={{ 
                backgroundColor: "transparent"
              }}
              customDarkSquareStyle={{ 
                backgroundColor: "transparent"
              }}
              customPieces={{
                            wP: ({ squareWidth }) => (
                              <Image src="https://lichess1.org/assets/piece/pixel/wP.svg" alt="P" width={squareWidth} height={squareWidth} style={{ width: squareWidth, height: squareWidth, pointerEvents: "none" }} unoptimized />
                            ),
                            wN: ({ squareWidth }) => (
                              <Image src="https://lichess1.org/assets/piece/pixel/wN.svg" alt="N" width={squareWidth} height={squareWidth} style={{ width: squareWidth, height: squareWidth, pointerEvents: "none" }} unoptimized />
                            ),
                            wB: ({ squareWidth }) => (
                              <Image src="https://lichess1.org/assets/piece/pixel/wB.svg" alt="B" width={squareWidth} height={squareWidth} style={{ width: squareWidth, height: squareWidth, pointerEvents: "none" }} unoptimized />
                            ),
                            wR: ({ squareWidth }) => (
                              <Image src="https://lichess1.org/assets/piece/pixel/wR.svg" alt="R" width={squareWidth} height={squareWidth} style={{ width: squareWidth, height: squareWidth, pointerEvents: "none" }} unoptimized />
                            ),
                            wQ: ({ squareWidth }) => (
                              <Image src="https://lichess1.org/assets/piece/pixel/wQ.svg" alt="Q" width={squareWidth} height={squareWidth} style={{ width: squareWidth, height: squareWidth, pointerEvents: "none" }} unoptimized />
                            ),
                            wK: ({ squareWidth }) => (
                              <Image src="https://lichess1.org/assets/piece/pixel/wK.svg" alt="K" width={squareWidth} height={squareWidth} style={{ width: squareWidth, height: squareWidth, pointerEvents: "none" }} unoptimized />
                            ),
                            bP: ({ squareWidth }) => (
                              <Image src="https://lichess1.org/assets/piece/pixel/bP.svg" alt="p" width={squareWidth} height={squareWidth} style={{ width: squareWidth, height: squareWidth, pointerEvents: "none" }} unoptimized />
                            ),
                            bN: ({ squareWidth }) => (
                              <Image src="https://lichess1.org/assets/piece/pixel/bN.svg" alt="n" width={squareWidth} height={squareWidth} style={{ width: squareWidth, height: squareWidth, pointerEvents: "none" }} unoptimized />
                            ),
                            bB: ({ squareWidth }) => (
                              <Image src="https://lichess1.org/assets/piece/pixel/bB.svg" alt="b" width={squareWidth} height={squareWidth} style={{ width: squareWidth, height: squareWidth, pointerEvents: "none" }} unoptimized />
                            ),
                            bR: ({ squareWidth }) => (
                              <Image src="https://lichess1.org/assets/piece/pixel/bR.svg" alt="r" width={squareWidth} height={squareWidth} style={{ width: squareWidth, height: squareWidth, pointerEvents: "none" }} unoptimized />
                            ),
                            bQ: ({ squareWidth }) => (
                              <Image src="https://lichess1.org/assets/piece/pixel/bQ.svg" alt="q" width={squareWidth} height={squareWidth} style={{ width: squareWidth, height: squareWidth, pointerEvents: "none" }} unoptimized />
                            ),
                            bK: ({ squareWidth }) => (
                              <Image src="https://lichess1.org/assets/piece/pixel/bK.svg" alt="k" width={squareWidth} height={squareWidth} style={{ width: squareWidth, height: squareWidth, pointerEvents: "none" }} unoptimized />
                            )
                          }}
            />
            </div>
          </div>
        </div>

        {/* Panneau de droite - Contrôles et historique */}
        <div className="col-span-1 md:col-span-3 grid grid-rows-[auto_1fr_auto] gap-2" ref={rightPanelRef} style={containerHeight > 0 ? { height: `${containerHeight}px` } : undefined}>
          {/* Titre et info joueur */}
          <Card className="rounded-none">
            <CardContent className="pt-0 p-2 sm:p-3">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <div className="w-7 h-7 bg-muted flex items-center justify-center">
                    <User className="h-3.5 w-3.5" />
                  </div>
                  <div>
                    <p className="font-semibold text-xs">
                      Vous ({playerColor === "white" ? "Blancs" : "Noirs"})
                    </p>
                  </div>
                </div>
                {/* Timer joueur */}
                <div className={`text-lg font-mono font-bold ${
                  ((game.turn() === 'w' && playerColor === 'white') || (game.turn() === 'b' && playerColor === 'black')) && isTimerActive
                    ? 'text-foreground' 
                    : 'text-muted-foreground'
                }`}>
                  {formatTime(playerTime)}
                </div>
              </div>
              {/* Pièces capturées par le joueur */}
              <div className="flex flex-wrap gap-1 min-h-[20px]">
                {playerColor === "white" ? (
                  capturedPieces.white.map((piece, idx) => (
                    <span key={idx} className="text-base">{piece}</span>
                  ))
                ) : (
                  capturedPieces.black.map((piece, idx) => (
                    <span key={idx} className="text-base">{piece}</span>
                  ))
                )}
              </div>
            </CardContent>
          </Card>

          {/* Zone centrale - Historique des coups */}
          <Card className="rounded-none overflow-hidden flex flex-col">
            <CardHeader className="pb-2 px-3 pt-3 flex-shrink-0">
              <CardTitle className="text-sm">Coups</CardTitle>
              <CardDescription className="text-[10px]">
                {moveHistory.length} coup{moveHistory.length > 1 ? 's' : ''}
              </CardDescription>
            </CardHeader>
            <CardContent className="p-3 pt-0 flex-1 overflow-y-auto">
              <div className="space-y-1">
                {moveHistory.length === 0 ? (
                  <p className="text-sm text-muted-foreground text-center py-4">
                    Aucun coup joué
                  </p>
                ) : (
                  moveHistory.map((move, idx) => (
                    <div 
                      key={idx} 
                      className={`flex items-center gap-2 p-2 text-sm cursor-pointer hover:bg-muted/50 rounded ${
                        idx === currentMoveIndex ? 'bg-primary/20' : idx % 2 === 0 ? 'bg-muted/30' : ''
                      }`}
                      onClick={() => goToMove(idx)}
                    >
                      <span className="text-muted-foreground font-mono text-xs w-8">
                        {Math.floor(idx / 2) + 1}.
                      </span>
                      <span className="font-semibold flex-1 text-base">{move}</span>
                    </div>
                  ))
                )}
              </div>
            </CardContent>
          </Card>

          {/* Contrôles de navigation */}
          <Card className="rounded-none">
            <CardContent className="p-2 sm:p-3 space-y-2">
              <div className="flex items-center justify-center gap-1">
                <Button
                  variant="outline"
                  size="sm"
                  className="h-7 w-7 p-0"
                  onClick={() => goToMove(-1)}
                  disabled={currentMoveIndex === -1}
                >
                  <ChevronsLeft className="h-3 w-3" />
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  className="h-7 w-7 p-0"
                  onClick={() => goToMove(Math.max(-1, currentMoveIndex - 1))}
                  disabled={currentMoveIndex === -1}
                >
                  <ChevronLeft className="h-3 w-3" />
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  className="h-7 w-7 p-0"
                  onClick={() => goToMove(Math.min(moveHistory.length - 1, currentMoveIndex + 1))}
                  disabled={currentMoveIndex >= moveHistory.length - 1}
                >
                  <ChevronRight className="h-3 w-3" />
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  className="h-7 w-7 p-0"
                  onClick={() => goToMove(moveHistory.length - 1)}
                  disabled={currentMoveIndex >= moveHistory.length - 1}
                >
                  <ChevronsRight className="h-3 w-3" />
                </Button>
              </div>

              {/* Statut du tour */}
              <div className="text-center">
                <p className="text-xs font-semibold">
                  {isAiThinking ? (
                    <span className="flex items-center justify-center gap-1">
                      <Loader2 className="h-3 w-3 animate-spin" />
                      L&apos;IA RÉFLÉCHIT...
                    </span>
                  ) : (
                    ((game.turn() === 'w' && playerColor === 'white') || (game.turn() === 'b' && playerColor === 'black'))
                      ? "VOTRE TOUR" 
                      : "TOUR DE L'IA"
                  )}
                </p>
              </div>

              {/* Statut de la partie */}
              {gameStatus && (
                <div className="p-2 bg-primary/20 text-center text-xs font-semibold">
                  {gameStatus}
                </div>
              )}

              {/* Boutons d'action */}
              <div className="space-y-1">
                <div className="grid grid-cols-2 gap-1">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={resetGame}
                    className="h-7 text-xs"
                  >
                    Nouvelle partie
                  </Button>
                  <Button
                    variant="destructive"
                    size="sm"
                    onClick={resign}
                    disabled={isGameOver}
                    className="h-7 text-xs"
                  >
                    <Flag className="h-3 w-3 mr-1" />
                    Abandonner
                  </Button>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    setGameMode("image")
                    setIsTimerActive(false)
                    setGame(new Chess())
                    setMoveHistory([])
                    setCapturedPieces({ white: [], black: [] })
                    setGameStatus("")
                    setCurrentMoveIndex(-1)
                  }}
                  className="w-full h-7 text-xs"
                >
                  <Camera className="h-3 w-3 mr-1" />
                  Nouvelle détection
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>

      </div>
        )}
      </div>
    </div>
  )
}
