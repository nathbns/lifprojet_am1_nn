/**
 * Implémentation de l'algorithme Minimax avec élagage Alpha-Beta
 * Adapté pour fonctionner avec chess.js
 */

import { Chess } from "chess.js"
import type { Move, Square } from "chess.js"

// Valeurs des pièces pour l'évaluation
const PIECE_VALUES: { [key: string]: number } = {
  p: 100,   // Pion
  n: 320,   // Cavalier
  b: 330,   // Fou
  r: 500,   // Tour
  q: 900,   // Reine
  k: 20000  // Roi
}

// Tables de position pour encourager les bonnes positions (blancs)
const PAWN_TABLE = [
  0,  0,  0,  0,  0,  0,  0,  0,
  50, 50, 50, 50, 50, 50, 50, 50,
  10, 10, 20, 30, 30, 20, 10, 10,
  5,  5, 10, 25, 25, 10,  5,  5,
  0,  0,  0, 20, 20,  0,  0,  0,
  5, -5,-10,  0,  0,-10, -5,  5,
  5, 10, 10,-20,-20, 10, 10,  5,
  0,  0,  0,  0,  0,  0,  0,  0
]

const KNIGHT_TABLE = [
  -50,-40,-30,-30,-30,-30,-40,-50,
  -40,-20,  0,  0,  0,  0,-20,-40,
  -30,  0, 10, 15, 15, 10,  0,-30,
  -30,  5, 15, 20, 20, 15,  5,-30,
  -30,  0, 15, 20, 20, 15,  0,-30,
  -30,  5, 10, 15, 15, 10,  5,-30,
  -40,-20,  0,  5,  5,  0,-20,-40,
  -50,-40,-30,-30,-30,-30,-40,-50
]

const BISHOP_TABLE = [
  -20,-10,-10,-10,-10,-10,-10,-20,
  -10,  0,  0,  0,  0,  0,  0,-10,
  -10,  0,  5, 10, 10,  5,  0,-10,
  -10,  5,  5, 10, 10,  5,  5,-10,
  -10,  0, 10, 10, 10, 10,  0,-10,
  -10, 10, 10, 10, 10, 10, 10,-10,
  -10,  5,  0,  0,  0,  0,  5,-10,
  -20,-10,-10,-10,-10,-10,-10,-20
]

const ROOK_TABLE = [
  0,  0,  0,  0,  0,  0,  0,  0,
  5, 10, 10, 10, 10, 10, 10,  5,
  -5,  0,  0,  0,  0,  0,  0, -5,
  -5,  0,  0,  0,  0,  0,  0, -5,
  -5,  0,  0,  0,  0,  0,  0, -5,
  -5,  0,  0,  0,  0,  0,  0, -5,
  -5,  0,  0,  0,  0,  0,  0, -5,
  0,  0,  0,  5,  5,  0,  0,  0
]

const QUEEN_TABLE = [
  -20,-10,-10, -5, -5,-10,-10,-20,
  -10,  0,  0,  0,  0,  0,  0,-10,
  -10,  0,  5,  5,  5,  5,  0,-10,
  -5,  0,  5,  5,  5,  5,  0, -5,
  0,  0,  5,  5,  5,  5,  0, -5,
  -10,  5,  5,  5,  5,  5,  0,-10,
  -10,  0,  5,  0,  0,  0,  0,-10,
  -20,-10,-10, -5, -5,-10,-10,-20
]

const KING_TABLE = [
  -30,-40,-40,-50,-50,-40,-40,-30,
  -30,-40,-40,-50,-50,-40,-40,-30,
  -30,-40,-40,-50,-50,-40,-40,-30,
  -30,-40,-40,-50,-50,-40,-40,-30,
  -20,-30,-30,-40,-40,-30,-30,-20,
  -10,-20,-20,-20,-20,-20,-20,-10,
  20, 20,  0,  0,  0,  0, 20, 20,
  20, 30, 10,  0,  0, 10, 30, 20
]

// Obtenir la table de position pour une pièce
function getPiecePositionTable(piece: string): number[] {
  const type = piece.toLowerCase()
  switch (type) {
    case 'p': return PAWN_TABLE
    case 'n': return KNIGHT_TABLE
    case 'b': return BISHOP_TABLE
    case 'r': return ROOK_TABLE
    case 'q': return QUEEN_TABLE
    case 'k': return KING_TABLE
    default: return Array(64).fill(0)
  }
}

// Convertir une case (ex: "e4") en index 0-63
function squareToIndex(square: Square): number {
  const file = square.charCodeAt(0) - 97 // a=0, b=1, etc.
  const rank = parseInt(square.charAt(1)) - 1 // 1=0, 2=1, etc.
  return rank * 8 + file
}

/**
 * Évalue la position actuelle du point de vue des blancs
 * Positif = avantage blanc, Négatif = avantage noir
 */
export function evaluatePosition(game: Chess): number {
  if (game.isCheckmate()) {
    // Si les blancs sont en échec et mat, c'est très mauvais
    // Si les noirs sont en échec et mat, c'est très bon
    return game.turn() === 'w' ? -999999 : 999999
  }
  
  if (game.isDraw() || game.isStalemate() || game.isThreefoldRepetition()) {
    return 0
  }

  let score = 0
  const board = game.board()

  // Parcourir l'échiquier
  for (let rank = 0; rank < 8; rank++) {
    for (let file = 0; file < 8; file++) {
      const piece = board[rank][file]
      
      if (piece) {
        const pieceValue = PIECE_VALUES[piece.type]
        const positionTable = getPiecePositionTable(piece.type)
        
        // Calculer l'index de position
        // Pour les blancs: utiliser l'index normal
        // Pour les noirs: inverser verticalement (miroir)
        const posIndex = piece.color === 'w' 
          ? (7 - rank) * 8 + file  // Blancs jouent du bas
          : rank * 8 + file         // Noirs jouent du haut
        
        const positionBonus = positionTable[posIndex]
        
        // Ajouter ou soustraire selon la couleur
        if (piece.color === 'w') {
          score += pieceValue + positionBonus
        } else {
          score -= pieceValue + positionBonus
        }
      }
    }
  }

  return score
}

export interface MinimaxResult {
  move: Move | null
  score: number
  nodesEvaluated: number
}

/**
 * Classe pour l'IA utilisant Minimax avec élagage Alpha-Beta
 */
export class MinimaxAI {
  private depth: number
  private nodesEvaluated: number

  constructor(depth: number = 3) {
    this.depth = depth
    this.nodesEvaluated = 0
  }

  /**
   * Trouve le meilleur coup pour la position actuelle
   */
  getBestMove(game: Chess): MinimaxResult {
    this.nodesEvaluated = 0
    const isMaximizing = game.turn() === 'w'
    
    let bestMove: Move | null = null
    let bestScore = isMaximizing ? -Infinity : Infinity
    let alpha = -Infinity
    let beta = Infinity

    const moves = game.moves({ verbose: true })

    if (moves.length === 0) {
      return { move: null, score: 0, nodesEvaluated: 0 }
    }

    // Évaluer chaque coup possible
    for (const move of moves) {
      game.move(move)
      const score = this.minimax(game, this.depth - 1, alpha, beta, !isMaximizing)
      game.undo()

      if (isMaximizing) {
        if (score > bestScore) {
          bestScore = score
          bestMove = move
        }
        alpha = Math.max(alpha, score)
      } else {
        if (score < bestScore) {
          bestScore = score
          bestMove = move
        }
        beta = Math.min(beta, score)
      }
    }

    return {
      move: bestMove,
      score: bestScore,
      nodesEvaluated: this.nodesEvaluated
    }
  }

  /**
   * Algorithme Minimax avec élagage Alpha-Beta
   */
  private minimax(
    game: Chess,
    depth: number,
    alpha: number,
    beta: number,
    isMaximizing: boolean
  ): number {
    this.nodesEvaluated++

    // Condition d'arrêt
    if (depth === 0 || game.isGameOver()) {
      return evaluatePosition(game)
    }

    const moves = game.moves({ verbose: true })

    if (isMaximizing) {
      let maxEval = -Infinity

      for (const move of moves) {
        game.move(move)
        const evaluation = this.minimax(game, depth - 1, alpha, beta, false)
        game.undo()

        maxEval = Math.max(maxEval, evaluation)
        alpha = Math.max(alpha, evaluation)

        // Élagage Beta
        if (beta <= alpha) {
          break
        }
      }

      return maxEval
    } else {
      let minEval = Infinity

      for (const move of moves) {
        game.move(move)
        const evaluation = this.minimax(game, depth - 1, alpha, beta, true)
        game.undo()

        minEval = Math.min(minEval, evaluation)
        beta = Math.min(beta, evaluation)

        // Élagage Alpha
        if (beta <= alpha) {
          break
        }
      }

      return minEval
    }
  }

  /**
   * Définir la profondeur de recherche
   */
  setDepth(depth: number) {
    this.depth = depth
  }

  /**
   * Obtenir la profondeur actuelle
   */
  getDepth(): number {
    return this.depth
  }
}

