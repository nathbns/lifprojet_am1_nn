"use client"

import * as React from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { Upload, Play, RotateCcw, Download, FileText, Image as ImageIcon, Mic } from "lucide-react"

export default function TestPage() {
  const [selectedModel, setSelectedModel] = React.useState<string>("")
  const [inputText, setInputText] = React.useState<string>("")
  const [isLoading, setIsLoading] = React.useState(false)
  const [results, setResults] = React.useState<any>(null)

  const models = [
    { id: "crnn", name: "CRNN", type: "Vision" },
    { id: "TBA", name: "a determiner", type: "TBA" },
    { id: "TBA", name: "a determiner", type: "TBA" },
  ]

  const handleTest = async () => {
    setIsLoading(true)
    // Simuler un appel API
    setTimeout(() => {
      setResults({
        confidence: Math.random() * 100,
        prediction: "Résultat de prédiction simulé",
        processing_time: Math.random() * 1000,
        accuracy: Math.random() * 100
      })
      setIsLoading(false)
    }, 2000)
  }

  const handleReset = () => {
    setInputText("")
    setResults(null)
    setSelectedModel("")
  }

  return (
    <div className="container max-w-5xl mx-auto py-8 px-4">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-4">Test des Modèles IA</h1>
        <p className="text-xl text-muted-foreground">
          Testez vos modèles d'intelligence artificielle avec différents types d'entrées et analysez les résultats.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Configuration Panel */}
        <div className="lg:col-span-1 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Play className="h-5 w-5" />
                Configuration
              </CardTitle>
              <CardDescription>
                Sélectionnez votre modèle et configurez les paramètres
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-sm font-medium mb-2 block">Modèle</label>
                <Select value={selectedModel} onValueChange={setSelectedModel}>
                  <SelectTrigger>
                    <SelectValue placeholder="Choisir un modèle" />
                  </SelectTrigger>
                  <SelectContent>
                    {models.map((model) => (
                      <SelectItem key={model.id} value={model.id}>
                        <div className="flex items-center gap-2">
                          <span>{model.name}</span>
                          <Badge variant="secondary" className="text-xs">
                            {model.type}
                          </Badge>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="text-sm font-medium mb-2 block">Seuil de Confiance</label>
                <Input type="number" placeholder="0.8" min="0" max="1" step="0.1" />
              </div>

              <div>
                <label className="text-sm font-medium mb-2 block">Batch Size</label>
                <Input type="number" placeholder="32" min="1" max="128" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Actions Rapides</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <Button 
                onClick={handleTest} 
                disabled={!selectedModel || isLoading} 
                className="w-full"
              >
                {isLoading ? (
                  <>
                    <RotateCcw className="mr-2 h-4 w-4 animate-spin" />
                    Test en cours...
                  </>
                ) : (
                  <>
                    <Play className="mr-2 h-4 w-4" />
                    Lancer le Test
                  </>
                )}
              </Button>
              
              <Button variant="outline" onClick={handleReset} className="w-full">
                <RotateCcw className="mr-2 h-4 w-4" />
                Réinitialiser
              </Button>
              
              <Button variant="outline" className="w-full">
                <Download className="mr-2 h-4 w-4" />
                Exporter Résultats
              </Button>
            </CardContent>
          </Card>
        </div>

        {/* Input Panel */}
        <div className="lg:col-span-2 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Upload className="h-5 w-5" />
                Données d'Entrée
              </CardTitle>
              <CardDescription>
                Fournissez les données à tester selon le type de votre modèle
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {/* File Upload */}
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <ImageIcon className="h-4 w-4" />
                    <label className="text-sm font-medium">Upload d'Image</label>
                  </div>
                  <div className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-8 text-center hover:border-muted-foreground/50 transition-colors cursor-pointer">
                    <Upload className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
                    <p className="text-sm text-muted-foreground mb-2">
                      Glissez-déposez une image ou cliquez pour sélectionner
                    </p>
                    <p className="text-xs text-muted-foreground">
                      PNG, JPG jusqu'à 10MB
                    </p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Results Panel */}
          {results && (
            <Card>
              <CardHeader>
                <CardTitle>Résultats du Test</CardTitle>
                <CardDescription>
                  Analyse des performances et prédictions du modèle
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm font-medium">Confiance</span>
                      <span className="text-sm">{results.confidence.toFixed(2)}%</span>
                    </div>
                    <div className="w-full bg-secondary rounded-full h-2">
                      <div 
                        className="bg-primary h-2 rounded-full transition-all duration-500" 
                        style={{ width: `${results.confidence}%` }}
                      ></div>
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm font-medium">Précision</span>
                      <span className="text-sm">{results.accuracy.toFixed(2)}%</span>
                    </div>
                    <div className="w-full bg-secondary rounded-full h-2">
                      <div 
                        className="bg-green-500 h-2 rounded-full transition-all duration-500" 
                        style={{ width: `${results.accuracy}%` }}
                      ></div>
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <div>
                    <label className="text-sm font-medium mb-2 block">Prédiction</label>
                    <div className="p-4 bg-muted rounded-lg">
                      <code className="text-sm">{results.prediction}</code>
                    </div>
                  </div>
                  
                  <div className="flex items-center justify-between text-sm text-muted-foreground">
                    <span>Temps de traitement: {results.processing_time.toFixed(0)}ms</span>
                    <Badge variant="outline">Succès</Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}
