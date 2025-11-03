"use client"

import * as React from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import {Play, RotateCcw, X, Settings } from "lucide-react"
import { FileUpload } from "@/components/ui/file-upload"
import { Highlighter } from "@/components/ui/highlighter"
import Image from "next/image"
import { predictYoloFromSpace } from "@/lib/gradio"

interface YoloResult {
  result?: {
    data?: Array<{
      url?: string;
      [key: string]: unknown;
    } | string>;
  };
  error?: string;
}

export default function YoloPage() {
  const [isLoading, setIsLoading] = React.useState(false)
  const [results, setResults] = React.useState<YoloResult | null>(null)
  const [imageDataUrl, setImageDataUrl] = React.useState<string>("")
  const [threshold, setThreshold] = React.useState<number>(0.3)
  const [iou, setIou] = React.useState<number>(0.5)
  const [selectedModel, setSelectedModel] = React.useState<"yolov1" | "yolov3">("yolov1")

  const handleTest = async () => {
    if (!imageDataUrl) return
    setIsLoading(true)
    try {
      // Appel direct Gradio côté client (évite la limite de taille Vercel)
      const result = await predictYoloFromSpace(imageDataUrl, {
        confidence_threshold: threshold,
        iou_threshold: iou,
        show_confidence: true,
        model: selectedModel,
      })
      console.log("Résultats complets:", result)
      const resultData = result as { data?: Array<unknown> }
      console.log("Data:", resultData?.data)
      if (resultData?.data) {
        console.log("Stats (index 0):", resultData.data[0])
        console.log("Stats (index 1):", resultData.data[1])
        console.log("Stats (index 2):", resultData.data[2])
      }
      setResults({ result: resultData as YoloResult['result'] })
    } catch (error) {
      console.error("Erreur YOLO:", error)
      setResults({ error: error instanceof Error ? error.message : "Erreur lors de l&apos;appel API" })
    } finally {
      setIsLoading(false)
    }
  }

  const handleReset = () => {
    setResults(null)
    setImageDataUrl("")
  }

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
          Nos modèles{"    "}
          <Highlighter action="highlight" color="#c9c9c9" padding={3} >
            <p className="text-background">YOLO</p>
          </Highlighter>
          </h1>
        <p className="text-muted-foreground">
          <Highlighter action="underline" color="#c9c9c9" padding={3} >
          Glissez-déposez
          </Highlighter>
          {" "}une image, choisissez un modèle et lancez la détection
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Upload Section */}
        <div className="space-y-6">
          <div className="relative">
            <Card>
              <CardContent>
                {!imageDataUrl ? (
                  <FileUpload onChange={handleFileUpload} />
                ) : (
                  <div className="space-y-4">
                    <div className="relative inline-block">
                      <Image 
                        src={imageDataUrl} 
                        alt="preview" 
                        width={800} 
                        height={600} 
                        className="max-h-80 w-auto h-auto mx-auto" 
                        style={{ objectFit: 'contain' }}
                      />
                      <Button
                        variant="destructive"
                        size="sm"
                        className="absolute -top-2 -right-2 h-8 w-8 p-0"
                        onClick={() => setImageDataUrl("")}
                      >
                        <X className="h-2 w-2" />
                      </Button>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Settings & Actions */}
        <div className="space-y-6">
          <div className="relative">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Settings className="h-5 w-5" />
                  Paramètres
                </CardTitle>
                <CardDescription>Ajustez les seuils de détection</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-4">
                  {/* Sélecteur de modèle */}
                  <div>
                    <label className="text-sm font-medium mb-2 block">Modèle YOLO</label>
                    <div className="grid grid-cols-2 gap-3">
                      <button
                        onClick={() => setSelectedModel("yolov1")}
                        className={`p-4 border-2 transition-all ${
                          selectedModel === "yolov1"
                            ? "border border-foreground"
                            : "border-muted"
                        }`}
                      >
                        <div className="text-center">
                          <div className={`font-bold text-lg ${selectedModel === "yolov1" ? "text-foreground" : ""}`}>
                            YOLOv1
                          </div>
                        </div>
                      </button>
                      <button
                        onClick={() => setSelectedModel("yolov3")}
                        className={`p-4 border-2 transition-all ${
                          selectedModel === "yolov3"
                            ? "border border-foreground"
                            : "border-muted"
                        }`}
                      >
                        <div className="text-center">
                          <div className={`font-bold text-lg ${selectedModel === "yolov3" ? "text-foreground" : ""}`}>
                            YOLOv3
                          </div>
                        </div>
                      </button>
                    </div>
                  </div>
                  
                  <div>
                    <label className="text-sm font-medium mb-2 block">Seuil de confiance</label>
                    <div className="space-y-2">
                      <Input 
                        type="number" 
                        value={threshold} 
                        min={0.05} 
                        max={0.95} 
                        step={0.05} 
                        onChange={(e)=>setThreshold(parseFloat(e.target.value))} 
                      />
                      <div className="flex justify-between text-xs text-muted-foreground">
                        <span>Plus bas = plus de détections</span>
                        <span>{threshold}</span>
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    <label className="text-sm font-medium mb-2 block">Seuil IoU</label>
                    <div className="space-y-2">
                      <Input 
                        type="number" 
                        value={iou} 
                        min={0.1} 
                        max={0.9} 
                        step={0.05} 
                        onChange={(e)=>setIou(parseFloat(e.target.value))} 
                      />
                      <div className="flex justify-between text-xs text-muted-foreground">
                        <span>Plus haut = garde plus de boxes</span>
                        <span>{iou}</span>
                      </div>
                    </div>
                  </div>
                  
                </div>

                <div className="flex gap-3">
                  <Button 
                    onClick={handleTest} 
                    disabled={!imageDataUrl || isLoading} 
                    className="flex-1"
                    size="lg"
                  >
                    {isLoading ? (
                      <>
                        <RotateCcw className="mr-2 h-4 w-4 animate-spin" />
                        Analyse en cours...
                      </>
                    ) : (
                      <>
                        <Play className="mr-2 h-4 w-4" />
                        Analyser l&apos;image
                      </>
                    )}
                  </Button>
                  <Button 
                    variant="outline" 
                    onClick={handleReset} 
                    size="lg"
                  >
                    <RotateCcw className="mr-2 h-4 w-4" />
                    Reset
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>

      {/* Results Section */}
      {results && (
        <div className="mt-8">
          <div className="relative">
            <Card>
              <CardHeader>
                <CardTitle>
                  Résultats de détection
                </CardTitle>
                <CardDescription>Image annotée avec les objets détectés</CardDescription>
              </CardHeader>
              <CardContent>
                {Array.isArray(results?.result?.data) && (
                  <div className="space-y-6">
                    {/* Image annotée seulement */}
                    <div className="flex justify-center">
                      <div className="max-w-2xl">
                        <p className="text-sm font-medium mb-3 text-center">Résultat de détection</p>
                        {(() => {
                          // YOLOv3 retourne 2 éléments: [0]=image, [1]=stats
                          // YOLOv1 retourne 3 éléments: [0]=?, [1]=image, [2]=stats
                          const imageIndex = selectedModel === "yolov3" ? 0 : 1;
                          const imageUrl = (results.result.data[imageIndex] as { url?: string })?.url;
                          return imageUrl ? (
                            <Image 
                              src={imageUrl} 
                              alt="annotated" 
                              width={640} 
                              height={480} 
                              className="w-full border" 
                            />
                          ) : null;
                        })()}
                      </div>
                    </div>
                    
                    {/* Stats de détection */}
                    {(() => {
                      const statsIndex = selectedModel === "yolov3" ? 1 : 2;
                      const isYOLOv1 = selectedModel === "yolov1";
                      return results.result.data[statsIndex] && (
                      <div className="space-y-4">
                        {(() => {
                          const dataItem = results.result.data[statsIndex];
                          // Vérifier que l'élément est une chaîne de caractères
                          if (typeof dataItem !== 'string') {
                            return null;
                          }
                          const markdownText = dataItem;
                          
                          // Parse les données du markdown
                          const lines = markdownText.split('\n').filter((line: string) => line.trim());
                          let detectedCount = 0;
                          let inferenceTime = '';
                          let avgConfidence = '';
                          
                          // Stocker toutes les détections individuelles (même si du même type)
                          const detectedObjects: Array<{name: string, confidence: number, index?: number}> = [];
                          
                          lines.forEach((line: string) => {
                            // Nombre d'objets détectés
                            const countMatch = line.match(/\*\*(\d+)\s+objet\(s\)\s+détecté\(s\)\*\*/i);
                            if (countMatch) detectedCount = parseInt(countMatch[1]);
                            
                            // Temps d&apos;inférence
                            const timeMatch = line.match(/Temps d&apos;inférence:\s*\*\*([^*]+)\*\*/i);
                            if (timeMatch) inferenceTime = timeMatch[1];
                            
                            // Confiance moyenne
                            const confMatch = line.match(/Confiance moyenne:\s*\*\*([^*]+)\*\*/i);
                            if (confMatch) avgConfidence = confMatch[1];
                            
                            // Ignorer les lignes de statistiques générales
                            if (line.match(/confiance|temps|objet|détecté|moyenne|inférence/i)) {
                              return;
                            }
                            
                            // Pour YOLOv1 seulement: ignorer les lignes avec des guillemets ou des tirets (format suspect)
                            if (isYOLOv1 && (line.match(/^[\s-]*["']/) || line.match(/["'].*:/))) {
                              return;
                            }
                            
                            // Objets individuels avec confiance (format: "1. cat: 83.5%" ou "cat: 0.83")
                            // Prioriser le format avec numéro car plus fiable
                            const objMatchWithNum = line.match(/(\d+)\.\s*([^:]+):\s*([\d.]+)%?/);
                            let name: string | undefined;
                            let confidenceStr: string | undefined;
                            let detectionIndex: number | undefined;
                            
                            if (objMatchWithNum) {
                              // Format avec numéro: "1. cat: 83.5%"
                              detectionIndex = parseInt(objMatchWithNum[1]);
                              name = objMatchWithNum[2]?.trim().toLowerCase();
                              confidenceStr = objMatchWithNum[3];
                            } else {
                              // Format sans numéro: "cat: 0.83" ou "cat: 83.5%"
                              // Pour YOLOv1: ignorer si la ligne commence par un tiret ou contient des guillemets
                              // Pour YOLOv3: être plus permissif
                              const objMatchWithoutNum = isYOLOv1 
                                ? line.match(/^([^:"'\s-]+):\s*([\d.]+)%?/)
                                : line.match(/([^:]+):\s*([\d.]+)%?/);
                              if (objMatchWithoutNum) {
                                name = objMatchWithoutNum[1]?.trim().toLowerCase();
                                confidenceStr = objMatchWithoutNum[2];
                              }
                            }
                            
                            if (name && confidenceStr) {
                              // Ignorer les noms qui ressemblent à des mots-clés de statistiques
                              if (name.match(/^(confiance|temps|objet|détecté|moyenne|inférence)$/i)) {
                                return;
                              }
                              
                              // Si la confiance est entre 0 et 1 (format 0.83), multiplier par 100
                              let confidence = parseFloat(confidenceStr);
                              if (confidence <= 1 && confidence > 0) {
                                confidence = confidence * 100;
                              }
                              
                              // Filtrer les confiances invalides
                              if (confidence <= 0 || confidence > 100 || !isFinite(confidence)) {
                                return;
                              }
                              
                              // Pour YOLOv1 seulement: ignorer les confiances exactement à 100% si elles ne viennent pas du format avec numéro
                              // Pour YOLOv3: accepter toutes les confiances valides
                              if (isYOLOv1 && confidence === 100 && !objMatchWithNum) {
                                return;
                              }
                              
                              // Ajouter cette détection individuelle (même si on a déjà une détection du même type)
                              detectedObjects.push({
                                name: name,
                                confidence: confidence,
                                index: detectionIndex
                              });
                            }
                          });
                          
                          // Trier par confiance décroissante (ou par index si disponible)
                          detectedObjects.sort((a, b) => {
                            if (a.index !== undefined && b.index !== undefined) {
                              return a.index - b.index; // Trier par index si disponible
                            }
                            return b.confidence - a.confidence; // Sinon par confiance décroissante
                          });
                          
                          // Si le nombre d'objets n'a pas été trouvé via regex, utiliser la longueur de la liste
                          if (detectedCount === 0 && detectedObjects.length > 0) {
                            detectedCount = detectedObjects.length;
                          }
                          
                          return (
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                              {/* Nombre d'objets */}
                              <div className="text-center p-4 bg-muted/50 border">
                                <div className="text-2xl font-bold text-primary mb-1">{detectedCount}</div>
                                <div className="text-xs text-muted-foreground">Objet(s) détecté(s)</div>
                              </div>
                              
                              {/* Temps d'inférence */}
                              {inferenceTime && (
                                <div className="text-center p-4 bg-muted/50 border">
                                  <div className="text-2xl font-bold mb-1">{inferenceTime}</div>
                                  <div className="text-xs text-muted-foreground">Temps d&apos;inférence</div>
                                </div>
                              )}
                              
                              {/* Confiance moyenne */}
                              {avgConfidence && (
                                <div className="text-center p-4 bg-muted/50 border">
                                  <div className="text-2xl font-bold mb-1">{avgConfidence}</div>
                                  <div className="text-xs text-muted-foreground">Confiance moyenne</div>
                                </div>
                              )}
                            </div>
                          );
                        })()}
                        
                        {/* Liste des objets détectés */}
                        {(() => {
                          const dataItem = results.result.data[statsIndex];
                          // Vérifier que l'élément est une chaîne de caractères
                          if (typeof dataItem !== 'string') {
                            return null;
                          }
                          const markdownText = dataItem;
                          const lines = markdownText.split('\n').filter((line: string) => line.trim());
                          
                          // Stocker toutes les détections individuelles (même si du même type)
                          const detectedObjects: Array<{name: string, confidence: number, index?: number}> = [];
                          
                          lines.forEach((line: string) => {
                            // Ignorer les lignes de statistiques générales
                            if (line.match(/confiance|temps|objet|détecté|moyenne|inférence/i)) {
                              return;
                            }
                            
                            // Pour YOLOv1 seulement: ignorer les lignes avec des guillemets ou des tirets (format suspect)
                            if (isYOLOv1 && (line.match(/^[\s-]*["']/) || line.match(/["'].*:/))) {
                              return;
                            }
                            
                            // Objets individuels avec confiance (format: "1. cat: 83.5%" ou "cat: 0.83")
                            // Prioriser le format avec numéro car plus fiable
                            const objMatchWithNum = line.match(/(\d+)\.\s*([^:]+):\s*([\d.]+)%?/);
                            let name: string | undefined;
                            let confidenceStr: string | undefined;
                            let detectionIndex: number | undefined;
                            
                            if (objMatchWithNum) {
                              // Format avec numéro: "1. cat: 83.5%"
                              detectionIndex = parseInt(objMatchWithNum[1]);
                              name = objMatchWithNum[2]?.trim().toLowerCase();
                              confidenceStr = objMatchWithNum[3];
                            } else {
                              // Format sans numéro: "cat: 0.83" ou "cat: 83.5%"
                              // Pour YOLOv1: ignorer si la ligne commence par un tiret ou contient des guillemets
                              // Pour YOLOv3: être plus permissif
                              const objMatchWithoutNum = isYOLOv1 
                                ? line.match(/^([^:"'\s-]+):\s*([\d.]+)%?/)
                                : line.match(/([^:]+):\s*([\d.]+)%?/);
                              if (objMatchWithoutNum) {
                                name = objMatchWithoutNum[1]?.trim().toLowerCase();
                                confidenceStr = objMatchWithoutNum[2];
                              }
                            }
                            
                            if (name && confidenceStr) {
                              // Ignorer les noms qui ressemblent à des mots-clés de statistiques
                              if (name.match(/^(confiance|temps|objet|détecté|moyenne|inférence)$/i)) {
                                return;
                              }
                              
                              // Si la confiance est entre 0 et 1 (format 0.83), multiplier par 100
                              let confidence = parseFloat(confidenceStr);
                              if (confidence <= 1 && confidence > 0) {
                                confidence = confidence * 100;
                              }
                              
                              // Filtrer les confiances invalides
                              if (confidence <= 0 || confidence > 100 || !isFinite(confidence)) {
                                return;
                              }
                              
                              // Pour YOLOv1 seulement: ignorer les confiances exactement à 100% si elles ne viennent pas du format avec numéro
                              // Pour YOLOv3: accepter toutes les confiances valides
                              if (isYOLOv1 && confidence === 100 && !objMatchWithNum) {
                                return;
                              }
                              
                              // Ajouter cette détection individuelle (même si on a déjà une détection du même type)
                              detectedObjects.push({
                                name: name,
                                confidence: confidence,
                                index: detectionIndex
                              });
                            }
                          });
                          
                          // Trier par confiance décroissante (ou par index si disponible)
                          detectedObjects.sort((a, b) => {
                            if (a.index !== undefined && b.index !== undefined) {
                              return a.index - b.index; // Trier par index si disponible
                            }
                            return b.confidence - a.confidence; // Sinon par confiance décroissante
                          });
                          
                          if (detectedObjects.length > 0) {
                            return (
                              <div className="space-y-3">
                                <h4 className="font-semibold text-sm">Objets détectés</h4>
                                <div className="space-y-2">
                                  {detectedObjects.map((obj, index) => (
                                    <div key={index} className="flex items-center justify-between p-3 bg-muted/50 border">
                                      <span className="text-sm font-medium capitalize">{obj.name}</span>
                                      <div className="flex items-center gap-3">
                                        <div className="w-24 bg-muted h-2">
                                          <div 
                                            className="bg-primary h-2 transition-all duration-500" 
                                            style={{ width: `${obj.confidence}%` }}
                                          ></div>
                                        </div>
                                        <span className="text-sm font-mono w-12 text-right">{obj.confidence}%</span>
                                      </div>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            );
                          }
                          return null;
                        })()}
                      </div>
                      );
                    })()}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      )}
      </div>
    </div>
  )
}





