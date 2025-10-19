"use client"

import * as React from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import {Play, RotateCcw, X, Settings } from "lucide-react"
import { GlowingEffect } from "@/components/ui/glowing-effect"
import { FileUpload } from "@/components/ui/file-upload"
import { Highlighter } from "@/components/ui/highlighter"
import Image from "next/image"

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

  const handleTest = async () => {
    if (!imageDataUrl) return
    setIsLoading(true)
    try {
      const res = await fetch("/api/yolo", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          imageDataUrl,
          confidence_threshold: threshold,
          iou_threshold: iou,
          show_confidence: true,
        }),
      })
      const json = await res.json()
      setResults(json)
    } catch {
      setResults({ error: "Erreur lors de l&apos;appel API" })
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
          Nos modeles{"    "}
          <Highlighter action="highlight" color="#15803d" padding={3} >
            YOLO
          </Highlighter>
          </h1>
        <p className="text-muted-foreground">
          <Highlighter action="underline" color="#15803d" padding={3} >
          Glissez-déposez
          </Highlighter>
          {" "}une image, puis lancez la détection
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
                        className="max-h-80 w-auto h-auto mx-auto rounded-lg" 
                        style={{ objectFit: 'contain' }}
                      />
                      <Button
                        variant="destructive"
                        size="sm"
                        className="absolute -top-2 -right-2 rounded-full h-8 w-8 p-0"
                        onClick={() => setImageDataUrl("")}
                      >
                        <X className="h-2 w-2" />
                      </Button>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
            <GlowingEffect className="rounded-xl" borderWidth={1} />
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
            <GlowingEffect className="rounded-xl" borderWidth={1} />
          </div>
        </div>
      </div>

      {/* Results Section */}
      {results && (
        <div className="mt-8">
          <div className="relative">
            <Card>
              <CardHeader>
                <CardTitle>Résultats de détection</CardTitle>
                <CardDescription>Image annotée avec les objets détectés</CardDescription>
              </CardHeader>
              <CardContent>
                {Array.isArray(results?.result?.data) && (
                  <div className="space-y-6">
                    {/* Image annotée seulement */}
                    <div className="flex justify-center">
                      <div className="max-w-2xl">
                        <p className="text-sm font-medium mb-3 text-center">Résultat de détection</p>
                        {(results.result.data[1] as { url?: string })?.url && (
                          <Image 
                            src={(results.result.data[1] as { url?: string }).url!} 
                            alt="annotated" 
                            width={640} 
                            height={480} 
                            className="w-full rounded-lg border" 
                          />
                        )}
                      </div>
                    </div>
                    
                    {/* Stats de détection */}
                    {results.result.data[2] && (
                      <div className="space-y-4">
                        {(() => {
                          const markdownText = results.result.data[2] as string;
                          
                          // Parse les données du markdown
                          const lines = markdownText.split('\n').filter((line: string) => line.trim());
                          let detectedCount = 0;
                          let inferenceTime = '';
                          let avgConfidence = '';
                          const detectedObjects: Array<{name: string, confidence: number}> = [];
                          
                          lines.forEach((line: string) => {
                            // Nombre d'objets détectés
                            const countMatch = line.match(/\*\*(\d+)\s+objet\(s\)\s+détecté\(s\)\*\*/);
                            if (countMatch) detectedCount = parseInt(countMatch[1]);
                            
                            // Temps d&apos;inférence
                            const timeMatch = line.match(/Temps d&apos;inférence:\s*\*\*([^*]+)\*\*/);
                            if (timeMatch) inferenceTime = timeMatch[1];
                            
                            // Confiance moyenne
                            const confMatch = line.match(/Confiance moyenne:\s*\*\*([^*]+)\*\*/);
                            if (confMatch) avgConfidence = confMatch[1];
                            
                            // Objets individuels avec confiance
                            const objMatch = line.match(/(\d+)\.\s*([^:]+):\s*([\d.]+)%/);
                            if (objMatch) {
                              detectedObjects.push({
                                name: objMatch[2].trim(),
                                confidence: parseFloat(objMatch[3])
                              });
                            }
                          });
                          
                          return (
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                              {/* Nombre d'objets */}
                              <div className="text-center p-4 bg-muted/50 rounded-lg border">
                                <div className="text-2xl font-bold text-primary mb-1">{detectedCount}</div>
                                <div className="text-xs text-muted-foreground">Objet(s) détecté(s)</div>
                              </div>
                              
                              {/* Temps d'inférence */}
                              {inferenceTime && (
                                <div className="text-center p-4 bg-muted/50 rounded-lg border">
                                  <div className="text-2xl font-bold mb-1">{inferenceTime}</div>
                                  <div className="text-xs text-muted-foreground">Temps d&apos;inférence</div>
                                </div>
                              )}
                              
                              {/* Confiance moyenne */}
                              {avgConfidence && (
                                <div className="text-center p-4 bg-muted/50 rounded-lg border">
                                  <div className="text-2xl font-bold mb-1">{avgConfidence}</div>
                                  <div className="text-xs text-muted-foreground">Confiance moyenne</div>
                                </div>
                              )}
                            </div>
                          );
                        })()}
                        
                        {/* Liste des objets détectés */}
                        {(() => {
                          const markdownText = results.result.data[2] as string;
                          const lines = markdownText.split('\n').filter((line: string) => line.trim());
                          const detectedObjects: Array<{name: string, confidence: number}> = [];
                          
                          lines.forEach((line: string) => {
                            const objMatch = line.match(/(\d+)\.\s*([^:]+):\s*([\d.]+)%/);
                            if (objMatch) {
                              detectedObjects.push({
                                name: objMatch[2].trim(),
                                confidence: parseFloat(objMatch[3])
                              });
                            }
                          });
                          
                          if (detectedObjects.length > 0) {
                            return (
                              <div className="space-y-3">
                                <h4 className="font-semibold text-sm">Objets détectés</h4>
                                <div className="space-y-2">
                                  {detectedObjects.map((obj, index) => (
                                    <div key={index} className="flex items-center justify-between p-3 bg-muted/50 rounded-lg border">
                                      <span className="text-sm font-medium capitalize">{obj.name}</span>
                                      <div className="flex items-center gap-3">
                                        <div className="w-24 bg-muted rounded-full h-2">
                                          <div 
                                            className="bg-primary h-2 rounded-full transition-all duration-500" 
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
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
            <GlowingEffect className="rounded-xl" borderWidth={1} />
          </div>
        </div>
      )}
      </div>
    </div>
  )
}


