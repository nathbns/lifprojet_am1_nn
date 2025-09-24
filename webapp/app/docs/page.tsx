import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { Book, Code, Zap, Settings, FileText, TestTube2 } from "lucide-react"

export default function DocsPage() {
  return (
    <div className="container max-w-5xl mx-auto py-8 px-4">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-4">Documentation</h1>
        <p className="text-xl text-muted-foreground">
          Guide complet pour utiliser la plateforme de test de modèles IA
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        {/* Sidebar Navigation */}
        <div className="lg:col-span-1">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Navigation</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <a href="#getting-started" className="block p-2 rounded hover:bg-muted transition-colors">
                <div className="flex items-center gap-2">
                  <Zap className="h-4 w-4" />
                  <span className="text-sm">Démarrage Rapide</span>
                </div>
              </a>
              <a href="#models" className="block p-2 rounded hover:bg-muted transition-colors">
                <div className="flex items-center gap-2">
                  <TestTube2 className="h-4 w-4" />
                  <span className="text-sm">Modèles Disponibles</span>
                </div>
              </a>
              <a href="#api" className="block p-2 rounded hover:bg-muted transition-colors">
                <div className="flex items-center gap-2">
                  <Code className="h-4 w-4" />
                  <span className="text-sm">API Reference</span>
                </div>
              </a>
              <a href="#configuration" className="block p-2 rounded hover:bg-muted transition-colors">
                <div className="flex items-center gap-2">
                  <Settings className="h-4 w-4" />
                  <span className="text-sm">Configuration</span>
                </div>
              </a>
            </CardContent>
          </Card>
        </div>

        {/* Main Content */}
        <div className="lg:col-span-3 space-y-8">
          {/* Getting Started */}
          <section id="getting-started">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Zap className="h-5 w-5" />
                  Démarrage Rapide
                </CardTitle>
                <CardDescription>
                  Commencez à tester vos modèles en quelques étapes simples
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <div className="flex items-start gap-3">
                    <Badge className="mt-1">1</Badge>
                    <div>
                      <h4 className="font-medium">Sélectionnez un modèle</h4>
                      <p className="text-sm text-muted-foreground">
                        Choisissez parmi nos modèles pré-entraînés ou uploadez le vôtre
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <Badge className="mt-1">2</Badge>
                    <div>
                      <h4 className="font-medium">Préparez vos données</h4>
                      <p className="text-sm text-muted-foreground">
                        Saisissez du texte, uploadez des images ou des fichiers audio
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <Badge className="mt-1">3</Badge>
                    <div>
                      <h4 className="font-medium">Lancez le test</h4>
                      <p className="text-sm text-muted-foreground">
                        Analysez les résultats et exportez les métriques
                      </p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </section>

          <Separator />

          {/* Models */}
          <section id="models">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TestTube2 className="h-5 w-5" />
                  Modèles Disponibles
                </CardTitle>
                <CardDescription>
                  Liste des modèles pré-configurés et leurs spécifications
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="border rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-medium">CRNN - Reconnaissance de Texte</h4>
                      <Badge variant="secondary">Vision</Badge>
                    </div>
                    <p className="text-sm text-muted-foreground mb-2">
                      Réseau de neurones convolutionnel récurrent pour la reconnaissance de texte dans les images.
                    </p>
                    <div className="flex gap-2">
                      <Badge variant="outline" className="text-xs">Images</Badge>
                      <Badge variant="outline" className="text-xs">OCR</Badge>
                      <Badge variant="outline" className="text-xs">CTC Loss</Badge>
                    </div>
                  </div>

                  <div className="border rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-medium">Transformer - NLP</h4>
                      <Badge variant="secondary">Texte</Badge>
                    </div>
                    <p className="text-sm text-muted-foreground mb-2">
                      Modèle transformer pour le traitement du langage naturel et l'analyse de sentiment.
                    </p>
                    <div className="flex gap-2">
                      <Badge variant="outline" className="text-xs">NLP</Badge>
                      <Badge variant="outline" className="text-xs">Classification</Badge>
                      <Badge variant="outline" className="text-xs">Attention</Badge>
                    </div>
                  </div>

                  <div className="border rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-medium">Modèle Personnalisé</h4>
                      <Badge variant="secondary">Custom</Badge>
                    </div>
                    <p className="text-sm text-muted-foreground mb-2">
                      Uploadez et testez vos propres modèles avec notre interface flexible.
                    </p>
                    <div className="flex gap-2">
                      <Badge variant="outline" className="text-xs">Custom</Badge>
                      <Badge variant="outline" className="text-xs">Flexible</Badge>
                      <Badge variant="outline" className="text-xs">Upload</Badge>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </section>

          <Separator />

          {/* API Reference */}
          <section id="api">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Code className="h-5 w-5" />
                  API Reference
                </CardTitle>
                <CardDescription>
                  Intégrez nos modèles dans vos applications
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <h4 className="font-medium mb-2">Endpoint de Test</h4>
                    <div className="bg-muted p-3 rounded-lg">
                      <code className="text-sm">POST /api/test</code>
                    </div>
                    <p className="text-sm text-muted-foreground mt-2">
                      Testez un modèle avec vos données d'entrée
                    </p>
                  </div>

                  <div>
                    <h4 className="font-medium mb-2">Paramètres</h4>
                    <div className="space-y-2">
                      <div className="flex justify-between items-center p-2 border rounded">
                        <span className="text-sm font-mono">model_id</span>
                        <Badge variant="outline">string</Badge>
                      </div>
                      <div className="flex justify-between items-center p-2 border rounded">
                        <span className="text-sm font-mono">input_data</span>
                        <Badge variant="outline">object</Badge>
                      </div>
                      <div className="flex justify-between items-center p-2 border rounded">
                        <span className="text-sm font-mono">confidence_threshold</span>
                        <Badge variant="outline">number</Badge>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </section>

          <Separator />

          {/* Configuration */}
          <section id="configuration">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Settings className="h-5 w-5" />
                  Configuration
                </CardTitle>
                <CardDescription>
                  Personnalisez les paramètres de test selon vos besoins
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <h4 className="font-medium mb-2">Paramètres Généraux</h4>
                    <ul className="space-y-2 text-sm text-muted-foreground">
                      <li>• <strong>Seuil de confiance :</strong> Définit le niveau minimum de confiance pour les prédictions</li>
                      <li>• <strong>Batch size :</strong> Nombre d'échantillons traités simultanément</li>
                      <li>• <strong>Timeout :</strong> Durée maximale d'attente pour les résultats</li>
                    </ul>
                  </div>

                  <div>
                    <h4 className="font-medium mb-2">Formats Supportés</h4>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div className="border rounded p-3">
                        <FileText className="h-5 w-5 mb-2" />
                        <h5 className="font-medium text-sm">Texte</h5>
                        <p className="text-xs text-muted-foreground">UTF-8, Plain text</p>
                      </div>
                      <div className="border rounded p-3">
                        <span className="text-lg mb-2 block">🖼️</span>
                        <h5 className="font-medium text-sm">Images</h5>
                        <p className="text-xs text-muted-foreground">PNG, JPG, WebP</p>
                      </div>
                      <div className="border rounded p-3">
                        <span className="text-lg mb-2 block">🎵</span>
                        <h5 className="font-medium text-sm">Audio</h5>
                        <p className="text-xs text-muted-foreground">WAV, MP3, FLAC</p>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </section>
        </div>
      </div>
    </div>
  )
}
