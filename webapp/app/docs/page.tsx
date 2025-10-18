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
    </div>
  )
}
