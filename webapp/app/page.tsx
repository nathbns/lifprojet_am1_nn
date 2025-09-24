import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Brain, FileText, TestTube2, Zap, Target, Users } from "lucide-react";

export default function Home() {
  return (
    <div className="flex flex-col min-h-screen max-w-5xl mx-auto">
      {/* Hero Section */}
      <section className="flex-1 flex items-center justify-center px-4 py-16">
        <div className="container max-w-6xl mx-auto text-center">
          <div className="flex items-center justify-center mb-8">
            <Brain className="h-16 w-16 text-primary mr-4" />
            <h1 className="text-5xl md:text-7xl font-bold bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text text-transparent">
              LIF Project
            </h1>
          </div>
          
          <p className="text-xl md:text-2xl text-muted-foreground mb-8 max-w-3xl mx-auto">
            Plateforme de test et d'évaluation de modèles d'intelligence artificielle.
            Testez, analysez et optimisez vos modèles avec notre interface moderne.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            <Button asChild size="lg" className="w-full sm:w-auto">
              <Link href="/test">
                <TestTube2 className="mr-2 h-5 w-5" />
                Tester les Modèles
              </Link>
            </Button>
            <Button variant="outline" size="lg" asChild className="w-full sm:w-auto">
              <Link href="/docs">
                <FileText className="mr-2 h-5 w-5" />
                Documentation
              </Link>
            </Button>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-16 px-4">
        <div className="container max-w-6xl mx-auto">
          <h2 className="text-3xl md:text-4xl font-bold text-center mb-12">
            Fonctionnalités Principales
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <Card className="border-2 hover:border-primary/20 transition-colors">
              <CardHeader>
                <Zap className="h-12 w-12 text-primary mb-4" />
                <CardTitle>Test Rapide</CardTitle>
                <CardDescription>
                  Interface intuitive pour tester rapidement vos modèles avec différents types d'entrées.
                </CardDescription>
              </CardHeader>
            </Card>
            
            <Card className="border-2 hover:border-primary/20 transition-colors">
              <CardHeader>
                <Target className="h-12 w-12 text-primary mb-4" />
                <CardTitle>Analyse Précise</CardTitle>
                <CardDescription>
                  Obtenez des métriques détaillées et des analyses de performance pour vos modèles.
                </CardDescription>
              </CardHeader>
            </Card>
            
            <Card className="border-2 hover:border-primary/20 transition-colors">
              <CardHeader>
                <Users className="h-12 w-12 text-primary mb-4" />
                <CardTitle>Interface Moderne</CardTitle>
                <CardDescription>
                  Design épuré avec mode sombre/clair et composants shadcn/ui pour une expérience optimale.
                </CardDescription>
              </CardHeader>
            </Card>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-16 px-4">
        <div className="container max-w-4xl mx-auto text-center">
          <h2 className="text-3xl md:text-4xl font-bold mb-6">
            Prêt à tester vos modèles ?
          </h2>
          <p className="text-xl text-muted-foreground mb-8">
            Commencez dès maintenant avec notre interface de test intuitive.
          </p>
          <Button asChild size="lg">
            <Link href="/test">
              <Brain className="mr-2 h-5 w-5" />
              Commencer le Test
            </Link>
          </Button>
        </div>
      </section>
    </div>
  );
}
