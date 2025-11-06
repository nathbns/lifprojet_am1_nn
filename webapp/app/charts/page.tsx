"use client";
import { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import "chart.js/auto";
import type { ChartData } from "chart.js";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

const Line = dynamic(() => import("react-chartjs-2").then((mod) => mod.Line), {
  ssr: false,
});

interface EpochData {
  epoch: number;
  train_loss: number;
  learning_rate: number;
}

interface ExperimentData {
  experiment_name: string;
  epochs: EpochData[];
}

export default function GraphPage() {
  const [chart1, setChart1] = useState<ChartData<"line"> | null>(null);
  const [chart2, setChart2] = useState<ChartData<"line"> | null>(null);
  const [isDark, setIsDark] = useState(false);

  // Helpers pour récupérer les couleurs du thème courant (CSS Vars)
  function getCssColor(variable: string, fallback: string): string {
    if (typeof window === 'undefined') return fallback;
    const val = getComputedStyle(document.documentElement).getPropertyValue(variable).trim();
    if (!val) return fallback;
    const temp = document.createElement('div');
    temp.style.color = val;
    document.body.appendChild(temp);
    const rgb = getComputedStyle(temp).color;
    document.body.removeChild(temp);
    return rgb || fallback;
  }

  const colors = useMemo(() => {
    const foreground = getCssColor('--foreground', 'rgb(17,24,39)');
    const mutedFg = getCssColor('--muted-foreground', 'rgb(107,114,128)');
    const primary = getCssColor('--primary', 'rgb(37,99,235)');
    const secondary = getCssColor('--secondary', 'rgb(16,185,129)');
    return { foreground, mutedFg, primary, secondary };
  }, [isDark]);

  useEffect(() => {
    const updateIsDark = () => setIsDark(document.documentElement.classList.contains('dark'));
    updateIsDark();
    const observer = new MutationObserver(updateIsDark);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    async function fetchData() {
      try {
        // --- Dataset 1 : Pascal VOC ---
        const res1 = await fetch("/datasets/yolov3_pascal_voc_1760281309.json");
        const json1: ExperimentData = await res1.json();

        const labels1 = json1.epochs.map((e) => e.epoch);
        const lossData1 = json1.epochs.map((e) => e.train_loss);
        const lrData1 = json1.epochs.map((e) => e.learning_rate);

        setChart1({
          labels: labels1,
          datasets: [
            {
              label: "Training Loss",
              data: lossData1,
              borderColor: colors.primary,
              backgroundColor: isDark ? "rgba(37, 99, 235, 0.2)" : "rgba(37, 99, 235, 0.12)",
              tension: 0.25,
            },
          ],
        });

        // --- Dataset 2 : Chess ---
        const res2 = await fetch("/datasets/yolov3_chess_1760367124.json");
        const json2: ExperimentData = await res2.json();

        const labels2 = json2.epochs.map((e) => e.epoch);
        const lossData2 = json2.epochs.map((e) => e.train_loss);
        const lrData2 = json2.epochs.map((e) => e.learning_rate);

        setChart2({
          labels: labels2,
          datasets: [
            {
              label: " YOLO Chess",
              data: lossData2,
              borderColor: colors.secondary, 
              backgroundColor: isDark ? "rgba(16, 185, 129, 0.2)" : "rgba(16, 185, 129, 0.12)",
              tension: 0.25,
            },
            {
              label: "YOLO Pascal Voc",
              data: lossData1,
              borderColor: "rgba(235, 93, 37, 1)",
              backgroundColor: isDark ? "rgba(235, 93, 37, 0.2)" : "rgba(235, 93, 37, 0.12)",
              tension: 0.25,
            },
          ],
        });
      } catch (err) {
        console.error("Erreur lors du chargement des datasets :", err);
      }
    }

    fetchData();
  }, [isDark, colors]);

  return (
    <div className="min-h-screen bg-background text-foreground p-10">
      <div className="flex items-center justify-between mb-12 max-w-6xl mx-auto">
        <h1 className="text-4xl font-bold tracking-tight">
          Analyse des modèles YOLOv3
        </h1>
        <span className="text-sm text-muted-foreground italic">
          Comparaison : Pascal VOC vs Chess
        </span>
      </div>

      {!chart1 || !chart2 ? (
        <p className="text-center text-muted-foreground">Chargement des données...</p>
      ) : (
        <div className="grid md:grid-cols-2 gap-8 max-w-6xl mx-auto">
          {/* Graphique 1 : Pascal VOC */}
          <Card className="bg-background border border-border shadow-md hover:shadow-lg transition-all">
            <CardHeader>
              <CardTitle className="text-lg font-semibold text-foreground">
                YOLOv3 – Pascal VOC
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-2">
              <Line data={chart1} options={{
                responsive: true,
                plugins: {
                  legend: {
                    labels: { color: colors.foreground }
                  },
                },
                scales: {
                  x: {
                    ticks: { color: colors.mutedFg },
                    grid: { color: isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.06)' }
                  },
                  y: {
                    ticks: { color: colors.mutedFg },
                    grid: { color: isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.06)' }
                  }
                }
              }} />
            </CardContent>
          </Card>

          {/* Graphique 2 : Chess */}
          <Card className="bg-background border border-border shadow-md hover:shadow-lg transition-all">
            <CardHeader>
              <CardTitle className="text-lg font-semibold text-foreground">
                YOLOv3 Training Loss
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-2">
              <Line data={chart2} options={{
                responsive: true,
                plugins: {
                  legend: {
                    labels: { color: colors.foreground }
                  },
                },
                scales: {
                  x: {
                    ticks: { color: colors.mutedFg },
                    grid: { color: isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.06)' }
                  },
                  y: {
                    ticks: { color: colors.mutedFg },
                    grid: { color: isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.06)' }
                  }
                }
              }} />
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
