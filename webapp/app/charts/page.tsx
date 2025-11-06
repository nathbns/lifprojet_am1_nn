"use client";
import { useEffect, useState } from "react";
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
              borderColor: "rgba(37, 99, 235, 1)",
              backgroundColor: "rgba(37, 99, 235, 0.15)",
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
              borderColor: "rgba(16, 185, 129, 1)", 
              backgroundColor: "rgba(16, 185, 129, 0.15)",
              tension: 0.25,
            },
            {
              label: "YOLO Pascal Voc",
              data: lossData1,
              borderColor: "rgba(235, 93, 37, 1)",
              backgroundColor: "rgba(37, 99, 235, 0.15)",
              tension: 0.25,
            },
          ],
        });
      } catch (err) {
        console.error("Erreur lors du chargement des datasets :", err);
      }
    }

    fetchData();
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-white text-gray-800 p-10">
      <div className="flex items-center justify-between mb-12 max-w-6xl mx-auto">
        <h1 className="text-4xl font-bold text-gray-900 tracking-tight">
          Analyse des modèles YOLOv3
        </h1>
        <span className="text-sm text-gray-500 italic">
          Comparaison : Pascal VOC vs Chess
        </span>
      </div>

      {!chart1 || !chart2 ? (
        <p className="text-center text-gray-500">Chargement des données...</p>
      ) : (
        <div className="grid md:grid-cols-2 gap-8 max-w-6xl mx-auto">
          {/* Graphique 1 : Pascal VOC */}
          <Card className="bg-white border border-gray-200 shadow-md hover:shadow-lg transition-all rounded-2xl">
            <CardHeader>
              <CardTitle className="text-lg font-semibold text-gray-700">
                YOLOv3 – Pascal VOC
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-2">
              <Line data={chart1} />
            </CardContent>
          </Card>

          {/* Graphique 2 : Chess */}
          <Card className="bg-white border border-gray-200 shadow-md hover:shadow-lg transition-all rounded-2xl">
            <CardHeader>
              <CardTitle className="text-lg font-semibold text-gray-700">
                YOLOv3 Training Loss
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-2">
              <Line data={chart2} />
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
