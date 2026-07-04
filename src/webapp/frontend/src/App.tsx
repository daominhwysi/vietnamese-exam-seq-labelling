import { useState, useEffect } from "react";
import { Header } from "./components/Header";
import { DashboardView } from "./components/DashboardView";
import { ExamView } from "@/components/ExamView";
import { DatasetView } from "./components/DatasetView";
import { InferencePlayground } from "./components/InferencePlayground";

export default function App() {
  const [route, setRoute] = useState<string>("/");
  const [selectedExamId, setSelectedExamId] = useState<string | null>(null);
  const [selectedSplit, setSelectedSplit] = useState<string>("train");

  useEffect(() => {
    const handleHashChange = () => {
      const hash = window.location.hash || "#/";
      if (hash.startsWith("#/exam/")) {
        setSelectedExamId(hash.replace("#/exam/", ""));
        setRoute("/exam");
      } else if (hash.startsWith("#/dataset/")) {
        setSelectedSplit(hash.replace("#/dataset/", ""));
        setRoute("/dataset-viewer");
      } else if (hash === "#/dataset") {
        setRoute("/dataset");
      } else if (hash === "#/inference") {
        setRoute("/inference");
      } else {
        setRoute("/");
      }
    };

    window.addEventListener("hashchange", handleHashChange);
    handleHashChange();
    return () => window.removeEventListener("hashchange", handleHashChange);
  }, []);

  const navigate = (hash: string) => {
    window.location.hash = hash;
  };

  return (
    <div className="min-h-screen bg-zinc-50/50 flex flex-col font-sans text-zinc-950">
      <Header route={route} navigate={navigate} />

      <main className="flex-grow flex flex-col">
        {route === "/" && <DashboardView navigate={navigate} />}
        {route === "/exam" && selectedExamId && <ExamView examId={selectedExamId} navigate={navigate} />}
        {route === "/dataset" && <DatasetView route="/dataset" navigate={navigate} />}
        {route === "/dataset-viewer" && <DatasetView route="/dataset-viewer" selectedSplit={selectedSplit} navigate={navigate} />}
        {route === "/inference" && <InferencePlayground />}
      </main>
    </div>
  );
}
