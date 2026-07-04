import React from "react";
import { FileText, Database, Play } from "lucide-react";
import { Button } from "./ui/button";

interface HeaderProps {
  route: string;
  navigate: (hash: string) => void;
}

export const Header: React.FC<HeaderProps> = ({ route, navigate }) => {
  return (
    <header className="sticky top-0 z-40 bg-white border-b border-zinc-200/80 px-6 py-3 flex items-center justify-between shadow-sm">
      <div className="flex items-center space-x-2.5 cursor-pointer" onClick={() => navigate("#/")}>
        <div className="h-7 w-7 rounded bg-zinc-900 flex items-center justify-center text-zinc-50 font-semibold text-sm">A</div>
        <div>
          <h1 className="text-xs font-bold text-zinc-900 leading-tight tracking-tight">Sequence Labeling</h1>
          <p className="text-[9px] text-zinc-550 font-semibold uppercase tracking-wider">Antigravity Workspace</p>
        </div>
      </div>

      <nav className="flex space-x-1">
        <Button
          variant={route === "/" || route === "/exam" ? "secondary" : "ghost"}
          size="sm"
          onClick={() => navigate("#/")}
          className="flex items-center gap-1.5"
        >
          <FileText size={13} />
          Đề Thi
        </Button>
        <Button
          variant={route.startsWith("/dataset") ? "secondary" : "ghost"}
          size="sm"
          onClick={() => navigate("#/dataset")}
          className="flex items-center gap-1.5"
        >
          <Database size={13} />
          Tập Dữ Liệu
        </Button>
        <Button
          variant={route === "/inference" ? "secondary" : "ghost"}
          size="sm"
          onClick={() => navigate("#/inference")}
          className="flex items-center gap-1.5"
        >
          <Play size={13} />
          Inference
        </Button>
      </nav>
    </header>
  );
};
