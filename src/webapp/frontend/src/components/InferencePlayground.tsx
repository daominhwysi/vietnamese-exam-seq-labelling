import React, { useState, useEffect } from "react";
import { CheckCircle, Clock, AlertCircle, BookOpen } from "lucide-react";
import { type ModelStatus } from "../types";
import { TAG_COLOR_CLASSES } from "../utils/textUtils";
import { Button } from "./ui/button";

export const InferencePlayground: React.FC = () => {
  const [status, setStatus] = useState<ModelStatus>({
    model_loaded: false,
    model_path: "",
    base_model_name: "",
    device: "",
    loading: false,
    error: null
  });

  const [modelPathInput, setModelPathInput] = useState<string>("output/lora_adapter");
  const [baseModelInput, setBaseModelInput] = useState<string>("aisingapore/SEA-LION-ModernBERT-300M");
  const [deviceChoice, setDeviceChoice] = useState<string>("auto");

  const [rawTextInput, setRawTextInput] = useState<string>("Dán văn bản OCR vào đây để chạy suy luận...");
  const [inferenceResult, setInferenceResult] = useState<any>(null);
  const [runningInference, setRunningInference] = useState<boolean>(false);

  const [presets, setPresets] = useState<string[]>([]);
  const [selectedPreset, setSelectedPreset] = useState<string>("");

  useEffect(() => {
    checkModelStatus();
    loadPresets();
    const timer = setInterval(checkModelStatus, 5000);
    return () => clearInterval(timer);
  }, []);

  const checkModelStatus = () => {
    fetch("/api/model-status")
      .then(res => res.json())
      .then(data => setStatus(data))
      .catch(err => console.error("Error reading model status:", err));
  };

  const loadPresets = () => {
    fetch("/api/presets")
      .then(res => res.json())
      .then(data => setPresets(data))
      .catch(err => console.error("Error loading presets:", err));
  };

  const handleLoadPreset = (filename: string) => {
    setSelectedPreset(filename);
    if (!filename) return;
    fetch(`/api/presets/${filename}`)
      .then(res => res.json())
      .then(data => setRawTextInput(data.content))
      .catch(err => alert("Lỗi khi nạp mẫu thử: " + err.message));
  };

  const handleLoadModelSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setStatus(prev => ({ ...prev, loading: true, error: null }));
    fetch("/api/load-model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model_path: modelPathInput,
        base_model_name: baseModelInput,
        device_choice: deviceChoice
      })
    })
      .then(res => res.json())
      .then(() => checkModelStatus())
      .catch(err => alert("Error: " + err.message));
  };

  const runInference = (e: React.FormEvent) => {
    e.preventDefault();
    if (!rawTextInput.trim()) return;
    setRunningInference(true);
    setInferenceResult(null);

    fetch("/api/run-inference", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ raw_text: rawTextInput })
    })
      .then(res => {
        if (!res.ok) throw new Error("Yêu cầu suy luận thất bại.");
        return res.json();
      })
      .then(data => {
        setInferenceResult(data);
        setRunningInference(false);
      })
      .catch(err => {
        alert(err.message);
        setRunningInference(false);
      });
  };

  const renderInferenceHighlight = () => {
    if (!inferenceResult) return null;
    const rawText = inferenceResult.raw_text || "";
    const spans = inferenceResult.spans || [];
    const sortedSpans = [...spans].sort((a, b) => a.start - b.start);
    let htmlStr = "";
    let lastIdx = 0;

    sortedSpans.forEach(span => {
      if (span.start > lastIdx) {
        htmlStr += rawText.substring(lastIdx, span.start);
      }
      const spanText = rawText.substring(span.start, span.end);
      const colorClass = TAG_COLOR_CLASSES[span.label] || "bg-zinc-100";
      htmlStr += `<span class="tag-token ${colorClass} px-0.5 rounded border-b text-xs inline-block font-serif" title="${span.label}">${spanText}</span>`;
      lastIdx = span.end;
    });

    if (lastIdx < rawText.length) {
      htmlStr += rawText.substring(lastIdx);
    }

    return (
      <div
        className="p-6 border border-zinc-250 rounded-xl bg-white font-serif leading-relaxed text-sm whitespace-pre-wrap text-zinc-950 shadow-sm"
        dangerouslySetInnerHTML={{ __html: htmlStr }}
      />
    );
  };

  return (
    <div className="max-w-7xl mx-auto w-full px-6 py-6 space-y-6">
      <div className="space-y-1">
        <h2 className="text-xl font-bold tracking-tight text-zinc-900">Inference Playground</h2>
        <p className="text-xs text-zinc-550">Chạy thử nghiệm suy luận mô hình sequence labeling trên văn bản đề thi.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-5">
        <div className="lg:col-span-1 space-y-4">
          <form onSubmit={handleLoadModelSubmit} className="p-4 border border-zinc-200 bg-white rounded-xl shadow-sm space-y-4 text-xs">
            <span className="block text-[10px] font-bold text-zinc-400 uppercase tracking-wider">Cấu hình mô hình</span>
            <div className="space-y-2">
              <div className="space-y-1">
                <label className="font-semibold text-zinc-700">LoRA Adapter Path:</label>
                <input
                  type="text"
                  value={modelPathInput}
                  onChange={e => setModelPathInput(e.target.value)}
                  className="w-full h-8 px-2 border rounded-md focus-visible:outline-zinc-800"
                />
              </div>
              <div className="space-y-1">
                <label className="font-semibold text-zinc-700">Base Model Name:</label>
                <input
                  type="text"
                  value={baseModelInput}
                  onChange={e => setBaseModelInput(e.target.value)}
                  className="w-full h-8 px-2 border rounded-md focus-visible:outline-zinc-800"
                />
              </div>
              <div className="space-y-1">
                <label className="font-semibold text-zinc-700">Thiết bị (Device):</label>
                <select
                  value={deviceChoice}
                  onChange={e => setDeviceChoice(e.target.value)}
                  className="w-full h-8 px-2 border rounded-md"
                >
                  <option value="auto">Auto (CUDA / CPU)</option>
                  <option value="cuda">CUDA</option>
                  <option value="cpu">CPU</option>
                </select>
              </div>
            </div>
            <Button type="submit" disabled={status.loading} className="w-full justify-center">
              {status.loading ? "Đang tải..." : "Tải mô hình"}
            </Button>
          </form>

          {/* Model Status block */}
          <div className="p-4 border border-zinc-200 rounded-xl bg-white shadow-sm space-y-2 text-xs">
            <span className="block text-[9.5px] font-bold text-zinc-400 uppercase tracking-wider">Trạng thái</span>
            {status.model_loaded ? (
              <div className="bg-emerald-50 text-emerald-800 p-3 rounded-lg border border-emerald-200/50 space-y-1">
                <h4 className="font-bold flex items-center gap-1.5 text-xs">
                  <CheckCircle size={14} className="text-emerald-600" /> Hoạt động
                </h4>
                <p className="text-[10px] text-emerald-700">Đã load adapter lên thiết bị: {status.device}</p>
              </div>
            ) : status.loading ? (
              <div className="bg-amber-50 text-amber-850 p-3 rounded-lg border border-amber-200/50 space-y-1">
                <h4 className="font-bold flex items-center gap-1.5 text-xs">
                  <Clock size={14} className="animate-spin text-amber-600" /> Đang tải...
                </h4>
                <p className="text-[10px] text-amber-700">Mô hình đang được nạp vào bộ nhớ...</p>
              </div>
            ) : (
              <div className="bg-zinc-50 text-zinc-500 p-3 rounded-lg border border-zinc-200 space-y-1">
                <h4 className="font-bold flex items-center gap-1.5 text-xs">
                  <AlertCircle size={14} className="text-zinc-400" /> Chưa nạp
                </h4>
                <p className="text-[10px] text-zinc-400">Vui lòng nạp mô hình để khởi tạo suy luận.</p>
              </div>
            )}
          </div>
        </div>

        {/* Workspace Canvas */}
        <div className="lg:col-span-3 space-y-4">
          <form onSubmit={runInference} className="p-4 border border-zinc-200 rounded-xl bg-white shadow-sm space-y-3 flex flex-col">
            <div className="flex justify-between items-center">
              <span className="block text-[10px] font-bold text-zinc-400 uppercase tracking-wider">Văn bản kiểm tra</span>
              <div className="flex items-center gap-1.5 text-xs">
                <BookOpen size={13} className="text-zinc-450" />
                <select
                  value={selectedPreset}
                  onChange={e => handleLoadPreset(e.target.value)}
                  className="flex h-7 rounded-md border border-zinc-250 bg-white px-2 text-xs"
                >
                  <option value="">Chọn mẫu văn bản</option>
                  {presets.map(p => (
                    <option key={p} value={p}>{p}</option>
                  ))}
                </select>
              </div>
            </div>

            <textarea
              value={rawTextInput}
              onChange={e => setRawTextInput(e.target.value)}
              placeholder="Nhập hoặc dán nội dung OCR..."
              rows={12}
              className="w-full p-4 border border-zinc-200 bg-zinc-50/20 rounded-lg text-sm font-serif focus:outline-none focus:bg-white transition"
            />

            <Button
              type="submit"
              disabled={runningInference || !status.model_loaded || !rawTextInput.trim()}
              className="self-end gap-1.5"
            >
              {runningInference ? "Đang chạy..." : "Trích xuất nhãn"}
            </Button>
          </form>

          {inferenceResult && (
            <div className="space-y-3">
              <span className="block text-[10px] font-bold text-zinc-400 uppercase tracking-wider">Bản đồ thực thể</span>
              {renderInferenceHighlight()}
              <div className="bg-zinc-50 border border-zinc-200 rounded-xl p-4 space-y-1.5 font-mono text-xs">
                <span className="block text-[9px] font-bold text-zinc-400 uppercase">Định dạng XML Markup</span>
                <pre className="overflow-x-auto p-3 bg-white border border-zinc-200 rounded-lg text-zinc-800 whitespace-pre-wrap max-h-[250px]">
                  {inferenceResult.xml_content}
                </pre>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
