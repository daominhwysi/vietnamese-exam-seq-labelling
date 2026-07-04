import React, { useState, useEffect, useRef } from "react";
import { Undo2, RefreshCw, Save, ArrowLeft } from "lucide-react";
import { type Span } from "../types";
import {
  mapDomOffsetToRawOffset,
  parseSpansToQuestions,
  TAG_COLOR_CLASSES
} from "../utils/textUtils";
import { Button } from "./ui/button";

interface ExamViewProps {
  examId: string;
  navigate: (h: string) => void;
}

export const ExamView: React.FC<ExamViewProps> = ({ examId, navigate }) => {
  const [exam, setExam] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [viewMode, setViewMode] = useState<"standard" | "tag">("standard");
  const [annotated, setAnnotated] = useState<boolean>(false);

  // Annotator settings
  const [activeBrush, setActiveBrush] = useState<string>("stem");
  const [currentSpans, setCurrentSpans] = useState<Span[]>([]);
  const [originalSpans, setOriginalSpans] = useState<Span[]>([]);
  const [historyStack, setHistoryStack] = useState<Span[][]>([]);
  const [selectionStatus, setSelectionStatus] = useState<string>("Chưa bôi đen");

  const annotatorRef = useRef<HTMLDivElement>(null);

  const toggleAnnotatedStatus = async () => {
    const nextVal = !annotated;
    try {
      const response = await fetch(`/api/exam/${examId}/annotated`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ annotated: nextVal })
      });
      if (!response.ok) throw new Error("Không thể cập nhật trạng thái.");
      setAnnotated(nextVal);
    } catch (err: any) {
      alert("Lỗi: " + err.message);
    }
  };

  useEffect(() => {
    setLoading(true);
    fetch(`/api/exams/${examId}`)
      .then(res => {
        if (!res.ok) throw new Error("Không tìm thấy đề thi");
        return res.json();
      })
      .then(data => {
        setExam(data);
        setAnnotated(data.annotated || false);
        if (data.is_real) {
          const q = Object.values(data.sections)[0] as any[];
          const spans = q[0]?.spans || [];
          setCurrentSpans([...spans].sort((a, b) => a.start - b.start));
          setOriginalSpans(JSON.parse(JSON.stringify(spans)));
          setHistoryStack([]);
          setViewMode("tag");
        } else {
          setViewMode("standard");
        }
        setLoading(false);
      })
      .catch(err => {
        alert(err.message);
        navigate("#/");
      });
  }, [examId]);

  // MathJax integration helper
  useEffect(() => {
    if (!loading && viewMode !== "tag" && (window as any).MathJax) {
      (window as any).MathJax.typesetPromise();
    }
  }, [loading, viewMode]);

  // Hotkey interface configuration
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (viewMode !== "tag" || !exam?.is_real) return;
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
        undo();
        e.preventDefault();
        return;
      }

      const keys: Record<string, string> = {
        "1": "question_label",
        "2": "stem",
        "3": "option_label",
        "4": "option_text",
        "5": "context",
        "6": "section",
        "7": "explanation",
        "0": "clear"
      };

      if (keys[e.key]) {
        setActiveBrush(keys[e.key]);
        e.preventDefault();
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [viewMode, exam, currentSpans, historyStack]);

  const saveStateToHistory = () => {
    setHistoryStack(prev => {
      const copy = [...prev];
      if (copy.length >= 50) copy.shift();
      copy.push(JSON.parse(JSON.stringify(currentSpans)));
      return copy;
    });
  };

  const undo = () => {
    if (historyStack.length === 0) return;
    setHistoryStack(prev => {
      const copy = [...prev];
      const previousState = copy.pop();
      if (previousState) {
        setCurrentSpans(previousState);
      }
      return copy;
    });
  };

  const resetAnnotations = () => {
    if (!confirm("Khôi phục về trạng thái nhãn ban đầu?")) return;
    saveStateToHistory();
    setCurrentSpans(JSON.parse(JSON.stringify(originalSpans)));
  };

  const handleTextSelection = () => {
    const container = annotatorRef.current;
    if (!container || !exam) return;

    const selection = window.getSelection();
    if (!selection || selection.rangeCount === 0) return;
    const range = selection.getRangeAt(0);

    if (!container.contains(range.startContainer) || !container.contains(range.endContainer)) {
      return;
    }

    const getSpanWrapper = (node: Node): HTMLElement | null => {
      let curr: Node | null = node;
      while (curr && curr !== container) {
        if (curr.nodeType === Node.ELEMENT_NODE && (curr as HTMLElement).hasAttribute('data-start')) {
          return curr as HTMLElement;
        }
        curr = curr.parentNode;
      }
      return null;
    };

    const startSpan = getSpanWrapper(range.startContainer);
    const endSpan = getSpanWrapper(range.endContainer);

    if (!startSpan || !endSpan) return;

    const baseStart = parseInt(startSpan.getAttribute('data-start') || "0");
    const baseEnd = parseInt(endSpan.getAttribute('data-start') || "0");
    const limitStart = parseInt(startSpan.getAttribute('data-end') || "0");
    const limitEnd = parseInt(endSpan.getAttribute('data-end') || "0");

    const domTextStart = range.startContainer.nodeValue || "";
    const domOffsetStart = range.startOffset;
    const rawSubStart = (exam.raw_text || "").substring(baseStart, limitStart);
    const startOffset = mapDomOffsetToRawOffset(rawSubStart, domTextStart, domOffsetStart);
    const start = baseStart + startOffset;

    const domTextEnd = range.endContainer.nodeValue || "";
    const domOffsetEnd = range.endOffset;
    const rawSubEnd = (exam.raw_text || "").substring(baseEnd, limitEnd);
    const endOffset = mapDomOffsetToRawOffset(rawSubEnd, domTextEnd, domOffsetEnd);
    const end = baseEnd + endOffset;

    if (start < end) {
      setSelectionStatus(`Chọn: [${start} - ${end}]`);
      if (activeBrush === "clear") {
        clearSpanRange(start, end);
      } else {
        addSpan(start, end, activeBrush);
      }
      selection.removeAllRanges();
      setTimeout(() => setSelectionStatus("Chưa bôi đen"), 1000);
    }
  };

  const addSpan = (start: number, end: number, label: string) => {
    saveStateToHistory();
    const newSpans: Span[] = [];
    const newSpan: Span = {
      start,
      end,
      label,
      text: (exam.raw_text || "").substring(start, end)
    };

    currentSpans.forEach(span => {
      if (span.start >= start && span.end <= end) return;
      if (span.start < start && span.end > end) {
        newSpans.push({ start: span.start, end: start, label: span.label });
        newSpans.push({ start: end, end: span.end, label: span.label });
        return;
      }
      if (span.start < start && span.end > start && span.end <= end) {
        newSpans.push({ start: span.start, end: start, label: span.label });
        return;
      }
      if (span.start >= start && span.start < end && span.end > end) {
        newSpans.push({ start: end, end: span.end, label: span.label });
        return;
      }
      newSpans.push(span);
    });

    newSpans.push(newSpan);
    setCurrentSpans(newSpans.sort((a, b) => a.start - b.start));
  };

  const clearSpanRange = (start: number, end: number) => {
    saveStateToHistory();
    const newSpans: Span[] = [];

    currentSpans.forEach(span => {
      if (span.start >= start && span.end <= end) return;
      if (span.start < start && span.end > end) {
        newSpans.push({ start: span.start, end: start, label: span.label });
        newSpans.push({ start: end, end: span.end, label: span.label });
        return;
      }
      if (span.start < start && span.end > start && span.end <= end) {
        newSpans.push({ start: span.start, end: start, label: span.label });
        return;
      }
      if (span.start >= start && span.start < end && span.end > end) {
        newSpans.push({ start: end, end: span.end, label: span.label });
        return;
      }
      newSpans.push(span);
    });

    setCurrentSpans(newSpans.sort((a, b) => a.start - b.start));
  };

  const clickSpan = (event: React.MouseEvent, idx: number) => {
    event.stopPropagation();
    const span = currentSpans[idx];
    if (!span) return;

    if (confirm(`Bạn muốn xóa nhãn [${span.label}] cho đoạn text: "${(span.text || exam.raw_text || "").substring(span.start, span.start + 30)}..."?`)) {
      saveStateToHistory();
      setCurrentSpans(prev => {
        const copy = [...prev];
        copy.splice(idx, 1);
        return copy;
      });
    }
  };

  const saveNewAnnotations = async () => {
    if (!confirm("Lưu các thay đổi trên cấu trúc nhãn gán?")) return;

    try {
      const response = await fetch(`/api/exam/${examId}/spans`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ spans: currentSpans })
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Không thể lưu nhãn.");
      }

      alert("Lưu thành công.");
      setOriginalSpans(JSON.parse(JSON.stringify(currentSpans)));
    } catch (err: any) {
      alert("Lỗi khi lưu: " + err.message);
    }
  };

  const renderAnnotatorHtml = () => {
    const rawText = exam.raw_text || "";
    let lastIdx = 0;
    const elements: React.ReactNode[] = [];

    currentSpans.forEach((span, idx) => {
      if (span.start > lastIdx) {
        const textSlice = rawText.substring(lastIdx, span.start);
        elements.push(
          <span key={`text-${lastIdx}`} data-text-node="true" data-start={lastIdx} data-end={span.start}>
            {textSlice}
          </span>
        );
      }

      const spanText = rawText.substring(span.start, span.end);
      const colorClass = TAG_COLOR_CLASSES[span.label] || "bg-zinc-100";

      elements.push(
        <span
          key={`span-${idx}`}
          onClick={e => clickSpan(e, idx)}
          className={`tag-token ${colorClass} px-0.5 rounded cursor-pointer hover:opacity-80 border-b font-serif relative`}
          title="Click to remove tag"
          data-start={span.start}
          data-end={span.end}
        >
          {spanText}
        </span>
      );
      lastIdx = span.end;
    });

    if (lastIdx < rawText.length) {
      const textSlice = rawText.substring(lastIdx);
      elements.push(
        <span key={`text-${lastIdx}`} data-text-node="true" data-start={lastIdx} data-end={rawText.length}>
          {textSlice}
        </span>
      );
    }

    return elements;
  };

  if (loading) {
    return <div className="py-20 text-center text-zinc-500 text-xs font-mono">Đang tải tài liệu...</div>;
  }

  return (
    <div className="flex-grow flex flex-col max-w-7xl mx-auto w-full px-6 py-6 gap-5">
      {/* Header bar */}
      <div className="flex items-center justify-between pb-3 border-b border-zinc-200">
        <div className="flex items-center gap-2.5">
          <Button variant="outline" size="sm" onClick={() => navigate("#/")} className="gap-1.5">
            <ArrowLeft size={13} /> Quay về
          </Button>
          <span className="text-zinc-200">|</span>
          <div className="inline-flex rounded-lg bg-zinc-100 p-0.5 border border-zinc-200 text-xs">
            <button
              onClick={() => setViewMode("standard")}
              className={`px-3 py-0.5 rounded-md font-medium transition ${viewMode === "standard" ? "bg-white text-zinc-950 shadow-sm" : "text-zinc-550"}`}
            >
              Học sinh
            </button>
            <button
              onClick={() => setViewMode("tag")}
              className={`px-3 py-0.5 rounded-md font-medium transition ${viewMode === "tag" ? "bg-white text-zinc-950 shadow-sm" : "text-zinc-550"}`}
            >
              Nhãn Spans
            </button>
          </div>
        </div>
        <div className="flex items-center gap-2 font-mono text-xs text-zinc-400">
          <span>{exam?.exam_id}</span>
          <span className="bg-zinc-100 text-zinc-800 px-1.5 py-0.2 rounded font-sans">{exam?.subject_display || exam?.subject}</span>
        </div>
      </div>

      {/* Main Container */}
      {viewMode === "standard" ? (
        <div className="border border-zinc-200 rounded-xl p-8 bg-white font-serif whitespace-pre-wrap shadow-sm">
          <div className="border-b border-zinc-200 pb-5 mb-6 text-xs font-sans text-zinc-750 flex justify-between">
            <div>
              <p className="uppercase font-bold text-zinc-400">ĐỀ THI GỐC OCR</p>
              <p className="text-sm font-bold text-zinc-900 mt-1">{(exam?.subject_display || "").toUpperCase()}</p>
            </div>
            <div className="text-right">
              <p className="font-semibold">Khối lớp {exam?.grade}</p>
              <p className="text-zinc-400">Thời gian làm bài: 50 phút</p>
            </div>
          </div>

          <div className="space-y-6">
            {exam.is_real ? (
              (() => {
                const parsedQs = parseSpansToQuestions(currentSpans, exam.raw_text || "");
                if (parsedQs.length > 0) {
                  return parsedQs.map((pq, idx) => (
                    <div key={idx} className="p-4 rounded-lg border border-zinc-200 bg-zinc-50/20 space-y-3 font-serif">
                      {pq.context && (
                        <div className="p-3 bg-zinc-100 rounded border border-zinc-200 text-zinc-600 text-xs">
                          {pq.context}
                        </div>
                      )}
                      <div className="text-sm">
                        <span className="font-sans font-bold text-zinc-950">Câu {idx + 1}: </span>
                        {pq.stem}
                      </div>
                      {pq.options?.length > 0 && (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-2 pl-4 text-xs">
                          {pq.options.map((opt, optIdx) => (
                            <div key={optIdx} className="flex gap-2">
                              <span className="font-sans font-bold text-zinc-400">{String.fromCharCode(65 + optIdx)}.</span>
                              <span>{opt}</span>
                            </div>
                          ))}
                        </div>
                      )}
                      {pq.explanation && (
                        <div className="text-xs text-emerald-800 bg-emerald-50 border border-emerald-100 px-2 py-1 rounded w-fit">
                          {pq.explanation}
                        </div>
                      )}
                    </div>
                  ));
                }
                return <div className="p-4 bg-zinc-50 rounded border text-sm">{exam.raw_text}</div>;
              })()
            ) : (
              Object.entries(exam.sections).map(([sectionTitle, questions]: [any, any]) => (
                <div key={sectionTitle} className="space-y-3">
                  <h2 className="text-xs font-bold text-zinc-400 uppercase font-sans">{sectionTitle}</h2>
                  {questions.map((q: any) => (
                    <div key={q.start_number} className="p-4 rounded-lg border border-zinc-250 bg-zinc-50/20 space-y-2">
                      <div className="text-sm">Câu {q.start_number}: {q.stem}</div>
                      {q.options?.length > 0 && (
                        <div className="grid grid-cols-2 gap-2 pl-4 text-xs">
                          {q.options.map((o: string, idx: number) => (
                            <div key={idx}>{String.fromCharCode(65 + idx)}. {o}</div>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ))
            )}
          </div>
        </div>
      ) : (
        /* Dynamic Tagging Suite */
        <div className="flex-grow flex flex-col lg:flex-row gap-5">
          <div className="lg:w-1/4 space-y-4 flex flex-col">
            <div className="p-4 border border-zinc-200 rounded-xl bg-white shadow-sm space-y-2.5">
              <span className="block text-[10px] font-bold text-zinc-400 uppercase tracking-wider font-sans">Bảng mã thực thể</span>
              <div className="space-y-1">
                {Object.keys(TAG_COLOR_CLASSES).map((label, index) => {
                  const isActive = activeBrush === label;
                  return (
                    <button
                      key={label}
                      onClick={() => setActiveBrush(label)}
                      className={`w-full text-left px-3 py-1 rounded-md text-[11px] font-medium border flex items-center justify-between transition ${isActive ? "bg-zinc-100 border-zinc-300 text-zinc-950" : "bg-white border-transparent text-zinc-650 hover:bg-zinc-50"}`}
                    >
                      <span>{label}</span>
                      <span className="font-mono text-[9px] bg-zinc-200/50 px-1 rounded">{index + 1}</span>
                    </button>
                  );
                })}
                <button
                  onClick={() => setActiveBrush("clear")}
                  className={`w-full text-left px-3 py-1 rounded-md text-[11px] font-medium border flex items-center justify-between transition text-red-650 ${activeBrush === "clear" ? "bg-red-50 border-red-200" : "border-transparent hover:bg-red-50/50"}`}
                >
                  <span>🧹 Xóa Nhãn</span>
                  <span className="font-mono text-[9px] bg-zinc-200/50 px-1 rounded">0</span>
                </button>
              </div>
            </div>

            <div className="p-4 border border-zinc-200 rounded-xl bg-white shadow-sm space-y-2">
              <span className="block text-[10px] font-bold text-zinc-400 uppercase tracking-wider font-sans">Thao tác</span>
              <Button onClick={saveNewAnnotations} className="w-full justify-center gap-1.5">
                <Save size={13} /> Lưu Nhãn
              </Button>
              <Button
                variant="outline"
                className={`w-full justify-center ${annotated ? "bg-emerald-50 text-emerald-800 border-emerald-200" : "bg-amber-50 text-amber-800 border-amber-200"}`}
                onClick={toggleAnnotatedStatus}
              >
                {annotated ? "✓ Đã đánh dấu gán" : "✗ Chưa đánh dấu gán"}
              </Button>
              <div className="flex gap-2 pt-1">
                <Button variant="outline" size="sm" onClick={undo} disabled={historyStack.length === 0} className="flex-1 gap-1">
                  <Undo2 size={12} /> Undo
                </Button>
                <Button variant="outline" size="sm" onClick={resetAnnotations} className="flex-1 gap-1 text-red-650">
                  <RefreshCw size={12} /> Reset
                </Button>
              </div>
            </div>
          </div>

          <div className="lg:w-3/4 flex flex-col space-y-2">
            <div className="flex justify-between items-center bg-white px-4 py-2 rounded-xl border border-zinc-200 text-xs text-zinc-500 shadow-sm">
              <span>Rê chuột và bôi đen phần văn bản OCR để gán nhãn thực thể</span>
              <span className="font-mono bg-zinc-100 border px-1.5 py-0.2 rounded text-[10px] text-zinc-800">{selectionStatus}</span>
            </div>

            <div
              ref={annotatorRef}
              onMouseUp={handleTextSelection}
              className="p-8 rounded-xl border border-zinc-200 bg-white font-serif leading-relaxed text-sm select-text whitespace-pre-wrap text-zinc-950 min-h-[500px] max-h-[600px] overflow-y-auto shadow-inner"
              style={{ lineHeight: "2.3" }}
            >
              {renderAnnotatorHtml()}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
