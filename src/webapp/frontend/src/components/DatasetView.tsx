import React, { useState, useEffect } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { SUBJECT_DISPLAY, TAG_COLOR_CLASSES } from "../utils/textUtils";
import { Button } from "./ui/button";

interface DatasetViewProps {
  route: string;
  navigate: (h: string) => void;
  selectedSplit?: string;
}

export const DatasetView: React.FC<DatasetViewProps> = ({ route, navigate, selectedSplit = "train" }) => {
  if (route === "/dataset") {
    return <DatasetDashboard navigate={navigate} />;
  }
  return <DatasetSplitViewer split={selectedSplit} navigate={navigate} />;
};

function DatasetDashboard({ navigate }: { navigate: (h: string) => void }) {
  const [stats, setStats] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    fetch("/api/dataset/stats")
      .then(res => res.json())
      .then(data => {
        setStats(data);
        setLoading(false);
      })
      .catch(err => console.error(err));
  }, []);

  if (loading) return <div className="py-20 text-center text-zinc-550 text-xs font-mono">Đang nạp phân bố dữ liệu...</div>;

  return (
    <div className="max-w-7xl mx-auto w-full px-6 py-6 space-y-6">
      <div className="space-y-1">
        <h2 className="text-xl font-bold tracking-tight text-zinc-900">Thông tin Dataset</h2>
        <p className="text-xs text-zinc-550">Xem phân bố mẫu huấn luyện và thống kê chi tiết các tập cắt dữ liệu.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {["train", "val", "test"].map(split => {
          const splitStats = stats[split] || {};
          return (
            <div key={split} className="bg-white border border-zinc-200 rounded-xl p-5 flex flex-col justify-between space-y-4 hover:border-zinc-350 shadow-sm transition">
              <div className="space-y-3">
                <div className="flex justify-between items-center border-b pb-2">
                  <span className="uppercase text-xs font-bold text-zinc-900 font-mono">{split} Split</span>
                  <span className="bg-zinc-100 text-[10px] font-mono border px-1.5 rounded">{splitStats.file_size_mb || 0} MB</span>
                </div>
                <div>
                  <span className="text-[9px] font-bold text-zinc-400 uppercase">Mẫu câu hỏi</span>
                  <p className="text-xl font-bold text-zinc-900">{splitStats.total_samples || 0}</p>
                </div>
                {splitStats.subjects && (
                  <div className="space-y-1 max-h-[150px] overflow-y-auto text-xs pr-1">
                    {Object.entries(splitStats.subjects).map(([k, v]: [string, any]) => (
                      <div key={k} className="flex justify-between text-zinc-550 border-b border-zinc-50 pb-0.5">
                        <span className="truncate">{SUBJECT_DISPLAY[k] || k}</span>
                        <span className="font-semibold text-zinc-850">{v}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
              <Button onClick={() => navigate(`#/dataset/${split}`)} className="w-full justify-center">
                Xem chi tiết
              </Button>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function DatasetSplitViewer({ split, navigate }: { split: string; navigate: (h: string) => void }) {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [page, setPage] = useState<number>(1);
  const [subject, setSubject] = useState<string>("");
  const [grade, setGrade] = useState<string>("");

  useEffect(() => {
    setLoading(true);
    const query = new URLSearchParams({ page: page.toString(), page_size: "15" });
    if (subject) query.append("subject", subject);
    if (grade) query.append("grade", grade);

    fetch(`/api/dataset/${split}?${query.toString()}`)
      .then(res => res.json())
      .then(resData => {
        setData(resData);
        setLoading(false);
      })
      .catch(err => console.error(err));
  }, [split, page, subject, grade]);

  return (
    <div className="max-w-7xl mx-auto w-full px-6 py-6 space-y-4">
      <div className="pb-3 border-b border-zinc-200 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={() => navigate("#/dataset")}>
            ← Bảng split
          </Button>
          <span className="text-zinc-200">|</span>
          <span className="uppercase text-xs font-bold text-zinc-900 font-mono">Tập dữ liệu {split}</span>
        </div>

        <div className="flex items-center gap-2">
          <select
            value={subject}
            onChange={e => { setSubject(e.target.value); setPage(1); }}
            className="flex h-8 rounded-md border border-zinc-200 bg-white px-2 text-xs"
          >
            <option value="">Môn học</option>
            {data?.available_subjects?.map((s: string) => (
              <option key={s} value={s}>{SUBJECT_DISPLAY[s] || s}</option>
            ))}
          </select>
          <select
            value={grade}
            onChange={e => { setGrade(e.target.value); setPage(1); }}
            className="flex h-8 rounded-md border border-zinc-200 bg-white px-2 text-xs"
          >
            <option value="">Lớp</option>
            {data?.available_grades?.map((g: number) => (
              <option key={g} value={g}>Lớp {g}</option>
            ))}
          </select>
        </div>
      </div>

      {loading ? (
        <div className="py-20 text-center text-zinc-550 text-xs font-mono">Đang nạp dữ liệu...</div>
      ) : (
        <div className="space-y-4">
          <div className="border border-zinc-200 rounded-xl overflow-hidden divide-y bg-white shadow-sm">
            {data?.samples?.map((sample: any, idx: number) => (
              <div key={idx} className="p-4 flex flex-col md:flex-row gap-4 hover:bg-zinc-50/20 transition">
                <div className="md:w-1/5 space-y-1.5 text-xs flex-shrink-0">
                  <div className="font-bold text-zinc-900">{SUBJECT_DISPLAY[sample.metadata?.subject] || sample.metadata?.subject}</div>
                  <div className="flex gap-1.5">
                    <span className="bg-zinc-100 text-[10px] border px-1.5 py-0.2 rounded">Khối {sample.metadata?.grade}</span>
                    <span className="bg-zinc-100 text-[10px] border px-1.5 py-0.2 rounded">{sample.tokens?.length} Tokens</span>
                  </div>
                </div>
                <div className="md:w-4/5 leading-relaxed max-h-[140px] overflow-y-auto scrollbar-thin select-text">
                  {sample.tokens?.map((token: string, tIdx: number) => {
                    const tag = sample.tags?.[tIdx] || "O";
                    let classNames = "text-zinc-800";
                    if (tag !== "O") {
                      const base = tag.replace("B-", "").replace("I-", "");
                      classNames = `${TAG_COLOR_CLASSES[base] || "bg-zinc-100 text-zinc-900"} px-0.5 rounded font-medium border-b`;
                    }
                    return (
                      <span key={tIdx} className={`${classNames} inline-block mr-1 text-xs font-serif`} title={tag}>
                        {token}
                      </span>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>

          {data && data.total_pages > 1 && (
            <div className="flex items-center justify-between border bg-white rounded-xl p-3 text-xs text-zinc-500 shadow-sm">
              <span>Hiển thị {data.start_sample_idx}-{data.end_sample_idx} trên {data.total_matching} mẫu</span>
              <div className="flex items-center gap-1">
                <Button variant="outline" size="sm" disabled={page <= 1} onClick={() => setPage(p => p - 1)}>
                  <ChevronLeft size={14} />
                </Button>
                <span className="font-semibold text-zinc-900 px-3">Trang {data.page} / {data.total_pages}</span>
                <Button variant="outline" size="sm" disabled={page >= data.total_pages} onClick={() => setPage(p => p + 1)}>
                  <ChevronRight size={14} />
                </Button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
