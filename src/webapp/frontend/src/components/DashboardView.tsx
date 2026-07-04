import React, { useState, useEffect, useMemo } from "react";
import {
  Grid,
  List,
  Search,
  CheckCircle,
  AlertCircle,
} from "lucide-react";
import { type Exam } from "../types";
import { SUBJECT_DISPLAY } from "../utils/textUtils";
import { Checkbox } from "./ui/checkbox";
import { Button } from "./ui/button";

interface DashboardViewProps {
  navigate: (h: string) => void;
}

export const DashboardView: React.FC<DashboardViewProps> = ({ navigate }) => {
  const [exams, setExams] = useState<Exam[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  // Dense vs Grid view toggle
  const [layoutMode, setLayoutMode] = useState<"table" | "grid">(
    () => (localStorage.getItem("dashboard_layoutMode") as "table" | "grid") || "table"
  );

  // Filter States
  const [search, setSearch] = useState<string>("");
  const [debouncedSearch, setDebouncedSearch] = useState<string>("");
  const [filterSubject, setFilterSubject] = useState<string>(
    () => localStorage.getItem("dashboard_filterSubject") || ""
  );
  const [filterType, setFilterType] = useState<string>(
    () => localStorage.getItem("dashboard_filterType") || "all"
  );
  const [annotationFilter, setAnnotationFilter] = useState<string>(
    () => localStorage.getItem("dashboard_annotationFilter") || "all"
  );

  // Selection states
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  // Save layout and filters to localStorage
  useEffect(() => {
    localStorage.setItem("dashboard_layoutMode", layoutMode);
  }, [layoutMode]);

  useEffect(() => {
    localStorage.setItem("dashboard_filterSubject", filterSubject);
  }, [filterSubject]);

  useEffect(() => {
    localStorage.setItem("dashboard_filterType", filterType);
  }, [filterType]);

  useEffect(() => {
    localStorage.setItem("dashboard_annotationFilter", annotationFilter);
  }, [annotationFilter]);

  // Debounce search input for DOM performance
  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedSearch(search);
    }, 200);
    return () => clearTimeout(handler);
  }, [search]);

  useEffect(() => {
    fetch("/api/exams")
      .then(res => res.json())
      .then(data => {
        setExams(data);
        setLoading(false);
      })
      .catch(err => console.error("Error loading exams:", err));
  }, []);

  const filteredExams = useMemo(() => {
    return exams.filter(ex => {
      if (debouncedSearch && !ex.exam_id.toLowerCase().includes(debouncedSearch.toLowerCase())) return false;
      if (filterSubject && ex.subject !== filterSubject) return false;
      if (filterType === "real" && !ex.is_real) return false;
      if (filterType === "synthetic" && ex.is_real) return false;

      const isAnnotated = ex.annotated;
      if (annotationFilter === "annotated" && (!ex.is_real || !isAnnotated)) return false;
      if (annotationFilter === "unannotated" && (!ex.is_real || isAnnotated)) return false;

      return true;
    });
  }, [exams, debouncedSearch, filterSubject, filterType, annotationFilter]);

  const stats = useMemo(() => {
    const real = exams.filter(e => e.is_real);
    return {
      total: exams.length,
      real: real.length,
      synthetic: exams.length - real.length,
      annotated: real.filter(e => e.annotated).length,
      unannotated: real.filter(e => !e.annotated).length,
    };
  }, [exams]);

  // Pagination states
  const [currentPage, setCurrentPage] = useState<number>(1);
  const pageSize = 25;

  // Reset to page 1 when search or filters change
  useEffect(() => {
    setCurrentPage(1);
  }, [debouncedSearch, filterSubject, filterType, annotationFilter]);

  const totalPages = Math.ceil(filteredExams.length / pageSize) || 1;

  const paginatedExams = useMemo(() => {
    const startIdx = (currentPage - 1) * pageSize;
    return filteredExams.slice(startIdx, startIdx + pageSize);
  }, [filteredExams, currentPage]);

  // Bulk Actions Handlers
  const handleSelectAll = (checked: boolean) => {
    if (checked) {
      const pageIds = paginatedExams.map(e => e.exam_id);
      setSelectedIds(new Set(pageIds));
    } else {
      setSelectedIds(new Set());
    }
  };

  const handleSelectOne = (examId: string, checked: boolean) => {
    const updated = new Set(selectedIds);
    if (checked) {
      updated.add(examId);
    } else {
      updated.delete(examId);
    }
    setSelectedIds(updated);
  };

  const handleBatchAnnotate = async (status: boolean) => {
    if (selectedIds.size === 0) return;
    const targetIds = Array.from(selectedIds);

    try {
      setLoading(true);
      const responses = await Promise.all(
        targetIds.map(id =>
          fetch(`/api/exam/${id}/annotated`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ annotated: status })
          })
        )
      );

      if (responses.some(r => !r.ok)) {
        throw new Error("One or more requests returned an error.");
      }

      // Update local state locally
      setExams(prev =>
        prev.map(ex =>
          targetIds.includes(ex.exam_id) ? { ...ex, annotated: status } : ex
        )
      );
      setSelectedIds(new Set());
    } catch (err: any) {
      alert("Error processing updates: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  const isAllSelected = paginatedExams.length > 0 && paginatedExams.every(e => selectedIds.has(e.exam_id));
  const isSomeSelected = paginatedExams.some(e => selectedIds.has(e.exam_id)) && !isAllSelected;

  return (
    <div className="max-w-7xl mx-auto w-full px-6 py-6 space-y-6">
      {/* Header section */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div className="space-y-1">
          <h2 className="text-xl font-bold tracking-tight text-zinc-900">Bảng điều khiển</h2>
          <p className="text-xs text-zinc-550">Quản lý và thiết lập trạng thái gán nhãn thực thể cho dữ liệu OCR đề thi.</p>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              const realExams = exams.filter(e => e.is_real);
              if (realExams.length > 0) {
                const rand = realExams[Math.floor(Math.random() * realExams.length)];
                navigate(`#/exam/${rand.exam_id}`);
              }
            }}
          >
            🎲 Ngẫu nhiên đề OCR
          </Button>
        </div>
      </div>

      {/* Stats Board */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="rounded-lg border border-zinc-200 bg-white p-4 shadow-sm space-y-0.5">
          <span className="text-[9px] font-bold text-zinc-450 uppercase tracking-wider block">Tổng số đề</span>
          <p className="text-xl font-bold text-zinc-950">{stats.total}</p>
        </div>
        <div className="rounded-lg border border-zinc-200 bg-white p-4 shadow-sm space-y-0.5">
          <span className="text-[9px] font-bold text-zinc-450 uppercase tracking-wider block font-sans">Đề thực tế (OCR)</span>
          <p className="text-xl font-bold text-zinc-950 flex items-center gap-1.5">
            {stats.real}
            <span className="text-[10px] font-medium text-zinc-500 font-mono">({stats.annotated} gán / {stats.unannotated} chưa)</span>
          </p>
        </div>
        <div className="rounded-lg border border-zinc-200 bg-white p-4 shadow-sm space-y-0.5">
          <span className="text-[9px] font-bold text-zinc-450 uppercase tracking-wider block">Đề giả lập (LLM)</span>
          <p className="text-xl font-bold text-zinc-950">{stats.synthetic}</p>
        </div>
        <div className="rounded-lg border border-zinc-200 bg-white p-4 shadow-sm space-y-0.5">
          <span className="text-[9px] font-bold text-zinc-450 uppercase tracking-wider block">Đã chọn</span>
          <p className="text-xl font-bold text-zinc-950">{selectedIds.size} đề</p>
        </div>
      </div>

      {/* Control Panel */}
      <div className="space-y-3 bg-white p-4 rounded-xl border border-zinc-200/80 shadow-sm/5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-2">
            <div className="relative">
              <Search className="absolute left-2.5 top-2.5 h-3.5 w-3.5 text-zinc-400" />
              <input
                type="text"
                placeholder="Tìm mã đề thi..."
                value={search}
                onChange={e => setSearch(e.target.value)}
                className="flex h-8 w-44 rounded-md border border-zinc-200 bg-white pl-8 pr-3 py-1 text-xs shadow-sm/5 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-zinc-900 text-zinc-950"
              />
            </div>

            <select
              value={filterSubject}
              onChange={e => setFilterSubject(e.target.value)}
              className="flex h-8 rounded-md border border-zinc-200 bg-white px-2 py-1 text-xs shadow-sm/5 text-zinc-700 focus-visible:outline-none"
            >
              <option value="">Tất cả môn học</option>
              {Object.entries(SUBJECT_DISPLAY).map(([k, v]) => (
                <option key={k} value={k}>{v}</option>
              ))}
            </select>

            <div className="inline-flex rounded-md bg-zinc-100 p-0.5 border border-zinc-200 text-xs">
              <button
                onClick={() => setFilterType("all")}
                className={`px-2.5 py-0.5 rounded-sm font-medium transition ${filterType === "all" ? "bg-white text-zinc-950 shadow-sm" : "text-zinc-550 hover:text-zinc-950"}`}
              >
                Tất cả
              </button>
              <button
                onClick={() => setFilterType("real")}
                className={`px-2.5 py-0.5 rounded-sm font-medium transition ${filterType === "real" ? "bg-white text-zinc-950 shadow-sm" : "text-zinc-550 hover:text-zinc-950"}`}
              >
                OCR
              </button>
              <button
                onClick={() => setFilterType("synthetic")}
                className={`px-2.5 py-0.5 rounded-sm font-medium transition ${filterType === "synthetic" ? "bg-white text-zinc-950 shadow-sm" : "text-zinc-550 hover:text-zinc-950"}`}
              >
                Giả lập
              </button>
            </div>

            <div className="inline-flex rounded-md bg-zinc-100 p-0.5 border border-zinc-200 text-xs">
              <button
                onClick={() => setAnnotationFilter("all")}
                className={`px-2.5 py-0.5 rounded-sm font-medium transition ${annotationFilter === "all" ? "bg-white text-zinc-950 shadow-sm" : "text-zinc-550"}`}
              >
                Tất cả trạng thái
              </button>
              <button
                onClick={() => setAnnotationFilter("unannotated")}
                className={`px-2.5 py-0.5 rounded-sm font-medium transition ${annotationFilter === "unannotated" ? "bg-white text-zinc-950 shadow-sm" : "text-zinc-550"}`}
              >
                Chưa gán
              </button>
              <button
                onClick={() => setAnnotationFilter("annotated")}
                className={`px-2.5 py-0.5 rounded-sm font-medium transition ${annotationFilter === "annotated" ? "bg-white text-zinc-950 shadow-sm" : "text-zinc-550"}`}
              >
                Đã gán
              </button>
            </div>
          </div>

          {/* Grid/Dense List switches */}
          <div className="flex items-center gap-1.5">
            <button
              onClick={() => setLayoutMode("table")}
              className={`p-1.5 rounded-md border transition ${layoutMode === "table" ? "bg-zinc-100 text-zinc-950 border-zinc-300" : "bg-white text-zinc-400 border-zinc-200"}`}
              title="Danh sách rút gọn (Bảng)"
            >
              <List size={14} />
            </button>
            <button
              onClick={() => setLayoutMode("grid")}
              className={`p-1.5 rounded-md border transition ${layoutMode === "grid" ? "bg-zinc-100 text-zinc-950 border-zinc-300" : "bg-white text-zinc-400 border-zinc-200"}`}
              title="Dạng lưới"
            >
              <Grid size={14} />
            </button>
          </div>
        </div>

        {/* Batch action status bar */}
        {selectedIds.size > 0 && (
          <div className="flex items-center justify-between border-t border-zinc-100 pt-3 bg-zinc-50/50 -mx-4 -mb-4 p-4 rounded-b-xl">
            <div className="flex items-center gap-2 text-xs font-medium text-zinc-700">
              <span>Đã chọn <b>{selectedIds.size}</b> tài liệu</span>
            </div>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                className="bg-white"
                onClick={() => handleBatchAnnotate(true)}
              >
                ✓ Đánh dấu Đã gán nhãn
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="bg-white text-amber-700 hover:text-amber-800"
                onClick={() => handleBatchAnnotate(false)}
              >
                ✗ Đánh dấu Chưa gán nhãn
              </Button>
            </div>
          </div>
        )}
      </div>

      {/* Main List */}
      {loading ? (
        <div className="py-20 text-center text-zinc-550 text-xs font-mono">Đang tải danh sách tài liệu...</div>
      ) : layoutMode === "table" ? (
        /* ULTRA-DENSE TABLE VIEW (Fixes spacing waste) */
        <div className="border border-zinc-200 rounded-xl bg-white shadow-sm overflow-hidden">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="bg-zinc-50 text-zinc-500 font-semibold border-b border-zinc-200">
                <th className="p-3 w-10 text-center">
                  <Checkbox
                    checked={isAllSelected}
                    indeterminate={isSomeSelected}
                    onChange={(e) => handleSelectAll(e.target.checked)}
                  />
                </th>
                <th className="p-3">Mã đề / ID</th>
                <th className="p-3">Môn học</th>
                <th className="p-3 w-20">Khối lớp</th>
                <th className="p-3 w-24">Số câu</th>
                <th className="p-3 w-24">Nguồn</th>
                <th className="p-3 w-28">Trạng thái</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-100">
              {paginatedExams.map(ex => {
                const isChecked = selectedIds.has(ex.exam_id);
                return (
                  <tr
                    key={ex.exam_id}
                    className={`hover:bg-zinc-50/60 transition cursor-pointer ${isChecked ? "bg-zinc-50/40" : ""}`}
                    onClick={() => navigate(`#/exam/${ex.exam_id}`)}
                  >
                    <td className="p-3 text-center" onClick={(e) => e.stopPropagation()}>
                      <Checkbox
                        checked={isChecked}
                        onChange={(e) => handleSelectOne(ex.exam_id, e.target.checked)}
                      />
                    </td>
                    <td className="p-3 font-mono text-[11px] text-zinc-900 font-semibold">{ex.exam_id}</td>
                    <td className="p-3 font-medium text-zinc-800">{ex.subject_display}</td>
                    <td className="p-3 text-zinc-550">Lớp {ex.grade}</td>
                    <td className="p-3 text-zinc-550 font-mono text-[11px]">{ex.question_count} câu</td>
                    <td className="p-3">
                      {ex.is_real ? (
                        <span className="inline-flex items-center rounded bg-rose-50 px-1.5 py-0.2 text-[10px] font-semibold text-rose-700 border border-rose-200/50">OCR</span>
                      ) : (
                        <span className="inline-flex items-center rounded bg-blue-50 px-1.5 py-0.2 text-[10px] font-semibold text-blue-700 border border-blue-200/50">LLM</span>
                      )}
                    </td>
                    <td className="p-3">
                      {ex.is_real ? (
                        ex.annotated ? (
                          <span className="inline-flex items-center gap-1.5 text-emerald-700 text-[10.5px] font-semibold">
                            <CheckCircle size={12} /> Đã gán
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1.5 text-amber-700 text-[10.5px] font-semibold">
                            <AlertCircle size={12} /> Chưa gán
                          </span>
                        )
                      ) : (
                        <span className="text-zinc-400 font-mono text-[10px]">-</span>
                      )}
                    </td>
                  </tr>
                );
              })}
              {filteredExams.length === 0 && (
                <tr>
                  <td colSpan={7} className="p-8 text-center text-zinc-450">Không có tài liệu nào phù hợp với bộ lọc.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      ) : (
        /* COMPACT GRID VIEW */
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
          {paginatedExams.map(ex => {
            const isChecked = selectedIds.has(ex.exam_id);
            return (
              <div
                key={ex.exam_id}
                className={`group rounded-xl border p-4 bg-white hover:border-zinc-350 hover:shadow-sm/5 transition flex flex-col justify-between space-y-3 cursor-pointer ${isChecked ? "border-zinc-800 bg-zinc-50/10" : "border-zinc-200"}`}
                onClick={() => navigate(`#/exam/${ex.exam_id}`)}
              >
                <div className="space-y-1.5">
                  <div className="flex justify-between items-start">
                    <span className="text-[10px] text-zinc-400 font-mono font-medium">ID: {ex.exam_id}</span>
                    <div className="flex items-center gap-1.5" onClick={e => e.stopPropagation()}>
                      <Checkbox
                        checked={isChecked}
                        onChange={(e) => handleSelectOne(ex.exam_id, e.target.checked)}
                      />
                    </div>
                  </div>
                  <h3 className="text-xs font-bold text-zinc-900 group-hover:text-black line-clamp-1">{ex.subject_display}</h3>
                  <div className="flex flex-wrap gap-1.5 pt-0.5">
                    <span className="inline-flex items-center rounded bg-zinc-100 px-1.5 py-0.2 text-[9.5px] text-zinc-600 border border-zinc-200/50">Lớp {ex.grade}</span>
                    <span className="inline-flex items-center rounded bg-zinc-100 px-1.5 py-0.2 text-[9.5px] text-zinc-600 border border-zinc-200/50">{ex.question_count} Câu</span>
                    {ex.is_real ? (
                      ex.annotated ? (
                        <span className="inline-flex items-center rounded bg-emerald-50 px-1.5 py-0.2 text-[9.5px] font-bold text-emerald-700 border border-emerald-250">Đã gán</span>
                      ) : (
                        <span className="inline-flex items-center rounded bg-amber-50 px-1.5 py-0.2 text-[9.5px] font-bold text-amber-700 border border-amber-250">Chưa gán</span>
                      )
                    ) : (
                      <span className="inline-flex items-center rounded bg-blue-50 px-1.5 py-0.2 text-[9.5px] font-bold text-blue-700 border border-blue-250">LLM</span>
                    )}
                  </div>
                </div>

                <div className="border-t border-zinc-100 pt-2.5 flex justify-between items-center text-[9.5px] text-zinc-400 font-mono">
                  <span>Ngày khởi tạo</span>
                  <span>{ex.created_at ? ex.created_at.substring(0, 10) : "N/A"}</span>
                </div>
              </div>
            );
          })}
          {filteredExams.length === 0 && (
            <div className="col-span-full py-16 text-center text-zinc-450 text-xs">Không tìm thấy tài liệu phù hợp.</div>
          )}
        </div>
      )}

      {/* Pagination Controls */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between border border-zinc-200 bg-white px-4 py-3 sm:px-6 rounded-xl shadow-sm/5 text-xs text-zinc-700">
          <div className="flex flex-1 justify-between sm:hidden">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
              disabled={currentPage === 1}
            >
              Trước
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
              disabled={currentPage === totalPages}
            >
              Sau
            </Button>
          </div>
          <div className="hidden sm:flex sm:flex-1 sm:items-center sm:justify-between">
            <div>
              <p className="text-xs text-zinc-500">
                Hiển thị <span className="font-semibold text-zinc-900">{(currentPage - 1) * pageSize + 1}</span> đến{" "}
                <span className="font-semibold text-zinc-900">{Math.min(currentPage * pageSize, filteredExams.length)}</span> trong số{" "}
                <span className="font-semibold text-zinc-900">{filteredExams.length}</span> kết quả
              </p>
            </div>
            <div className="flex items-center gap-1.5">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setCurrentPage(1)}
                disabled={currentPage === 1}
                className="h-8 w-8 p-0"
              >
                «
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
                disabled={currentPage === 1}
                className="h-8 px-2"
              >
                Trước
              </Button>
              <span className="px-2.5 py-1 rounded-md bg-zinc-50 border border-zinc-250 font-medium font-mono text-[11px]">
                {currentPage} / {totalPages}
              </span>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
                disabled={currentPage === totalPages}
                className="h-8 px-2"
              >
                Sau
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setCurrentPage(totalPages)}
                disabled={currentPage === totalPages}
                className="h-8 w-8 p-0"
              >
                »
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
