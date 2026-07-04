import type { Span, StructuredQuestion } from "../types";

/**
 * Maps a standard DOM text offset back to the raw source index,
 * normalizing carriage returns and line endings on the fly.
 */
export const mapDomOffsetToRawOffset = (
  rawSub: string,
  domText: string,
  domOffset: number,
): number => {
  let rawIdx = 0;
  let domIdx = 0;
  const rawLen = rawSub.length;

  while (domIdx < domOffset && rawIdx < rawLen) {
    const rawChar = rawSub[rawIdx];
    const domChar = domText[domIdx];

    if (rawChar === domChar) {
      rawIdx++;
      domIdx++;
    } else if (rawChar === "\r" && rawSub[rawIdx + 1] === "\n") {
      rawIdx += 2;
      domIdx++;
    } else if (rawChar === "\r" || rawChar === "\n") {
      rawIdx++;
      if (domChar === "\n" || domChar === "\r") {
        domIdx++;
      }
    } else {
      rawIdx++;
      domIdx++;
    }
  }

  return rawIdx;
};

/**
 * Compiles flat entity label segments into structured questionnaire blocks.
 */
export const parseSpansToQuestions = (
  spans: Span[],
  rawText: string,
): StructuredQuestion[] => {
  const sortedSpans = [...spans].sort((a, b) => a.start - b.start);
  const questions: any[] = [];
  let currentContext = "";
  let currentExplanation = "";
  let currentQuestion: any = null;

  for (let i = 0; i < sortedSpans.length; i++) {
    const span = sortedSpans[i];
    const label = span.label;
    const text = (
      span.text ||
      rawText.substring(span.start, span.end) ||
      ""
    ).trim();
    if (!text) continue;

    if (label === "context") {
      currentContext = text;
    } else if (label === "explanation") {
      currentExplanation = text;
      if (currentQuestion) {
        currentQuestion.explanation = text;
      }
    } else if (label === "question_label") {
      if (currentQuestion) {
        questions.push(currentQuestion);
      }
      currentQuestion = {
        question_label: text,
        context: currentContext,
        stem: "",
        options: [],
        explanation: currentExplanation || "",
        current_option_label: "",
      };
    } else if (label === "stem") {
      if (!currentQuestion) {
        currentQuestion = {
          question_label: "",
          context: currentContext,
          stem: text,
          options: [],
          explanation: currentExplanation || "",
          current_option_label: "",
        };
      } else {
        currentQuestion.stem = currentQuestion.stem
          ? `${currentQuestion.stem}\n${text}`
          : text;
      }
    } else if (label === "option_label") {
      if (currentQuestion) {
        currentQuestion.current_option_label = text;
      }
    } else if (label === "option_text") {
      if (currentQuestion) {
        const optLbl = currentQuestion.current_option_label || "";
        const prefix = optLbl ? `${optLbl} ` : "";
        currentQuestion.options.push(prefix + text);
        currentQuestion.current_option_label = "";
      }
    } else if (label === "section") {
      currentContext = "";
      currentExplanation = "";
    }
  }

  if (currentQuestion) {
    questions.push(currentQuestion);
  }

  return questions;
};

export const SUBJECT_DISPLAY: Record<string, string> = {
  economics_law: "GD Kinh tế & Pháp luật",
  geography: "Địa lý",
  history: "Lịch sử",
  math_algebra: "Toán (Đại số)",
  math_geometry: "Toán (Hình học)",
  physics: "Vật lý",
  chemistry: "Hóa học",
  english: "Tiếng Anh",
  literature: "Ngữ văn",
};

export const TAG_COLOR_CLASSES: Record<string, string> = {
  question_label: "bg-blue-50 text-blue-700 border-blue-200",
  stem: "bg-purple-50 text-purple-700 border-purple-200",
  option_label: "bg-amber-50 text-amber-700 border-amber-200",
  option_text: "bg-emerald-50 text-emerald-700 border-emerald-200",
  context: "bg-rose-50 text-rose-700 border-rose-200",
  section: "bg-zinc-100 text-zinc-700 border-zinc-200",
  explanation: "bg-orange-50 text-orange-700 border-orange-200",
};
