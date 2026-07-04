export interface Span {
  start: number;
  end: number;
  label: string;
  text?: string;
}

export interface StructuredQuestion {
  question_label: string;
  context: string;
  stem: string;
  options: string[];
  explanation: string;
}

export interface Exam {
  exam_id: string;
  subject: string;
  subject_display: string;
  grade: number;
  created_at: string;
  question_count: number;
  is_real: boolean;
  annotated?: boolean;
}

export interface ModelStatus {
  model_loaded: boolean;
  model_path: string;
  base_model_name: string;
  device: string;
  loading: boolean;
  error: string | null;
}
