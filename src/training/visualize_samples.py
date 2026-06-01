import json
import sys
from pathlib import Path

# Add local path
sys.path.append(str(Path(__file__).parent.parent))

def generate_visualization(jsonl_path: str, output_html_path: str, max_embed_samples: int = 1000):
    jsonl_file = Path(jsonl_path)
    if not jsonl_file.exists():
        print(f"Error: Dataset file '{jsonl_path}' not found.")
        sys.exit(1)
        
    samples = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for _ in range(max_embed_samples):
            line = f.readline()
            if not line:
                break
            samples.append(json.loads(line))
            
    # Serialize samples to embedded JSON
    embedded_dataset_json = json.dumps(samples, ensure_ascii=False)
            
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>XLM-RoBERTa Token Span Visualization</title>
    <style>
        :root {{
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --text-color: #f1f5f9;
            --border-color: #334155;
            
            --color-question_label: rgba(59, 130, 246, 0.25);
            --border-question_label: #3b82f6;
            --text-question_label: #93c5fd;
            
            --color-stem: rgba(139, 92, 246, 0.25);
            --border-stem: #8b5cf6;
            --text-stem: #c084fc;
            
            --color-option_label: rgba(245, 158, 11, 0.25);
            --border-option_label: #f59e0b;
            --text-option_label: #fde047;
            
            --color-option_text: rgba(16, 185, 129, 0.25);
            --border-option_text: #10b981;
            --text-option_text: #6ee7b7;
            
            --color-context: rgba(244, 63, 94, 0.25);
            --border-context: #f43f5e;
            --text-context: #fda4af;
            
            --color-ordering_item_label: rgba(20, 184, 166, 0.25);
            --border-ordering_item_label: #14b8a6;
            --text-ordering_item_label: #99f6e4;
            
            --color-ordering_item_text: rgba(99, 102, 241, 0.25);
            --border-ordering_item_text: #6366f1;
            --text-ordering_item_text: #c7d2fe;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 40px 20px;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1000px;
            margin: 0 auto;
        }}
        
        header {{
            text-align: center;
            margin-bottom: 40px;
        }}
        
        h1 {{
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(to right, #60a5fa, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }}
        
        .subtitle {{
            color: #94a3b8;
            font-size: 1.1rem;
        }}
        
        .control-panel {{
            display: flex;
            justify-content: center;
            margin-bottom: 30px;
        }}
        
        .btn {{
            background: linear-gradient(135deg, #3b82f6, #8b5cf6);
            color: white;
            border: none;
            padding: 14px 28px;
            font-size: 1rem;
            font-weight: 700;
            border-radius: 9999px;
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            outline: none;
        }}
        
        .btn:hover {{
            transform: translateY(-2px) scale(1.02);
            box-shadow: 0 6px 20px rgba(139, 92, 246, 0.6);
        }}
        
        .btn:active {{
            transform: translateY(0) scale(0.98);
        }}
        
        /* Legend styling */
        .legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 30px;
            justify-content: center;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.9rem;
            font-weight: 500;
            padding: 4px 10px;
            border-radius: 6px;
        }}
        
        /* Tag classes */
        .tag-question_label {{
            background-color: var(--color-question_label);
            border: 1px solid var(--border-question_label);
            color: var(--text-question_label);
        }}
        .tag-stem {{
            background-color: var(--color-stem);
            border: 1px solid var(--border-stem);
            color: var(--text-stem);
        }}
        .tag-option_label {{
            background-color: var(--color-option_label);
            border: 1px solid var(--border-option_label);
            color: var(--text-option_label);
        }}
        .tag-option_text {{
            background-color: var(--color-option_text);
            border: 1px solid var(--border-option_text);
            color: var(--text-option_text);
        }}
        .tag-context {{
            background-color: var(--color-context);
            border: 1px solid var(--border-context);
            color: var(--text-context);
        }}
        .tag-ordering_item_label {{
            background-color: var(--color-ordering_item_label);
            border: 1px solid var(--border-ordering_item_label);
            color: var(--text-ordering_item_label);
        }}
        .tag-ordering_item_text {{
            background-color: var(--color-ordering_item_text);
            border: 1px solid var(--border-ordering_item_text);
            color: var(--text-ordering_item_text);
        }}
        
        .tag-o {{
            color: #94a3b8;
        }}
        
        /* Sample card styling */
        .sample-card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
            animation: fadeIn 0.4s ease-out;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .sample-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 12px;
            margin-bottom: 20px;
        }}
        
        .sample-title {{
            font-size: 1.25rem;
            font-weight: 700;
            color: #38bdf8;
        }}
        
        .sample-meta {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }}
        
        .badge {{
            font-size: 0.75rem;
            font-weight: 600;
            background-color: #334155;
            color: #cbd5e1;
            padding: 4px 10px;
            border-radius: 9999px;
            text-transform: uppercase;
        }}
        
        .text-renderer {{
            font-family: inherit;
            font-size: 1.15rem;
            white-space: pre-wrap;
            word-break: break-word;
        }}
        
        .token {{
            display: inline-block;
            position: relative;
            cursor: help;
            border-radius: 3px;
            padding: 1px 2px;
            margin: 0 0.5px;
            transition: all 0.2s ease;
        }}
        
        .token:hover {{
            filter: brightness(1.2);
            box-shadow: 0 0 8px currentColor;
            transform: scale(1.05);
            z-index: 10;
        }}
        
        /* Tooltip styling */
        .token::after {{
            content: attr(data-tag);
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%) translateY(-5px);
            background-color: #020617;
            color: #ffffff;
            font-size: 0.75rem;
            font-weight: bold;
            padding: 4px 8px;
            border-radius: 4px;
            white-space: nowrap;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.2s, transform 0.2s;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
            border: 1px solid #475569;
        }}
        
        .token:hover::after {{
            opacity: 1;
            transform: translateX(-50%) translateY(-2px);
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>XLM-RoBERTa Token Span Alignment</h1>
            <div class="subtitle">Interactive visualization of subword token labels in sequence labelling mock exams</div>
        </header>
        
        <div class="control-panel">
            <button id="random-btn" class="btn">Get 5 Random Samples</button>
        </div>
        
        <div class="legend">
            <div class="legend-item tag-question_label">Question Label</div>
            <div class="legend-item tag-stem">Stem (Question Text)</div>
            <div class="legend-item tag-option_label">Option Label</div>
            <div class="legend-item tag-option_text">Option Text</div>
            <div class="legend-item tag-context">Context Passage</div>
            <div class="legend-item tag-ordering_item_label">Ordering Label</div>
            <div class="legend-item tag-ordering_item_text">Ordering Text</div>
        </div>

        <div id="samples-container"></div>
    </div>

    <script>
        const dataset = {embedded_dataset_json};

        function escapeHtml(str) {{
            return str.replace(/&/g, "&amp;")
                      .replace(/</g, "&lt;")
                      .replace(/>/g, "&gt;")
                      .replace(/"/g, "&quot;")
                      .replace(/'/g, "&#039;");
        }}

        function renderSamples(samples) {{
            const container = document.getElementById("samples-container");
            container.innerHTML = "";
            
            samples.forEach((sample, idx) => {{
                const metadata = sample.metadata || {{}};
                const source = metadata.source_file || sample.metadata.source_file || "unknown";
                const subject = metadata.subject || "N/A";
                const qtype = metadata.question_type || "N/A";
                const grade = metadata.grade || "N/A";
                const diff = metadata.difficulty || "N/A";
                
                const card = document.createElement("div");
                card.className = "sample-card";
                
                // Header
                const header = document.createElement("div");
                header.className = "sample-header";
                header.innerHTML = `
                    <span class="sample-title">Sample #${{idx+1}} &mdash; <span style="font-size: 0.85rem; font-family: monospace; color: #94a3b8;">${{source}}</span></span>
                    <div class="sample-meta">
                        <span class="badge">${{subject}}</span>
                        <span class="badge">${{qtype}}</span>
                        <span class="badge">Grade ${{grade}}</span>
                        <span class="badge">${{diff}}</span>
                    </div>
                `;
                card.appendChild(header);
                
                // Text renderer
                const textRenderer = document.createElement("div");
                textRenderer.className = "text-renderer";
                
                const tokens = sample.tokens || [];
                const tags = sample.tags || [];
                
                let htmlStr = "";
                
                for (let i = 0; i < tokens.length; i++) {{
                    let tok = tokens[i];
                    const tag = tags[i];
                    
                    if (tok === "<s>" || tok === "</s>" || tok === "<pad>" || tok === "<unk>" || tok === "<mask>") {{
                        continue;
                    }}
                    if (tag === "IGNORE") {{
                        continue;
                    }}
                    
                    let hasSpace = false;
                    // XLM-RoBERTa space prefix is U+2581 (lower half block)
                    if (tok.startsWith(" ") || tok.startsWith(" ")) {{
                        hasSpace = true;
                        tok = tok.substring(1);
                    }}
                    
                    if (!tok) {{
                        if (hasSpace) htmlStr += " ";
                        continue;
                    }}
                    
                    const baseTag = tag !== "O" ? tag.split("-").pop().toLowerCase() : "o";
                    const spanClass = `tag-${{baseTag}}`;
                    
                    const spacePrefix = hasSpace ? " " : "";
                    const escapedTok = escapeHtml(tok);
                    
                    if (baseTag !== "o") {{
                        htmlStr += `${{spacePrefix}}<span class="token ${{spanClass}}" data-tag="${{tag}}">${{escapedTok}}</span>`;
                    }} else {{
                        htmlStr += `${{spacePrefix}}${{escapedTok}}`;
                    }}
                }}
                
                textRenderer.innerHTML = htmlStr;
                card.appendChild(textRenderer);
                container.appendChild(card);
            }});
        }}

        function getRandomSamples(num) {{
            const shuffled = [...dataset].sort(() => 0.5 - Math.random());
            return shuffled.slice(0, num);
        }}

        document.getElementById("random-btn").addEventListener("click", () => {{
            const randomSamples = getRandomSamples(5);
            renderSamples(randomSamples);
        }});

        // Render initial 5 samples on load
        window.addEventListener("DOMContentLoaded", () => {{
            const initialSamples = getRandomSamples(5);
            renderSamples(initialSamples);
        }});
    </script>
</body>
</html>
"""
    
    out_file = Path(output_html_path)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Successfully generated visualizer at: {out_file.absolute()}")

if __name__ == "__main__":
    generate_visualization("dataset_output/train.jsonl", "dataset_output/sample_visualization.html")
