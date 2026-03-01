"use client";

import Image from "next/image";
import { AnimatePresence, motion } from "framer-motion";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type SlotStatus = "idle" | "loading" | "done" | "error";
type Theme = "dark" | "light";

type Slot = {
  title: string;
  style: string;
  status: SlotStatus;
  imageUrl: string | null;
  error: string | null;
  layoutId: string | null;
};

type GalleryItem = {
  id: string;
  imageUrl: string;
  style: string;
  model: string;
  prompt: string;
  createdAt: number;
};

type ActivePreview = {
  layoutId: string;
  imageUrl: string;
  title: string;
};

type StreamEvent =
  | {
      type: "result";
      index: number;
      style: string;
      image?: string;
      error?: string;
    }
  | {
      type: "complete";
    };

const GALLERY_STORAGE_KEY = "thumbly-gallery-v1";
const ASPECT_RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4", "3:2", "2:3", "4:5", "5:4", "21:9"];
const MODEL_OPTIONS = [
  { id: "google-ai-studio/gemini-2.5-flash-image", label: "Google AI Studio: Gemini Image" },
  { id: "nvidia/nemotron-nano-12b-v2-vl:free", label: "NVIDIA: Nemotron Nano 12B VL (Free)" },
  { id: "sourceful/riverflow-v2-fast", label: "Sourceful: Riverflow V2 Fast" },
  { id: "sourceful/riverflow-v2-pro", label: "Sourceful: Riverflow V2 Pro" },
  { id: "openai/gpt-image-1", label: "OpenAI: GPT Image 1" },
];
const STYLE_LABELS = ["Cinematic realism", "Studio editorial", "High-energy social"];

function buildInitialSlots(): Slot[] {
  return STYLE_LABELS.map((style, idx) => ({
    title: `Variant ${idx + 1}`,
    style,
    status: "idle",
    imageUrl: null,
    error: null,
    layoutId: null,
  }));
}

function parseStreamChunk(chunk: string, onEvent: (event: StreamEvent) => void): string {
  const lines = chunk.split("\n");
  const remainder = lines.pop() ?? "";

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      continue;
    }

    try {
      const event = JSON.parse(trimmed) as StreamEvent;
      onEvent(event);
    } catch {
      // Ignore malformed lines in stream.
    }
  }

  return remainder;
}

function sanitizeError(raw: string): string {
  const compact = raw.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
  if (!compact) {
    return "Generation failed. Please try again.";
  }

  if (compact.toLowerCase().includes("incorrect api key")) {
    return "Invalid API key for current provider. Check your .env and model provider.";
  }

  if (compact.toLowerCase().includes("unauthorized") || compact.toLowerCase().includes("401")) {
    return "Unauthorized request. Verify API key and provider base URL.";
  }

  return compact.length > 180 ? `${compact.slice(0, 180)}...` : compact;
}

function loadGallery(): GalleryItem[] {
  if (typeof window === "undefined") {
    return [];
  }

  try {
    const raw = window.localStorage.getItem(GALLERY_STORAGE_KEY);
    if (!raw) {
      return [];
    }

    const parsed = JSON.parse(raw) as GalleryItem[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function saveGallery(items: GalleryItem[]) {
  if (typeof window === "undefined") {
    return;
  }

  try {
    window.localStorage.setItem(GALLERY_STORAGE_KEY, JSON.stringify(items.slice(0, 60)));
  } catch {
    // ignore storage errors
  }
}

export default function Home() {
  const [theme, setTheme] = useState<Theme>("dark");
  const [aspect, setAspect] = useState("16:9");
  const [model, setModel] = useState("google-ai-studio/gemini-2.5-flash-image");
  const [prompt, setPrompt] = useState("");
  const [referenceImage, setReferenceImage] = useState<File | null>(null);
  const [slots, setSlots] = useState<Slot[]>(() => buildInitialSlots());
  const [history, setHistory] = useState<string[]>([]);
  const [gallery, setGallery] = useState<GalleryItem[]>([]);
  const [activePreview, setActivePreview] = useState<ActivePreview | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);

  const currentRunRef = useRef("");
  const abortRef = useRef<AbortController | null>(null);

  const canGenerate = useMemo(() => prompt.trim().length > 0 && !isGenerating, [prompt, isGenerating]);

  useEffect(() => {
    setGallery(loadGallery());
    return () => abortRef.current?.abort();
  }, []);

  useEffect(() => {
    saveGallery(gallery);
  }, [gallery]);

  const downloadImage = useCallback((imageUrl: string, name: string) => {
    const a = document.createElement("a");
    a.href = imageUrl;
    a.download = `${name}.png`;
    a.rel = "noopener noreferrer";
    document.body.appendChild(a);
    a.click();
    a.remove();
  }, []);

  const applyStreamEvent = useCallback(
    (event: StreamEvent, runId: string, activePrompt: string, activeModel: string) => {
      if (runId !== currentRunRef.current) {
        return;
      }

      if (event.type === "result") {
        const layoutId = `result-${runId}-${event.index}`;

        setSlots((current) => {
          if (!current[event.index]) {
            return current;
          }

          const next = [...current];
          next[event.index] = {
            ...next[event.index],
            style: event.style,
            status: event.error ? "error" : "done",
            imageUrl: event.image ?? null,
            error: event.error ? sanitizeError(event.error) : null,
            layoutId: event.image ? layoutId : null,
          };
          return next;
        });

        if (event.image) {
          const imageUrl = event.image;
          setGallery((current) => {
            const id = `${runId}-${event.index}`;
            if (current.some((item) => item.id === id)) {
              return current;
            }

            return [
              {
                id,
                imageUrl,
                style: event.style,
                model: activeModel,
                prompt: activePrompt,
                createdAt: Date.now(),
              },
              ...current,
            ].slice(0, 60);
          });
        }
        return;
      }

      if (event.type === "complete") {
        setIsGenerating(false);
      }
    },
    [],
  );

  const generateAll = useCallback(async () => {
    if (!canGenerate) {
      return;
    }

    const activePrompt = prompt.trim();
    const activeModel = model;
    const runId = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    currentRunRef.current = runId;

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setIsGenerating(true);
    setSlots(buildInitialSlots().map((slot) => ({ ...slot, status: "loading" })));
    setHistory((current) => [activePrompt, ...current.filter((item) => item !== activePrompt)].slice(0, 8));

    const formData = new FormData();
    formData.append("prompt", activePrompt);
    formData.append("aspect", aspect);
    formData.append("model", activeModel);
    if (referenceImage) {
      formData.append("referenceImage", referenceImage);
    }

    try {
      const response = await fetch("/api/generate", {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });

      if (!response.ok) {
        let raw = "Generation request failed.";
        const type = response.headers.get("content-type") || "";

        try {
          if (type.includes("application/json")) {
            const data = (await response.json()) as { error?: string; message?: string };
            raw = data.error || data.message || raw;
          } else {
            raw = await response.text();
          }
        } catch {
          raw = `Request failed with status ${response.status}`;
        }

        throw new Error(sanitizeError(raw));
      }

      if (!response.body) {
        throw new Error("Streaming is not available in this browser.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        buffer = parseStreamChunk(buffer, (event) => applyStreamEvent(event, runId, activePrompt, activeModel));
      }

      const tail = buffer.trim();
      if (tail) {
        try {
          applyStreamEvent(JSON.parse(tail) as StreamEvent, runId, activePrompt, activeModel);
        } catch {
          // ignore trailing invalid chunk
        }
      }

      if (runId === currentRunRef.current) {
        setIsGenerating(false);
      }
    } catch (error) {
      if (controller.signal.aborted) {
        return;
      }

      const message = sanitizeError(error instanceof Error ? error.message : "Failed to generate images.");
      if (runId === currentRunRef.current) {
        setSlots((current) =>
          current.map((slot) => ({
            ...slot,
            status: "error",
            error: message,
          })),
        );
        setIsGenerating(false);
      }
    }
  }, [applyStreamEvent, aspect, canGenerate, model, prompt, referenceImage]);

  return (
    <div className="page-wrap" data-theme={theme}>
      <main className="shell">
        <header className="topbar">
          <div>
            <p className="kicker">THUMBLY // AI THUMBNAIL STUDIO</p>
            <h1>Build 3 thumbnail styles in parallel</h1>
          </div>
          <button
            type="button"
            className="theme-toggle"
            onClick={() => setTheme((current) => (current === "dark" ? "light" : "dark"))}
          >
            {theme === "dark" ? "Light mode" : "Dark mode"}
          </button>
        </header>

        <section className="workspace">
          <aside className="panel side">
            <div className="block">
              <h2>Aspect Ratio</h2>
              <div className="ratio-grid">
                {ASPECT_RATIOS.map((item) => (
                  <button
                    key={item}
                    type="button"
                    className={`chip ${aspect === item ? "active" : ""}`}
                    onClick={() => setAspect(item)}
                  >
                    {item}
                  </button>
                ))}
              </div>
            </div>

            <div className="block">
              <h2>Model</h2>
              <div className="model-list">
                {MODEL_OPTIONS.map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    className={`chip model-chip ${model === item.id ? "active" : ""}`}
                    onClick={() => setModel(item.id)}
                  >
                    {item.label}
                  </button>
                ))}
              </div>
            </div>
          </aside>

          <section className="main-col">
            <div className="panel composer">
              <p className="composer-help">Prompt + reference image + ratio {"=>"} 3 generated variants as each completes.</p>

              <label className="upload-row" htmlFor="reference-image">
                <span>Reference</span>
                <input
                  id="reference-image"
                  type="file"
                  accept="image/png,image/jpeg,image/webp"
                  onChange={(event) => setReferenceImage(event.target.files?.[0] ?? null)}
                />
              </label>

              <textarea
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
                placeholder="Describe subject, mood, camera angle, lighting, and intended click impact..."
              />

              <div className="composer-footer">
                <span className="reference-state">
                  {referenceImage ? `Attached: ${referenceImage.name}` : "No reference image attached"}
                </span>
                <button type="button" className="generate-btn" disabled={!canGenerate} onClick={generateAll}>
                  {isGenerating ? "Generating..." : "Generate 3"}
                </button>
              </div>
            </div>

            <section className="results-grid">
              {slots.map((slot, idx) => (
                <article key={slot.title} className="panel result-card">
                  <div className="result-head">
                    <h3>{slot.title}</h3>
                    <span>{slot.style}</span>
                  </div>

                  {slot.status === "idle" && <div className="result-placeholder">Waiting</div>}
                  {slot.status === "loading" && <div className="result-placeholder">Generating...</div>}
                  {slot.status === "error" && <div className="result-placeholder error">{slot.error ?? "Generation failed"}</div>}

                  {slot.status === "done" && slot.imageUrl && slot.layoutId && (
                    <>
                      <motion.button
                        type="button"
                        className="image-button"
                        onClick={() =>
                          setActivePreview({
                            layoutId: slot.layoutId!,
                            imageUrl: slot.imageUrl!,
                            title: `${slot.title} - ${slot.style}`,
                          })
                        }
                      >
                        <motion.div layoutId={slot.layoutId} className="result-image-wrap">
                          <Image
                            className="result-image"
                            src={slot.imageUrl}
                            alt={`Generated thumbnail ${idx + 1}`}
                            width={1024}
                            height={1024}
                            unoptimized
                          />
                        </motion.div>
                      </motion.button>

                      <button
                        type="button"
                        className="download-btn"
                        onClick={() => downloadImage(slot.imageUrl!, `thumbly-${Date.now()}-${idx + 1}`)}
                      >
                        Download
                      </button>
                    </>
                  )}
                </article>
              ))}
            </section>
          </section>

          <aside className="panel side history-panel">
            <div className="history-head">
              <h2>History</h2>
              <span>/{history.length}</span>
            </div>

            <div className="history-list">
              {history.length === 0 && <div className="history-item muted">No prompts yet</div>}
              {history.map((item, idx) => (
                <button key={`${item}-${idx}`} type="button" className="history-item" onClick={() => setPrompt(item)}>
                  {item}
                </button>
              ))}
            </div>
          </aside>
        </section>

        <section className="panel gallery-panel">
          <div className="gallery-head">
            <h2>Gallery</h2>
            <span>{gallery.length} images</span>
          </div>

          <div className="gallery-grid">
            {gallery.length === 0 && <div className="gallery-empty">Generated images will appear here.</div>}
            {gallery.map((item) => {
              const layoutId = `gallery-${item.id}`;
              return (
                <article key={item.id} className="gallery-item">
                  <button
                    type="button"
                    className="gallery-image-button"
                    onClick={() =>
                      setActivePreview({
                        layoutId,
                        imageUrl: item.imageUrl,
                        title: `${item.style} - ${item.model}`,
                      })
                    }
                  >
                    <motion.div layoutId={layoutId} className="gallery-image-wrap">
                      <Image src={item.imageUrl} alt={item.style} width={1024} height={1024} className="gallery-image" unoptimized />
                    </motion.div>
                  </button>

                  <div className="gallery-meta">
                    <p>{item.style}</p>
                    <span>{item.model}</span>
                  </div>

                  <button
                    type="button"
                    className="download-btn"
                    onClick={() => downloadImage(item.imageUrl, `thumbly-gallery-${item.id}`)}
                  >
                    Download
                  </button>
                </article>
              );
            })}
          </div>
        </section>

        <AnimatePresence>
          {activePreview && (
            <motion.div
              className="preview-backdrop"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setActivePreview(null)}
            >
              <motion.div
                className="preview-content"
                initial={{ opacity: 0, scale: 0.96 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.97 }}
                transition={{ duration: 0.22, ease: "easeOut" }}
                onClick={(event) => event.stopPropagation()}
              >
                <div className="preview-head">
                  <h3>{activePreview.title}</h3>
                  <button type="button" className="close-btn" onClick={() => setActivePreview(null)}>
                    Close
                  </button>
                </div>

                <motion.div layoutId={activePreview.layoutId} className="preview-image-wrap">
                  <Image src={activePreview.imageUrl} alt={activePreview.title} width={1536} height={1024} className="preview-image" unoptimized />
                </motion.div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}
