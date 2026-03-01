import OpenAI from "openai";

export const runtime = "nodejs";

type SupportedSize = "1024x1024" | "1536x1024" | "1024x1536";

type Variant = {
  label: string;
  direction: string;
};

type ProviderMode = "openrouter" | "openai" | "google";

type ProviderConfig = {
  apiKey: string;
  mode: ProviderMode;
  missingEnvHint?: string;
};
const NEMOTRON_FREE_MODEL = "nvidia/nemotron-nano-12b-v2-vl:free";
const DEFAULT_GOOGLE_IMAGE_MODEL = "google-ai-studio/gemini-2.5-flash-image";
const DEFAULT_IMAGE_MODEL = DEFAULT_GOOGLE_IMAGE_MODEL;
const GOOGLE_PREFIX = "google-ai-studio/";

const VARIANTS: Variant[] = [
  {
    label: "Cinematic realism",
    direction: "Photorealistic lighting, dramatic depth, crisp subject focus, high emotional impact.",
  },
  {
    label: "Studio editorial",
    direction: "Clean product-ad framing, polished color balance, minimal but premium visual hierarchy.",
  },
  {
    label: "High-energy social",
    direction: "Bold color contrast, dynamic composition, modern social-media momentum and clarity.",
  },
];

function resolveProviderConfig(model: string): ProviderConfig | null {
  const isGoogle = model.startsWith(GOOGLE_PREFIX);
  const isFree = model === NEMOTRON_FREE_MODEL;
  const isFast = model === "sourceful/riverflow-v2-fast";
  const isPro = model === "sourceful/riverflow-v2-pro";

  if (isGoogle) {
    const apiKey = process.env.GOOGLE_AI_STUDIO_API_KEY?.trim();
    if (!apiKey) {
      return { apiKey: "", mode: "google", missingEnvHint: "GOOGLE_AI_STUDIO_API_KEY" };
    }
    return { apiKey, mode: "google" };
  }

  if (isFree) {
    const apiKey = process.env.FREE_MODEL_API_KEY?.trim() || process.env.OPENROUTER_API_KEY?.trim();
    if (!apiKey) {
      return { apiKey: "", mode: "openrouter", missingEnvHint: "FREE_MODEL_API_KEY" };
    }
    return { apiKey, mode: "openrouter" };
  }

  if (isFast) {
    const apiKey = process.env.RIVERFLOW_V2_FAST?.trim() || process.env.OPENROUTER_API_KEY?.trim();
    if (!apiKey) {
      return { apiKey: "", mode: "openrouter", missingEnvHint: "RIVERFLOW_V2_FAST" };
    }
    return { apiKey, mode: "openrouter" };
  }

  if (isPro) {
    const apiKey = process.env.RIVERFLOW_V2_PRO?.trim() || process.env.OPENROUTER_API_KEY?.trim();
    if (!apiKey) {
      return { apiKey: "", mode: "openrouter", missingEnvHint: "RIVERFLOW_V2_PRO" };
    }
    return { apiKey, mode: "openrouter" };
  }

  const openAiKey = process.env.OPENAI_API_KEY?.trim();
  if (openAiKey) {
    return { apiKey: openAiKey, mode: "openai" };
  }

  const openRouterFallback = process.env.OPENROUTER_API_KEY?.trim();
  if (openRouterFallback) {
    return { apiKey: openRouterFallback, mode: "openrouter" };
  }

  return null;
}

function getOpenAIClient(apiKey: string): OpenAI {
  const baseURL = process.env.OPENAI_BASE_URL?.trim();
  return new OpenAI({ apiKey, baseURL });
}

function normalizeSize(aspect: string): SupportedSize {
  if (aspect === "1:1") {
    return "1024x1024";
  }

  const [w, h] = aspect.split(":").map(Number);
  if (!Number.isFinite(w) || !Number.isFinite(h) || h === 0) {
    return "1536x1024";
  }

  return w / h > 1 ? "1536x1024" : "1024x1536";
}

function buildPrompt(basePrompt: string, variant: Variant, aspect: string): string {
  return [
    `Create a high-performing thumbnail composition for aspect ratio ${aspect}.`,
    `Style intent: ${variant.label}. ${variant.direction}`,
    "Make subject separation clear and keep composition readable at small sizes.",
    "Prioritize one strong focal point with cinematic contrast and clean lighting.",
    "Do not include watermarks, logos, random text artifacts, or cluttered backgrounds.",
    `User concept: ${basePrompt}`,
  ].join(" ");
}

function asDataUrl(image: { b64_json?: string | null; url?: string | null }): string | null {
  if (image.b64_json) {
    return `data:image/png;base64,${image.b64_json}`;
  }

  if (image.url) {
    return image.url;
  }

  return null;
}

function toDataUrl(bytes: ArrayBuffer, mimeType: string): string {
  const base64 = Buffer.from(bytes).toString("base64");
  return `data:${mimeType};base64,${base64}`;
}

function toBase64(bytes: ArrayBuffer): string {
  return Buffer.from(bytes).toString("base64");
}

function toGoogleModelId(model: string): string {
  return model.startsWith(GOOGLE_PREFIX) ? model.slice(GOOGLE_PREFIX.length) : model;
}

function extractOpenRouterImage(result: unknown): string | null {
  if (!result || typeof result !== "object") {
    return null;
  }

  const message = (result as { choices?: Array<{ message?: { images?: Array<{ image_url?: { url?: string } }> } }> })
    .choices?.[0]?.message;

  const url = message?.images?.[0]?.image_url?.url;
  return typeof url === "string" ? url : null;
}

function extractOpenRouterText(result: unknown): string | null {
  if (!result || typeof result !== "object") {
    return null;
  }

  const content = (result as { choices?: Array<{ message?: { content?: string } }> }).choices?.[0]?.message?.content;
  return typeof content === "string" ? content.trim() : null;
}

function extractGoogleImage(result: unknown): string | null {
  if (!result || typeof result !== "object") {
    return null;
  }

  const parts = (result as { candidates?: Array<{ content?: { parts?: Array<unknown> } }> }).candidates?.[0]?.content?.parts;
  if (!Array.isArray(parts)) {
    return null;
  }

  for (const part of parts) {
    if (!part || typeof part !== "object") {
      continue;
    }

    const inlineData = (part as { inlineData?: { data?: string; mimeType?: string }; inline_data?: { data?: string; mime_type?: string } })
      .inlineData;
    const snakeInline = (part as { inline_data?: { data?: string; mime_type?: string } }).inline_data;
    const data = inlineData?.data || snakeInline?.data;
    const mime = inlineData?.mimeType || snakeInline?.mime_type || "image/png";
    if (data) {
      return `data:${mime};base64,${data}`;
    }
  }

  return null;
}

async function generateWithOpenRouter(params: {
  apiKey: string;
  model: string;
  prompt: string;
  referenceDataUrl?: string;
}): Promise<string | null> {
  const referer = process.env.OPENROUTER_SITE_URL?.trim() || "http://localhost:3000";
  const title = process.env.OPENROUTER_APP_NAME?.trim() || "Thumbly";

  const content = params.referenceDataUrl
    ? [
        { type: "text", text: params.prompt },
        { type: "image_url", image_url: { url: params.referenceDataUrl } },
      ]
    : params.prompt;

  const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${params.apiKey}`,
      "Content-Type": "application/json",
      "HTTP-Referer": referer,
      "X-Title": title,
    },
    body: JSON.stringify({
      model: params.model,
      messages: [{ role: "user", content }],
      modalities: ["image"],
    }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `OpenRouter request failed (${response.status}).`);
  }

  const result = (await response.json()) as unknown;
  return extractOpenRouterImage(result);
}

async function generateWithGoogleAIStudio(params: {
  apiKey: string;
  model: string;
  prompt: string;
  referenceBytes?: ArrayBuffer;
  referenceMimeType?: string;
}): Promise<string | null> {
  const modelId = toGoogleModelId(params.model);
  const parts: Array<Record<string, unknown>> = [{ text: params.prompt }];

  if (params.referenceBytes && params.referenceMimeType) {
    parts.push({
      inline_data: {
        mime_type: params.referenceMimeType,
        data: toBase64(params.referenceBytes),
      },
    });
  }

  const response = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/${modelId}:generateContent?key=${params.apiKey}`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        contents: [{ parts }],
        generationConfig: {
          responseModalities: ["TEXT", "IMAGE"],
        },
      }),
    },
  );

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Google AI Studio request failed (${response.status}).`);
  }

  const result = (await response.json()) as unknown;
  return extractGoogleImage(result);
}

async function refinePromptWithOpenRouter(params: {
  apiKey: string;
  model: string;
  prompt: string;
}): Promise<string | null> {
  const referer = process.env.OPENROUTER_SITE_URL?.trim() || "http://localhost:3000";
  const title = process.env.OPENROUTER_APP_NAME?.trim() || "Thumbly";

  const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${params.apiKey}`,
      "Content-Type": "application/json",
      "HTTP-Referer": referer,
      "X-Title": title,
    },
    body: JSON.stringify({
      model: params.model,
      messages: [
        {
          role: "system",
          content:
            "Rewrite user prompts for image generation. Keep output concise, visual, and production-ready. Return only the rewritten prompt.",
        },
        {
          role: "user",
          content: params.prompt,
        },
      ],
    }),
  });

  if (!response.ok) {
    return null;
  }

  const result = (await response.json()) as unknown;
  return extractOpenRouterText(result);
}

function normalizeErrorMessage(raw: unknown): string {
  const text = typeof raw === "string" ? raw : raw instanceof Error ? raw.message : "Generation failed.";
  const compact = text.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();

  if (!compact) {
    return "Generation failed.";
  }

  if (compact.toLowerCase().includes("incorrect api key")) {
    return "Invalid API key for this provider.";
  }

  return compact.length > 180 ? `${compact.slice(0, 180)}...` : compact;
}

function line(data: unknown): Uint8Array {
  return new TextEncoder().encode(`${JSON.stringify(data)}\n`);
}

export async function POST(request: Request) {
  const formData = await request.formData();
  const prompt = String(formData.get("prompt") ?? "").trim();
  const aspect = String(formData.get("aspect") ?? "16:9").trim();
  const model = String(formData.get("model") ?? DEFAULT_GOOGLE_IMAGE_MODEL).trim();
  const referenceImage = formData.get("referenceImage");

  if (model.includes("embed")) {
    return new Response(
      JSON.stringify({
        error: `${model} is an embedding model. Thumbnail generation requires an image-generation model.`,
      }),
      {
        status: 400,
        headers: { "Content-Type": "application/json" },
      },
    );
  }

  if (!prompt) {
    return new Response(JSON.stringify({ error: "Prompt is required." }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  const provider = resolveProviderConfig(model);
  if (!provider) {
    return new Response(JSON.stringify({ error: "Missing API key. Configure OPENAI_API_KEY or OpenRouter keys." }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }

  if (!provider.apiKey) {
    return new Response(JSON.stringify({ error: `Missing API key in .env: ${provider.missingEnvHint}` }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }

  const openai = provider.mode === "openai" ? getOpenAIClient(provider.apiKey) : null;
  const effectiveModel = provider.mode === "openrouter" && model === "gpt-image-1" ? "openai/gpt-image-1" : model;
  const size = normalizeSize(aspect);

  const referenceMeta =
    referenceImage instanceof File && referenceImage.size > 0
      ? {
          name: referenceImage.name,
          type: referenceImage.type || "image/png",
          bytes: await referenceImage.arrayBuffer(),
        }
      : null;

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const send = (payload: unknown) => controller.enqueue(line(payload));

      const tasks = VARIANTS.map(async (variant, index) => {
        try {
          const basePrompt = buildPrompt(prompt, variant, aspect);
          let image: string | null = null;

          if (provider.mode === "openrouter") {
            let finalPrompt = basePrompt;
            let imageModel = effectiveModel;

            if (effectiveModel === NEMOTRON_FREE_MODEL) {
              const refined = await refinePromptWithOpenRouter({
                apiKey: provider.apiKey,
                model: NEMOTRON_FREE_MODEL,
                prompt: basePrompt,
              });
              finalPrompt = refined || basePrompt;
              imageModel = process.env.FREE_IMAGE_FALLBACK_MODEL?.trim() || DEFAULT_IMAGE_MODEL;
            }

            if (imageModel.startsWith(GOOGLE_PREFIX)) {
              const googleKey = process.env.GOOGLE_AI_STUDIO_API_KEY?.trim();
              if (!googleKey) {
                throw new Error("Missing GOOGLE_AI_STUDIO_API_KEY for Google image fallback model.");
              }
              image = await generateWithGoogleAIStudio({
                apiKey: googleKey,
                model: imageModel,
                prompt: finalPrompt,
                referenceBytes: referenceMeta?.bytes,
                referenceMimeType: referenceMeta?.type,
              });
            } else {
              image = await generateWithOpenRouter({
                apiKey: provider.apiKey,
                model: imageModel,
                prompt: finalPrompt,
                referenceDataUrl: referenceMeta ? toDataUrl(referenceMeta.bytes, referenceMeta.type) : undefined,
              });
            }
          } else if (provider.mode === "google") {
            image = await generateWithGoogleAIStudio({
              apiKey: provider.apiKey,
              model: effectiveModel,
              prompt: basePrompt,
              referenceBytes: referenceMeta?.bytes,
              referenceMimeType: referenceMeta?.type,
            });
          } else if (openai) {
            const finalPrompt = basePrompt;
            const imageResult = referenceMeta
              ? await openai.images.edit({
                  model: effectiveModel,
                  image: new File([referenceMeta.bytes], referenceMeta.name, { type: referenceMeta.type }),
                  prompt: finalPrompt,
                  size,
                })
              : await openai.images.generate({
                  model: effectiveModel,
                  prompt: finalPrompt,
                  size,
                });

            const data = imageResult.data?.[0];
            image = data ? asDataUrl(data) : null;
          }

          if (!image) {
            send({
              type: "result",
              index,
              style: variant.label,
              error: "Image generation returned no usable data.",
            });
            return;
          }

          send({
            type: "result",
            index,
            style: variant.label,
            image,
          });
        } catch (error) {
          send({
            type: "result",
            index,
            style: variant.label,
            error: normalizeErrorMessage(error),
          });
        }
      });

      await Promise.allSettled(tasks);
      send({ type: "complete" });
      controller.close();
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "application/x-ndjson; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
    },
  });
}
