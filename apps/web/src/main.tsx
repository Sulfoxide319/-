import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { Download, FileAudio, Mic, Pause, Play, Plus, RefreshCw, Scissors, Trash2, Upload, Wand2 } from "lucide-react";
import "./styles.css";

const API = "";
const DEFAULT_WHISPER_PROMPT = "这是中文语音拼接测试，请准确识别人名、短语和标点。";

type Phrase = {
  id: string;
  text: string;
  start: number;
  end: number;
  segments?: Array<{ start: number; end: number }>;
  removedGaps?: Array<{ start: number; end: number; reason?: string }>;
  boundaryCuts?: Array<{ at: number; with?: string; reason?: string }>;
  anchor?: { start: number; end: number; source?: string };
  scores?: {
    noiseFloorDb?: number;
    speechPeakDb?: number;
    onsetThresholdDb?: number;
    releaseThresholdDb?: number;
    vadSource?: string;
    vadAvailable?: boolean;
  };
  ownership?: string;
  pauseAfterMs?: number;
  kind: "clip" | "pause";
  quality: "good" | "warn" | "bad";
  source?: string;
};

type Recording = {
  id: string;
  name: string;
  audioUrl: string;
  text: string;
  duration: number;
  words?: unknown[];
  phrases: Phrase[];
  engine: string;
  engineNote?: string;
  whisperModel?: string;
  enableBoundaryRefine?: boolean;
  enablePhraseMerge?: boolean;
  textPostprocessMode?: string;
};

type WhisperModelPreset = "small" | "medium" | "large-v3" | "custom";

type RenderSettings = {
  enableAudioPostprocess: boolean;
  marginMs: number;
  enableGainNormalize: boolean;
  targetDbfs: number;
  maxGainDb: number;
  enableClipFade: boolean;
  fadeMs: number;
  enableCrossfade: boolean;
  crossfadeMs: number;
  enableFinalNormalize: boolean;
};

const DEFAULT_RENDER_SETTINGS: RenderSettings = {
  enableAudioPostprocess: false,
  marginMs: 0,
  enableGainNormalize: false,
  targetDbfs: -18,
  maxGainDb: 8,
  enableClipFade: false,
  fadeMs: 0,
  enableCrossfade: false,
  crossfadeMs: 0,
  enableFinalNormalize: false,
};

type TrackItem =
  | { id: string; type: "clip"; recordingId: string; phraseId: string; text: string; recordingName: string }
  | { id: string; type: "pause"; durationMs: number; text: string };

type Health = {
  ok: boolean;
  whisperxAvailable: boolean;
  cudaAvailable: boolean;
  torchVersion?: string | null;
};

function newItemId() {
  return `item_${Math.random().toString(16).slice(2)}_${Date.now()}`;
}

function phraseSegments(phrase: Phrase) {
  if (phrase.segments?.length) {
    return phrase.segments.filter((segment) => Number.isFinite(segment.start) && Number.isFinite(segment.end) && segment.end > segment.start);
  }
  return [{ start: phrase.start, end: phrase.end }];
}

async function responseMessage(res: Response) {
  const text = await res.text();
  if (!text) return res.statusText || "请求失败";
  try {
    const payload = JSON.parse(text);
    return payload.detail || payload.message || text;
  } catch {
    return text;
  }
}

function App() {
  const [recordings, setRecordings] = useState<Recording[]>([]);
  const [outputItems, setOutputItems] = useState<TrackItem[]>([]);
  const [manualText, setManualText] = useState("");
  const [autoTranscribe, setAutoTranscribe] = useState(false);
  const [whisperModelPreset, setWhisperModelPreset] = useState<WhisperModelPreset>("medium");
  const [customWhisperModel, setCustomWhisperModel] = useState("");
  const [whisperPrompt, setWhisperPrompt] = useState(DEFAULT_WHISPER_PROMPT);
  const [enableBoundaryRefine, setEnableBoundaryRefine] = useState(false);
  const [enablePhraseMerge, setEnablePhraseMerge] = useState(false);
  const [renderSettings] = useState<RenderSettings>(DEFAULT_RENDER_SETTINGS);
  const [health, setHealth] = useState<Health | null>(null);
  const [busy, setBusy] = useState("");
  const [message, setMessage] = useState("");
  const [selected, setSelected] = useState<{ recordingId: string; phrase: Phrase } | null>(null);
  const [renderResult, setRenderResult] = useState<{ wavUrl: string; mp3Url: string; durationMs: number } | null>(null);
  const [recordingNow, setRecordingNow] = useState(false);
  const [micLevel, setMicLevel] = useState(0);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const audioContextRef = useRef<AudioContext | null>(null);
  const levelFrameRef = useRef<number | null>(null);

  useEffect(() => {
    void refresh();
    void fetch(`${API}/api/health`).then((res) => res.json()).then(setHealth).catch(() => setHealth(null));
    return () => stopLevelMeter();
  }, []);

  async function refresh() {
    const res = await fetch(`${API}/api/recordings`);
    setRecordings(await res.json());
  }

  async function clearData() {
    if (!window.confirm("清空所有录音、转写和渲染结果？")) return;
    setBusy("clear");
    const res = await fetch(`${API}/api/data`, { method: "DELETE" });
    if (!res.ok) {
      setMessage(await responseMessage(res));
    } else {
      const result = await res.json();
      setRecordings([]);
      setOutputItems([]);
      setSelected(null);
      setRenderResult(null);
      setMessage(`已清理：录音 ${result.deleted.recordings}，转写 ${result.deleted.transcriptions}，渲染 ${result.deleted.renders}`);
    }
    setBusy("");
  }

  function selectedWhisperModel() {
    return whisperModelPreset === "custom" ? customWhisperModel.trim() || "medium" : whisperModelPreset;
  }

  function appendTranscriptionSettings(form: FormData) {
    form.append("manualText", manualText);
    form.append("autoTranscribe", String(autoTranscribe));
    form.append("whisperModel", selectedWhisperModel());
    form.append("whisperPrompt", whisperPrompt.trim() || DEFAULT_WHISPER_PROMPT);
    form.append("enableTextPostprocess", "false");
    form.append("enableBoundaryRefine", String(enableBoundaryRefine));
    form.append("enablePhraseMerge", String(enablePhraseMerge));
  }

  function phraseMode(recording: Recording) {
    const sources = new Set(recording.phrases.map((phrase) => phrase.source || "unknown"));
    if (recording.textPostprocessMode) return recording.textPostprocessMode;
    if (sources.has("phrase-merged")) return "phrase-merged";
    if (sources.has("boundary-refined")) return "boundary-refined";
    if (sources.has("whisperx-word")) return "raw";
    if (sources.has("whisperx-postprocess") || sources.has("whisperx")) return "legacy-postprocess";
    if (sources.has("fallback")) return "fallback";
    return "unknown";
  }

  function sourceSummary(recording: Recording) {
    const counts = recording.phrases.reduce<Record<string, number>>((acc, phrase) => {
      const source = phrase.source || "unknown";
      acc[source] = (acc[source] || 0) + 1;
      return acc;
    }, {});
    return Object.entries(counts).map(([source, count]) => `${source}:${count}`).join(" / ");
  }

  function busyText() {
    if (busy === "upload") return "正在上传并转写...";
    if (busy === "transcribe") return "正在重新转写...";
    if (busy === "render") return "正在生成音频...";
    if (busy === "clear") return "正在清理数据...";
    return "正在处理...";
  }

  async function uploadFile(file: File) {
    setBusy("upload");
    setMessage("正在保存并生成短语块...");
    const form = new FormData();
    form.append("file", file);
    appendTranscriptionSettings(form);
    const res = await fetch(`${API}/api/recordings`, { method: "POST", body: form });
    if (!res.ok) {
      setMessage(await responseMessage(res));
    } else {
      const recording = await res.json();
      setRecordings((prev) => [recording, ...prev]);
      setManualText("");
      setMessage(recording.engineNote || "录音已生成短语块");
    }
    setBusy("");
  }

  async function retranscribe(recordingId: string) {
    setBusy("transcribe");
    setMessage("正在重新转写...");
    const form = new FormData();
    appendTranscriptionSettings(form);
    const res = await fetch(`${API}/api/recordings/${recordingId}/transcribe`, { method: "POST", body: form });
    if (!res.ok) {
      setMessage(await responseMessage(res));
    } else {
      const updated = await res.json();
      setRecordings((prev) => prev.map((recording) => (recording.id === recordingId ? updated : recording)));
      setMessage(updated.engineNote || "重新转写完成");
    }
    setBusy("");
  }

  async function startRecording() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    startLevelMeter(stream);
    const recorder = new MediaRecorder(stream);
    chunksRef.current = [];
    recorder.ondataavailable = (event) => chunksRef.current.push(event.data);
    recorder.onstop = async () => {
      stream.getTracks().forEach((track) => track.stop());
      const blob = new Blob(chunksRef.current, { type: "audio/webm" });
      await uploadFile(new File([blob], `browser-recording-${Date.now()}.webm`, { type: "audio/webm" }));
    };
    recorderRef.current = recorder;
    recorder.start();
    setRecordingNow(true);
  }

  function stopRecording() {
    recorderRef.current?.stop();
    stopLevelMeter();
    setRecordingNow(false);
  }

  function startLevelMeter(stream: MediaStream) {
    stopLevelMeter();
    const audioWindow = window as typeof window & { webkitAudioContext?: typeof AudioContext };
    const AudioContextClass = audioWindow.AudioContext || audioWindow.webkitAudioContext;
    if (!AudioContextClass) return;
    const audioContext = new AudioContextClass();
    const analyser = audioContext.createAnalyser();
    analyser.fftSize = 1024;
    analyser.smoothingTimeConstant = 0.82;
    audioContext.createMediaStreamSource(stream).connect(analyser);
    const samples = new Uint8Array(analyser.fftSize);
    audioContextRef.current = audioContext;
    const update = () => {
      analyser.getByteTimeDomainData(samples);
      let sum = 0;
      for (const sample of samples) {
        const centered = (sample - 128) / 128;
        sum += centered * centered;
      }
      setMicLevel(Math.min(1, Math.sqrt(sum / samples.length) * 4));
      levelFrameRef.current = window.requestAnimationFrame(update);
    };
    update();
  }

  function stopLevelMeter() {
    if (levelFrameRef.current !== null) window.cancelAnimationFrame(levelFrameRef.current);
    levelFrameRef.current = null;
    if (audioContextRef.current) void audioContextRef.current.close();
    audioContextRef.current = null;
    setMicLevel(0);
  }

  function dragPayload(recording: Recording, phrase: Phrase) {
    return JSON.stringify({ type: "clip", recordingId: recording.id, phraseId: phrase.id, text: phrase.text, recordingName: recording.name });
  }

  function onPhraseDrag(event: React.DragEvent, recording: Recording, phrase: Phrase) {
    event.dataTransfer.setData("application/json", dragPayload(recording, phrase));
  }

  function insertOutputItem(data: Record<string, unknown>, index: number) {
    setOutputItems((prev) => {
      const next = [...prev];
      if (typeof data.reorderId === "string") {
        const from = next.findIndex((item) => item.id === data.reorderId);
        if (from < 0) return prev;
        const [moved] = next.splice(from, 1);
        next.splice(Math.max(0, Math.min(next.length, from < index ? index - 1 : index)), 0, moved);
        return next;
      }
      next.splice(Math.max(0, Math.min(next.length, index)), 0, { id: newItemId(), ...data } as TrackItem);
      return next;
    });
  }

  function onOutputDrop(event: React.DragEvent) {
    event.preventDefault();
    const raw = event.dataTransfer.getData("application/json");
    if (raw) insertOutputItem(JSON.parse(raw), outputItems.length);
  }

  function outputInsertDrop(event: React.DragEvent, index: number) {
    event.preventDefault();
    event.stopPropagation();
    const raw = event.dataTransfer.getData("application/json");
    if (raw) insertOutputItem(JSON.parse(raw), index);
  }

  function insertPause(durationMs = 260) {
    setOutputItems((prev) => [...prev, { id: newItemId(), type: "pause", durationMs, text: `${durationMs}ms` }]);
  }

  async function render() {
    setBusy("render");
    setMessage("正在拼接...");
    const payload = {
      items: outputItems.map((item) => (item.type === "pause" ? { type: "pause", durationMs: item.durationMs } : { type: "clip", recordingId: item.recordingId, phraseId: item.phraseId })),
      settings: renderSettings,
    };
    const res = await fetch(`${API}/api/render`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    if (!res.ok) {
      setMessage(await responseMessage(res));
    } else {
      const result = await res.json();
      setRenderResult(result);
      setMessage(`已生成 ${Math.round(result.durationMs / 10) / 100}s 音频`);
    }
    setBusy("");
  }

  async function patchPhrase(recordingId: string, phrase: Phrase, patch: Partial<Phrase>) {
    const res = await fetch(`${API}/api/recordings/${recordingId}/phrases/${phrase.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    });
    const updated = await res.json();
    setRecordings((prev) =>
      prev.map((recording) =>
        recording.id === recordingId ? { ...recording, phrases: recording.phrases.map((item) => (item.id === phrase.id ? updated : item)) } : recording
      )
    );
    setSelected({ recordingId, phrase: updated });
  }

  const outputText = useMemo(() => outputItems.map((item) => item.text).join(""), [outputItems]);

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <h1>语音活字印刷</h1>
          <p>{health ? envText(health) : "正在检查本地环境..."}</p>
        </div>
        <div className="actions">
          <label className="toggle">
            <input type="checkbox" checked={autoTranscribe} onChange={(event) => setAutoTranscribe(event.target.checked)} />
            <span>WhisperX</span>
          </label>
          <button onClick={recordingNow ? stopRecording : startRecording} className={recordingNow ? "danger" : ""}>
            {recordingNow ? <Pause size={18} /> : <Mic size={18} />}
            {recordingNow ? "停止" : "录音"}
          </button>
          <div className={`mic-meter ${recordingNow ? "active" : ""}`} aria-label="麦克风输入音量">
            <span className="mic-meter-fill" style={{ width: `${Math.round(micLevel * 100)}%` }} />
            <strong>{recordingNow ? `${Math.round(micLevel * 100)}%` : "0%"}</strong>
          </div>
          <label className="button">
            <Upload size={18} />
            上传
            <input hidden type="file" accept="audio/*" onChange={(event) => event.target.files?.[0] && uploadFile(event.target.files[0])} />
          </label>
          <button className="danger" onClick={() => void clearData()} disabled={!!busy}>
            <Trash2 size={18} />
            清理数据
          </button>
          <button onClick={() => void refresh()}>
            <RefreshCw size={18} />
            刷新
          </button>
        </div>
      </header>

      {busy && (
        <div className="processing">
          <span>{busyText()}</span>
          <div className="processing-track" aria-label={busyText()}>
            <div className="processing-bar" />
          </div>
        </div>
      )}

      <section className="panel settings-panel">
        <div className="panel-head settings-head">
          <div>
            <h2>测试配置</h2>
            <p>默认保留 WhisperX 原始切分；需要优化时显式打开边界优化和连词合并。</p>
          </div>
          <div className="settings-grid">
            <label className="toggle">
              <input type="checkbox" checked={autoTranscribe} onChange={(event) => setAutoTranscribe(event.target.checked)} />
              <span>WhisperX</span>
            </label>
            <label>
              模型
              <select value={whisperModelPreset} onChange={(event) => setWhisperModelPreset(event.target.value as WhisperModelPreset)}>
                <option value="small">快速 small</option>
                <option value="medium">平衡 medium</option>
                <option value="large-v3">高精度 large-v3</option>
                <option value="custom">自定义</option>
              </select>
            </label>
            {whisperModelPreset === "custom" && (
              <label>
                自定义模型名
                <input value={customWhisperModel} onChange={(event) => setCustomWhisperModel(event.target.value)} placeholder="medium" />
              </label>
            )}
            <label className="wide">
              WhisperX initial prompt
              <input value={whisperPrompt} onChange={(event) => setWhisperPrompt(event.target.value)} placeholder={DEFAULT_WHISPER_PROMPT} />
            </label>
            <label className="toggle">
              <input type="checkbox" checked={enableBoundaryRefine} onChange={(event) => setEnableBoundaryRefine(event.target.checked)} />
              <span>边界优化</span>
            </label>
            <label className="toggle">
              <input type="checkbox" checked={enablePhraseMerge} onChange={(event) => setEnablePhraseMerge(event.target.checked)} />
              <span>连词合并</span>
            </label>
          </div>
        </div>
      </section>

      <main className="workspace">
        <section className="panel source-panel">
          <div className="panel-head">
            <div>
              <h2>录音轨</h2>
              <p>上传前可填转写文本；未启用 WhisperX 时按文本均分短语。</p>
            </div>
            <textarea value={manualText} onChange={(event) => setManualText(event.target.value)} placeholder="可选：输入这段录音的文字，例如：我是张三，他是李四" />
          </div>
          <div className="recording-list">
            {recordings.map((recording) => (
              <article className="recording" key={recording.id}>
                <div className="recording-title">
                  <FileAudio size={18} />
                  <strong>{recording.name}</strong>
                  <span title={recording.engineNote}>
                    {recording.engine}
                    {recording.whisperModel ? ` / ${recording.whisperModel}` : ""}
                    {` / ${phraseMode(recording)}`}
                  </span>
                  <audio src={recording.audioUrl} controls />
                </div>
                <div className="recording-tools">
                  <small>
                    words {recording.words?.length ?? 0} / phrases {recording.phrases.length} / {sourceSummary(recording)}
                  </small>
                  <button onClick={() => void retranscribe(recording.id)} disabled={!!busy}>
                    <RefreshCw size={14} />
                    重新转写
                  </button>
                </div>
                <RecordingWaveform recording={recording} onSelect={(phrase) => setSelected({ recordingId: recording.id, phrase })} />
                <div className="transcript">{recording.text}</div>
                <div className="phrases">
                  {recording.phrases.map((phrase) =>
                    phrase.kind === "pause" ? (
                      <button className="phrase pause" key={phrase.id} onClick={() => insertPause(phrase.pauseAfterMs ?? 260)}>
                        {phrase.text}
                      </button>
                    ) : (
                      <button
                        className={`phrase ${phrase.quality}`}
                        key={phrase.id}
                        draggable
                        onDragStart={(event) => onPhraseDrag(event, recording, phrase)}
                        onClick={() => setSelected({ recordingId: recording.id, phrase })}
                        title={`${phrase.start}s - ${phrase.end}s / ${phrase.source || "unknown"}`}
                      >
                        {phrase.text}
                      </button>
                    )
                  )}
                </div>
              </article>
            ))}
            {!recordings.length && <div className="empty">先上传或录制一段音频。</div>}
          </div>
        </section>

        <section className="panel output-panel">
          <div className="panel-head output-head">
            <div>
              <h2>输出轨</h2>
              <p>{outputText || "把上方短语拖到这里，或插入停顿。"}</p>
            </div>
            <div className="actions">
              <button onClick={() => insertPause(180)}>
                <Plus size={18} />
                短停顿
              </button>
              <button onClick={() => insertPause(420)}>
                <Plus size={18} />
                长停顿
              </button>
              <button onClick={() => setOutputItems([])}>
                <Trash2 size={18} />
                清空
              </button>
              <button className="primary" onClick={render} disabled={!outputItems.length || !!busy}>
                <Wand2 size={18} />
                生成
              </button>
            </div>
          </div>
          <div className="dropzone" onDragOver={(event) => event.preventDefault()} onDrop={onOutputDrop}>
            {outputItems.map((item, index) => (
              <React.Fragment key={item.id}>
                <div className="insert-slot" onDragOver={(event) => event.preventDefault()} onDrop={(event) => outputInsertDrop(event, index)} title="拖到这里插入" />
                <div className={`output-item ${item.type}`} draggable onDragStart={(event) => event.dataTransfer.setData("application/json", JSON.stringify({ reorderId: item.id }))}>
                  <span>{item.text}</span>
                  {item.type === "clip" && <small>{item.recordingName}</small>}
                  <button onClick={() => setOutputItems((prev) => prev.filter((candidate) => candidate.id !== item.id))}>
                    <Trash2 size={14} />
                  </button>
                </div>
              </React.Fragment>
            ))}
            {!!outputItems.length && <div className="insert-slot end" onDragOver={(event) => event.preventDefault()} onDrop={(event) => outputInsertDrop(event, outputItems.length)} title="拖到这里插入" />}
            {!outputItems.length && <div className="empty">输出轨为空</div>}
          </div>
          <OutputWaveTimeline items={outputItems} recordings={recordings} onRemove={(id) => setOutputItems((prev) => prev.filter((candidate) => candidate.id !== id))} onInsertDrop={outputInsertDrop} />
          {renderResult && (
            <div className="render-result">
              <audio src={renderResult.wavUrl} controls />
              <a href={renderResult.wavUrl} download>
                <Download size={18} />
                WAV
              </a>
              <a href={renderResult.mp3Url} download>
                <Download size={18} />
                MP3
              </a>
            </div>
          )}
        </section>

        <aside className="panel inspector">
          <h2>短语边界</h2>
          {selected ? (
            <PhraseInspector
              recording={recordings.find((recording) => recording.id === selected.recordingId)}
              phrase={selected.phrase}
              onPatch={(patch) => patchPhrase(selected.recordingId, selected.phrase, patch)}
            />
          ) : (
            <div className="empty">点击一个短语后可微调起止点。</div>
          )}
          <div className="status">{busy ? "处理中..." : message}</div>
        </aside>
      </main>
    </div>
  );
}

function OutputWaveTimeline({
  items,
  recordings,
  onRemove,
  onInsertDrop,
}: {
  items: TrackItem[];
  recordings: Recording[];
  onRemove: (id: string) => void;
  onInsertDrop: (event: React.DragEvent, index: number) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [status, setStatus] = useState("loading");
  const [segments, setSegments] = useState<Array<{ item: TrackItem; left: number; width: number; durationMs: number }>>([]);
  const [totalMs, setTotalMs] = useState(0);

  useEffect(() => {
    let cancelled = false;
    let audioContext: AudioContext | null = null;
    let resizeObserver: ResizeObserver | null = null;

    async function buildOutputWave() {
      const canvas = canvasRef.current;
      if (!canvas || !items.length) return;
      setStatus("loading");
      try {
        const audioWindow = window as typeof window & { webkitAudioContext?: typeof AudioContext };
        const AudioContextClass = audioWindow.AudioContext || audioWindow.webkitAudioContext;
        if (!AudioContextClass) {
          setStatus("unsupported");
          return;
        }
        audioContext = new AudioContextClass();
        const buffers = new Map<string, AudioBuffer>();
        const bufferFor = async (recording: Recording) => {
          const cached = buffers.get(recording.id);
          if (cached) return cached;
          const res = await fetch(recording.audioUrl);
          const data = await res.arrayBuffer();
          const buffer = await audioContext!.decodeAudioData(data.slice(0));
          buffers.set(recording.id, buffer);
          return buffer;
        };

        const sampleRate = audioContext.sampleRate;
        const chunks: Float32Array[] = [];
        const nextSegments: Array<{ item: TrackItem; startSample: number; sampleCount: number; durationMs: number }> = [];
        let totalSamples = 0;

        for (const item of items) {
          let chunk = new Float32Array(0);
          if (item.type === "pause") {
            chunk = new Float32Array(Math.max(1, Math.round((item.durationMs / 1000) * sampleRate)));
          } else {
            const recording = recordings.find((candidate) => candidate.id === item.recordingId);
            const phrase = recording?.phrases.find((candidate) => candidate.id === item.phraseId);
            if (recording && phrase) {
              const buffer = await bufferFor(recording);
              const channel = buffer.getChannelData(0);
              const slices = phraseSegments(phrase).map((segment) => {
                const start = Math.max(0, Math.floor(segment.start * buffer.sampleRate));
                const end = Math.min(channel.length, Math.ceil(segment.end * buffer.sampleRate));
                return channel.slice(start, Math.max(start + 1, end));
              });
              const sampleCount = slices.reduce((sum, slice) => sum + slice.length, 0);
              chunk = new Float32Array(Math.max(1, sampleCount));
              let cursor = 0;
              for (const slice of slices) {
                chunk.set(slice, cursor);
                cursor += slice.length;
              }
            }
          }
          chunks.push(chunk);
          nextSegments.push({ item, startSample: totalSamples, sampleCount: chunk.length, durationMs: Math.round((chunk.length / sampleRate) * 1000) });
          totalSamples += chunk.length;
        }

        const output = new Float32Array(Math.max(1, totalSamples));
        let writeCursor = 0;
        for (const chunk of chunks) {
          output.set(chunk, writeCursor);
          writeCursor += chunk.length;
        }

        const visibleSegments = nextSegments.map((segment) => ({
          item: segment.item,
          durationMs: segment.durationMs,
          left: totalSamples ? (segment.startSample / totalSamples) * 100 : 0,
          width: totalSamples ? Math.max(0.8, (segment.sampleCount / totalSamples) * 100) : 0,
        }));

        const redraw = () => {
          if (cancelled) return;
          const parentWidth = canvas.parentElement?.clientWidth || 720;
          const dpr = window.devicePixelRatio || 1;
          const width = Math.max(320, Math.floor(parentWidth));
          const height = 118;
          canvas.width = Math.floor(width * dpr);
          canvas.height = Math.floor(height * dpr);
          canvas.style.width = `${width}px`;
          canvas.style.height = `${height}px`;
          const ctx = canvas.getContext("2d");
          if (!ctx) return;
          ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
          ctx.clearRect(0, 0, width, height);
          ctx.fillStyle = "#f7faf8";
          ctx.fillRect(0, 0, width, height);
          ctx.strokeStyle = "#d9e0dc";
          ctx.strokeRect(0.5, 0.5, width - 1, height - 1);

          for (const segment of visibleSegments) {
            const x = (segment.left / 100) * width;
            const w = (segment.width / 100) * width;
            ctx.fillStyle = segment.item.type === "pause" ? "rgba(65, 82, 108, 0.11)" : "rgba(21, 111, 91, 0.12)";
            ctx.fillRect(x, 0, w, height);
            ctx.strokeStyle = segment.item.type === "pause" ? "rgba(65, 82, 108, 0.55)" : "rgba(21, 111, 91, 0.72)";
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
          }

          const samplesPerPixel = Math.max(1, Math.floor(output.length / width));
          const center = height * 0.54;
          const scale = height * 0.38;
          ctx.beginPath();
          ctx.strokeStyle = "#2f6759";
          ctx.lineWidth = 1;
          for (let x = 0; x < width; x += 1) {
            const start = x * samplesPerPixel;
            let min = 1;
            let max = -1;
            for (let i = 0; i < samplesPerPixel && start + i < output.length; i += 1) {
              const value = output[start + i];
              min = Math.min(min, value);
              max = Math.max(max, value);
            }
            ctx.moveTo(x + 0.5, center + min * scale);
            ctx.lineTo(x + 0.5, center + max * scale);
          }
          ctx.stroke();
        };

        if (cancelled) return;
        setSegments(visibleSegments);
        setTotalMs(Math.round((totalSamples / sampleRate) * 1000));
        redraw();
        resizeObserver = new ResizeObserver(redraw);
        if (canvas.parentElement) resizeObserver.observe(canvas.parentElement);
        setStatus("ready");
      } catch {
        if (!cancelled) setStatus("error");
      }
    }

    void buildOutputWave();
    return () => {
      cancelled = true;
      resizeObserver?.disconnect();
      void audioContext?.close();
    };
  }, [items, recordings]);

  if (!items.length) return null;

  return (
    <div className="output-timeline">
      <div className="output-timeline-head">
        <strong>输出音轨预览</strong>
        <span>{(totalMs / 1000).toFixed(2)}s</span>
      </div>
      <div className="output-wave">
        <canvas ref={canvasRef} />
        {status !== "ready" && <span className="output-wave-status">{status === "loading" ? "正在绘制输出波形..." : "输出波形不可用"}</span>}
        {segments.map((segment, index) => (
          <React.Fragment key={segment.item.id}>
            <div className="output-insert-marker" style={{ left: `${segment.left}%` }} onDragOver={(event) => event.preventDefault()} onDrop={(event) => onInsertDrop(event, index)} title="拖到这里插入" />
            <div
              className={`output-wave-item ${segment.item.type}`}
              draggable
              onDragStart={(event) => event.dataTransfer.setData("application/json", JSON.stringify({ reorderId: segment.item.id }))}
              style={{ left: `${segment.left}%`, width: `${segment.width}%` }}
              title={`${segment.item.text} / ${(segment.durationMs / 1000).toFixed(3)}s`}
            />
          </React.Fragment>
        ))}
        <div className="output-insert-marker end" style={{ left: "100%" }} onDragOver={(event) => event.preventDefault()} onDrop={(event) => onInsertDrop(event, items.length)} title="拖到这里插入" />
      </div>
      <div className="output-label-track">
        {segments.map((segment) => (
          <div
            className={`output-label-item ${segment.item.type}`}
            draggable
            key={segment.item.id}
            onDragStart={(event) => event.dataTransfer.setData("application/json", JSON.stringify({ reorderId: segment.item.id }))}
            style={{ left: `${segment.left}%`, width: `${segment.width}%` }}
            title={`${segment.item.text} / ${(segment.durationMs / 1000).toFixed(3)}s`}
          >
            <span>{segment.item.text}</span>
            <small>{(segment.durationMs / 1000).toFixed(2)}s</small>
            <button onClick={() => onRemove(segment.item.id)}>
              <Trash2 size={12} />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

function RecordingWaveform({ recording, onSelect }: { recording: Recording; onSelect: (phrase: Phrase) => void }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [status, setStatus] = useState("loading");
  const duration = recording.duration || 1;

  useEffect(() => {
    let cancelled = false;
    let audioContext: AudioContext | null = null;
    let resizeObserver: ResizeObserver | null = null;
    async function drawWaveform() {
      const canvas = canvasRef.current;
      if (!canvas) return;
      setStatus("loading");
      try {
        const res = await fetch(recording.audioUrl);
        const data = await res.arrayBuffer();
        const audioWindow = window as typeof window & { webkitAudioContext?: typeof AudioContext };
        const AudioContextClass = audioWindow.AudioContext || audioWindow.webkitAudioContext;
        if (!AudioContextClass) return setStatus("unsupported");
        audioContext = new AudioContextClass();
        const buffer = await audioContext.decodeAudioData(data.slice(0));
        if (cancelled) return;
        const redraw = () => {
          const parentWidth = canvas.parentElement?.clientWidth || 720;
          const dpr = window.devicePixelRatio || 1;
          const width = Math.max(320, Math.floor(parentWidth));
          const height = 92;
          canvas.width = Math.floor(width * dpr);
          canvas.height = Math.floor(height * dpr);
          canvas.style.width = `${width}px`;
          canvas.style.height = `${height}px`;
          const ctx = canvas.getContext("2d");
          if (!ctx) return;
          ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
          ctx.clearRect(0, 0, width, height);
          ctx.fillStyle = "#f7faf8";
          ctx.fillRect(0, 0, width, height);
          ctx.strokeStyle = "#d9e0dc";
          ctx.strokeRect(0.5, 0.5, width - 1, height - 1);

          const channel = buffer.getChannelData(0);
          const samplesPerPixel = Math.max(1, Math.floor(channel.length / width));
          const center = height * 0.48;
          const scale = height * 0.38;
          ctx.beginPath();
          ctx.strokeStyle = "#537367";
          ctx.lineWidth = 1;
          for (let x = 0; x < width; x += 1) {
            const start = x * samplesPerPixel;
            let min = 1;
            let max = -1;
            for (let i = 0; i < samplesPerPixel && start + i < channel.length; i += 1) {
              const value = channel[start + i];
              min = Math.min(min, value);
              max = Math.max(max, value);
            }
            ctx.moveTo(x + 0.5, center + min * scale);
            ctx.lineTo(x + 0.5, center + max * scale);
          }
          ctx.stroke();

          const sourceDuration = recording.duration || buffer.duration || 1;
          for (const phrase of recording.phrases) {
            if (phrase.kind !== "clip") continue;
            const startX = Math.max(0, Math.min(width, (phrase.start / sourceDuration) * width));
            const endX = Math.max(startX + 1, Math.min(width, (phrase.end / sourceDuration) * width));
            ctx.fillStyle = phrase.quality === "bad" ? "rgba(196, 77, 77, 0.2)" : phrase.quality === "warn" ? "rgba(224, 177, 63, 0.2)" : "rgba(21, 111, 91, 0.16)";
            ctx.fillRect(startX, 0, endX - startX, height);
            for (const segment of phraseSegments(phrase)) {
              const segmentStartX = Math.max(0, Math.min(width, (segment.start / sourceDuration) * width));
              const segmentEndX = Math.max(segmentStartX + 1, Math.min(width, (segment.end / sourceDuration) * width));
              ctx.fillStyle = phrase.quality === "bad" ? "rgba(196, 77, 77, 0.24)" : phrase.quality === "warn" ? "rgba(224, 177, 63, 0.24)" : "rgba(21, 111, 91, 0.24)";
              ctx.fillRect(segmentStartX, 0, segmentEndX - segmentStartX, height);
            }
            for (const gap of phrase.removedGaps || []) {
              const gapStartX = Math.max(0, Math.min(width, (gap.start / sourceDuration) * width));
              const gapEndX = Math.max(gapStartX + 1, Math.min(width, (gap.end / sourceDuration) * width));
              ctx.fillStyle = "rgba(90, 100, 110, 0.12)";
              ctx.fillRect(gapStartX, 0, gapEndX - gapStartX, height);
            }
            ctx.strokeStyle = phrase.quality === "bad" ? "rgba(196, 77, 77, 0.75)" : phrase.quality === "warn" ? "rgba(176, 133, 36, 0.75)" : "rgba(21, 111, 91, 0.7)";
            ctx.beginPath();
            ctx.moveTo(startX, 0);
            ctx.lineTo(startX, height);
            ctx.moveTo(endX, 0);
            ctx.lineTo(endX, height);
            ctx.stroke();
          }
        };
        redraw();
        resizeObserver = new ResizeObserver(redraw);
        if (canvas.parentElement) resizeObserver.observe(canvas.parentElement);
        setStatus("ready");
      } catch {
        if (!cancelled) setStatus("error");
      }
    }
    void drawWaveform();
    return () => {
      cancelled = true;
      resizeObserver?.disconnect();
      void audioContext?.close();
    };
  }, [recording.audioUrl, recording.duration, recording.phrases]);

  function handleClick(event: React.MouseEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    if (!canvas || !duration) return;
    const rect = canvas.getBoundingClientRect();
    const time = ((event.clientX - rect.left) / rect.width) * duration;
    const phrase = recording.phrases.find((item) => item.kind === "clip" && time >= item.start && time <= item.end);
    if (phrase) onSelect(phrase);
  }

  return (
    <div className="waveform">
      <canvas ref={canvasRef} onClick={handleClick} />
      <div className="waveform-words">
        {recording.phrases.filter((phrase) => phrase.kind === "clip").map((phrase) => {
          const left = Math.max(0, Math.min(100, (phrase.start / duration) * 100));
          const right = Math.max(left, Math.min(100, (phrase.end / duration) * 100));
          return (
            <button
              className={`waveform-word ${phrase.quality}`}
              draggable
              key={phrase.id}
              onClick={() => onSelect(phrase)}
              onDragStart={(event) => event.dataTransfer.setData("application/json", JSON.stringify({ type: "clip", recordingId: recording.id, phraseId: phrase.id, text: phrase.text, recordingName: recording.name }))}
              style={{ left: `${left}%`, width: `${Math.max(0.8, right - left)}%` }}
              title={`${phrase.text} / ${phrase.start}s - ${phrase.end}s`}
            >
              {phrase.text}
            </button>
          );
        })}
      </div>
      {status !== "ready" && <span>{status === "loading" ? "正在绘制波形..." : "波形不可用"}</span>}
    </div>
  );
}

function PhraseInspector({ recording, phrase, onPatch }: { recording?: Recording; phrase: Phrase; onPatch: (patch: Partial<Phrase>) => Promise<void> }) {
  const [text, setText] = useState(phrase.text);
  const [start, setStart] = useState(String(phrase.start));
  const [end, setEnd] = useState(String(phrase.end));

  useEffect(() => {
    setText(phrase.text);
    setStart(String(phrase.start));
    setEnd(String(phrase.end));
  }, [phrase]);

  const previewUrl = recording ? `${recording.audioUrl}#t=${Math.max(0, Number(start) || 0)},${Math.max(Number(start) || 0, Number(end) || 0)}` : "";

  return (
    <div className="inspector-form">
      <label>
        文本
        <input value={text} onChange={(event) => setText(event.target.value)} />
      </label>
      <label>
        起点
        <input type="number" step="0.01" value={start} onChange={(event) => setStart(event.target.value)} />
      </label>
      <label>
        终点
        <input type="number" step="0.01" value={end} onChange={(event) => setEnd(event.target.value)} />
      </label>
      <div className="nudge">
        <button onClick={() => setStart(String(Math.max(0, Number(start) - 0.02).toFixed(2)))}>
          <Scissors size={16} />
          左移
        </button>
        <button onClick={() => setEnd(String(Math.max(Number(start), Number(end) + 0.02).toFixed(2)))}>
          <Scissors size={16} />
          右扩
        </button>
      </div>
      {recording && <audio src={previewUrl} controls />}
      <div className="boundary-debug">
        {phrase.anchor && (
          <div>
            <strong>Anchor</strong>
            <span>
              {phrase.anchor.start}s - {phrase.anchor.end}s
            </span>
          </div>
        )}
        {!!phrase.segments?.length && (
          <div>
            <strong>Segments</strong>
            <span>{phrase.segments.map((segment) => `${segment.start}-${segment.end}`).join(" / ")}</span>
          </div>
        )}
        {!!phrase.removedGaps?.length && (
          <div>
            <strong>Gaps</strong>
            <span>{phrase.removedGaps.map((gap) => `${gap.start}-${gap.end}`).join(" / ")}</span>
          </div>
        )}
        {!!phrase.boundaryCuts?.length && (
          <div>
            <strong>Cuts</strong>
            <span>{phrase.boundaryCuts.map((cut) => `${cut.at}s`).join(" / ")}</span>
          </div>
        )}
        {phrase.scores && (
          <div>
            <strong>Signal</strong>
            <span>
              noise {phrase.scores.noiseFloorDb}dB / release {phrase.scores.releaseThresholdDb}dB / {phrase.scores.vadSource}
            </span>
          </div>
        )}
        {phrase.ownership && (
          <div>
            <strong>Ownership</strong>
            <span>{phrase.ownership}</span>
          </div>
        )}
      </div>
      <button className="primary" onClick={() => void onPatch({ text, start: Number(start), end: Number(end) })}>
        <Play size={18} />
        保存
      </button>
    </div>
  );
}

function envText(health: Health) {
  const whisper = health.whisperxAvailable ? "WhisperX 可用" : "WhisperX 未就绪";
  const cuda = health.cudaAvailable ? "CUDA 可用" : "CUDA 未连接";
  return `${whisper} · ${cuda}${health.torchVersion ? ` · torch ${health.torchVersion}` : ""}`;
}

createRoot(document.getElementById("root")!).render(<App />);
