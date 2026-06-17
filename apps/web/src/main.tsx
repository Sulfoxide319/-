import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Download,
  FileAudio,
  Mic,
  Pause,
  Play,
  Plus,
  RefreshCw,
  Scissors,
  Trash2,
  Upload,
  Wand2,
} from "lucide-react";
import "./styles.css";

const API = "";

type Phrase = {
  id: string;
  text: string;
  start: number;
  end: number;
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
  phrases: Phrase[];
  engine: string;
  engineNote?: string;
};

type TrackItem =
  | {
      id: string;
      type: "clip";
      recordingId: string;
      phraseId: string;
      text: string;
      recordingName: string;
    }
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

  async function uploadFile(file: File) {
    setBusy("upload");
    setMessage("正在保存并生成短语块...");
    const form = new FormData();
    form.append("file", file);
    form.append("manualText", manualText);
    form.append("autoTranscribe", String(autoTranscribe));
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

  async function startRecording() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    startLevelMeter(stream);
    const recorder = new MediaRecorder(stream);
    chunksRef.current = [];
    recorder.ondataavailable = (event) => chunksRef.current.push(event.data);
    recorder.onstop = async () => {
      stream.getTracks().forEach((track) => track.stop());
      const blob = new Blob(chunksRef.current, { type: "audio/webm" });
      const file = new File([blob], `browser-recording-${Date.now()}.webm`, { type: "audio/webm" });
      await uploadFile(file);
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
      const rms = Math.sqrt(sum / samples.length);
      setMicLevel(Math.min(1, rms * 4));
      levelFrameRef.current = window.requestAnimationFrame(update);
    };
    update();
  }

  function stopLevelMeter() {
    if (levelFrameRef.current !== null) {
      window.cancelAnimationFrame(levelFrameRef.current);
      levelFrameRef.current = null;
    }
    if (audioContextRef.current) {
      void audioContextRef.current.close();
      audioContextRef.current = null;
    }
    setMicLevel(0);
  }

  function onPhraseDrag(event: React.DragEvent, recording: Recording, phrase: Phrase) {
    event.dataTransfer.setData(
      "application/json",
      JSON.stringify({
        type: "clip",
        recordingId: recording.id,
        phraseId: phrase.id,
        text: phrase.text,
        recordingName: recording.name,
      })
    );
  }

  function onOutputDrop(event: React.DragEvent) {
    event.preventDefault();
    const raw = event.dataTransfer.getData("application/json");
    if (!raw) return;
    const data = JSON.parse(raw);
    if (data.reorderId) {
      const from = outputItems.findIndex((item) => item.id === data.reorderId);
      if (from < 0) return;
      const next = [...outputItems];
      const [moved] = next.splice(from, 1);
      next.push(moved);
      setOutputItems(next);
      return;
    }
    setOutputItems((prev) => [...prev, { id: newItemId(), ...data }]);
  }

  function reorderDrop(event: React.DragEvent, targetId: string) {
    event.preventDefault();
    const raw = event.dataTransfer.getData("application/json");
    if (!raw) return;
    const data = JSON.parse(raw);
    if (!data.reorderId) return;
    setOutputItems((prev) => {
      const from = prev.findIndex((item) => item.id === data.reorderId);
      const to = prev.findIndex((item) => item.id === targetId);
      if (from < 0 || to < 0 || from === to) return prev;
      const next = [...prev];
      const [moved] = next.splice(from, 1);
      next.splice(to, 0, moved);
      return next;
    });
  }

  function insertPause(durationMs = 260) {
    setOutputItems((prev) => [...prev, { id: newItemId(), type: "pause", durationMs, text: `${durationMs}ms` }]);
  }

  async function render() {
    setBusy("render");
    setMessage("正在拼接和后处理...");
    const payload = {
      items: outputItems.map((item) =>
        item.type === "pause"
          ? { type: "pause", durationMs: item.durationMs }
          : { type: "clip", recordingId: item.recordingId, phraseId: item.phraseId }
      ),
    };
    const res = await fetch(`${API}/api/render`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
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
        recording.id === recordingId
          ? {
              ...recording,
              phrases: recording.phrases.map((item) => (item.id === phrase.id ? updated : item)),
            }
          : recording
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
          <button onClick={() => void refresh()}>
            <RefreshCw size={18} />
            刷新
          </button>
        </div>
      </header>

      <main className="workspace">
        <section className="panel source-panel">
          <div className="panel-head">
            <div>
              <h2>录音轨</h2>
              <p>上传前可填转写文本；未启用 WhisperX 时按文本均分短语。</p>
            </div>
            <textarea
              value={manualText}
              onChange={(event) => setManualText(event.target.value)}
              placeholder="可选：输入这段录音的文字，例如：我是张三，他是李四"
            />
          </div>
          <div className="recording-list">
            {recordings.map((recording) => (
              <article className="recording" key={recording.id}>
                <div className="recording-title">
                  <FileAudio size={18} />
                  <strong>{recording.name}</strong>
                  <span>{recording.engine}</span>
                  <audio src={recording.audioUrl} controls />
                </div>
                <div className="transcript">{recording.text}</div>
                <div className="phrases">
                  {recording.phrases.map((phrase) =>
                    phrase.kind === "pause" ? (
                      <button className="phrase pause" key={phrase.id} onClick={() => insertPause(260)}>
                        {phrase.text}
                      </button>
                    ) : (
                      <button
                        className={`phrase ${phrase.quality}`}
                        key={phrase.id}
                        draggable
                        onDragStart={(event) => onPhraseDrag(event, recording, phrase)}
                        onClick={() => setSelected({ recordingId: recording.id, phrase })}
                        title={`${phrase.start}s - ${phrase.end}s`}
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
            {outputItems.map((item) => (
              <div
                className={`output-item ${item.type}`}
                key={item.id}
                draggable
                onDragStart={(event) => event.dataTransfer.setData("application/json", JSON.stringify({ reorderId: item.id }))}
                onDragOver={(event) => event.preventDefault()}
                onDrop={(event) => reorderDrop(event, item.id)}
              >
                <span>{item.text}</span>
                {item.type === "clip" && <small>{item.recordingName}</small>}
                <button onClick={() => setOutputItems((prev) => prev.filter((candidate) => candidate.id !== item.id))}>
                  <Trash2 size={14} />
                </button>
              </div>
            ))}
            {!outputItems.length && <div className="empty">输出轨为空</div>}
          </div>
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

function PhraseInspector({
  recording,
  phrase,
  onPatch,
}: {
  recording?: Recording;
  phrase: Phrase;
  onPatch: (patch: Partial<Phrase>) => Promise<void>;
}) {
  const [text, setText] = useState(phrase.text);
  const [start, setStart] = useState(String(phrase.start));
  const [end, setEnd] = useState(String(phrase.end));

  useEffect(() => {
    setText(phrase.text);
    setStart(String(phrase.start));
    setEnd(String(phrase.end));
  }, [phrase]);

  const previewUrl = recording
    ? `${recording.audioUrl}#t=${Math.max(0, Number(start) || 0)},${Math.max(Number(start) || 0, Number(end) || 0)}`
    : "";

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
      {recording && (
        <audio src={previewUrl} controls />
      )}
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
