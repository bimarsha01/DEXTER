import React, { useEffect, useMemo, useRef, useState } from "https://esm.sh/react@18";
import { createRoot } from "https://esm.sh/react-dom@18/client";
import htm from "https://esm.sh/htm@3";

const html = htm.bind(React.createElement);

const WS_URL = "ws://localhost:8765/ws";
const STAGE_ORDER = [
  "activate",
  "transcribe",
  "retrieve_context",
  "execute_tools",
  "generate_response",
  "speak",
];

const STAGE_LABELS = {
  activate: "Activate",
  transcribe: "Transcribe",
  retrieve_context: "Retrieve",
  execute_tools: "Tools",
  generate_response: "Generate",
  speak: "Speak",
};

const STAGE_TIMING_COLORS = {
  activate: "var(--tx3)",
  transcribe: "var(--blue)",
  retrieve_context: "var(--purple)",
  execute_tools: "var(--tx3)",
  generate_response: "var(--amber)",
  speak: "var(--green)",
};

const MOCK_MODE = (import.meta.env?.VITE_MOCK_MODE === "true") || new URLSearchParams(window.location.search).get("mock") === "1";

const MOCK_PROJECT = {
  name: "dexter-v2",
  source_path: "C:/Dexter/projects/dexter-v2",
  confidence: 0.92,
  last_confirmed_ts: Math.floor(Date.now() / 1000) - 120,
};

const MOCK_HEALTH = {
  service_name: "Dexter",
  overall_status: "healthy",
  gpu: {
    status: "healthy",
    device_name: "RTX 3060",
    total_vram_gb: 8.0,
    free_vram_gb: 2.8,
    updated_at: Math.floor(Date.now() / 1000),
  },
  rag: {
    status: "ready",
    doc_count: 824,
    last_updated_ts: Math.floor(Date.now() / 1000) - 3 * 24 * 3600,
    updated_at: Math.floor(Date.now() / 1000),
    details: "Index synchronized",
  },
  automation: {
    status: "ready",
  },
  providers: {
    groq: { current_status: "active" },
    openai: { current_status: "ready" },
    gemini: { current_status: "degraded" },
    ollama: { current_status: "offline" },
  },
  turn_stage_averages_ms: {
    activate: { average_ms: 320, sample_count: 7 },
    transcribe: { average_ms: 1180, sample_count: 7 },
    retrieve_context: { average_ms: 620, sample_count: 7 },
    execute_tools: { average_ms: 410, sample_count: 7 },
    generate_response: { average_ms: 1420, sample_count: 7 },
    speak: { average_ms: 880, sample_count: 7 },
  },
  updated_at: Math.floor(Date.now() / 1000),
};

const MOCK_TURNS = [
  {
    id: "t1",
    when: "14:32:10",
    provider: "groq",
    userText: "Start a focus session for 45 minutes.",
    responseText: "Focus timer set for 45 minutes. I will keep the workspace quiet and notify you at the end.",
    durationMs: 1860,
    project: "dexter-v2",
  },
  {
    id: "t2",
    when: "14:28:51",
    provider: "openai",
    userText: "Summarize the last test run output.",
    responseText: "The last suite completed with 16 passes and 1 retry. The retry was network related and resolved.",
    durationMs: 1290,
    project: "dexter-v2",
  },
  {
    id: "t3",
    when: "14:22:07",
    provider: "groq",
    userText: "Open the hardware diagnostics report.",
    responseText: "Diagnostics report opened. GPU memory usage is stable and within expected limits.",
    durationMs: 980,
    project: "dexter-v2",
  },
];

const MOCK_PROJECTS = [
  { id: "p1", name: "dexter-v2", count: 824, path: "C:/Dexter/projects/dexter-v2" },
  { id: "p2", name: "voice-labs", count: 312, path: "C:/Dexter/projects/voice-labs" },
  { id: "p3", name: "automation-core", count: 188, path: "C:/Dexter/projects/automation-core" },
];

const MOCK_CORRECTIONS = [
  { id: "c1", query: "open last transcript", tag: "wrong file" },
  { id: "c2", query: "provider fallback log", tag: "wrong file" },
];

const DEFAULT_PREFERENCE_PILLS = [
  { id: "pref-1", label: "mode", value: "adaptive" },
  { id: "pref-2", label: "voice", value: "studio" },
];

const MOCK_PREFERENCE_PILLS = [
  { id: "pref-1", label: "mode", value: "focus" },
  { id: "pref-2", label: "voice", value: "noir" },
];

const DEFAULT_HARDWARE_BARS = [
  { key: "cpu", label: "CPU temp", percent: 0, display: "--", color: "var(--tx3)" },
  { key: "gpu", label: "GPU temp", percent: 0, display: "--", color: "var(--tx3)" },
  { key: "ram", label: "RAM", percent: 0, display: "--", color: "var(--tx3)" },
];

const MOCK_HARDWARE = {
  cpu_temp_c: 62,
  gpu_temp_c: 71,
  ram_used_gb: 9.4,
  ram_total_gb: 16,
};

// Removed TAB_ITEMS

function formatNumber(value, options = {}) {
  if (!Number.isFinite(value)) return "--";
  return new Intl.NumberFormat("en-US", options).format(value);
}

function formatDurationMs(value) {
  if (!Number.isFinite(value)) return "--";
  if (value >= 1000) return `${(value / 1000).toFixed(1)}s`;
  return `${Math.round(value)}ms`;
}

function formatClock(tsSeconds) {
  if (!Number.isFinite(tsSeconds)) return "--";
  const date = new Date(tsSeconds * 1000);
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function formatAgo(tsSeconds) {
  if (!Number.isFinite(tsSeconds)) return "unknown";
  const diff = Math.max(0, Date.now() / 1000 - tsSeconds);
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.round(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.round(diff / 3600)}h ago`;
  return `${Math.round(diff / 86400)} days ago`;
}

function formatAgoParts(tsSeconds) {
  if (!Number.isFinite(tsSeconds)) return null;
  const diff = Math.max(0, Date.now() / 1000 - tsSeconds);
  if (diff < 60) return { text: "just now" };
  if (diff < 3600) return { value: Math.round(diff / 60), unit: "m ago" };
  if (diff < 86400) return { value: Math.round(diff / 3600), unit: "h ago" };
  return { value: Math.round(diff / 86400), unit: "days ago" };
}

function clamp(value, min = 0, max = 100) {
  return Math.min(max, Math.max(min, value));
}

function toNumber(value) {
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

function barColor(percent) {
  if (!Number.isFinite(percent)) return "var(--tx3)";
  if (percent >= 90) return "var(--red)";
  if (percent >= 70) return "var(--amber)";
  return "var(--green)";
}

function buildHardwareBars(summary, previousBars = DEFAULT_HARDWARE_BARS) {
  const baseline = Array.isArray(previousBars) && previousBars.length ? previousBars : DEFAULT_HARDWARE_BARS;
  const existing = new Map(baseline.map((bar) => [bar.key, bar]));
  const hardware = summary?.hardware || summary?.system || summary?.telemetry || {};

  const cpuTemp = toNumber(
    hardware.cpu_temp_c ??
    hardware.cpu_temp ??
    hardware.cpu_temperature_c ??
    hardware.cpu_temperature
  );
  const gpuTemp = toNumber(
    hardware.gpu_temp_c ??
    hardware.gpu_temp ??
    hardware.gpu_temperature_c ??
    hardware.gpu_temperature
  );
  const ramUsed = toNumber(
    hardware.ram_used_gb ??
    hardware.ram_used ??
    hardware.memory_used_gb ??
    hardware.memory_used
  );
  const ramTotal = toNumber(
    hardware.ram_total_gb ??
    hardware.ram_total ??
    hardware.memory_total_gb ??
    hardware.memory_total
  );

  const cpuPercent = cpuTemp != null ? clamp((cpuTemp / 100) * 100) : null;
  const gpuPercent = gpuTemp != null ? clamp((gpuTemp / 100) * 100) : null;
  const ramPercent = ramUsed != null && ramTotal ? clamp((ramUsed / ramTotal) * 100) : null;

  return [
    {
      key: "cpu",
      label: "CPU temp",
      percent: cpuPercent ?? existing.get("cpu")?.percent ?? 0,
      display: cpuTemp != null ? `${cpuTemp.toFixed(0)}C` : existing.get("cpu")?.display ?? "--",
      color: barColor(cpuPercent ?? existing.get("cpu")?.percent),
    },
    {
      key: "gpu",
      label: "GPU temp",
      percent: gpuPercent ?? existing.get("gpu")?.percent ?? 0,
      display: gpuTemp != null ? `${gpuTemp.toFixed(0)}C` : existing.get("gpu")?.display ?? "--",
      color: barColor(gpuPercent ?? existing.get("gpu")?.percent),
    },
    {
      key: "ram",
      label: "RAM",
      percent: ramPercent ?? existing.get("ram")?.percent ?? 0,
      display: ramUsed != null && ramTotal
        ? `${ramUsed.toFixed(1)} / ${ramTotal.toFixed(1)} GB`
        : existing.get("ram")?.display ?? "--",
      color: barColor(ramPercent ?? existing.get("ram")?.percent),
    },
  ];
}

function normalizePreferencePills(payload, previous) {
  if (!payload) return previous;
  const source = payload.pills || payload.preferences || payload.items;
  if (Array.isArray(source) && source.length) {
    return source.slice(0, 2).map((item, index) => ({
      id: item.id || item.key || `pref-${index}`,
      label: String(item.label || item.name || `preference ${index + 1}`),
      value: String(item.value ?? item.state ?? item.setting ?? item.text ?? "--"),
    }));
  }

  const primaryValue = payload.primary ?? payload.value ?? payload.first ?? payload.mode ?? payload.preference;
  const secondaryValue = payload.secondary ?? payload.value2 ?? payload.second ?? payload.voice ?? payload.profile;
  const primaryLabel = payload.primary_label ?? payload.label ?? payload.label1 ?? "mode";
  const secondaryLabel = payload.secondary_label ?? payload.label2 ?? "profile";

  if (primaryValue != null || secondaryValue != null) {
    return [
      { id: "pref-1", label: String(primaryLabel), value: String(primaryValue ?? "--") },
      { id: "pref-2", label: String(secondaryLabel), value: String(secondaryValue ?? "--") },
    ];
  }

  return previous;
}

function normalizeState(value) {
  const text = String(value || "").toLowerCase();
  if (text.includes("error")) return "error";
  if (text.includes("listen")) return "listening";
  if (text.includes("think")) return "thinking";
  if (text.includes("speak")) return "speaking";
  if (text.includes("idle")) return "idle";
  return "idle";
}

function useAnimatedNumber(value, duration = 500) {
  const [display, setDisplay] = useState(Number.isFinite(value) ? value : 0);
  const previous = useRef(Number.isFinite(value) ? value : 0);

  useEffect(() => {
    if (!Number.isFinite(value)) return;
    const start = performance.now();
    const from = previous.current;
    const to = value;
    let frame;

    const tick = (now) => {
      const progress = Math.min(1, (now - start) / duration);
      const next = from + (to - from) * progress;
      setDisplay(next);
      if (progress < 1) frame = requestAnimationFrame(tick);
    };

    frame = requestAnimationFrame(tick);
    previous.current = value;

    return () => cancelAnimationFrame(frame);
  }, [value, duration]);

  return display;
}

function useDexterData() {
  const [connection, setConnection] = useState(MOCK_MODE ? "mock" : "connecting");
  const [health, setHealth] = useState(MOCK_MODE ? MOCK_HEALTH : null);
  const [project, setProject] = useState(MOCK_MODE ? MOCK_PROJECT : null);
  const [assistantState, setAssistantState] = useState(MOCK_MODE ? "idle" : "idle");
  const [activeProvider, setActiveProvider] = useState(MOCK_MODE ? "groq" : null);
  const [turns, setTurns] = useState(MOCK_MODE ? MOCK_TURNS : []);
  const [turnCount, setTurnCount] = useState(MOCK_MODE ? 24 : 0);
  const [pipeline, setPipeline] = useState(() => {
    const next = {};
    STAGE_ORDER.forEach((stage) => {
      next[stage] = "idle";
    });
    return next;
  });
  const [stageTimings, setStageTimings] = useState(MOCK_MODE ? MOCK_HEALTH.turn_stage_averages_ms : {});
  const [lastTurnMs, setLastTurnMs] = useState(MOCK_MODE ? 1860 : null);
  const [fallbackLog, setFallbackLog] = useState(MOCK_MODE ? ["groq -> openai - rate limit - 14:32"] : []);
  const [projects, setProjects] = useState(MOCK_MODE ? MOCK_PROJECTS : []);
  const [corrections, setCorrections] = useState(MOCK_MODE ? MOCK_CORRECTIONS : []);
  const [preferencePills, setPreferencePills] = useState(MOCK_MODE ? MOCK_PREFERENCE_PILLS : DEFAULT_PREFERENCE_PILLS);
  const [hardwareBars, setHardwareBars] = useState(
    MOCK_MODE ? buildHardwareBars({ hardware: MOCK_HARDWARE }, DEFAULT_HARDWARE_BARS) : DEFAULT_HARDWARE_BARS
  );
  const [emergencyStop, setEmergencyStop] = useState(null);
  const currentTurnId = useRef(null);
  const pendingTranscript = useRef("");
  const activeProviderRef = useRef(activeProvider);
  const projectRef = useRef(project);

  useEffect(() => {
    activeProviderRef.current = activeProvider;
  }, [activeProvider]);

  useEffect(() => {
    projectRef.current = project;
  }, [project]);

  useEffect(() => {
    if (!MOCK_MODE) return;

    let stageIndex = 0;
    let stateIndex = 0;
    const states = ["idle", "listening", "thinking", "speaking"];

    const interval = window.setInterval(() => {
      stateIndex = (stateIndex + 1) % states.length;
      setAssistantState(states[stateIndex]);

      stageIndex = (stageIndex + 1) % STAGE_ORDER.length;
      setPipeline(() => {
        const next = {};
        STAGE_ORDER.forEach((stage, index) => {
          if (index < stageIndex) next[stage] = "done";
          else if (index === stageIndex) next[stage] = "active";
          else next[stage] = "idle";
        });
        return next;
      });
    }, 4200);

    return () => window.clearInterval(interval);
  }, []);

  useEffect(() => {
    if (MOCK_MODE) return;

    let socket;
    let retryTimer;
    let attempt = 0;

    const scheduleReconnect = () => {
      const delay = Math.min(1000 * Math.pow(2, attempt), 30000);
      setConnection("reconnecting");
      retryTimer = window.setTimeout(connect, delay);
      attempt = Math.min(attempt + 1, 10);
    };

    const applyHealth = (summary) => {
      if (!summary) return;
      setHealth(summary);
      setHardwareBars((prev) => buildHardwareBars(summary, prev));
      if (summary.turn_stage_averages_ms) {
        setStageTimings(summary.turn_stage_averages_ms);
      }
    };

    const applySnapshot = (snapshot) => {
      if (!snapshot) return;
      if (snapshot.project) setProject(snapshot.project);
      if (snapshot.active_provider) setActiveProvider(snapshot.active_provider);
      if (snapshot.assistant_state) setAssistantState(normalizeState(snapshot.assistant_state));
      if (snapshot.health) applyHealth(snapshot.health);
    };

    const updatePipeline = (stage, status, turnId, durationMs) => {
      if (!stage) return;
      if (turnId && turnId !== currentTurnId.current) {
        currentTurnId.current = turnId;
        setPipeline(() => {
          const reset = {};
          STAGE_ORDER.forEach((item) => {
            reset[item] = "idle";
          });
          return reset;
        });
      }

      const stageIndex = STAGE_ORDER.indexOf(stage);
      if (stageIndex === -1) return;

      setPipeline((prev) => {
        const next = { ...prev };
        if (status === "start") {
          STAGE_ORDER.forEach((item, index) => {
            if (index < stageIndex) next[item] = "done";
            else if (index === stageIndex) next[item] = "active";
            else if (next[item] !== "done") next[item] = "idle";
          });
        } else if (status === "done") {
          next[stage] = "done";
          if (Number.isFinite(durationMs)) {
            setStageTimings((prev) => {
              const entry = prev?.[stage];
              const count = Number.isFinite(entry?.sample_count) ? entry.sample_count : 0;
              const avg = Number.isFinite(entry?.average_ms) ? entry.average_ms : durationMs;
              const nextAvg = count > 0 ? (avg * count + durationMs) / (count + 1) : durationMs;
              return {
                ...prev,
                [stage]: {
                  average_ms: nextAvg,
                  sample_count: count + 1,
                },
              };
            });
          }
        } else if (status === "error") {
          next[stage] = "active";
        }
        return next;
      });
    };

    const addTurn = (payload, timestamp, provider) => {
      const id = payload.event_id || payload.correlation_id || `${timestamp}-${Math.random()}`;
      const userText = payload.user_text || payload.user || payload.prompt || pendingTranscript.current || "Voice command received.";
      const responseText = payload.response_text || payload.response || payload.text || "Response ready.";
      const when = formatClock(timestamp || Date.now() / 1000);

      pendingTranscript.current = "";

      setTurns((prev) => {
        const next = [
          {
            id,
            when,
            provider: provider || activeProviderRef.current || "unknown",
            userText,
            responseText,
            durationMs: payload.duration_ms,
            project: projectRef.current?.name || "dexter",
          },
          ...prev,
        ];
        return next.slice(0, 20);
      });
      setTurnCount((prev) => prev + 1);

      if (payload.duration_ms) setLastTurnMs(payload.duration_ms);
    };

    const appendFallback = (entry) => {
      if (!entry) return;
      setFallbackLog((prev) => {
        const next = [...prev, entry];
        return next.slice(-6);
      });
    };

    const handleEvent = (event) => {
      const payload = event.payload || {};
      switch (event.type) {
        case "assistant_state":
          setAssistantState(normalizeState(payload.state ?? payload.value ?? payload));
          break;
        case "state_changed":
          setAssistantState(normalizeState(payload.state));
          break;
        case "turn_stage":
          updatePipeline(payload.stage, payload.status, payload.turn_id, payload.duration_ms);
          if (payload.status === "done" && payload.stage === "speak" && payload.duration_ms) {
            setLastTurnMs(payload.duration_ms);
          }
          break;
        case "turn_complete":
          addTurn(payload, event.timestamp, payload.provider);
          break;
        case "turn_stage_error":
          setAssistantState("error");
          break;
        case "provider_fallback": {
          const fallback = payload.fallback_to || "unknown";
          setActiveProvider(fallback);
          const when = formatClock(event.timestamp || Date.now() / 1000);
          appendFallback(`${payload.provider || "unknown"} -> ${fallback} - ${payload.reason || "unknown"} - ${when}`);
          break;
        }
        case "llm_call_started":
        case "llm_stream_started":
          if (payload.provider) setActiveProvider(payload.provider);
          break;
        case "response_completed":
        case "response_complete":
          addTurn(payload, event.timestamp, payload.provider);
          break;
        case "transcript_ready":
          if (payload.text) pendingTranscript.current = payload.text;
          break;
        case "preference_update":
          setPreferencePills((prev) => normalizePreferencePills(payload, prev));
          break;
        case "hardware_emergency_stop":
          setEmergencyStop({
            reason: payload.reason || payload.message || payload.detail || "Hardware emergency stop",
          });
          break;
        case "rag_search_failed":
          if (payload.query) {
            setCorrections((prev) => {
              const next = [{ id: payload.event_id || `${Date.now()}`, query: payload.query, tag: "wrong file" }, ...prev];
              return next.slice(0, 4);
            });
          }
          break;
        default:
          break;
      }
    };

    const handleMessage = (data) => {
      if (!data) return;
      if (data.type === "health_summary") {
        applyHealth(data.payload || {});
        return;
      }
      if (data.type === "dashboard_snapshot") {
        applySnapshot(data.payload || {});
        return;
      }
      if (!data.type) {
        applyHealth(data);
        return;
      }
      handleEvent(data);
    };

    const connect = () => {
      socket = new WebSocket(WS_URL);
      setConnection("connecting");

      socket.addEventListener("open", () => {
        attempt = 0;
        setConnection("connected");
      });

      socket.addEventListener("message", (event) => {
        try {
          const parsed = JSON.parse(event.data);
          handleMessage(parsed);
        } catch (error) {
          setConnection("error");
        }
      });

      socket.addEventListener("close", () => {
        setConnection("reconnecting");
        scheduleReconnect();
      });

      socket.addEventListener("error", () => {
        setConnection("reconnecting");
        try {
          socket.close();
        } catch (error) {
          return;
        }
      });
    };

    connect();

    return () => {
      if (socket) socket.close();
      if (retryTimer) window.clearTimeout(retryTimer);
    };
  }, []);

  useEffect(() => {
    if (project?.name) {
      setProjects((prev) => {
        if (!prev.length) return [{ id: project.name, name: project.name, count: health?.rag?.doc_count || 0 }];
        return prev;
      });
    }
  }, [project, health]);

  return {
    connection,
    health,
    project,
    assistantState,
    activeProvider,
    turns,
    pipeline,
    stageTimings,
    lastTurnMs,
    fallbackLog,
    projects,
    corrections,
    turnCount,
    preferencePills,
    hardwareBars,
    emergencyStop,
    clearEmergencyStop: () => setEmergencyStop(null),
  };
}

// --- Icons ---
function MicIcon() { return html`<svg class="lucide" viewBox="0 0 24 24"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/></svg>`; }
function SettingsIcon() { return html`<svg class="lucide" viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1Z"/></svg>`; }
function HistoryIcon() { return html`<svg class="lucide" viewBox="0 0 24 24"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l4 2"/></svg>`; }
function CameraIcon() { return html`<svg class="lucide" viewBox="0 0 24 24"><path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/><circle cx="12" cy="13" r="3"/></svg>`; }
function GlobeIcon() { return html`<svg class="lucide" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/><path d="M2 12h20"/></svg>`; }
function CloudIcon() { return html`<svg class="lucide" viewBox="0 0 24 24"><path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z"/></svg>`; }

// --- Top Bar ---
function TopBar({ assistantState, project, hardwareBars, providerName }) {
  const isListening = assistantState === "listening";
  const isThinking = assistantState === "thinking";
  const isSpeaking = assistantState === "speaking";
  const isError = assistantState === "error";
  
  let dotClass = "";
  if (isListening) dotClass = "listening";
  if (isThinking) dotClass = "thinking";
  if (isSpeaking) dotClass = "speaking";
  if (isError) dotClass = "error";
  
  return html`
    <header class="top-bar">
      <div class="tb-left">
        <div class="brand">Dexter</div>
        <div class="status-indicator">
          <div class=${`status-dot ${dotClass}`}></div>
          <span>${assistantState}</span>
        </div>
      </div>
      
      <div class="tb-center">
        <div class="active-project mono">${project?.name || "No Project Active"}</div>
      </div>
      
      <div class="tb-right">
        <div class="hw-mini-bars">
          ${hardwareBars.map(bar => html`
            <div class="hw-mini">
              <span>${bar.key.toUpperCase()}</span>
              <div class="hw-mini-track">
                <div class="hw-mini-fill" style=${{ width: `${bar.percent}%`, background: bar.color }}></div>
              </div>
            </div>
          `)}
        </div>
        <div class="provider-icon mono">${providerName}</div>
        <div class="user-avatar">LOQ</div>
      </div>
    </header>
  `;
}

// --- Left Panel ---
function Orb({ state }) {
  return html`
    <div class="orb-container">
      <div class="orb" data-state=${state}>
        <div class="orb-core"></div>
        <div class="orb-ring"></div>
      </div>
      <div class=${`orb-label mono ${state}`}>${state}</div>
    </div>
  `;
}

function LeftPanel({ assistantState }) {
  return html`
    <aside class="panel panel-left">
      <${Orb} state=${assistantState} />
      <div class="quick-actions">
        <button class="btn-ptt">
          <${MicIcon} /> Push to Talk
        </button>
        <div class="action-grid">
          <button class="btn-action"><${HistoryIcon} /> History</button>
          <button class="btn-action"><${SettingsIcon} /> Settings</button>
        </div>
      </div>
    </aside>
  `;
}

// --- Center Column ---
function PipelineBanner({ pipeline }) {
  return html`
    <div class="feed-header">
      <div class="pipeline-banner">
        ${STAGE_ORDER.map((stage) => {
          const status = pipeline[stage] || "idle";
          return html`
            <div class=${`pipe-step ${status}`}>
              <div class="pipe-dot"></div>
              <span>${STAGE_LABELS[stage]}</span>
            </div>
          `;
        })}
      </div>
    </div>
  `;
}

function ChatFeed({ turns }) {
  const feedRef = useRef(null);
  
  // Auto-scroll to bottom
  useEffect(() => {
    if (feedRef.current) {
      feedRef.current.scrollTop = feedRef.current.scrollHeight;
    }
  }, [turns]);

  return html`
    <div class="chat-feed" ref=${feedRef}>
      ${turns.slice().reverse().map(turn => html`
        <div class="msg-wrapper user">
          <div class="msg-bubble">${turn.userText}</div>
          <div class="msg-meta mono">
            <span>${turn.when}</span>
          </div>
        </div>
        <div class="msg-wrapper dexter">
          <div class="msg-bubble">${turn.responseText}</div>
          <div class="msg-meta mono">
            <span>${formatDurationMs(turn.durationMs)}</span>
            <span>${turn.provider}</span>
            <span>[${turn.project}]</span>
          </div>
        </div>
      `)}
      ${turns.length === 0 ? html`
        <div style=${{ textAlign: 'center', color: 'var(--tx-sub)', marginTop: '40px' }}>
          <div class="serif" style=${{ fontSize: '1.5rem', marginBottom: '8px' }}>Awaiting Command</div>
          <div class="mono subtle">Ready to process voice or text input.</div>
        </div>
      ` : null}
    </div>
  `;
}

function CenterColumn({ pipeline, turns }) {
  return html`
    <main class="panel panel-center">
      <${PipelineBanner} pipeline=${pipeline} />
      <${ChatFeed} turns=${turns} />
    </main>
  `;
}

// --- Right Panel ---
function HealthCard({ label, value, percent, color }) {
  const sparklines = Array.from({ length: 15 }).map(() => Math.random() * 80 + 20); // Dummy sparkline data
  
  return html`
    <div class="health-card">
      <div class="hc-info">
        <span class="hc-label mono">${label}</span>
        <span class="hc-val mono">${value}</span>
      </div>
      <div class="hc-sparkline">
        ${sparklines.map((h, i) => {
          const isLast = i === sparklines.length - 1;
          const bg = isLast ? color : 'var(--tx-muted)';
          const hVal = isLast ? percent : h;
          return html`<div class="hc-bar" style=${{ height: `${hVal}%`, background: bg }}></div>`;
        })}
      </div>
    </div>
  `;
}

function RightPanel({ health, docCount, project, hardwareBars }) {
  const lastIndexedParts = formatAgoParts(health?.rag?.last_updated_ts);
  
  return html`
    <aside class="panel panel-right">
      <div>
        <div class="section-title">System Health</div>
        <div class="health-grid">
          ${hardwareBars.map(bar => html`
            <${HealthCard} label=${bar.label} value=${bar.display} percent=${bar.percent} color=${bar.color} />
          `)}
        </div>
      </div>
      
      <div>
        <div class="section-title">Current Context</div>
        <div class="info-card">
          <div class="ic-row mono subtle">Project</div>
          <div class="ic-row ic-val">${project?.name || 'dexter-v2'}</div>
          <div class="ic-row mono subtle" style=${{ marginTop: '12px' }}>Knowledge Base</div>
          <div class="ic-row">
            <span class="subtle">Indexed Files:</span>
            <span class="mono">${formatNumber(docCount)}</span>
          </div>
          <div class="ic-row">
            <span class="subtle">Updated:</span>
            <span class="mono">${lastIndexedParts ? `${lastIndexedParts.value} ${lastIndexedParts.unit}` : 'unknown'}</span>
          </div>
        </div>
      </div>
      
      <div>
        <div class="section-title">Quick Tools</div>
        <div class="tools-grid">
          <div class="tool-btn">
            <${CameraIcon} />
            <span class="mono" style=${{ fontSize: '0.65rem' }}>Screen</span>
          </div>
          <div class="tool-btn">
            <${GlobeIcon} />
            <span class="mono" style=${{ fontSize: '0.65rem' }}>Browser</span>
          </div>
          <div class="tool-btn">
            <${CloudIcon} />
            <span class="mono" style=${{ fontSize: '0.65rem' }}>Weather</span>
          </div>
        </div>
      </div>
    </aside>
  `;
}

// --- Main App ---
function App() {
  const {
    connection,
    health,
    project,
    assistantState,
    activeProvider,
    turns,
    pipeline,
    hardwareBars,
    emergencyStop,
    clearEmergencyStop,
  } = useDexterData();

  const orbState = useMemo(() => {
    const normalized = normalizeState(assistantState);
    if (normalized !== "idle") return normalized;
    const activeStage = STAGE_ORDER.find((stage) => pipeline[stage] === "active");
    if (activeStage === "speak") return "speaking";
    if (activeStage === "transcribe" || activeStage === "activate") return "listening";
    if (activeStage === "retrieve_context" || activeStage === "execute_tools" || activeStage === "generate_response") return "thinking";
    return "idle";
  }, [assistantState, pipeline]);

  const docCount = health?.rag?.doc_count ?? 0;
  const providerName = activeProvider || "unknown";

  return html`
    <div style=${{ display: "flex", flexDirection: "column", height: "100%", width: "100%" }}>
      ${connection !== "connected" && connection !== "mock"
        ? html`<div style=${{ position: 'absolute', top: 0, left: 0, width: '100%', padding: '4px', textAlign: 'center', background: 'var(--accent-red)', color: 'white', zIndex: 100 }} class="mono">Reconnecting...</div>`
        : null}
        
      <${TopBar} 
        assistantState=${orbState} 
        project=${project} 
        hardwareBars=${hardwareBars} 
        providerName=${providerName} 
      />
      
      <div class="dashboard-core">
        <${LeftPanel} assistantState=${orbState} />
        <${CenterColumn} pipeline=${pipeline} turns=${turns} />
        <${RightPanel} health=${health} docCount=${docCount} project=${project} hardwareBars=${hardwareBars} />
      </div>

      ${emergencyStop ? html`
        <div style=${{ position: 'fixed', bottom: '20px', left: '50%', transform: 'translateX(-50%)', background: 'var(--accent-red)', color: 'white', padding: '16px 24px', borderRadius: '8px', zIndex: 100, cursor: 'pointer' }} onClick=${clearEmergencyStop}>
          <div class="mono" style=${{ fontWeight: 'bold' }}>HARDWARE EMERGENCY STOP</div>
          <div>${emergencyStop.reason}</div>
          <div class="mono" style=${{ fontSize: '0.8rem', marginTop: '8px', opacity: 0.8 }}>Tap to dismiss</div>
        </div>
      ` : null}
    </div>
  `;
}

const root = createRoot(document.getElementById("root"));
root.render(html`<${App} />`);
