"use client";

import React, { useState, useRef, useEffect } from "react";
import { Sidebar, TopContextBar } from "@/components/Navigation";
import { Play, Layers, Cpu, Activity, Database, AlertCircle, Send, User, Bot, Loader2 } from "lucide-react";
import BinarySampleLoader from "@/components/visualizations/BinarySampleLoader";
import AttentionHeatmap from "@/components/visualizations/AttentionHeatmap";
import LayerSlider from "@/components/visualizations/LayerSlider";
import TokenSelector from "@/components/visualizations/TokenSelector";
import LayerInputView from "@/components/visualizations/LayerInputView";
import EmbeddingView from "@/components/visualizations/EmbeddingView";
import OutputView from "@/components/visualizations/OutputView";
import { useWorkspace } from "@/features/collaboration/useWorkspace";
import { WorkspacePanel } from "@/features/collaboration/WorkspacePanel";
import { ViewState } from "@/lib/collaboration-types";

// Best-effort cleanup; server TTL handles closed browsers and interrupted networks.
const releasePersonalHandle = (handle: string) => fetch("/api/v1/models/current", {
  method: "DELETE", headers: { "X-Model-Handle": handle }, keepalive: true,
}).catch(() => null);

export default function ExplorePage() {
  const workspace = useWorkspace();
  const appliedAnalysis = useRef("");
  const personalHandle = useRef<string | null>(null);
  const personalSession = useRef<string | null>(null);
  const personalGeneration = useRef(0);
  const mounted = useRef(true);
  const activeRoom = useRef<string | null>(null);
  activeRoom.current = workspace.identity?.roomId || workspace.room?.id || null;
  const personalRequestIsCurrent = (generation: number) =>
    mounted.current && !activeRoom.current && personalGeneration.current === generation;
  const [checkpointPath, setCheckpointPath] = useState("");
  const [modelArchitecture, setModelArchitecture] = useState("auto");
  const [modelLoaded, setModelLoaded] = useState(false);
  const [modelName, setModelName] = useState("");
  const [loadedModelPath, setLoadedModelPath] = useState("");
  const [device, setDevice] = useState("cuda");
  const [targetDevice, setTargetDevice] = useState("cuda");
  const [capabilities, setCapabilities] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [inspectorTab, setInspectorTab] = useState<"details" | "assistant">("details");

  // Visualization States
  const [selectedToken, setSelectedToken] = useState(0);
  const [currentLayer, setCurrentLayer] = useState(0);
  const [maxLayer, setMaxLayer] = useState(3);
  const [tokenCount, setTokenCount] = useState(16);
  const [modelInputSchema, setModelInputSchema] = useState<any>(null);
  const [modelOutputSchema, setModelOutputSchema] = useState<any>(null);
  const [modelDimension, setModelDimension] = useState<number | null>(null);
  
  const [inputFeatures, setInputFeatures] = useState<number[] | null>(null);
  const [embedding, setEmbedding] = useState<number[] | null>(null);
  const [layerInput, setLayerInput] = useState<number[] | null>(null);
  const [attentionMap, setAttentionMap] = useState<Record<number, number[][]>>({});
  const [output, setOutput] = useState<number[] | null>(null);
  const [outputShape, setOutputShape] = useState<number[]>([]);

  // Chat
  const [messages, setMessages] = useState<{role: string, content: string}[]>([]);
  const [inputMessage, setInputMessage] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const endOfMessagesRef = useRef<HTMLDivElement>(null);
  const chatContext = `${activeRoom.current || "personal"}:${workspace.room?.analysisRevision ?? ""}:${personalHandle.current || ""}`;
  const currentChatContext = useRef(chatContext);
  currentChatContext.current = chatContext;
  useEffect(() => {
    setMessages([]);
    setChatLoading(false);
  }, [chatContext]);

  const clearPersonalView = () => {
    personalHandle.current = null; personalSession.current = null;
    setModelLoaded(false); setModelName(""); setLoadedModelPath(""); setModelInputSchema(null);
    setCapabilities(null); setAttentionMap({}); setOutput(null); setModelOutputSchema(null);
    setInputFeatures(null); setEmbedding(null); setLayerInput(null);
    setTokenCount(16); setMaxLayer(3); setSelectedToken(0); setCurrentLayer(0);
  };
  const reportPersonalError = (data: any, fallback: string) => {
    if (["MODEL_EXPIRED", "MODEL_HANDLE_REQUIRED"].includes(data.error?.code)) clearPersonalView();
    setError(data.error?.message || fallback);
  };
  useEffect(() => {
    mounted.current = true;
    const release = () => {
      if (personalHandle.current) void releasePersonalHandle(personalHandle.current);
    };
    window.addEventListener("pagehide", release);
    return () => {
      mounted.current = false;
      window.removeEventListener("pagehide", release);
      release();
    };
  }, []);

  useEffect(() => {
    const room = workspace.room;
    if (!room) {
      if (appliedAnalysis.current) {
        setModelLoaded(false); setModelName(""); setLoadedModelPath(""); setModelInputSchema(null);
        setAttentionMap({}); setOutput(null); setCapabilities(null);
        setTokenCount(16); setMaxLayer(3); setSelectedToken(0); setCurrentLayer(0);
      }
      appliedAnalysis.current = ""; return;
    }
    if (personalHandle.current) {
      void releasePersonalHandle(personalHandle.current);
      personalHandle.current = null; personalSession.current = null;
    }
    personalGeneration.current++;
    setLoading(false);
    const key = `${room.id}:${room.analysisRevision}`;
    if (appliedAnalysis.current === key) return;
    appliedAnalysis.current = key;
    const a = room.analysis;
    setModelLoaded(!!a.model); setModelName(a.modelName || "");
    setLoadedModelPath(a.model?.checkpoint_path === "dummy" ? a.modelName || "" : a.model?.checkpoint_path || "");
    setDevice(a.model?.device || "cuda"); setTargetDevice(a.model?.device || "cuda");
    setCheckpointPath(a.model?.checkpoint_path === "dummy" ? "" : a.model?.checkpoint_path || "");
    setModelArchitecture(String(a.model?.manifest_spec?.name || (a.model?.checkpoint_path === "dummy" ? "transOriginal" : "auto")));
    setCapabilities(a.capabilities || null); setModelInputSchema(a.inputSchema || null);
    setAttentionMap(a.attention); setOutput(a.output.length ? a.output : null);
    setTokenCount(a.tokenCount); setMaxLayer(a.maxLayer);
    setSelectedToken(0); setCurrentLayer(0);
    setInputFeatures(null); setEmbedding(null); setLayerInput(null);
    setModelOutputSchema(null);
    setOutputShape(a.outputShape || []);
  }, [workspace.room]);

  useEffect(() => {
    if (workspace.room && workspace.following && !workspace.isHost) {
      setSelectedToken(workspace.room.view.token); setCurrentLayer(workspace.room.view.layer);
    }
  }, [workspace.following, workspace.isHost, workspace.room?.view.token, workspace.room?.view.layer, workspace.room?.analysisRevision]);

  useEffect(() => {
    if (workspace.isHost && workspace.room) {
      setSelectedToken(workspace.room.view.token); setCurrentLayer(workspace.room.view.layer);
      workspace.setFollowing(false);
    }
  }, [workspace.isHost]);

  const navigateView = (view: ViewState) => {
    workspace.setFollowing(false);
    setSelectedToken(view.token); setCurrentLayer(view.layer);
    if (workspace.isHost && workspace.room) void workspace.act({ type: "view", view, analysisRevision: workspace.room.analysisRevision });
  };

  const handleLoadModel = async (requestedCheckpoint?: string) => {
    const path = requestedCheckpoint ?? checkpointPath;
    const manifestSpec = !path ? { name: modelArchitecture === "auto" ? "transOriginal" : modelArchitecture }
      : modelArchitecture === "auto" ? undefined : { name: modelArchitecture };
    setLoading(true);
    setError(null);
    if (workspace.room) {
      await workspace.act({ type: "load", analysisRevision: workspace.room.analysisRevision, model: {
        checkpoint_path: path || "dummy", manifest_spec: manifestSpec, device: targetDevice,
      } });
      setLoading(false); return;
    }
    if (activeRoom.current) { setLoading(false); return; }
    const generation = ++personalGeneration.current;
    const previousHandle = personalHandle.current;
    try {
      const res = await fetch("/api/v1/models/load", {
        method: "POST",
        headers: { "Content-Type": "application/json", ...(previousHandle ? { "X-Model-Handle": previousHandle } : {}) },
        body: JSON.stringify({ checkpoint_path: path || "dummy", manifest_spec: manifestSpec, device: targetDevice }),
      });
      const data = await res.json();
      if (!personalRequestIsCurrent(generation)) {
        if (data.model_handle) void releasePersonalHandle(data.model_handle);
        return;
      }
      if (!res.ok || data.error) {
        reportPersonalError(data, "Failed to load model");
      } else {
        if (!data.model_handle) throw new Error("模型服務版本不相容，請重新啟動前後端服務。");
        personalHandle.current = data.model_handle;
        personalSession.current = null;
        setModelLoaded(true); setModelName(data.model_name);
        setLoadedModelPath(path || data.model_name);
        setDevice(data.device || targetDevice); setCapabilities(data.capabilities);
        setModelInputSchema(data.input_schema || null);
        setTokenCount(data.input_schema?.seq_len || 16);
        setMaxLayer(Math.max(0, data.layer_count - 1));
        setSelectedToken(0); setCurrentLayer(0);
        setAttentionMap({}); setOutput(null); setModelOutputSchema(null);
        setInputFeatures(null); setEmbedding(null); setLayerInput(null);
      }
    } catch (err: any) {
      if (personalRequestIsCurrent(generation)) setError(err.message);
    } finally {
      if (personalRequestIsCurrent(generation)) setLoading(false);
    }
  };

  const handleReleaseModel = async () => {
    const handle = personalHandle.current;
    if (!handle) return;
    const generation = ++personalGeneration.current;
    setLoading(true); setError(null);
    try {
      const response = await releasePersonalHandle(handle);
      if (!personalRequestIsCurrent(generation)) return;
      if (!response) throw new Error("無法釋放模型，請稍後重試。");
      const data = await response.json();
      if (!response.ok && data.error?.code !== "MODEL_EXPIRED") {
        reportPersonalError(data, "無法釋放模型。"); return;
      }
      clearPersonalView();
    } catch (err: any) {
      if (personalRequestIsCurrent(generation)) setError(err.message);
    } finally {
      if (personalRequestIsCurrent(generation)) setLoading(false);
    }
  };

  const requestedCheckpointApplied = useRef(false);
  useEffect(() => {
    if (requestedCheckpointApplied.current) return;
    const query = new URLSearchParams(window.location.search);
    if (query.has("room")) return;
    const path = query.get("checkpoint");
    if (!path) return;
    requestedCheckpointApplied.current = true;
    setCheckpointPath(path);
    void handleLoadModel(path);
    // The URL request is applied once when this page opens.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleBinaryFeature = async (feature: number[][], tokenCountVal: number, meta?: Record<string, unknown>) => {
    setLoading(true);
    setError(null);
    if (workspace.room) {
      await workspace.act({ type: "infer", analysisRevision: workspace.room.analysisRevision, input: feature, tokenCount: tokenCountVal, sample: meta });
      setLoading(false); return;
    }
    const handle = personalHandle.current;
    if (activeRoom.current || !handle) {
      setLoading(false);
      if (!activeRoom.current) setError("請先在此分頁載入模型。");
      return;
    }
    const generation = ++personalGeneration.current;
    const headers = { "Content-Type": "application/json", "X-Model-Handle": handle };
    let resultSession: string | undefined;
    let accepted = false;
    try {
      const res = await fetch("/api/v1/inference-sessions", {
        method: "POST", headers,
        body: JSON.stringify({ input_data: feature, requested_probes: ["attention"] }),
      });
      const data = await res.json();
      resultSession = data.session_id;
      if (!personalRequestIsCurrent(generation)) return;
      if (!res.ok || data.error) { reportPersonalError(data, "Inference failed"); return; }
      const actRes = await fetch(`/api/v1/inference-sessions/${data.session_id}/activations`, { headers });
      const actData = await actRes.json();
      if (!personalRequestIsCurrent(generation)) return;
      if (!actRes.ok || actData.error) { reportPersonalError(actData, "無法讀取分析結果。"); return; }
      const newAttn: Record<number, number[][]> = {};
      for (const [key, value] of Object.entries(actData.activations || {})) {
        if (key.startsWith("attention_layer_")) newAttn[Number(key.replace("attention_layer_", ""))] = value as number[][];
      }
      const count = Math.min(tokenCountVal, modelInputSchema?.seq_len || tokenCountVal);
      setTokenCount(count); setSelectedToken(Math.min(selectedToken, count - 1));
      setAttentionMap(newAttn); setOutput(data.output_sample || null);
      setOutputShape(data.output_shape || []);
      // Until the model declares output semantics, display raw values without invented classes.
      setModelOutputSchema(null);
      const previousSession = personalSession.current;
      personalSession.current = data.session_id;
      accepted = true;
      if (previousSession) void fetch(`/api/v1/inference-sessions/${previousSession}`, { method: "DELETE", headers }).catch(() => {});
    } catch (err: any) {
      if (personalRequestIsCurrent(generation)) setError(err.message);
    } finally {
      if (resultSession && !accepted) void fetch(`/api/v1/inference-sessions/${resultSession}`, { method: "DELETE", headers }).catch(() => {});
      if (personalRequestIsCurrent(generation)) setLoading(false);
      else if (personalHandle.current !== handle || !mounted.current) void releasePersonalHandle(handle);
    }
  };

  const sendChatMessage = async () => {
    if (!inputMessage.trim() || chatLoading) return;
    const context = currentChatContext.current;
    const isCurrent = () => mounted.current && currentChatContext.current === context;

    const userMsg = inputMessage;
    setInputMessage("");
    const newMessages = [...messages, { role: "user", content: userMsg }].slice(-40);
    setMessages(newMessages);
    setChatLoading(true);

    try {
      const res = await fetch("/api/v1/assistant/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: newMessages, page_context: {
          model_path: loadedModelPath,
          selected_token: selectedToken, current_layer: currentLayer,
          // Send only fields accepted by the assistant's strict DenseInputSchema.
          ...(modelInputSchema?.seq_len <= 512 && modelInputSchema?.feature_dim >= 58 && modelInputSchema?.feature_dim <= 20000 ? {
            model_input_schema: {
              dtype: "float32", rank: 3, batch_size: 1,
              seq_len: modelInputSchema.seq_len, feature_dim: modelInputSchema.feature_dim,
              shape: [1, modelInputSchema.seq_len, modelInputSchema.feature_dim],
            },
          } : {}),
        } }),
      });
      const data = await res.json();
      if (!isCurrent()) return;
      if (!res.ok || !data.success) throw new Error(data.error?.message ||
        (typeof data.detail === "string" ? data.detail : "分析對話失敗，請稍後重試。"));
      if (data.kind !== "chat" || typeof data.reply !== "string") throw new Error("分析對話回應格式不正確。");
      setMessages(prev => [...prev, { role: "assistant", content: data.reply }]);
    } catch (err: any) {
      if (isCurrent()) setMessages(prev => [...prev, { role: "assistant", content: err.message || "無法連線至分析對話服務。" }]);
    } finally {
      if (isCurrent()) setChatLoading(false);
    }
  };

  useEffect(() => {
    if (inspectorTab === "assistant") {
      endOfMessagesRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, inspectorTab]);

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar activeTab="explore" />
      <div className="flex-1 flex flex-col min-w-0 bg-slate-50">
        <TopContextBar modelName={modelName || undefined} device={device} />

        <div className="explore-content flex-1 flex overflow-hidden">
          <main className="explore-main flex-1 overflow-y-auto p-6 space-y-6">
            <WorkspacePanel workspace={workspace} view={{ token: selectedToken, layer: currentLayer }} onNavigate={navigateView} />
            <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="font-semibold text-slate-900 flex items-center space-x-2">
                  <Cpu className="w-5 h-5 text-blue-600" />
                  <span>Model Control & Source Selection</span>
                </h2>
              </div>
              <fieldset disabled={!!workspace.identity && (!workspace.room || !workspace.isHost || workspace.room.busy) || loading} className="flex flex-wrap gap-3 min-w-0">
                <select
                  value={targetDevice}
                  onChange={(e) => setTargetDevice(e.target.value)}
                  className="px-3 py-2 text-sm border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  title="Target compute device"
                >
                  <option value="cpu">CPU</option>
                  <option value="cuda">CUDA (GPU)</option>
                </select>
                <select
                  value={modelArchitecture}
                  onChange={(e) => setModelArchitecture(e.target.value)}
                  className="px-3 py-2 text-sm border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  title="Model architecture"
                >
                  <option value="auto">Auto detect (transOriginal for dummy)</option>
                  <option value="trans">trans</option>
                  <option value="transTest">transTest</option>
                  <option value="transOriginal">transOriginal</option>
                  <option value="transBertBase">transBertBase</option>
                  <option value="transBertLarge">transBertLarge</option>
                  <option value="transVOG">transVOG</option>
                  <option value="cnn">cnn</option>
                  <option value="resnet10">resnet10</option>
                  <option value="resnet14">resnet14</option>
                  <option value="resnet18">resnet18</option>
                  <option value="resnet22">resnet22</option>
                  <option value="conformerTest">conformerTest</option>
                  <option value="conformerSmall">conformerSmall</option>
                  <option value="conformerOriginal">conformerOriginal</option>
                  <option value="conformerLarge">conformerLarge</option>
                </select>
                <input
                  type="text"
                  value={checkpointPath}
                  onChange={(e) => setCheckpointPath(e.target.value)}
                  placeholder="Enter model checkpoint .pth path (leave blank for dummy model)"
                  className="flex-1 px-3 py-2 text-sm font-mono border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <button
                  onClick={() => void handleLoadModel()}
                  disabled={loading}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
                  >
                  <span>{loading ? "Loading..." : "Load Model"}</span>
                </button>
                {!workspace.identity && modelLoaded && <button type="button" onClick={() => void handleReleaseModel()}
                  className="px-4 py-2 border border-slate-300 rounded-lg text-sm">Release Model</button>}
              </fieldset>
              {error && <div className="p-3 bg-rose-50 border border-rose-200 rounded-lg text-xs text-rose-700">{error}</div>}
              {modelLoaded && <p role="status" className="text-xs text-slate-600">Loaded {modelName}.</p>}
              {modelLoaded && (!workspace.room || workspace.isHost) && !workspace.room?.busy && !loading && (
                <BinarySampleLoader
                  key={workspace.room ? `${workspace.room.id}:${workspace.room.analysisRevision}` : personalHandle.current}
                  modelShape={modelInputSchema}
                  onFeatureLoaded={handleBinaryFeature}
                  onTenhouUrl={(url) => {}}
                />
              )}
            </div>

            <div className="grid grid-cols-1 gap-6">
              <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-3">
                 <div className="flex space-x-4 mb-4">
                    <TokenSelector tokenCount={tokenCount} selectedToken={selectedToken} onSelect={token => navigateView({ token, layer: currentLayer })} />
                    <LayerSlider maxLayer={maxLayer} currentLayer={currentLayer} onChange={layer => navigateView({ token: selectedToken, layer })} />
                 </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-3">
                  <h3 className="font-semibold text-sm text-slate-800">Embedding</h3>
                  <div className="border border-slate-200 rounded-lg overflow-hidden">
                     {capabilities?.has_embedding === true ?
                       <EmbeddingView embedding={embedding} tokenIndex={selectedToken} /> :
                       <p className="p-4 text-sm text-slate-500">{modelLoaded
                         ? capabilities?.unavailable_probes?.embedding || "目前模型的分析服務不支援 Embedding probe。"
                         : "載入模型後可查看支援的分析項目。"}</p>}
                  </div>
                </div>
                <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-3">
                  <h3 className="font-semibold text-sm text-slate-800">Layer {currentLayer} Attention Map</h3>
                  <div className="border border-slate-200 rounded-lg overflow-x-auto text-[10px]">
                     <AttentionHeatmap attention={attentionMap[currentLayer] || null} tokenIndex={selectedToken} />
                  </div>
                </div>
              </div>
              
              <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-3">
                  <h3 className="font-semibold text-sm text-slate-800">Output Layer</h3>
                  <div className="border border-slate-200 rounded-lg overflow-hidden p-2">
                     <OutputView logits={output} schema={modelOutputSchema} shape={outputShape} />
                  </div>
              </div>
            </div>
          </main>

          <aside className="explore-assistant w-80 bg-white border-l border-slate-200 flex flex-col shrink-0 relative">
            <div className="flex border-b border-slate-200 p-2 space-x-1 shrink-0 bg-white z-10 sticky top-0">
              <button
                onClick={() => setInspectorTab("assistant")}
                className="flex-1 px-3 py-1.5 text-sm font-medium rounded-md transition-colors bg-blue-50 text-blue-700"
              >
                Assistant
              </button>
            </div>
            <div className="flex-1 flex flex-col items-center">
              <div className="flex-1 w-full overflow-y-auto p-4 space-y-4">
                 {messages.map((m, i) => (
                    <div key={i} className={`text-xs p-3 rounded-lg leading-relaxed ${m.role === 'user' ? 'bg-blue-50 text-blue-950 ml-4' : 'bg-slate-100 text-slate-800 mr-4'}`}>
                      {m.content}
                    </div>
                 ))}
                 <div ref={endOfMessagesRef} />
              </div>
              <div className="p-3 w-full border-t border-slate-200">
                 <form onSubmit={(e) => { e.preventDefault(); sendChatMessage(); }} className="flex gap-2">
                   <input type="text" maxLength={8000} aria-label="Assistant message" className="min-w-0 flex-1 rounded-lg border border-slate-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500" value={inputMessage} onChange={e => setInputMessage(e.target.value)} />
                   <button type="submit" disabled={chatLoading} className="rounded-lg bg-blue-600 px-3 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50">Send</button>
                 </form>
              </div>
            </div>
          </aside>
        </div>
      </div>
    </div>
  );
}
