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

export default function ExplorePage() {
  const [checkpointPath, setCheckpointPath] = useState("");
  const [dummyModelType, setDummyModelType] = useState("transTest");
  const [modelLoaded, setModelLoaded] = useState(false);
  const [modelName, setModelName] = useState("");
  const [device, setDevice] = useState("cpu");
  const [targetDevice, setTargetDevice] = useState("cpu");
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

  // Chat
  const [messages, setMessages] = useState<{role: string, content: string}[]>([]);
  const [inputMessage, setInputMessage] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const endOfMessagesRef = useRef<HTMLDivElement>(null);

  const handleLoadModel = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/v1/models/load", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          checkpoint_path: checkpointPath || "dummy",
          manifest_spec: !checkpointPath ? { name: dummyModelType } : undefined,
          device: targetDevice
        }),
      });
      const data = await res.json();
      if (!res.ok || data.error) {
        setError(data.error?.message || "Failed to load model");
      } else {
        setModelLoaded(true);
        setModelName(data.model_name);
        setDevice(data.device || targetDevice);
        setCapabilities(data.capabilities);
        // Add schemas directly for the binary sample loader
        setModelInputSchema({ seq_len: 512, feature_dim: 256 }); // Mocks or get from capabilities if possible
        
        // Add context message
        setMessages(prev => [...prev, {
          role: "assistant",
          content: `Loaded ${data.model_name}. I can help you analyze the attention heatmap and feature projections. What would you like to explore?`
        }]);
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleBinaryFeature = async (feature: number[][], tokenCountVal: number, meta?: Record<string, unknown>) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/v1/inference-sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input_data: feature, requested_probes: ["attention"] }),
      });
      const data = await res.json();
      if (!res.ok || data.error) {
         setError(data.error?.message || "Inference failed");
         return;
      }
      setTokenCount(tokenCountVal);
      setSelectedToken(Math.min(selectedToken, tokenCountVal - 1));
      
      const actRes = await fetch(`/api/v1/inference-sessions/${data.session_id}/activations`);
      const actData = await actRes.json();
      
      const newAttn: Record<number, number[][]> = {};
      if (actData.activations) {
        Object.keys(actData.activations).forEach(key => {
          if (key.startsWith("attention_layer_")) {
            const idx = parseInt(key.replace("attention_layer_", ""));
            newAttn[idx] = actData.activations[key];
          }
        });
      }
      if (Object.keys(newAttn).length > 0) setMaxLayer(Math.max(...Object.keys(newAttn).map(Number)));
      setAttentionMap(newAttn);
      setOutput(data.output_sample || null);
      if (data.output_shape) setModelOutputSchema({ output_type: "multiclass_logits", class_names: ["1", "2"] });
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const sendChatMessage = async () => {
    if (!inputMessage.trim()) return;

    const userMsg = inputMessage;
    setInputMessage("");
    const newMessages = [...messages, { role: "user", content: userMsg }];
    setMessages(newMessages);
    setChatLoading(true);

    try {
      const res = await fetch("/api/v1/assistant/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: newMessages, context: { modelName, capabilities } }),
      });
      const data = await res.json();
      if (data.role === 'assistant') {
        setMessages(prev => [...prev, { role: "assistant", content: data.content }]);
      } else {
        setMessages(prev => [...prev, { role: "assistant", content: JSON.stringify(data) }]);
      }
    } catch (err: any) {
      console.error("Chat error:", err);
      setMessages(prev => [...prev, { role: "assistant", content: "Sorry, an error occurred while connecting to the assistant API." }]);
    } finally {
      setChatLoading(false);
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
        <TopContextBar modelName={modelName || undefined} device={device} sidecarHealthy={true} />

        <div className="flex-1 flex overflow-hidden">
          <main className="flex-1 overflow-y-auto p-6 space-y-6">
            <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="font-semibold text-slate-900 flex items-center space-x-2">
                  <Cpu className="w-5 h-5 text-blue-600" />
                  <span>Model Control & Source Selection</span>
                </h2>
              </div>
              <div className="flex space-x-3">
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
                  value={dummyModelType}
                  onChange={(e) => setDummyModelType(e.target.value)}
                  disabled={checkpointPath.length > 0}
                  className="px-3 py-2 text-sm border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:bg-slate-100"
                  title="Dummy model to load if path is empty"
                >
                  <option value="transTest">transTest</option>
                  <option value="transOriginal">transOriginal</option>
                  <option value="transBertBase">transBertBase</option>
                  <option value="transBertLarge">transBertLarge</option>
                  <option value="transVOG">transVOG</option>
                  <option value="cnn">cnn</option>
                </select>
                <input
                  type="text"
                  value={checkpointPath}
                  onChange={(e) => setCheckpointPath(e.target.value)}
                  placeholder="Enter model checkpoint .pth path (leave blank for dummy model)"
                  className="flex-1 px-3 py-2 text-sm font-mono border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <button
                  onClick={handleLoadModel}
                  disabled={loading}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
                  >
                  <span>{loading ? "Loading..." : "Load Model"}</span>
                </button>
              </div>
              {error && <div className="p-3 bg-rose-50 border border-rose-200 rounded-lg text-xs text-rose-700">{error}</div>}
              {modelLoaded && (
                <BinarySampleLoader
                  modelShape={modelInputSchema}
                  onFeatureLoaded={handleBinaryFeature}
                  onTenhouUrl={(url) => {}}
                />
              )}
            </div>

            <div className="grid grid-cols-1 gap-6">
              <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-3">
                 <div className="flex space-x-4 mb-4">
                    <TokenSelector tokenCount={tokenCount} selectedToken={selectedToken} onSelect={setSelectedToken} />
                    <LayerSlider maxLayer={maxLayer} currentLayer={currentLayer} onChange={setCurrentLayer} />
                 </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-3">
                  <h3 className="font-semibold text-sm text-slate-800">Embedding</h3>
                  <div className="border border-slate-200 rounded-lg overflow-hidden">
                     <EmbeddingView embedding={embedding} tokenIndex={selectedToken} />
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
                     <OutputView logits={output} schema={modelOutputSchema} />
                  </div>
              </div>
            </div>
          </main>

          <aside className="w-80 bg-white border-l border-slate-200 flex flex-col shrink-0 relative">
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
                    <div key={i} className={`text-xs p-2 rounded ${m.role === 'user' ? 'bg-blue-100 ml-4' : 'bg-slate-100 mr-4'}`}>
                      {m.content}
                    </div>
                 ))}
                 <div ref={endOfMessagesRef} />
              </div>
              <div className="p-3 w-full border-t">
                 <form onSubmit={(e) => { e.preventDefault(); sendChatMessage(); }} className="flex">
                   <input type="text" className="flex-1 text-sm border p-1" value={inputMessage} onChange={e => setInputMessage(e.target.value)} />
                   <button type="submit" className="ml-2 bg-blue-600 text-white px-2 py-1 rounded">Send</button>
                 </form>
              </div>
            </div>
          </aside>
        </div>
      </div>
    </div>
  );
}
