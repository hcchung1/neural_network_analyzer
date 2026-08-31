"use client";

import React, { useState, useRef, useEffect } from "react";
import { Sidebar, TopContextBar } from "@/components/Navigation";
import { Play, Layers, Cpu, Activity, Database, AlertCircle, Send, User, Bot, Loader2 } from "lucide-react";

export default function ExplorePage() {
  const [checkpointPath, setCheckpointPath] = useState("");
  const [modelLoaded, setModelLoaded] = useState(false);
  const [modelName, setModelName] = useState("transOriginal");
  const [capabilities, setCapabilities] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [inspectorTab, setInspectorTab] = useState<"details" | "assistant">("details");

  // Assistant Chat State
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
        body: JSON.stringify({ checkpoint_path: checkpointPath || undefined }),
      });
      const data = await res.json();
      if (!res.ok || data.error) {
        setError(data.error?.message || "Failed to load model");
      } else {
        setModelLoaded(true);
        setModelName(data.model_name);
        setCapabilities(data.capabilities);
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
      if (data.content) {
        setMessages(prev => [...prev, { role: "assistant", content: data.content }]);
      }
    } catch (err: any) {
      console.error("Chat error:", err);
      setMessages(prev => [...prev, { role: "assistant", content: "Sorry, an error occurred while connecting to the assistant API." }]);
    } finally {
      setChatLoading(false);
    }
  };

  useEffect(() => {
    if (inspectorTab === 'assistant') {
      endOfMessagesRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, inspectorTab]);

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar activeTab="explore" />
      <div className="flex-1 flex flex-col min-w-0 bg-slate-50">
        <TopContextBar modelName={modelName} sidecarHealthy={true} />

        <div className="flex-1 flex overflow-hidden">
          {/* Central Analysis Canvas */}
          <main className="flex-1 overflow-y-auto p-6 space-y-6">
            {/* Control Header Card */}
            <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="font-semibold text-slate-900 flex items-center space-x-2">
                  <Cpu className="w-5 h-5 text-blue-600" />
                  <span>Model Control & Source Selection</span>
                </h2>
                {modelLoaded && (
                  <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-emerald-50 text-emerald-700 border border-emerald-200">
                    Strict Mode Verified
                  </span>
                )}
              </div>

              <div className="flex space-x-3">
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
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-colors disabled:opacity-50 flex items-center space-x-2 shrink-0"
                >
                  <Play className="w-4 h-4 fill-current" />
                  <span>{loading ? "Loading..." : "Load Model"}</span>
                </button>
              </div>

              {error && (
                <div className="p-3 bg-rose-50 border border-rose-200 rounded-lg text-xs text-rose-700 flex items-start space-x-2">
                  <AlertCircle className="w-4 h-4 text-rose-500 shrink-0 mt-0.5" />
                  <span>{error}</span>
                </div>
              )}
            </div>

            {/* Capability-driven Panels Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Input Features / Embedding Panel */}
              <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-3">
                <h3 className="font-semibold text-sm text-slate-800 flex items-center space-x-2">
                  <Database className="w-4 h-4 text-slate-500" />
                  <span>Input Features & Projection</span>
                </h3>
                <div className="h-48 bg-slate-50 rounded-lg border border-dashed border-slate-300 flex items-center justify-center text-xs text-slate-400 p-4 text-center">
                  {capabilities?.has_input_token ? "Dense Input Features Loaded. Select an activation to inspect token embeddings." : "No Active Features Loaded. Load a model to begin."}
                </div>
              </div>

              {/* Attention Heatmap Panel */}
              <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-3">
                <h3 className="font-semibold text-sm text-slate-800 flex items-center space-x-2">
                  <Layers className="w-4 h-4 text-slate-500" />
                  <span>Attention Matrix (Layer 0)</span>
                </h3>
                {capabilities && !capabilities.has_attention_heatmap ? (
                  <div className="h-48 bg-slate-100/50 rounded-lg border border-slate-200 flex items-center justify-center text-xs text-slate-500 italic p-4 text-center">
                    Attention heatmap is not available for {modelName} ({capabilities.family} family)
                  </div>
                ) : (
                  <div className="h-48 bg-slate-50 rounded-lg border border-dashed border-slate-300 flex items-center justify-center text-xs text-slate-400">
                    {modelLoaded ? "Attention Heatmap Ready. Awaiting inference session." : "Waiting for model load."}
                  </div>
                )}
              </div>
            </div>
          </main>

          {/* Right Inspector Drawer */}
          <aside className="w-80 bg-white border-l border-slate-200 flex flex-col shrink-0 relative">
            <div className="flex border-b border-slate-200 p-2 space-x-1 shrink-0 bg-white z-10 sticky top-0">
              <button
                onClick={() => setInspectorTab("details")}
                className={`flex-1 px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${
                  inspectorTab === "details"
                    ? "bg-slate-100 text-slate-900"
                    : "text-slate-500 hover:text-slate-700 hover:bg-slate-50"
                }`}
              >
                Details
              </button>
              <button
                onClick={() => setInspectorTab("assistant")}
                className={`flex-1 px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${
                  inspectorTab === "assistant"
                    ? "bg-slate-100 text-slate-900 bg-blue-50/50 text-blue-700 border border-blue-100"
                    : "text-slate-500 hover:text-slate-700 hover:bg-slate-50"
                }`}
              >
                Assistant
              </button>
            </div>

            <div className="flex-1 overflow-y-auto w-full relative">
              {inspectorTab === "details" ? (
                <div className="p-4 space-y-5">
                  <div className="space-y-3">
                    <h3 className="font-semibold text-sm text-slate-900 border-b border-slate-200 pb-2 flex items-center space-x-2">
                       <Activity className="w-4 h-4 text-slate-400" />
                       <span>Capabilities</span>
                    </h3>
                    {capabilities ? (
                      <div className="bg-slate-50 border border-slate-100 rounded-lg p-3 text-xs text-slate-600 space-y-2.5 font-mono shadow-sm">
                        <div className="flex justify-between">
                          <span className="text-slate-400">Family:</span>
                          <span className="text-slate-800 font-semibold">{capabilities.family || "Transformer"}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-400">Input Token:</span>
                          <span className={capabilities.has_input_token ? "text-emerald-600" : "text-slate-400"}>
                            {capabilities.has_input_token ? "Supported" : "No"}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-400">Attention:</span>
                          <span className={capabilities.has_attention_heatmap ? "text-emerald-600" : "text-slate-400"}>
                            {capabilities.has_attention_heatmap ? "Yes" : "No"}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-400">Conv Map:</span>
                          <span className={capabilities.has_convolution_feature_map ? "text-emerald-600" : "text-slate-400"}>
                             {capabilities.has_convolution_feature_map ? "Yes" : "No"}
                          </span>
                        </div>
                      </div>
                    ) : (
                      <div className="text-xs text-slate-400 italic py-2">No model capabilities loaded.</div>
                    )}
                  </div>
                </div>
              ) : (
                <div className="h-full flex flex-col divide-y divide-slate-100 absolute inset-0">
                  <div className="flex-1 overflow-y-auto p-4 space-y-4 text-sm bg-slate-50/50">
                    {messages.length === 0 ? (
                      <div className="h-full flex flex-col items-center justify-center text-slate-400 space-y-3 px-4 text-center">
                        <Bot className="w-10 h-10 text-slate-300" />
                        <p className="text-sm">Contextual chat is ready. Load a model and select an artifact to begin.</p>
                      </div>
                    ) : (
                      messages.map((msg, i) => (
                        <div key={i} className={`flex space-x-2 items-start ${msg.role === 'user' ? 'justify-end' : ''}`}>
                          {msg.role === 'assistant' && (
                            <div className="w-6 h-6 rounded bg-blue-100 flex items-center justify-center shrink-0 mt-0.5">
                              <Bot className="w-3.5 h-3.5 text-blue-600" />
                            </div>
                          )}
                          <div className={`p-3 rounded-xl max-w-[85%] text-[13px] leading-relaxed ${
                            msg.role === 'user'
                              ? 'bg-blue-600 text-white rounded-tr-sm'
                              : 'bg-white border border-slate-200 text-slate-700 shadow-sm rounded-tl-sm'
                          }`}>
                            {msg.content}
                          </div>
                          {msg.role === 'user' && (
                            <div className="w-6 h-6 rounded bg-slate-200 flex items-center justify-center shrink-0 mt-0.5">
                              <User className="w-3.5 h-3.5 text-slate-600" />
                            </div>
                          )}
                        </div>
                      ))
                    )}
                    {chatLoading && (
                      <div className="flex space-x-2 items-start">
                        <div className="w-6 h-6 rounded bg-blue-100 flex items-center justify-center shrink-0 mt-0.5">
                          <Loader2 className="w-3.5 h-3.5 text-blue-600 animate-spin" />
                        </div>
                        <div className="p-3 bg-white border border-slate-200 text-slate-400 rounded-xl rounded-tl-sm text-[13px] shadow-sm">
                          Analyzing context...
                        </div>
                      </div>
                    )}
                    <div ref={endOfMessagesRef} />
                  </div>

                  {/* Chat Input */}
                  <div className="p-3 bg-white">
                    <form
                      onSubmit={(e) => { e.preventDefault(); sendChatMessage(); }}
                      className="flex items-center space-x-2"
                    >
                      <input
                        type="text"
                        value={inputMessage}
                        onChange={(e) => setInputMessage(e.target.value)}
                        placeholder="Ask about attention patterns..."
                        className="flex-1 px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:border-blue-300 focus:ring-1 focus:ring-blue-300 bg-slate-50 transition-all font-sans"
                        disabled={chatLoading}
                      />
                      <button
                        type="submit"
                        disabled={!inputMessage.trim() || chatLoading}
                        className="p-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg disabled:opacity-50 transition-colors shadow-sm disabled:shadow-none"
                      >
                        <Send className="w-4 h-4" />
                      </button>
                    </form>
                  </div>
                </div>
              )}
            </div>
          </aside>
        </div>
      </div>
    </div>
  );
}
