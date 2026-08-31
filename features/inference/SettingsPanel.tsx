import React, { useState, useEffect } from "react";
import { Settings, Server, HardDrive, RefreshCw } from "lucide-react";

export function SettingsPanel() {
  const [healthStatus, setHealthStatus] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [artifactRoot, setArtifactRoot] = useState("output");
  const [isScanning, setIsScanning] = useState(false);
  const [scanStatus, setScanStatus] = useState<any>(null);

  const checkSidecar = async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/v1/health").catch(() => null);
      if (res && res.status === 200) {
        setHealthStatus({ status: "healthy", latency: "..." });
      } else {
        setHealthStatus({ status: "error", timestamp: new Date().toISOString() });
      }
    } catch {
      setHealthStatus({ status: "disconnected" });
    } finally {
      setLoading(false);
    }
  };

  const triggerScan = async () => {
    setIsScanning(true);
    setScanStatus(null);
    try {
      const res = await fetch("/api/v1/runs/scan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ folder_path: artifactRoot })
      });

      if (!res.ok) throw new Error("Failed to start scan");

      const reader = res.body?.getReader();
      if (!reader) return;

      const decoder = new TextDecoder();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = JSON.parse(line.slice(6));
            setScanStatus(data);
          }
        }
      }
    } catch (err: any) {
      setScanStatus({ status: "error", message: err.message });
    } finally {
      setIsScanning(false);
    }
  };

  useEffect(() => {
    checkSidecar();
  }, []);

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-6">
      <div className="flex items-center space-x-2 pb-2 mb-4 border-b border-slate-200">
        <Settings className="w-6 h-6 text-slate-800" />
        <h1 className="text-2xl font-bold text-slate-900">Platform Settings</h1>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6 space-y-6">
          <div className="flex items-center space-x-2">
            <Server className="w-5 h-5 text-blue-500" />
            <h3 className="font-semibold text-slate-800">Python Sidecar Configuration</h3>
          </div>

          <div className="space-y-4">
            <div className="flex items-center justify-between p-3 bg-slate-50 rounded border border-slate-100">
              <div>
                <div className="font-medium text-slate-700 text-sm">Connection Status</div>
                <div className="text-xs text-slate-500 mt-1 capitalize">{healthStatus?.status || "Checking..."}</div>
              </div>
              <button
                onClick={checkSidecar}
                disabled={loading}
                className="p-2 text-slate-400 hover:text-blue-600 bg-white border border-slate-200 rounded shadow-sm transition-colors"
              >
                <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
              </button>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-semibold text-slate-700">Sidecar Internal URL</label>
              <input
                type="text"
                disabled
                value="http://127.0.0.1:8080/internal/v1"
                className="w-full px-3 py-2 text-sm bg-slate-50 border border-slate-200 rounded-lg text-slate-500 font-mono"
              />
              <p className="text-xs text-slate-400">Environment variable: PYTHON_SIDECAR_URL</p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6 space-y-6">
          <div className="flex items-center space-x-2">
            <HardDrive className="w-5 h-5 text-emerald-500" />
            <h3 className="font-semibold text-slate-800">Artifact Catalog Configuration</h3>
          </div>

          <div className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-semibold text-slate-700">Root Directory (Relative to Next.js)</label>
              <input
                type="text"
                value={artifactRoot}
                onChange={(e) => setArtifactRoot(e.target.value)}
                className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg font-mono focus:outline-none focus:ring-2 focus:ring-emerald-500"
              />
            </div>

            <button
              onClick={triggerScan}
              disabled={isScanning}
              className="w-full py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
            >
              {isScanning ? "Scanning..." : "Trigger Full Catalog Scan"}
            </button>

            {scanStatus && (
              <div className={`p-3 text-xs rounded border ${
                scanStatus.status === 'error' || scanStatus.code === 'SCAN_FAILED'
                  ? 'bg-rose-50 border-rose-200 text-rose-700'
                  : 'bg-slate-50 border-slate-200 text-slate-600 font-mono'
              }`}>
                {scanStatus.message || `Scanned ${scanStatus.scanned_files} files, updated ${scanStatus.runs_updated} runs.`}
                {scanStatus.progress && <div className="mt-1">Progress: {Math.round(scanStatus.progress * 100)}%</div>}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
