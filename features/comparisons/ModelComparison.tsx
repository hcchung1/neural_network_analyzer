import React, { useState } from "react";
import { GitCompare, Layers, Scale } from "lucide-react";

export function ModelComparison() {
  const [targetModelA, setTargetModelA] = useState("");
  const [targetModelB, setTargetModelB] = useState("");
  const [isComparing, setIsComparing] = useState(false);

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-6">
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6">
        <h2 className="font-bold text-slate-900 text-lg flex items-center space-x-2 mb-4">
          <GitCompare className="w-5 h-5 text-blue-600" />
          <span>Model Comparison</span>
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-2">
            <label className="text-sm font-semibold text-slate-700">Model A Checkpoint Path</label>
            <input
              type="text"
              value={targetModelA}
              onChange={e => setTargetModelA(e.target.value)}
              placeholder="e.g. output/checkpoint_A.pth"
              className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 font-mono"
            />
          </div>
          <div className="space-y-2">
            <label className="text-sm font-semibold text-slate-700">Model B Checkpoint Path</label>
            <input
              type="text"
              value={targetModelB}
              onChange={e => setTargetModelB(e.target.value)}
              placeholder="e.g. output/checkpoint_B.pth"
              className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 font-mono"
            />
          </div>
        </div>

        <div className="mt-4 flex justify-end">
          <button
            onClick={() => setIsComparing(true)}
            disabled={!targetModelA || !targetModelB}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
          >
            Run Comparison Analysis
          </button>
        </div>
      </div>

      {isComparing && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-4">
            <h3 className="font-semibold text-sm text-slate-800 flex items-center space-x-2 border-b border-slate-100 pb-2">
              <Scale className="w-4 h-4 text-slate-500" />
              <span>Weight Delta Metrics</span>
            </h3>
            <div className="h-64 bg-slate-50 rounded-lg border border-dashed border-slate-300 flex items-center justify-center text-xs text-slate-400">
              Parameter Distribution Shift Histogram
            </div>
          </div>
          <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-4">
            <h3 className="font-semibold text-sm text-slate-800 flex items-center space-x-2 border-b border-slate-100 pb-2">
              <Layers className="w-4 h-4 text-slate-500" />
              <span>Attention Matrix Cosine Similarity</span>
            </h3>
            <div className="h-64 bg-slate-50 rounded-lg border border-dashed border-slate-300 flex items-center justify-center text-xs text-slate-400">
              Heatmap Differences Overlay
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
