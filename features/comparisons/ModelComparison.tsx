import React, { useState, useCallback } from "react";
import { GitCompare, Layers, Scale, AlertCircle, Loader2, BarChart3 } from "lucide-react";
import dynamic from "next/dynamic";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

// ---------------------------------------------------------------------------
// Types (mirrors Python CompareResponse)
// ---------------------------------------------------------------------------

interface WeightDeltaLayer {
  layer_name: string;
  shape: number[];
  l2_norm_a: number;
  l2_norm_b: number;
  delta_l2_norm: number;
  delta_mean: number;
  delta_std: number;
  delta_abs_max: number;
  cosine_similarity: number;
  hist_counts: number[];
  hist_bin_edges: number[];
}

interface AttentionSimilarityLayer {
  layer_index: number;
  cosine_similarity: number;
  diff_matrix: number[][];
  matrix_size: number;
}

interface CompareResult {
  status: string;
  model_a_name: string;
  model_b_name: string;
  total_params_a: number;
  total_params_b: number;
  weight_deltas: WeightDeltaLayer[];
  attention_similarity: AttentionSimilarityLayer[];
  mean_weight_cosine: number;
  mean_attention_cosine: number | null;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ModelComparison() {
  const [targetModelA, setTargetModelA] = useState("");
  const [targetModelB, setTargetModelB] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<CompareResult | null>(null);

  // Which weight-delta layer is selected for detail histogram
  const [selectedDeltaIdx, setSelectedDeltaIdx] = useState(0);

  // Which attention layer is selected for detail heatmap
  const [selectedAttnIdx, setSelectedAttnIdx] = useState(0);

  const runComparison = useCallback(async () => {
    if (!targetModelA || !targetModelB) return;
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch("/api/v1/models/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          checkpoint_a: targetModelA.trim(),
          checkpoint_b: targetModelB.trim(),
          device: "cpu",
        }),
      });

      const data = await res.json();

      if (!res.ok) {
        const msg =
          data?.error?.message || data?.detail?.error?.message || "Unknown error";
        throw new Error(msg);
      }

      setResult(data);
      setSelectedDeltaIdx(0);
      setSelectedAttnIdx(0);
    } catch (e: any) {
      setError(e.message || "Failed to compare models");
    } finally {
      setIsLoading(false);
    }
  }, [targetModelA, targetModelB]);

  // ---------------------------------------------------------------------------
  // Derived chart data
  // ---------------------------------------------------------------------------

  // Weight delta overview: bar chart of cosine similarity per layer
  const weightCosineTrace = result
    ? {
        x: result.weight_deltas.map((d) => d.layer_name.replace(/\./g, ".​")),
        y: result.weight_deltas.map((d) => d.cosine_similarity),
        type: "bar" as const,
        marker: {
          color: result.weight_deltas.map((d) =>
            d.cosine_similarity > 0.99
              ? "#22c55e"
              : d.cosine_similarity > 0.9
              ? "#f59e0b"
              : "#ef4444"
          ),
        },
        hovertemplate: "%{x}<br>Cosine Sim: %{y:.6f}<extra></extra>",
      }
    : null;

  // Selected weight layer histogram
  const selectedDelta =
    result && result.weight_deltas.length > selectedDeltaIdx
      ? result.weight_deltas[selectedDeltaIdx]
      : null;

  const histTrace = selectedDelta
    ? {
        x: selectedDelta.hist_bin_edges
          .slice(0, -1)
          .map((e, i) => (e + selectedDelta.hist_bin_edges[i + 1]) / 2),
        y: selectedDelta.hist_counts,
        type: "bar" as const,
        marker: { color: "#6366f1" },
        hovertemplate: "Δ: %{x:.5f}<br>Count: %{y}<extra></extra>",
      }
    : null;

  // Attention diff heatmap
  const selectedAttn =
    result && result.attention_similarity.length > selectedAttnIdx
      ? result.attention_similarity[selectedAttnIdx]
      : null;

  const heatmapTrace = selectedAttn
    ? {
        z: selectedAttn.diff_matrix,
        type: "heatmap" as const,
        colorscale: [
          [0, "#2563eb"],
          [0.5, "#fafafa"],
          [1, "#dc2626"],
        ] as Array<[number, string]>,
        hovertemplate: "Row %{y}, Col %{x}<br>Δ: %{z:.5f}<extra></extra>",
        showscale: true,
        colorbar: { title: "A − B", thickness: 12 },
      }
    : null;

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-6">
      {/* ---- Input card ---- */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6">
        <h2 className="font-bold text-slate-900 text-lg flex items-center space-x-2 mb-4">
          <GitCompare className="w-5 h-5 text-blue-600" />
          <span>Model Comparison</span>
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-2">
            <label className="text-sm font-semibold text-slate-700">
              Model A Checkpoint Path
            </label>
            <input
              type="text"
              value={targetModelA}
              onChange={(e) => setTargetModelA(e.target.value)}
              placeholder="e.g. output/checkpoint_A.pth"
              className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 font-mono"
            />
          </div>
          <div className="space-y-2">
            <label className="text-sm font-semibold text-slate-700">
              Model B Checkpoint Path
            </label>
            <input
              type="text"
              value={targetModelB}
              onChange={(e) => setTargetModelB(e.target.value)}
              placeholder="e.g. output/checkpoint_B.pth"
              className="w-full px-3 py-2 text-sm border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 font-mono"
            />
          </div>
        </div>

        <div className="mt-4 flex justify-end">
          <button
            onClick={runComparison}
            disabled={!targetModelA || !targetModelB || isLoading}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-colors disabled:opacity-50 flex items-center space-x-2"
          >
            {isLoading && <Loader2 className="w-4 h-4 animate-spin" />}
            <span>{isLoading ? "Comparing…" : "Run Comparison Analysis"}</span>
          </button>
        </div>
      </div>

      {/* ---- Error ---- */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-4 flex items-start space-x-3">
          <AlertCircle className="w-5 h-5 text-red-500 mt-0.5 shrink-0" />
          <div>
            <p className="text-sm font-semibold text-red-800">
              Comparison Failed
            </p>
            <p className="text-sm text-red-700 mt-1">{error}</p>
          </div>
        </div>
      )}

      {/* ---- Summary cards ---- */}
      {result && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <SummaryCard
            label="Model A"
            value={result.model_a_name}
            sub={`${(result.total_params_a / 1e6).toFixed(2)}M params`}
          />
          <SummaryCard
            label="Model B"
            value={result.model_b_name}
            sub={`${(result.total_params_b / 1e6).toFixed(2)}M params`}
          />
          <SummaryCard
            label="Mean Weight Cosine"
            value={result.mean_weight_cosine.toFixed(6)}
            sub={`${result.weight_deltas.length} layers`}
          />
          <SummaryCard
            label="Mean Attn Cosine"
            value={
              result.mean_attention_cosine != null
                ? result.mean_attention_cosine.toFixed(6)
                : "N/A"
            }
            sub={`${result.attention_similarity.length} layers`}
          />
        </div>
      )}

      {/* ---- Weight Delta section ---- */}
      {result && result.weight_deltas.length > 0 && (
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-4">
          <h3 className="font-semibold text-sm text-slate-800 flex items-center space-x-2 border-b border-slate-100 pb-2">
            <Scale className="w-4 h-4 text-slate-500" />
            <span>Weight Delta Metrics</span>
          </h3>

          {/* Overview bar chart */}
          {weightCosineTrace && (
            <Plot
              data={[weightCosineTrace]}
              layout={{
                title: { text: "Per-Layer Cosine Similarity (A vs B)", font: { size: 13 } },
                xaxis: { title: "Layer", tickangle: -45, automargin: true, tickfont: { size: 9 } },
                yaxis: { title: "Cosine Similarity", range: [Math.min(...result!.weight_deltas.map(d => d.cosine_similarity)) - 0.01, 1.005] },
                margin: { l: 60, r: 20, t: 40, b: 120 },
                height: 320,
                paper_bgcolor: "transparent",
                plot_bgcolor: "transparent",
              }}
              config={{ responsive: true, displayModeBar: false }}
              style={{ width: "100%" }}
            />
          )}

          {/* Layer picker + detail histogram */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 pt-2">
            <div className="space-y-2">
              <label className="text-xs font-medium text-slate-500 uppercase tracking-wide">
                Select Layer
              </label>
              <select
                className="w-full text-sm border border-slate-300 rounded-lg px-2 py-1.5 font-mono focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={selectedDeltaIdx}
                onChange={(e) => setSelectedDeltaIdx(Number(e.target.value))}
              >
                {result!.weight_deltas.map((d, i) => (
                  <option key={i} value={i}>
                    {d.layer_name} — cos {d.cosine_similarity.toFixed(4)}
                  </option>
                ))}
              </select>

              {selectedDelta && (
                <div className="bg-slate-50 rounded-lg p-3 text-xs font-mono text-slate-600 space-y-1">
                  <div>Shape: [{selectedDelta.shape.join(", ")}]</div>
                  <div>‖A‖₂: {selectedDelta.l2_norm_a.toFixed(4)}</div>
                  <div>‖B‖₂: {selectedDelta.l2_norm_b.toFixed(4)}</div>
                  <div>‖Δ‖₂: {selectedDelta.delta_l2_norm.toFixed(4)}</div>
                  <div>Δ mean: {selectedDelta.delta_mean.toExponential(3)}</div>
                  <div>Δ std: {selectedDelta.delta_std.toExponential(3)}</div>
                  <div>Δ |max|: {selectedDelta.delta_abs_max.toExponential(3)}</div>
                </div>
              )}
            </div>

            <div>
              {histTrace && (
                <Plot
                  data={[histTrace]}
                  layout={{
                    title: { text: `Δ Distribution: ${selectedDelta!.layer_name}`, font: { size: 12 } },
                    xaxis: { title: "Weight Δ (A − B)" },
                    yaxis: { title: "Count" },
                    margin: { l: 50, r: 20, t: 36, b: 50 },
                    height: 260,
                    paper_bgcolor: "transparent",
                    plot_bgcolor: "transparent",
                    bargap: 0.05,
                  }}
                  config={{ responsive: true, displayModeBar: false }}
                  style={{ width: "100%" }}
                />
              )}
            </div>
          </div>
        </div>
      )}

      {/* ---- Attention Cosine Similarity section ---- */}
      {result && result.attention_similarity.length > 0 && (
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-4">
          <h3 className="font-semibold text-sm text-slate-800 flex items-center space-x-2 border-b border-slate-100 pb-2">
            <Layers className="w-4 h-4 text-slate-500" />
            <span>Attention Matrix Cosine Similarity</span>
          </h3>

          {/* Overview bar chart for attention cosine */}
          <Plot
            data={[
              {
                x: result.attention_similarity.map((a) => `Layer ${a.layer_index}`),
                y: result.attention_similarity.map((a) => a.cosine_similarity),
                type: "bar" as const,
                marker: {
                  color: result.attention_similarity.map((a) =>
                    a.cosine_similarity > 0.99
                      ? "#22c55e"
                      : a.cosine_similarity > 0.9
                      ? "#f59e0b"
                      : "#ef4444"
                  ),
                },
                hovertemplate:
                  "Layer %{x}<br>Cosine Sim: %{y:.6f}<extra></extra>",
              },
            ]}
            layout={{
              title: { text: "Per-Layer Attention Cosine Similarity", font: { size: 13 } },
              xaxis: { title: "Attention Layer" },
              yaxis: {
                title: "Cosine Similarity",
                range: [
                  Math.min(
                    ...result.attention_similarity.map(
                      (a) => a.cosine_similarity
                    )
                  ) - 0.01,
                  1.005,
                ],
              },
              margin: { l: 60, r: 20, t: 40, b: 50 },
              height: 260,
              paper_bgcolor: "transparent",
              plot_bgcolor: "transparent",
            }}
            config={{ responsive: true, displayModeBar: false }}
            style={{ width: "100%" }}
          />

          {/* Layer picker + heatmap */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 pt-2">
            <div className="space-y-2">
              <label className="text-xs font-medium text-slate-500 uppercase tracking-wide">
                Select Attention Layer
              </label>
              <select
                className="w-full text-sm border border-slate-300 rounded-lg px-2 py-1.5 font-mono focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={selectedAttnIdx}
                onChange={(e) => setSelectedAttnIdx(Number(e.target.value))}
              >
                {result!.attention_similarity.map((a, i) => (
                  <option key={i} value={i}>
                    Layer {a.layer_index} — cos{" "}
                    {a.cosine_similarity.toFixed(4)}
                  </option>
                ))}
              </select>

              {selectedAttn && (
                <div className="bg-slate-50 rounded-lg p-3 text-xs font-mono text-slate-600 space-y-1">
                  <div>Layer Index: {selectedAttn.layer_index}</div>
                  <div>
                    Cosine Similarity: {selectedAttn.cosine_similarity.toFixed(6)}
                  </div>
                  <div>
                    Matrix Size: {selectedAttn.matrix_size} ×{" "}
                    {selectedAttn.matrix_size}
                  </div>
                </div>
              )}
            </div>

            <div>
              {heatmapTrace && (
                <Plot
                  data={[heatmapTrace]}
                  layout={{
                    title: {
                      text: `Attention Δ Heatmap: Layer ${selectedAttn!.layer_index}`,
                      font: { size: 12 },
                    },
                    xaxis: { title: "Key Position" },
                    yaxis: { title: "Query Position", autorange: "reversed" as const },
                    margin: { l: 50, r: 20, t: 36, b: 50 },
                    height: 340,
                    paper_bgcolor: "transparent",
                    plot_bgcolor: "transparent",
                  }}
                  config={{ responsive: true, displayModeBar: false }}
                  style={{ width: "100%" }}
                />
              )}
            </div>
          </div>
        </div>
      )}

      {/* ---- Empty state when comparison returned no attention data ---- */}
      {result && result.attention_similarity.length === 0 && (
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
          <h3 className="font-semibold text-sm text-slate-800 flex items-center space-x-2 border-b border-slate-100 pb-2">
            <Layers className="w-4 h-4 text-slate-500" />
            <span>Attention Matrix Cosine Similarity</span>
          </h3>
          <div className="h-32 flex items-center justify-center text-sm text-slate-400">
            <BarChart3 className="w-4 h-4 mr-2" />
            No attention data captured — the model may not expose attention
            weights via hooks.
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Small helper component
// ---------------------------------------------------------------------------

function SummaryCard({
  label,
  value,
  sub,
}: {
  label: string;
  value: string;
  sub: string;
}) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
      <p className="text-xs font-medium text-slate-500 uppercase tracking-wide">
        {label}
      </p>
      <p className="text-lg font-bold text-slate-900 mt-1 truncate" title={value}>
        {value}
      </p>
      <p className="text-xs text-slate-400 mt-0.5">{sub}</p>
    </div>
  );
}
