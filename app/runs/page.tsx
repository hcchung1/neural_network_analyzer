"use client";

import React, { useEffect, useState } from "react";
import { Sidebar, TopContextBar } from "@/components/Navigation";
import { FolderKanban, RefreshCw, FileText, Activity } from "lucide-react";

export default function RunsPage() {
  const [runs, setRuns] = useState<any[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [runDetails, setRunDetails] = useState<any | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingDetails, setLoadingDetails] = useState(false);

  const fetchRuns = async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/v1/runs");
      const data = await res.json();
      if (data.runs) {
        setRuns(data.runs);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const fetchRunDetails = async (runId: string) => {
    setLoadingDetails(true);
    setSelectedRunId(runId);
    setRunDetails(null);
    try {
      const res = await fetch(`/api/v1/runs/${runId}`);
      const data = await res.json();
      if (data.run) {
        setRunDetails(data.run);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoadingDetails(false);
    }
  };

  useEffect(() => {
    fetchRuns();
  }, []);

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar activeTab="runs" />
      <div className="flex-1 flex flex-col min-w-0 bg-slate-50">
        <TopContextBar />

        <main className="flex-1 overflow-y-auto p-6 space-y-6 flex gap-6">
          <div className="w-1/3 bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden flex flex-col">
            <div className="flex items-center justify-between p-4 border-b border-slate-200">
              <h2 className="font-bold text-slate-900 text-lg flex items-center space-x-2">
                <FolderKanban className="w-5 h-5 text-blue-600" />
                <span>Runs Catalog</span>
              </h2>
              <button
                onClick={fetchRuns}
                disabled={loading}
                className="p-1.5 text-slate-400 hover:text-slate-600 hover:bg-slate-50 rounded transition-colors"
                title="Refresh runs"
              >
                <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto">
              <ul className="divide-y divide-slate-100">
                {runs.map((r, i) => (
                  <li key={i}>
                    <button
                      onClick={() => fetchRunDetails(r.run_id)}
                      className={`w-full text-left px-4 py-3 hover:bg-slate-50 transition-colors flex items-center justify-between ${selectedRunId === r.run_id ? 'bg-blue-50 border-l-4 border-blue-500 pl-3' : ''}`}
                    >
                      <div>
                        <div className="font-medium text-slate-800">{r.run_id}</div>
                        <div className="text-xs text-slate-500 flex items-center space-x-2 mt-1">
                          <span className="font-mono bg-slate-100 px-1.5 py-0.5 rounded">{r.model_family}</span>
                          <span>{new Date(r.created_at).toLocaleString()}</span>
                        </div>
                      </div>
                    </button>
                  </li>
                ))}
                {runs.length === 0 && (
                  <li className="px-4 py-8 text-center text-slate-400 text-sm italic">
                    No runs found. Rescan required.
                  </li>
                )}
              </ul>
            </div>
          </div>

          <div className="flex-1 flex flex-col min-h-0 bg-white rounded-xl border border-slate-200 shadow-sm p-5 space-y-4 overflow-hidden">
            {selectedRunId ? (
              loadingDetails ? (
                <div className="flex-1 flex items-center justify-center text-slate-400 flex-col space-y-2">
                  <RefreshCw className="w-6 h-6 animate-spin" />
                  <span className="text-sm">Loading details...</span>
                </div>
              ) : runDetails ? (
                <div className="flex flex-col h-full overflow-hidden">
                  <div className="border-b border-slate-200 pb-4 mb-4">
                    <h3 className="text-xl font-bold text-slate-900">{runDetails.run_id}</h3>
                    <div className="text-sm text-slate-500 flex items-center space-x-4 mt-2">
                      <span className="flex items-center space-x-1">
                        <Activity className="w-4 h-4" />
                        <span>{runDetails.model_family}</span>
                      </span>
                      <span>Created: {new Date(runDetails.created_at).toLocaleString()}</span>
                      <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-slate-100 text-slate-700">
                        {runDetails.is_legacy ? 'Legacy' : 'Migrated'}
                      </span>
                      <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-emerald-50 text-emerald-700">
                        {runDetails.confidence}
                      </span>
                    </div>
                  </div>

                  <h4 className="font-semibold text-slate-700 mb-3 text-sm">Artifacts ({runDetails.artifacts?.length || 0})</h4>
                  <div className="flex-1 overflow-y-auto">
                    <table className="w-full text-left text-xs text-slate-600">
                      <thead className="bg-slate-50 border-y border-slate-200 font-semibold text-slate-700 uppercase tracking-wider sticky top-0">
                        <tr>
                          <th className="px-4 py-2">Kind</th>
                          <th className="px-4 py-2">Path</th>
                          <th className="px-4 py-2">Size</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100 font-mono">
                        {runDetails.artifacts?.map((file: any, i: number) => (
                          <tr key={i} className="hover:bg-slate-50/80 transition-colors">
                            <td className="px-4 py-2 font-medium text-slate-800">
                              <div className="flex items-center space-x-2">
                                <FileText className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                                <span>{file.kind}</span>
                              </div>
                            </td>
                            <td className="px-4 py-2 text-slate-500 truncate max-w-[200px]" title={file.rel_path}>{file.rel_path}</td>
                            <td className="px-4 py-2 text-slate-700">{file.size.toLocaleString()} B</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ) : (
                <div className="flex-1 flex items-center justify-center text-slate-400 text-sm">
                  Run details could not be loaded.
                </div>
              )
            ) : (
              <div className="flex-1 flex items-center justify-center text-slate-400 text-sm italic">
                Select a run from the catalog to view details, predictions, and history.
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
