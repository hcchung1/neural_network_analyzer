import React, { useEffect, useState } from "react";
import { Database, FileText, Search } from "lucide-react";

export function DataSourceLibrary() {
  const [allArtifacts, setAllArtifacts] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [filterQuery, setFilterQuery] = useState("");

  const fetchData = async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/v1/runs");
      const data = await res.json();
      if (data.runs) {
        const artifactsPromises = data.runs.map((r: any) =>
          fetch(`/api/v1/runs/${r.run_id}`).then(res => res.json()).catch(() => ({ run: null }))
        );
        const details = await Promise.all(artifactsPromises);
        let artifacts: any[] = [];
        details.forEach(d => {
          if (d.run && d.run.artifacts) {
            d.run.artifacts.forEach((a: any) => {
              artifacts.push({ ...a, run_id: d.run.run_id });
            });
          }
        });
        setAllArtifacts(artifacts);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const dataArtifacts = allArtifacts.filter(a =>
    ["false_positive", "false_negative", "training_history", "prediction", "phase_summary", "turn_metrics", "comparison_table"].includes(a.kind)
  );

  const filteredArtifacts = dataArtifacts.filter(a =>
    a.rel_path.toLowerCase().includes(filterQuery.toLowerCase()) ||
    a.run_id.toLowerCase().includes(filterQuery.toLowerCase()) ||
    a.kind.toLowerCase().includes(filterQuery.toLowerCase())
  );

  return (
    <div className="flex-1 overflow-hidden p-6 flex flex-col space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="font-bold text-slate-900 text-lg flex items-center space-x-2">
          <Database className="w-5 h-5 text-blue-600" />
          <span>Data Source Library</span>
        </h2>
        <div className="flex space-x-3">
          <div className="relative">
            <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
            <input
              type="text"
              placeholder="Filter datasets..."
              value={filterQuery}
              onChange={e => setFilterQuery(e.target.value)}
              className="pl-9 pr-4 py-1.5 text-sm border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white"
            />
          </div>
          <button
            onClick={fetchData}
            disabled={loading}
            className="px-3 py-1.5 bg-white border border-slate-200 hover:bg-slate-50 text-slate-700 rounded-lg text-xs font-medium transition-colors cursor-pointer shadow-sm"
          >
            Refresh
          </button>
        </div>
      </div>

      <div className="flex-1 bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden flex flex-col">
        <div className="overflow-y-auto flex-1 p-0">
          <table className="w-full text-left text-xs text-slate-600">
            <thead className="bg-slate-50 border-b border-slate-200 font-semibold text-slate-700 uppercase tracking-wider sticky top-0">
              <tr>
                <th className="px-5 py-3">Dataset Kind</th>
                <th className="px-5 py-3">Source Run</th>
                <th className="px-5 py-3">Relative Path</th>
                <th className="px-5 py-3">Size</th>
                <th className="px-5 py-3">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {loading && allArtifacts.length === 0 ? (
                <tr>
                  <td colSpan={5} className="px-5 py-12 text-center text-slate-400">Loading catalog...</td>
                </tr>
              ) : filteredArtifacts.length === 0 ? (
                <tr>
                  <td colSpan={5} className="px-5 py-12 text-center text-slate-400 italic">No CSV/binary catalogs match your search.</td>
                </tr>
              ) : filteredArtifacts.map((file, i) => (
                <tr key={i} className="hover:bg-slate-50/80 transition-colors">
                  <td className="px-5 py-3 font-semibold text-slate-800">
                    <div className="flex items-center space-x-2">
                      <FileText className="w-4 h-4 text-blue-500 shrink-0" />
                      <span className="capitalize">{file.kind.replace(/_/g, ' ')}</span>
                    </div>
                  </td>
                  <td className="px-5 py-3 text-slate-500 font-mono text-[11px]">{file.run_id}</td>
                  <td className="px-5 py-3 text-slate-500">{file.rel_path}</td>
                  <td className="px-5 py-3 font-mono text-slate-600">{(file.size / 1024).toFixed(1)} KB</td>
                  <td className="px-5 py-3">
                    <button className="text-blue-600 hover:text-blue-800 font-medium font-sans px-2 py-1 bg-blue-50 rounded hover:bg-blue-100 transition-colors">
                      Inspect
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
