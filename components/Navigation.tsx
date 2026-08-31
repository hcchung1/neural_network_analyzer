import React from "react";
import Link from "next/link";
import { Compass, FolderKanban, Database, GitCompare, Settings, Activity } from "lucide-react";

export function Sidebar({ activeTab }: { activeTab: string }) {
  const navItems = [
    { id: "explore", label: "Explore", icon: Compass, href: "/explore" },
    { id: "runs", label: "Runs", icon: FolderKanban, href: "/runs" },
    { id: "data", label: "Data", icon: Database, href: "/data" },
    { id: "compare", label: "Compare", icon: GitCompare, href: "/compare" },
    { id: "settings", label: "Settings", icon: Settings, href: "/settings" },
  ];

  return (
    <aside className="w-64 bg-slate-900 text-slate-200 border-r border-slate-800 flex flex-col h-screen select-none">
      <div className="p-4 border-b border-slate-800 flex items-center space-x-3">
        <Activity className="w-6 h-6 text-blue-500" />
        <div>
          <h1 className="font-bold text-sm text-slate-100 tracking-wide">ArchAnalyzer</h1>
          <p className="text-xs text-slate-400">Research Workbench</p>
        </div>
      </div>

      <nav className="flex-1 p-3 space-y-1">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <Link
              key={item.id}
              href={item.href}
              className={`flex items-center space-x-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                isActive
                  ? "bg-blue-600 text-white shadow-sm"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
              }`}
            >
              <Icon className="w-4 h-4" />
              <span>{item.label}</span>
            </Link>
          );
        })}
      </nav>

      <div className="p-3 border-t border-slate-800 text-xs text-slate-500">
        Precision Light UI v0.2
      </div>
    </aside>
  );
}

export function TopContextBar({
  modelName = "transOriginal",
  device = "cpu",
  sidecarHealthy = true,
}: {
  modelName?: string;
  device?: string;
  sidecarHealthy?: boolean;
}) {
  return (
    <header className="h-12 bg-white border-b border-slate-200 px-4 flex items-center justify-between text-xs text-slate-600 font-mono">
      <div className="flex items-center space-x-4">
        <div className="flex items-center space-x-1.5">
          <span className="text-slate-400 font-sans">Model:</span>
          <span className="font-semibold text-slate-900 bg-slate-100 px-2 py-0.5 rounded border border-slate-200">{modelName}</span>
        </div>
        <div className="flex items-center space-x-1.5">
          <span className="text-slate-400 font-sans">Device:</span>
          <span className="bg-slate-100 px-2 py-0.5 rounded border border-slate-200 uppercase">{device}</span>
        </div>
      </div>

      <div className="flex items-center space-x-2 font-sans">
        <span className="text-slate-400">Sidecar Status:</span>
        <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${
          sidecarHealthy ? "bg-emerald-50 text-emerald-700 border border-emerald-200" : "bg-rose-50 text-rose-700 border border-rose-200"
        }`}>
          <span className={`w-1.5 h-1.5 rounded-full mr-1.5 ${sidecarHealthy ? "bg-emerald-500" : "bg-rose-500"}`} />
          {sidecarHealthy ? "Healthy" : "Offline"}
        </span>
      </div>
    </header>
  );
}
