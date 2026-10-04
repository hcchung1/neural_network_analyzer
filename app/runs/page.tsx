"use client";

import React, { useEffect, useState } from "react";
import { Sidebar, TopContextBar } from "@/components/Navigation";
import TrainingResultsComponent from "@/features/inference/TrainingResultsComponent";

export default function RunsPage() {
  const [requestedFolder, setRequestedFolder] = useState<{ path: string; requestId: number } | null>(null);
  useEffect(() => {
    const path = new URLSearchParams(window.location.search).get("folder");
    if (path) setRequestedFolder({ path, requestId: 1 });
  }, []);
  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar activeTab="runs" />
      <div className="flex-1 flex flex-col min-w-0 bg-slate-50">
        <TopContextBar />
        <main className="flex-1 overflow-auto p-4">
          <TrainingResultsComponent onBack={() => {}} requestedFolder={requestedFolder} />
        </main>
      </div>
    </div>
  );
}
