"use client";

import React from "react";
import { Sidebar, TopContextBar } from "@/components/Navigation";
import TrainingResultsComponent from "@/features/inference/TrainingResultsComponent";

export default function RunsPage() {
  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar activeTab="runs" />
      <div className="flex-1 flex flex-col min-w-0 bg-slate-50">
        <TopContextBar />
        <main className="flex-1 overflow-auto p-4">
          <TrainingResultsComponent onBack={() => {}} />
        </main>
      </div>
    </div>
  );
}
