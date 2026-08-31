"use client";

import React from "react";
import { Sidebar, TopContextBar } from "@/components/Navigation";
import { ModelComparison } from "@/features/comparisons/ModelComparison";

export default function ComparePage() {
  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar activeTab="compare" />
      <div className="flex-1 flex flex-col min-w-0 bg-slate-50">
        <TopContextBar />
        <main className="flex-1 flex flex-col min-h-0 overflow-y-auto">
          <ModelComparison />
        </main>
      </div>
    </div>
  );
}
