"use client";

import React from "react";
import { Sidebar, TopContextBar } from "@/components/Navigation";
import { DataSourceLibrary } from "@/features/sources/DataSourceLibrary";

export default function DataPage() {
  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar activeTab="data" />
      <div className="flex-1 flex flex-col min-w-0 bg-slate-50">
        <TopContextBar />
        <main className="flex-1 flex flex-col min-h-0 overflow-y-auto">
          <DataSourceLibrary />
        </main>
      </div>
    </div>
  );
}
