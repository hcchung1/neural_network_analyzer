"use client";

import React from "react";
import { Sidebar, TopContextBar } from "@/components/Navigation";
import { SettingsPanel } from "@/features/inference/SettingsPanel";

export default function SettingsPage() {
  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar activeTab="settings" />
      <div className="flex-1 flex flex-col min-w-0 bg-slate-50">
        <TopContextBar />
        <main className="flex-1 flex flex-col min-h-0 overflow-y-auto">
          <SettingsPanel />
        </main>
      </div>
    </div>
  );
}
