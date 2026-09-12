"use client";

import React from "react";
import { Sidebar, TopContextBar } from "@/components/Navigation";
import CsvReaderComponent from "@/features/sources/CsvReaderComponent";

export default function DataPage() {
  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar activeTab="data" />
      <div className="flex-1 flex flex-col min-w-0 bg-slate-50">
        <TopContextBar />
        <main className="flex-1 flex flex-col min-h-0 overflow-y-auto">
          <CsvReaderComponent 
            onBack={() => {}} 
            onAddModel={() => {}} 
            onAddTrainingResults={() => {}} 
          />
        </main>
      </div>
    </div>
  );
}
