import "@/app/globals.css";

export const metadata = {
  title: "ArchAnalyzer - Research Workbench",
  description: "Transformer single-token visualization & artifact workbench",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-slate-50 font-sans text-slate-900 antialiased">
        {children}
      </body>
    </html>
  );
}
