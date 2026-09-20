"use client";
import { useState } from "react";
import { Users, Link as LinkIcon, MessageSquare, Pin, Radio, LogOut } from "lucide-react";
import { WorkspaceController } from "./useWorkspace";
import { ViewState } from "@/lib/collaboration-types";

export function WorkspacePanel({ workspace: w, view, onNavigate }: {
  workspace: WorkspaceController; view: ViewState; onNavigate: (v: ViewState) => void;
}) {
  const [name, setName] = useState("");
  const [text, setText] = useState("");
  const [kind, setKind] = useState<"chat" | "annotation">("chat");
  const [sending, setSending] = useState(false);
  const [share, setShare] = useState("");
  const [shareMessage, setShareMessage] = useState("");
  const room = w.room;
  const button = "rounded-lg border border-slate-300 px-3 py-2 text-sm hover:bg-slate-50 disabled:opacity-40 disabled:cursor-not-allowed";
  const send = async (event: React.FormEvent) => {
    event.preventDefault(); if (!text.trim() || !room || sending) return;
    setSending(true);
    const result = await w.act({ type: kind, text, ...(kind === "annotation" ? { view, analysisRevision: room.analysisRevision } : {}) });
    if (result) setText("");
    setSending(false);
  };
  const copy = async () => {
    const link = window.location.href; setShare(link);
    try { await navigator.clipboard.writeText(link); setShareMessage("分享連結已複製"); }
    catch { setShareMessage("請複製下方連結"); }
  };
  return <section aria-label="共同協作工作區" className="rounded-xl border border-indigo-200 bg-white shadow-sm p-5 space-y-4">
    <div className="flex flex-wrap items-center justify-between gap-3">
      <div className="flex items-center gap-2"><Users className="w-5 h-5 text-indigo-600" /><h2 className="font-semibold text-slate-900">共同協作</h2>
        <span className="text-xs bg-indigo-50 text-indigo-700 px-2 py-1 rounded-full">混合模式</span></div>
      {room && <span role="status" className={`text-xs ${w.connected ? "text-emerald-700" : "text-amber-700"}`}>{w.connected ? "● 已連線" : "○ 重新連線中…"}{room.busy && " · 共同分析進行中"}</span>}
    </div>
    {!room ? <form onSubmit={e => { e.preventDefault(); void w.join(name, w.invite); }} className="flex flex-wrap gap-3 items-end">
      <div className="flex-1 min-w-48"><p className="text-sm text-slate-500 mb-3">{w.invite ? "你已收到工作區邀請。輸入暱稱即可加入。" : "建立工作區並分享連結。各自探索圖表，也能隨時跟隨主持人。"}</p>
        <label className="text-sm font-medium">暱稱<input aria-label="協作暱稱" value={name} onChange={e => setName(e.target.value)} maxLength={40} required className="ml-3 border border-slate-300 rounded-lg px-3 py-2" placeholder="例如：研究員小林" /></label></div>
      <button className="rounded-lg bg-indigo-600 text-white px-4 py-2 text-sm disabled:opacity-40" disabled={w.pending || !!w.identity || !name.trim()}>{w.pending ? "加入中…" : w.invite ? "加入工作區" : "建立工作區"}</button>
    </form> : <>
      <div className="flex flex-wrap gap-2 items-center">
        <span className="text-sm font-medium text-slate-700">{w.isHost ? "你是主持人" : w.following ? "正在跟隨主持人" : "獨立瀏覽"}</span>
        {!w.isHost && <button className={button} aria-pressed={w.following} onClick={() => w.setFollowing(!w.following)}><Radio className="w-4 h-4 inline mr-1" />{w.following ? "停止跟隨" : "跟隨主持人"}</button>}
        <button className={button} onClick={copy}><LinkIcon className="w-4 h-4 inline mr-1" />分享連結</button>
        <button className={button} onClick={() => void w.leave()} disabled={w.isHost && room.busy}><LogOut className="w-4 h-4 inline mr-1" />離開工作區</button>
        {!room.members.some(m => m.id === room.hostId && m.online) && <button className={button} disabled={room.busy} onClick={() => void w.act({ type: "claim" })}>接任主持人</button>}
      </div>
      {share && <div className="text-xs text-slate-500 space-y-1"><p role="status">{shareMessage}</p><input aria-label="工作區分享連結" readOnly value={share} onFocus={e => e.target.select()} className="w-full rounded border px-2 py-1" /><p>連結持有者可加入。跨電腦分享時，請使用大家可連線的伺服器網址。</p></div>}
      <div className="flex flex-wrap gap-2" aria-label="參與者">{room.members.filter(m => m.online || m.id === room.hostId).map(m => <span key={m.id} className="rounded-full bg-slate-100 text-xs text-slate-700 px-3 py-1.5">
        {m.online ? "●" : "○"} {m.name}{m.id === room.hostId ? " · 主持人" : ""}{m.id === w.identity?.memberId ? "（你）" : ""}
        {w.isHost && m.id !== room.hostId && m.online && <button disabled={room.busy} className="ml-2 text-indigo-600 disabled:opacity-40" aria-label={`交接主持人給 ${m.name}`} onClick={() => void w.act({ type: "transfer", target: m.id })}>交接</button>}
      </span>)}</div>
      <p className="text-xs text-slate-500">模型與樣本由主持人切換，全員共用分析結果。手動切換 token 或 layer 會停止跟隨。</p>
      {room.analysis.sample && <p className="text-xs text-indigo-700" data-testid="shared-sample">共同樣本：{String(room.analysis.sample.file_name || "已載入")} · record {String(room.analysis.sample.record_index ?? "—")}</p>}
      <details open className="border-t border-slate-100 pt-3">
        <summary className="text-sm font-medium cursor-pointer">討論與標註 <span className="text-slate-400">({room.entries.length})</span></summary>
        <div className="max-h-52 overflow-y-auto my-3 space-y-2" aria-label="共享討論" aria-live="polite">
          {!room.entries.length && <p className="text-sm text-slate-400 py-2">留下訊息，或標註目前的 token 與 layer。</p>}
          {room.entries.map(entry => <article key={entry.id} className="bg-slate-50 rounded-lg px-3 py-2 text-sm">
            <div className="flex flex-wrap gap-2 items-center text-xs text-slate-500"><span className="font-semibold text-slate-700">{entry.author}</span><time>{new Date(entry.createdAt).toLocaleTimeString()}</time>
              {entry.kind === "annotation" && <button className="text-indigo-600 disabled:text-slate-400" disabled={entry.analysisRevision !== room.analysisRevision} onClick={() => onNavigate(entry.view)}><Pin className="w-3 h-3 inline" /> Token {entry.view.token} · Layer {entry.view.layer}{entry.analysisRevision !== room.analysisRevision && " · 先前分析"}</button>}</div>
            <p className="whitespace-pre-wrap break-words mt-1">{entry.text}</p>
          </article>)}
        </div>
        <form onSubmit={send} className="flex flex-wrap gap-2">
          <select aria-label="訊息類型" value={kind} onChange={e => setKind(e.target.value as typeof kind)} className="border rounded-lg text-sm px-2"><option value="chat">聊天</option><option value="annotation">標註目前位置</option></select>
          <input aria-label="協作訊息" value={text} onChange={e => setText(e.target.value)} maxLength={4000} placeholder={kind === "annotation" ? `標註 Token ${view.token} / Layer ${view.layer}` : "傳送訊息給工作區…"} className="flex-1 min-w-40 border border-slate-300 rounded-lg px-3 py-2 text-sm" />
          <button className={button} disabled={!text.trim() || sending || !w.connected}><MessageSquare className="w-4 h-4 inline mr-1" />傳送</button>
        </form>
      </details>
    </>}
    {w.error && <p role="alert" className="text-sm text-rose-700 bg-rose-50 p-2 rounded">{w.error}</p>}
  </section>;
}
