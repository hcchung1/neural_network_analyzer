"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { RoomIdentity, RoomPoll, RoomSnapshot } from "@/lib/collaboration-types";
import { mergeRoomSnapshot, startWorkspacePolling } from "@/lib/workspace-polling";

export function useWorkspace() {
  const [identity, setIdentity] = useState<RoomIdentity | null>(null);
  const [room, setRoom] = useState<RoomSnapshot | null>(null);
  const [invite, setInvite] = useState("");
  const [error, setError] = useState("");
  const [connectionError, setConnectionError] = useState("");
  const [connected, setConnected] = useState(false);
  const [pending, setPending] = useState(false);
  const [following, setFollowing] = useState(false);
  const current = useRef<RoomSnapshot | null>(null);
  const activeIdentity = useRef<RoomIdentity | null>(null);
  const queue = useRef<Promise<unknown>>(Promise.resolve());
  const accept = useCallback((data: RoomPoll, who: RoomIdentity) => {
    if (activeIdentity.current !== who) return;
    const next = mergeRoomSnapshot(current.current, data);
    if (next === current.current) return;
    current.current = next;
    setRoom(next);
  }, []);
  useEffect(() => {
    const roomId = new URLSearchParams(window.location.search).get("room") || "";
    setInvite(roomId);
    if (roomId) {
      try {
        const saved = sessionStorage.getItem(`workspace:${roomId}`);
        if (saved) {
          const value = JSON.parse(saved);
          activeIdentity.current = value;
          setIdentity(value);
        }
      } catch { setError("無法還原加入資訊，請重新輸入暱稱。"); }
    }
  }, []);
  useEffect(() => {
    if (!identity) return;
    return startWorkspacePolling({
      hidden: () => document.hidden,
      poll: async signal => {
        const roomRevision = current.current?.revision ?? -1;
        const analysisRevision = current.current?.analysisRevision ?? -1;
        const response = await fetch(`/api/v1/workspaces/${identity.roomId}?revision=${roomRevision}&analysisRevision=${analysisRevision}`, {
          headers: { Authorization: `Bearer ${identity.token}` }, cache: "no-store", signal,
        });
        const data = await response.json();
        if (signal.aborted || activeIdentity.current !== identity) return false;
        if (!response.ok) {
          if (response.status === 401 || response.status === 404) {
            sessionStorage.removeItem(`workspace:${identity.roomId}`);
            activeIdentity.current = null; current.current = null;
            setIdentity(null); setRoom(null); setConnected(false);
            setConnectionError(data.error || "加入憑證已失效，請重新加入工作區。");
            return false;
          }
          throw new Error(data.error || "連線中斷，正在重連。");
        }
        accept(data, identity); setConnected(true); setConnectionError("");
      },
      onError: e => {
        if (activeIdentity.current !== identity) return;
        setConnected(false); setConnectionError(e instanceof Error ? e.message : "連線中斷，正在重連。");
      },
    });
  }, [identity, accept]);
  const join = async (name: string, existing?: string) => {
    setPending(true); setError(""); setConnectionError("");
    try {
      const response = await fetch("/api/v1/workspaces", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name, roomId: existing || undefined }) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error);
      activeIdentity.current = data.identity;
      current.current = null;
      setIdentity(data.identity); accept(data.room, data.identity);
      sessionStorage.setItem(`workspace:${data.identity.roomId}`, JSON.stringify(data.identity));
      const url = new URL(window.location.href); url.searchParams.set("room", data.identity.roomId);
      window.history.replaceState(null, "", url);
      setInvite(data.identity.roomId); setConnected(true); setFollowing(false);
    } catch (e) { setError(e instanceof Error ? e.message : "加入失敗。"); }
    finally { setPending(false); }
  };
  const act = useCallback((action: Record<string, unknown>): Promise<RoomSnapshot | null> => {
    const who = activeIdentity.current;
    if (!who) return Promise.resolve(null);
    const operation = async () => {
      if (activeIdentity.current !== who) return null;
      try {
        setError("");
        const response = await fetch(`/api/v1/workspaces/${who.roomId}`, {
          method: "POST", headers: { "Content-Type": "application/json", Authorization: `Bearer ${who.token}` }, body: JSON.stringify(action),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error);
        if (activeIdentity.current !== who) return null;
        accept(data, who); setConnected(true);
        return data as RoomSnapshot;
      } catch (e) { if (activeIdentity.current === who) setError(e instanceof Error ? e.message : "操作失敗，請重試。"); return null; }
    };
    const result = queue.current.then(operation, operation);
    queue.current = result;
    return result;
  }, [accept]);
  const leave = async () => {
    if (!identity || !await act({ type: "leave" })) return;
    sessionStorage.removeItem(`workspace:${identity.roomId}`);
    activeIdentity.current = null; current.current = null;
    setIdentity(null); setRoom(null); setInvite(""); setFollowing(false); setConnectionError("");
    const url = new URL(window.location.href); url.searchParams.delete("room");
    window.history.replaceState(null, "", url);
  };
  return { room, identity, invite, error: error || connectionError, connected, pending, following, setFollowing, join, act, leave,
    isHost: !!identity && room?.hostId === identity.memberId };
}
export type WorkspaceController = ReturnType<typeof useWorkspace>;
