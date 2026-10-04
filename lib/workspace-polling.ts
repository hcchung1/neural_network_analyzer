import { RoomPoll, RoomSnapshot } from "./collaboration-types";

const same = (left: unknown, right: unknown) => left === right || JSON.stringify(left) === JSON.stringify(right);

export function mergeRoomSnapshot(previous: RoomSnapshot | null, data: RoomPoll): RoomSnapshot | null {
  if (previous?.id === data.id && data.revision < previous.revision) return previous;
  const base = previous?.id === data.id ? previous : null;
  const view = data.view ?? base?.view;
  const entries = data.entries ?? base?.entries;
  const analysis = data.analysis ?? base?.analysis;
  if (!view || !entries || !analysis) return previous;
  const next: RoomSnapshot = {
    id: data.id, revision: data.revision, analysisRevision: data.analysisRevision,
    hostId: data.hostId, busy: data.busy,
    members: base && same(base.members, data.members) ? base.members : data.members,
    view: base && same(base.view, view) ? base.view : view,
    entries: base && same(base.entries, entries) ? base.entries : entries,
    analysis: base && same(base.analysis, analysis) ? base.analysis : analysis,
  };
  if (base && Object.keys(next).every(key => next[key as keyof RoomSnapshot] === base[key as keyof RoomSnapshot])) return base;
  return next;
}

type PollingOptions = {
  poll: (signal: AbortSignal) => Promise<void | boolean>;
  hidden?: () => boolean;
  onError?: (error: unknown) => void;
  random?: () => number;
  schedule?: (callback: () => void, delay: number) => ReturnType<typeof setTimeout>;
  cancel?: (timer: ReturnType<typeof setTimeout>) => void;
};

export function startWorkspacePolling(options: PollingOptions): () => void {
  const schedule = options.schedule ?? setTimeout;
  const cancel = options.cancel ?? clearTimeout;
  const random = options.random ?? Math.random;
  const controller = new AbortController();
  let stopped = false;
  let failures = 0;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const stop = () => {
    stopped = true;
    if (timer !== undefined) cancel(timer);
    controller.abort();
  };
  const poll = async () => {
    if (stopped) return;
    let delay = 1000;
    try {
      if (await options.poll(controller.signal) === false) { stop(); return; }
      failures = 0;
      delay = options.hidden?.() ? 5000 : 1000;
    } catch (error) {
      if (stopped || controller.signal.aborted) return;
      failures++;
      delay = Math.min(15000, 2000 * 2 ** Math.min(failures - 1, 4) * (0.8 + random() * 0.4));
      options.onError?.(error);
    }
    if (!stopped) timer = schedule(() => { void poll(); }, delay);
  };
  void poll();
  return stop;
}
