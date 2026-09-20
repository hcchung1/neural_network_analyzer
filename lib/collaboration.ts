import Database from "better-sqlite3";
import { randomBytes, randomUUID, createHash } from "crypto";
import fs from "fs";
import path from "path";
import { RoomSnapshot, SharedAnalysis, ViewState } from "./collaboration-types";

type Member = { id: string; name: string; secret: string; seen: number };
type Room = Omit<RoomSnapshot, "members" | "busy"> & {
  members: Member[]; job?: { id: string; expires: number };
};
let connection: Database.Database | undefined;
function db() {
  if (!connection) {
    fs.mkdirSync(path.join(process.cwd(), "data"), { recursive: true });
    connection = new Database(path.join(process.cwd(), "data", "collaboration.sqlite3"));
    connection.pragma("journal_mode = WAL");
    connection.pragma("busy_timeout = 5000");
    connection.exec("CREATE TABLE IF NOT EXISTS collaboration_rooms (id TEXT PRIMARY KEY, data TEXT NOT NULL)");
    connection.exec("CREATE TABLE IF NOT EXISTS collaboration_presence (room_id TEXT, member_id TEXT, seen INTEGER, PRIMARY KEY (room_id, member_id))");
  }
  return connection;
}
export class RoomError extends Error {
  constructor(message: string, public status = 400) { super(message); }
}
const hash = (token: string) => createHash("sha256").update(token).digest("hex");
const online = (member: Member) => Date.now() - member.seen < 15000;
const busy = (room: Room) => !!room.job && room.job.expires > Date.now();
function read(id: string): Room {
  const row = db().prepare("SELECT data FROM collaboration_rooms WHERE id = ?").get(id) as { data: string } | undefined;
  if (!row) throw new RoomError("找不到工作區，請確認分享連結。", 404);
  const room: Room = JSON.parse(row.data);
  const presence = db().prepare("SELECT member_id, seen FROM collaboration_presence WHERE room_id = ?").all(id) as { member_id: string; seen: number }[];
  for (const p of presence) {
    const m = room.members.find(m => m.id === p.member_id);
    if (m) m.seen = p.seen;
  }
  return room;
}
function save(room: Room) {
  db().prepare("INSERT OR REPLACE INTO collaboration_rooms (id, data) VALUES (?, ?)").run(room.id, JSON.stringify(room));
}
function snapshot(room: Room): RoomSnapshot {
  const { job, members, ...publicRoom } = room;
  return { ...publicRoom, busy: busy(room), members: members.map(m => ({ id: m.id, name: m.name, online: online(m) })) };
}
function member(room: Room, token: string) {
  const m = room.members.find(m => m.secret === hash(token));
  if (!m) throw new RoomError("加入憑證已失效，請重新加入工作區。", 401);
  m.seen = Date.now();
  db().prepare("INSERT OR REPLACE INTO collaboration_presence (room_id, member_id, seen) VALUES (?, ?, ?)").run(room.id, m.id, m.seen);
  return m;
}
function host(room: Room, id: string) {
  if (room.hostId !== id) throw new RoomError("只有主持人可以操作共同模型、樣本與主持人視圖。", 403);
}
function idle(room: Room) {
  if (busy(room)) throw new RoomError("工作區正在分析中，請稍候。", 409);
}
export function joinRoom(name: string, id?: string) {
  return db().transaction(() => {
    const token = randomBytes(32).toString("hex");
    const m: Member = { id: randomUUID(), name, secret: hash(token), seen: Date.now() };
    const room: Room = id ? read(id) : {
      id: randomBytes(18).toString("hex"), revision: 0, hostId: m.id,
      view: { token: 0, layer: 0 }, analysisRevision: 0,
      analysis: { tokenCount: 16, maxLayer: 3, attention: {}, output: [] }, members: [], entries: [],
    };
    if (room.members.filter(online).length >= 30) throw new RoomError("工作區已達 30 位線上參與者上限。", 409);
    room.members.push(m);
    room.revision++;
    save(room);
    return { identity: { roomId: room.id, memberId: m.id, token }, room: snapshot(room) };
  })();
}
export function getRoom(id: string, token: string) {
  return db().transaction(() => {
    const room = read(id);
    member(room, token);
    return snapshot(room);
  })();
}
export function updateRoom(id: string, token: string, action: {
  type: "view" | "chat" | "annotation" | "transfer" | "claim" | "leave";
  view?: ViewState; text?: string; target?: string; analysisRevision?: number;
}) {
  return db().transaction(() => {
    const room = read(id);
    const m = member(room, token);
    if (action.type === "view") {
      host(room, m.id);
      if (action.analysisRevision !== room.analysisRevision) throw new RoomError("分析已更新，請重試。", 409);
      const v = action.view!;
      room.view = { token: Math.min(v.token, room.analysis.tokenCount - 1), layer: Math.min(v.layer, room.analysis.maxLayer) };
    } else if (action.type === "chat" || action.type === "annotation") {
      if (action.type === "annotation" && (action.analysisRevision !== room.analysisRevision ||
        !action.view || action.view.token >= room.analysis.tokenCount || action.view.layer > room.analysis.maxLayer)) {
        throw new RoomError("標註位置已過期或超出分析範圍，請確認目前位置。", 409);
      }
      room.entries.push({ id: randomUUID(), author: m.name, text: action.text!, kind: action.type,
        view: action.view || room.view, analysisRevision: action.analysisRevision ?? room.analysisRevision, createdAt: Date.now() });
      room.entries = room.entries.slice(-500);
    } else if (action.type === "transfer") {
      host(room, m.id); idle(room);
      if (!room.members.some(p => p.id === action.target && online(p))) throw new RoomError("此參與者已離線。", 409);
      room.hostId = action.target!;
    } else if (action.type === "claim") {
      idle(room);
      if (room.members.some(p => p.id === room.hostId && online(p))) throw new RoomError("主持人仍在線上。", 409);
      room.hostId = m.id;
    } else {
      if (room.hostId === m.id) {
        idle(room);
        room.hostId = room.members.find(p => p.id !== m.id && online(p))?.id || "";
      }
      room.members = room.members.filter(p => p.id !== m.id);
      db().prepare("DELETE FROM collaboration_presence WHERE room_id = ? AND member_id = ?").run(room.id, m.id);
    }
    room.revision++;
    save(room);
    return snapshot(room);
  })();
}
export function beginAnalysis(id: string, token: string, expectedRevision: number) {
  return db().transaction(() => {
    const room = read(id);
    const m = member(room, token);
    host(room, m.id); idle(room);
    if (room.analysisRevision !== expectedRevision) throw new RoomError("共同模型已更新，請重試。", 409);
    const job = { id: randomUUID(), expires: Date.now() + 300000 };
    room.job = job; room.revision++; save(room);
    return { jobId: job.id, analysis: room.analysis };
  })();
}
export function finishAnalysis(id: string, jobId: string, analysis?: SharedAnalysis) {
  return db().transaction(() => {
    const room = read(id);
    if (room.job?.id !== jobId) throw new RoomError("分析作業已被取代。", 409);
    delete room.job;
    if (analysis) {
      room.analysis = analysis; room.analysisRevision++;
      room.view = { token: 0, layer: 0 };
    }
    room.revision++; save(room);
    return snapshot(room);
  })();
}
