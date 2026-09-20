import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import { beginAnalysis, finishAnalysis, getRoom, RoomError, updateRoom } from "@/lib/collaboration";
import { LoadModelRequestSchema } from "@/lib/contracts";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
type Context = { params: { roomId: string } };
const view = z.object({ token: z.number().int().min(0).max(100000), layer: z.number().int().min(0).max(10000) });
const revision = z.number().int().min(0);
const actionSchema = z.discriminatedUnion("type", [
  z.object({ type: z.literal("view"), view, analysisRevision: revision }),
  z.object({ type: z.literal("chat"), text: z.string().trim().min(1).max(4000) }),
  z.object({ type: z.literal("annotation"), text: z.string().trim().min(1).max(4000), view, analysisRevision: revision }),
  z.object({ type: z.literal("transfer"), target: z.string().uuid() }),
  z.object({ type: z.literal("claim") }), z.object({ type: z.literal("leave") }),
  z.object({ type: z.literal("load"), model: LoadModelRequestSchema, analysisRevision: revision }),
  z.object({ type: z.literal("infer"), input: z.array(z.array(z.number()).min(1)).min(1),
    sample: z.record(z.unknown()).optional(), tokenCount: z.number().int().positive(), analysisRevision: revision }),
]);
const token = (req: NextRequest) => req.headers.get("authorization")?.replace(/^Bearer /, "") || "";
function failure(e: unknown) {
  const invalid = e instanceof z.ZodError || e instanceof SyntaxError;
  return NextResponse.json({ error: e instanceof RoomError ? e.message : invalid ? "請求內容格式不正確。" : "協作服務暫時無法使用，請重試。" }, { status: e instanceof RoomError ? e.status : invalid ? 400 : 503 });
}
export async function GET(req: NextRequest, { params }: Context) {
  try {
    const room = getRoom(params.roomId, token(req));
    // Presence is refreshed independently of document revisions.
    const { analysis, ...metadata } = room;
    return NextResponse.json({ ...metadata, ...(req.nextUrl.searchParams.get("analysisRevision") === String(room.analysisRevision) ? {} : { analysis }) }, { headers: { "Cache-Control": "no-store" } });
  } catch (e) { return failure(e); }
}
export async function POST(req: NextRequest, { params }: Context) {
  let jobId: string | undefined;
  try {
    const raw = await req.text();
    if (raw.length > 16 * 1024 * 1024) throw new RoomError("樣本超過 16 MB 上限。", 413);
    const action = actionSchema.parse(JSON.parse(raw));
    if (action.type !== "load" && action.type !== "infer") return NextResponse.json(updateRoom(params.roomId, token(req), action));
    const job = beginAnalysis(params.roomId, token(req), action.analysisRevision);
    jobId = job.jobId;
    const model = action.type === "load" ? action.model : job.analysis.model;
    if (!model) throw new RoomError("請先載入共同模型。");
    const base = process.env.PYTHON_SIDECAR_URL || "http://127.0.0.1:8080/internal/v1";
    const response = await fetch(`${base}/collaboration/analyze`, {
      method: "POST", headers: { "Content-Type": "application/json" }, cache: "no-store",
      signal: AbortSignal.timeout(240000), body: JSON.stringify({ model, ...(action.type === "infer" ? { input_data: action.input } : {}) }),
    });
    const data = await response.json();
    if (!response.ok) throw new RoomError(typeof data.detail === "string" ? data.detail : "模型分析失敗，請確認模型與樣本。", 400);
    const analysis = action.type === "load" ? {
      model, modelName: data.model_name, inputSchema: data.input_schema, capabilities: data.capabilities,
      attention: {}, output: [], tokenCount: data.input_schema?.seq_len || 16, maxLayer: Math.max(0, data.layer_count - 1),
    } : {
      ...job.analysis, attention: data.attention, output: data.output,
      sample: action.sample, tokenCount: Math.min(action.tokenCount, data.input_schema?.seq_len || action.tokenCount),
      maxLayer: Math.max(0, data.layer_count - 1),
    };
    const result = finishAnalysis(params.roomId, jobId, analysis);
    jobId = undefined;
    return NextResponse.json(result);
  } catch (e) {
    if (jobId) { try { finishAnalysis(params.roomId, jobId); } catch {} }
    return failure(e);
  }
}
