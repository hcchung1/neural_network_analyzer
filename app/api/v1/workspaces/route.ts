import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import { joinRoom, RoomError } from "@/lib/collaboration";

export const runtime = "nodejs";
export async function POST(req: NextRequest) {
  try {
    const body = z.object({ name: z.string().trim().min(1).max(40), roomId: z.string().regex(/^[a-f0-9]{36}$/).optional() }).parse(await req.json());
    return NextResponse.json(joinRoom(body.name, body.roomId));
  } catch (e) {
    return NextResponse.json({ error: e instanceof RoomError ? e.message : "請輸入有效的暱稱與工作區連結。" }, { status: e instanceof RoomError ? e.status : 400 });
  }
}
