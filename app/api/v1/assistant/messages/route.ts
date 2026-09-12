import { NextRequest } from "next/server";
import { forwardToSidecar } from "@/lib/sidecar-client";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    return await forwardToSidecar("/assistant/chat", "POST", body);
  } catch (err: any) {
    return new Response(JSON.stringify({ error: { code: "SERVER_ERROR", message: err.message } }), { status: 500 });
  }
}
