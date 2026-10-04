import { forwardToSidecar } from "@/lib/sidecar-client";
import { NextRequest } from "next/server";

export async function GET(req: NextRequest) {
  return forwardToSidecar("/health", "GET", undefined, req.headers.get("X-Model-Handle"));
}
