import { NextRequest } from "next/server";
import { forwardToSidecar } from "@/lib/sidecar-client";

export async function DELETE(req: NextRequest) {
  return forwardToSidecar("/models/current", "DELETE", undefined, req.headers.get("X-Model-Handle"));
}
