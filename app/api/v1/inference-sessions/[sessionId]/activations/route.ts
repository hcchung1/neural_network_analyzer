import { NextRequest } from "next/server";
import { forwardToSidecar } from "@/lib/sidecar-client";

export async function GET(req: NextRequest, { params }: { params: { sessionId: string } }) {
  return forwardToSidecar(`/sessions/${params.sessionId}/activations`, "GET");
}
