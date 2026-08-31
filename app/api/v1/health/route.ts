import { forwardToSidecar } from "@/lib/sidecar-client";

export async function GET() {
  return forwardToSidecar("/health");
}
