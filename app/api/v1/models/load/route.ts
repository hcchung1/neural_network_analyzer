import { NextRequest, NextResponse } from "next/server";
import { LoadModelRequestSchema } from "@/lib/contracts";
import { forwardToSidecar } from "@/lib/sidecar-client";

export async function POST(req: NextRequest) {
  try {
    const json = await req.json();
    const parsed = LoadModelRequestSchema.safeParse(json);
    if (!parsed.success) {
      return NextResponse.json(
        {
          error: {
            code: "INVALID_REQUEST_PAYLOAD",
            message: "Failed to parse load model request payload",
            details: parsed.error.format(),
            recovery: [],
            evidence: [],
          },
        },
        { status: 400 }
      );
    }
    return forwardToSidecar("/models/load", "POST", parsed.data);
  } catch (err: any) {
    return NextResponse.json(
      {
        error: {
          code: "BAD_REQUEST",
          message: err.message,
        },
      },
      { status: 400 }
    );
  }
}
