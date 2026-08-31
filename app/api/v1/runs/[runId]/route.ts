import { NextRequest, NextResponse } from "next/server";
import { getRun } from "@/lib/catalog";

export async function GET(
  req: NextRequest,
  { params }: { params: { runId: string } }
) {
  try {
    const run = getRun(params.runId);
    if (!run) {
      return NextResponse.json(
        { error: { code: "NOT_FOUND", message: "Run not found" } },
        { status: 404 }
      );
    }
    return NextResponse.json({ run });
  } catch (err: any) {
    return NextResponse.json(
      { error: { code: "CATALOG_ERROR", message: err.message } },
      { status: 500 }
    );
  }
}
