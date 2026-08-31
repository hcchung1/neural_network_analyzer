import { NextRequest, NextResponse } from "next/server";
import { listRuns } from "@/lib/catalog";

export async function GET(req: NextRequest) {
  try {
    const data = listRuns();
    return NextResponse.json({ runs: data });
  } catch (err: any) {
    return NextResponse.json(
      { error: { code: "CATALOG_ERROR", message: err.message } },
      { status: 500 }
    );
  }
}
