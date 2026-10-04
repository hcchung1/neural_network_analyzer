import { NextRequest, NextResponse } from "next/server";
import { listRuns, listRunArtifacts } from "@/lib/catalog";

export async function GET(req: NextRequest) {
  try {
    const data = listRuns();
    return NextResponse.json({ runs: data, ...(req.nextUrl.searchParams.get("includeArtifacts") === "true"
      ? { artifacts: listRunArtifacts() } : {}) });
  } catch (err: any) {
    return NextResponse.json(
      { error: { code: "CATALOG_ERROR", message: err.message } },
      { status: 500 }
    );
  }
}
