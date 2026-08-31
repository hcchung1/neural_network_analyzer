import { NextResponse } from "next/server";

const SIDECAR_BASE_URL = process.env.PYTHON_SIDECAR_URL || "http://127.0.0.1:8080/internal/v1";

export async function forwardToSidecar(endpoint: string, method: string = "GET", body?: any) {
  const url = `${SIDECAR_BASE_URL}${endpoint}`;
  try {
    const res = await fetch(url, {
      method,
      headers: { "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined,
      cache: "no-store",
    });

    const data = await res.json();
    if (!res.ok) {
      return NextResponse.json(data, { status: res.status });
    }
    return NextResponse.json(data);
  } catch (err: any) {
    return NextResponse.json(
      {
        error: {
          code: "SIDECAR_UNREACHABLE",
          message: `Failed to connect to Python sidecar at ${url}: ${err.message}`,
          details: {},
          recovery: ["Ensure Python FastAPI sidecar is running on port 8080"],
          evidence: [],
        },
      },
      { status: 503 }
    );
  }
}
