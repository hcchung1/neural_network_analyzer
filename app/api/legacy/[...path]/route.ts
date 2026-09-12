import { NextRequest } from "next/server";
import { forwardToSidecar } from "@/lib/sidecar-client";

export async function GET(req: NextRequest, { params }: { params: { path: string[] } }) {
  const pathElements = await params.path;
  const endpoint = `/legacy/${pathElements.join("/")}` + req.nextUrl.search;
  return await forwardToSidecar(endpoint, "GET");
}

export async function POST(req: NextRequest, { params }: { params: { path: string[] } }) {
  const pathElements = await params.path;
  const endpoint = `/legacy/${pathElements.join("/")}` + req.nextUrl.search;
  let body;
  try {
    body = await req.json();
  } catch (e) {}
  return await forwardToSidecar(endpoint, "POST", body);
}
