import { NextRequest, NextResponse } from "next/server";
import { scanRootGenerator } from "@/lib/catalog";
import path from "path";

export async function POST(req: NextRequest) {
  try {
    const { folder_path } = await req.json();
    const targetFolder = folder_path || "output";

    // Create an absolute path resolved against process.cwd() or fallback
    const rootPath = path.isAbsolute(targetFolder) ? targetFolder : path.resolve(process.cwd(), targetFolder);

    const encoder = new TextEncoder();
    const customStream = new ReadableStream({
      async start(controller) {
        const sendEvent = (event: string, data: any) => {
          controller.enqueue(encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`));
        };

        sendEvent("progress", { stage: "initializing", progress: 0.1, message: "Initializing scan job..." });
        
        try {
          const generator = scanRootGenerator(rootPath);
          for await (const result of generator) {
            if (result.type === 'error') {
              sendEvent("error", { code: "SCAN_FAILED", message: result.message });
              break;
            } else if (result.type === 'progress') {
              sendEvent("progress", { stage: "scanning", progress: Math.max(0.1, (result.progress || 0) * 0.9), message: result.message });
            } else if (result.type === 'done') {
              sendEvent("complete", { status: "ok", scanned_files: result.scanned_files, runs_updated: result.runs_updated });
            }
          }
        } catch (err: any) {
          sendEvent("error", { code: "SCAN_FAILED", message: err.message });
        } finally {
          controller.close();
        }
      },
    });

    return new Response(customStream, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    });
  } catch (err: any) {
    return NextResponse.json({ error: { code: "BAD_REQUEST", message: err.message } }, { status: 400 });
  }
}
