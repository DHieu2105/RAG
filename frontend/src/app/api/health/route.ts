import { NextResponse } from "next/server";

// Health check endpoint for backend status
export async function GET() {
  const backendBaseUrl = process.env.UNIGUIDE_BACKEND_URL?.trim();

  if (!backendBaseUrl) {
    return NextResponse.json({ status: "offline", reason: "no_backend_url" }, { status: 503 });
  }

  try {
    const url = new URL(backendBaseUrl);
    url.pathname = "/health";
    url.search = "";
    url.hash = "";

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 3000);

    const res = await fetch(url.toString(), {
      method: "GET",
      signal: controller.signal,
      cache: "no-store",
    });

    clearTimeout(timer);

    if (res.ok) {
      return NextResponse.json({ status: "online" });
    }
    return NextResponse.json({ status: "offline", code: res.status }, { status: 503 });
  } catch (error) {
    return NextResponse.json({ status: "offline", error: String(error) }, { status: 503 });
  }
}

