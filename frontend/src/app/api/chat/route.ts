import { NextResponse } from "next/server";

const MAX_REQUEST_BYTES = 256_000;
const BACKEND_CHAT_PATH = "/api/v1/chat";

export const maxDuration = 300;

function backendChatUrl(rawBaseUrl: string): URL {
  const url = new URL(rawBaseUrl);
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new Error("Unsupported backend URL protocol");
  }
  if (url.username || url.password) {
    throw new Error("Backend URL must not contain credentials");
  }
  if (process.env.NODE_ENV === "production" && url.protocol !== "https:") {
    throw new Error("Production backend URL must use HTTPS");
  }

  url.pathname = BACKEND_CHAT_PATH;
  url.search = "";
  url.hash = "";
  return url;
}

export async function POST(request: Request) {
  const backendBaseUrl = process.env.UNIGUIDE_BACKEND_URL?.trim();
  const apiToken = process.env.UNIGUIDE_API_TOKEN?.trim();

  if (!backendBaseUrl || !apiToken) {
    return NextResponse.json(
      { detail: "Máy chủ proxy chưa được cấu hình." },
      { status: 503 },
    );
  }

  const declaredLength = Number(request.headers.get("content-length") ?? 0);
  if (Number.isFinite(declaredLength) && declaredLength > MAX_REQUEST_BYTES) {
    return NextResponse.json({ detail: "Yêu cầu quá lớn." }, { status: 413 });
  }

  const requestBody = await request.text();
  if (new TextEncoder().encode(requestBody).byteLength > MAX_REQUEST_BYTES) {
    return NextResponse.json({ detail: "Yêu cầu quá lớn." }, { status: 413 });
  }

  try {
    JSON.parse(requestBody);
  } catch {
    return NextResponse.json({ detail: "Dữ liệu JSON không hợp lệ." }, { status: 400 });
  }

  let chatUrl: URL;
  try {
    chatUrl = backendChatUrl(backendBaseUrl);
  } catch {
    return NextResponse.json(
      { detail: "Địa chỉ backend chưa được cấu hình hợp lệ." },
      { status: 503 },
    );
  }

  try {
    const backendResponse = await fetch(chatUrl, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiToken}`,
        "Content-Type": "application/json",
      },
      body: requestBody,
      cache: "no-store",
      signal: request.signal,
    });
    const responseBody = await backendResponse.text();

    return new Response(
      responseBody || JSON.stringify({ detail: "Backend không trả về dữ liệu." }),
      {
        status: backendResponse.status,
        headers: {
          "Cache-Control": "no-store",
          "Content-Type": backendResponse.headers.get("content-type") ?? "application/json",
        },
      },
    );
  } catch {
    return NextResponse.json(
      { detail: "Không thể kết nối tới máy xử lý. Hãy kiểm tra backend và Cloudflare Tunnel." },
      { status: 502 },
    );
  }
}
