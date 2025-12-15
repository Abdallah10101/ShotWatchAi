import { NextResponse } from "next/server"
import path from "path"
import fs from "fs"
import "@/lib/ensure-temp" // ensure temp dir exists

const LATEST_FRAME_PATH = path.join(process.cwd(), "temp", "latest_frame.jpg")
const LATEST_EVENT_PATH = path.join(process.cwd(), "temp", "latest_event.json")

export async function GET() {
  try {
    let frameBase64: string | null = null
    let event: any = null

    console.log("[stream] Checking frame path:", LATEST_FRAME_PATH)
    if (fs.existsSync(LATEST_FRAME_PATH)) {
      console.log("[stream] Frame file exists, reading")
      const frameBuf = fs.readFileSync(LATEST_FRAME_PATH)
      frameBase64 = `data:image/jpeg;base64,${frameBuf.toString("base64")}`
    } else {
      console.log("[stream] Frame file not found")
    }

    if (fs.existsSync(LATEST_EVENT_PATH)) {
      const eventBuf = fs.readFileSync(LATEST_EVENT_PATH, "utf-8")
      event = JSON.parse(eventBuf)
    }

    return NextResponse.json({ frame: frameBase64, detection: event })
  } catch (e) {
    console.error("[stream] Error:", e)
    return NextResponse.json({ frame: null, detection: null }, { status: 500 })
  }
}
