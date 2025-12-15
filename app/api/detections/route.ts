import { NextResponse } from "next/server"
import fs from "fs"
import path from "path"
import "@/lib/ensure-temp"

const DETECTIONS_PATH = path.join(process.cwd(), "temp", "detections.json")
const TEMP_DIR = path.join(process.cwd(), "temp")

function readManifest(): any[] {
  if (!fs.existsSync(DETECTIONS_PATH)) {
    return []
  }

  try {
    const raw = fs.readFileSync(DETECTIONS_PATH, "utf-8")
    const data = JSON.parse(raw)
    return Array.isArray(data) ? data : []
  } catch (error) {
    console.error("[detections] Failed to parse manifest", error)
    return []
  }
}

function loadThumbnail(entry: any): string | null {
  const providedPath = typeof entry.path === "string" ? entry.path : null
  const fullPath = providedPath && fs.existsSync(providedPath)
    ? providedPath
    : path.join(TEMP_DIR, entry.filename || "")

  try {
    if (fullPath && fs.existsSync(fullPath)) {
      const buffer = fs.readFileSync(fullPath)
      return `data:image/jpeg;base64,${buffer.toString("base64")}`
    }
  } catch (error) {
    console.error("[detections] Failed to load thumbnail", error)
  }

  return null
}

export async function GET() {
  try {
    const manifest = readManifest()
    const detections = manifest
      .slice(-30)
      .reverse()
      .map((entry, index) => {
        const timestamp = typeof entry.timestamp === "number" ? entry.timestamp : Date.now() / 1000
        const date = new Date(timestamp * 1000)
        return {
          id: entry.detection_id ?? `${timestamp}-${index}`,
          label: entry.label ?? "Unknown",
          confidence: entry.confidence ?? 0,
          quality: entry.quality ?? null,
          filename: entry.filename ?? null,
          source: entry.source ?? "audio",
          timestamp,
          isoTimestamp: date.toISOString(),
          thumbnail: loadThumbnail(entry),
          lat: entry.lat ?? null,
          lng: entry.lng ?? null,
          accuracy: entry.accuracy ?? null,
        }
      })

    return NextResponse.json({ detections })
  } catch (error) {
    console.error("[detections] Unexpected error", error)
    return NextResponse.json({ detections: [], error: "Failed to load detections" }, { status: 500 })
  }
}
