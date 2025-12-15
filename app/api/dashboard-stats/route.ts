import { NextResponse } from "next/server"
import fs from "fs"
import path from "path"
import "@/lib/ensure-temp"

const TEMP_DIR = path.join(process.cwd(), "temp")
const STATUS_PATH = path.join(TEMP_DIR, "ai_status.json")
const DETECTIONS_PATH = path.join(TEMP_DIR, "detections.json")

function readDetections(): any[] {
  if (!fs.existsSync(DETECTIONS_PATH)) {
    return []
  }
  try {
    const raw = fs.readFileSync(DETECTIONS_PATH, "utf-8")
    const data = JSON.parse(raw)
    return Array.isArray(data) ? data : []
  } catch (error) {
    console.error("[dashboard-stats] Failed to parse detections", error)
    return []
  }
}

function classifyConfidence(confidence = 0): string {
  if (confidence >= 0.9) return "confirmed"
  if (confidence >= 0.8) return "dispatched"
  if (confidence >= 0.65) return "pending"
  return "false-positive"
}

export async function GET() {
  try {
    const detections = readDetections()
    let statusData: any = {}

    if (fs.existsSync(STATUS_PATH)) {
      try {
        const raw = fs.readFileSync(STATUS_PATH, "utf-8")
        statusData = JSON.parse(raw)
      } catch (error) {
        console.error("[dashboard-stats] Failed to parse status file", error)
      }
    }

    const aiAccuracy =
      typeof statusData.ai_accuracy === "number"
        ? statusData.ai_accuracy
        : statusData.ai_accuracy ?? 0

    const summary = detections.reduce(
      (acc, entry) => {
        const status = classifyConfidence(entry.confidence)
        if (status === "confirmed") acc.confirmed += 1
        else if (status === "dispatched") acc.dispatched += 1
        else if (status === "pending") acc.pending += 1
        else acc.falsePositives += 1
        return acc
      },
      { total: detections.length, confirmed: 0, dispatched: 0, pending: 0, falsePositives: 0 }
    )

    const response = {
      activeAlerts: summary.total,
      todaysDetections: summary.total,
      systemStatus: statusData.system_status ?? "idle",
      systemMessage: statusData.system_message ?? "System ready · Monitoring events",
      aiAccuracy,
      detectionCount: summary.total,
      confirmed: summary.confirmed,
      dispatched: summary.dispatched,
      pending: summary.pending,
      falsePositives: summary.falsePositives,
    }

    return NextResponse.json(response)
  } catch (error) {
    console.error("[dashboard-stats] Unexpected error", error)
    return NextResponse.json(
      {
        activeAlerts: 0,
        todaysDetections: 0,
        systemStatus: "error",
        systemMessage: "Failed to load stats",
        aiAccuracy: 0,
        detectionCount: 0,
        confirmed: 0,
        dispatched: 0,
        pending: 0,
        falsePositives: 0,
      },
      { status: 500 }
    )
  }
}
