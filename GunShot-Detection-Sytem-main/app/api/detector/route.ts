import { NextResponse } from "next/server"
import { spawn, ChildProcess } from "child_process"
import path from "path"
import fs from "fs"

let detectorProcess: ChildProcess | null = null
const DETECTIONS_DIR = path.join(process.cwd(), "public", "detections")
fs.mkdirSync(DETECTIONS_DIR, { recursive: true })

export async function POST(req: Request) {
  console.log("[detector] POST received")
  try {
    const { action } = await req.json()
    console.log("[detector] Request action:", action)
    if (action === "start") {
      if (detectorProcess) {
        console.log("[detector] Already running")
        return NextResponse.json({ error: "Detector already running" }, { status: 400 })
      }
      const pythonScript = path.join(process.cwd(), "predict_audio_only.py")
      const modelPath = "C:\\Users\\sarma\\Desktop\\GunShot-Detection-Sytem-main\\GunShot-Detection-Sytem-main\\guntype_resnet50.pth"
      if (!fs.existsSync(pythonScript)) {
        console.error("[detector] Script not found:", pythonScript)
        return NextResponse.json({ error: "predict_audio_only.py not found" }, { status: 404 })
      }
      if (!fs.existsSync(modelPath)) {
        console.error("[detector] Model not found:", modelPath)
        return NextResponse.json({ error: "Model file not found" }, { status: 404 })
      }

      // Try to use venv Python first (Python 3.10), fallback to system Python 3.10
      const venvPython = path.join(process.cwd(), "venv", "Scripts", "python.exe")
      let pythonExecutable: string
      let pythonArgs: string[] = []
      
      if (fs.existsSync(venvPython)) {
        pythonExecutable = venvPython
      } else {
        // Use Python 3.10 launcher
        pythonExecutable = "py"
        pythonArgs = ["-3.10"]
      }
      
      console.log("[detector] Starting:", pythonScript, modelPath, DETECTIONS_DIR)
      console.log("[detector] Using Python:", pythonExecutable, pythonArgs.join(" "))
      detectorProcess = spawn(pythonExecutable, [...pythonArgs, pythonScript, "--model", modelPath, "--recordings-dir", DETECTIONS_DIR], {
        stdio: ["ignore", "pipe", "pipe"],
        cwd: process.cwd(),
      })

      console.log("[detector] Spawned process, PID:", detectorProcess.pid)
      detectorProcess.stdout?.on("data", (data) => {
        console.log(`[detector] ${data.toString().trim()}`)
      })
      detectorProcess.stderr?.on("data", (data) => {
        console.error(`[detector] ${data.toString().trim()}`)
      })
      detectorProcess.on("exit", (code, signal) => {
        console.log(`[detector] exited with code ${code}, signal ${signal}`)
        detectorProcess = null
      })
      detectorProcess.on("error", (err) => {
        console.error("[detector] spawn error:", err)
        detectorProcess = null
      })

      return NextResponse.json({ status: "started" })
    } else if (action === "stop") {
      if (!detectorProcess) {
        console.log("[detector] Not running")
        return NextResponse.json({ error: "Detector not running" }, { status: 400 })
      }
      console.log("[detector] Stopping")
      detectorProcess.kill("SIGTERM")
      detectorProcess = null
      return NextResponse.json({ status: "stopped" })
    } else {
      console.log("[detector] Invalid action:", action)
      return NextResponse.json({ error: "Invalid action" }, { status: 400 })
    }
  } catch (e) {
    console.error("[detector] POST error:", e)
    return NextResponse.json({ error: "Server error", details: String(e) }, { status: 500 })
  }
}

export async function GET() {
  console.log("[detector] GET received")
  return NextResponse.json({ running: detectorProcess !== null })
}
