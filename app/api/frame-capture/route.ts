import { NextResponse } from "next/server"
import path from "path"
import fs from "fs"
import { writeFile } from "fs/promises"

export async function POST(req: Request) {
  console.log("[frame-capture] POST received")
  try {
    const formData = await req.formData()
    const frame = formData.get("frame") as File
    const label = formData.get("label") as string
    const confidence = formData.get("confidence") as string
    const timestamp = formData.get("timestamp") as string
    const lat = formData.get("lat") as string | null
    const lng = formData.get("lng") as string | null
    const accuracy = formData.get("accuracy") as string | null

    if (!frame || !label || !confidence || !timestamp) {
      console.log("[frame-capture] Missing required fields")
      return NextResponse.json({ error: "Missing required fields" }, { status: 400 })
    }

    console.log(`[frame-capture] Processing frame: ${label} (${confidence})`)
    if (lat && lng) {
      console.log(`[frame-capture] Location: ${lat}, ${lng} (accuracy: ${accuracy}m)`)
    }

    // Read frame bytes
    const bytes = await frame.arrayBuffer()
    const buffer = Buffer.from(bytes)
    
    // Convert to base64 for sending to Python backend via WebSocket
    const base64Frame = buffer.toString("base64")
    
    // Send frame to Python backend for MediaPipe processing via WebSocket
    // The WebSocket server will process it with face/hand detection and save it
    const ts = parseFloat(timestamp)
    
    try {
      const WebSocket = (await import("ws")).default
      const ws = new WebSocket("ws://localhost:8765")
      
      const result = await new Promise<{ success: boolean; error?: string }>((resolve) => {
        const timeout = setTimeout(() => {
          ws.close()
          resolve({ success: false, error: "Connection timeout" })
        }, 3000)
        
        ws.on("open", () => {
      const message = {
        type: "frame",
        frame: base64Frame,
        detection: {
          label: label,
          confidence: parseFloat(confidence),
          timestamp: ts,
          lat: lat ? parseFloat(lat) : null,
          lng: lng ? parseFloat(lng) : null,
          accuracy: accuracy ? parseFloat(accuracy) : null,
        }
      }
          ws.send(JSON.stringify(message))
          console.log("[frame-capture] Frame sent to Python backend for MediaPipe processing")
          
          // Wait for confirmation
          ws.on("message", (data: Buffer) => {
            try {
              const response = JSON.parse(data.toString())
              if (response.type === "image_saved") {
                clearTimeout(timeout)
                ws.close()
                resolve({ success: true })
              }
            } catch (e) {
              // Ignore parse errors
            }
          })
        })
        
        ws.on("error", (error) => {
          clearTimeout(timeout)
          console.error("[frame-capture] WebSocket error:", error)
          resolve({ success: false, error: String(error) })
        })
      })
      
      if (!result.success) {
        throw new Error(result.error || "Failed to process frame")
      }
    } catch (error) {
      console.warn("[frame-capture] Could not send to Python backend, saving directly:", error)
      // Fallback: save directly without MediaPipe processing
      const date = new Date(ts * 1000)
      
      const dayOrdinal = (n: number) => {
        n = Math.floor(n)
        if (10 <= n % 100 && n % 100 <= 20) return "th"
        const remainder = n % 10
        if (remainder === 1) return "st"
        if (remainder === 2) return "nd"
        if (remainder === 3) return "rd"
        return "th"
      }
      
      const day = dayOrdinal(date.getDate())
      const month = date.toLocaleDateString("en-US", { month: "long" })
      const hour = date.toLocaleTimeString("en-US", { hour: "numeric", hour12: false })
      const ampm = date.toLocaleTimeString("en-US", { hour: "numeric", hour12: true }).split(" ")[1].toLowerCase()
      
      const filename = `${day} ${month} ${hour} ${ampm} gunshot fired ; ${label}.jpg`
      const detectionsDir = path.join(process.cwd(), "gunshot_detections")
      fs.mkdirSync(detectionsDir, { recursive: true })
      const filepath = path.join(detectionsDir, filename)
      await writeFile(filepath, buffer)
      
      console.log(`[frame-capture] Frame saved directly (no MediaPipe processing): ${filename}`)
    }

    // Update latest event JSON
    const eventData = {
      timestamp: ts,
      label: label,
      confidence: parseFloat(confidence),
      savedAt: new Date().toISOString()
    }

    const eventPath = path.join(process.cwd(), "temp", "latest_event.json")
    fs.mkdirSync(path.dirname(eventPath), { recursive: true })
    await writeFile(eventPath, JSON.stringify(eventData, null, 2))

    return NextResponse.json({ 
      success: true,
      message: "Frame received and sent for processing"
    })

  } catch (e) {
    console.error("[frame-capture] Error:", e)
    return NextResponse.json({ error: "Server error", details: String(e) }, { status: 500 })
  }
}
