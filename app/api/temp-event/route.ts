import { NextResponse } from "next/server"
import fs from "fs"
import path from "path"

export async function GET() {
  try {
    const eventPath = path.join(process.cwd(), "temp", "latest_event.json")
    
    if (fs.existsSync(eventPath)) {
      const eventContent = fs.readFileSync(eventPath, "utf-8")
      const event = JSON.parse(eventContent)
      
      return new NextResponse(eventContent, {
        headers: {
          'Content-Type': 'application/json',
          'Cache-Control': 'no-cache, no-store, must-revalidate',
          'Pragma': 'no-cache',
          'Expires': '0'
        }
      })
    } else {
      return NextResponse.json({}, { status: 404 })
    }
  } catch (e) {
    console.error("Error reading latest event:", e)
    return NextResponse.json({}, { status: 500 })
  }
}
