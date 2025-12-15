import { NextResponse } from "next/server"

/**
 * Camera stream endpoint
 * 
 * Option 1: Direct access to Flask server (recommended for MJPEG streaming)
 * Use: http://localhost:5000/camera-stream in your frontend
 * 
 * Option 2: Proxy through Next.js (this endpoint)
 * Use: http://localhost:3000/api/camera-stream
 * Note: Streaming through Next.js proxy may have limitations
 */
export async function GET() {
  try {
    // Proxy to Flask server running on port 5000
    // Use 127.0.0.1 instead of localhost to avoid IPv6 issues on Windows
    const flaskUrl = "http://127.0.0.1:5000/camera-stream"
    
    const response = await fetch(flaskUrl, {
      cache: "no-store",
      // Add timeout and connection options
      signal: AbortSignal.timeout(5000), // 5 second timeout
    })

    if (!response.ok) {
      return NextResponse.json(
        { 
          error: "Camera stream not available", 
          details: `Flask server returned ${response.status}`,
          hint: "Make sure the Flask server is running on port 5000"
        },
        { status: response.status }
      )
    }

    // Return the stream with proper MJPEG headers
    return new NextResponse(response.body, {
      headers: {
        "Content-Type": "multipart/x-mixed-replace; boundary=frame",
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
        "Connection": "keep-alive",
      },
    })
  } catch (error) {
    console.error("[camera-stream] Error proxying stream:", error)
    return NextResponse.json(
      { 
        error: "Failed to connect to camera stream",
        details: error instanceof Error ? error.message : String(error),
        hint: "Make sure the Flask server is running: python start_server.py"
      },
      { status: 503 }
    )
  }
}

