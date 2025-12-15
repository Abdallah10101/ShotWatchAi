import { WebSocketServer, WebSocket } from "ws"
import { NextRequest } from "next/server"
import { NextResponse } from "next/server"

let wss: WebSocketServer | null = null

export function getWebSocketServer() {
  if (!wss) {
    wss = new WebSocketServer({ noServer: true })
    wss.on("connection", (ws: WebSocket) => {
      console.log("[ws] Detector client connected")
      ws.on("message", (data: Buffer) => {
        try {
          const msg = JSON.parse(data.toString())
          console.log("[ws] Received:", msg.type)
          // Broadcast to all dashboard clients
          wss?.clients.forEach((client) => {
            if (client !== ws && client.readyState === WebSocket.OPEN) {
              client.send(JSON.stringify(msg))
            }
          })
        } catch (e) {
          console.error("[ws] Invalid JSON from detector", e)
        }
      })
      ws.on("close", () => console.log("[ws] Detector client disconnected"))
      ws.on("error", console.error)
    })
  }
  return wss
}

// Helper to upgrade Next.js request to WebSocket
export async function upgradeToWebSocket(req: NextRequest) {
  if (req.headers.get("upgrade") !== "websocket") {
    return new NextResponse("Expected websocket", { status: 426 })
  }
  // Note: In production, you’d need a custom server or adapter for true WebSocket upgrade.
  // For local development, we’ll expose a simple polling-based fallback.
  return new NextResponse(null, { status: 101 })
}
