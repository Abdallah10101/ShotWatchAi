import { NextRequest } from "next/server"
import { WebSocketServer, WebSocket } from "ws"
import { NextResponse } from "next/server"

export async function GET(req: NextRequest) {
  // Upgrade HTTP to WebSocket
  if (req.headers.get("upgrade") !== "websocket") {
    return new NextResponse("Expected websocket", { status: 426 })
  }

  const wss = new WebSocketServer({ noServer: true })
  const server = (global as any).wsServer
  if (!server) {
    throw new Error("WebSocket server not initialized")
  }

  return new Promise<Response>((resolve) => {
    server.handleUpgrade(req, req.headers.get("sec-websocket-key")!, req.headers.get("sec-websocket-protocol")!, req.headers.get("sec-websocket-extensions")!, (ws: WebSocket) => {
      wss.emit("connection", ws, req)
      resolve(new NextResponse(null))
    })
  })
}

// This will be initialized in a separate startup script
export function createWebSocketServer() {
  const wss = new WebSocketServer({ noServer: true })
  wss.on("connection", (ws: WebSocket) => {
    console.log("Detector client connected")
    ws.on("message", (data: Buffer) => {
      try {
        const msg = JSON.parse(data.toString())
        // Broadcast to dashboard clients if needed
        ;(global as any).wsDashboard?.clients?.forEach((client: WebSocket) => {
          if (client.readyState === WebSocket.OPEN) {
            client.send(JSON.stringify(msg))
          }
        })
      } catch (e) {
        console.error("Invalid message from detector", e)
      }
    })
    ws.on("close", () => console.log("Detector client disconnected"))
    ws.on("error", console.error)
  })
  ;(global as any).wsDetector = wss
  return wss
}
