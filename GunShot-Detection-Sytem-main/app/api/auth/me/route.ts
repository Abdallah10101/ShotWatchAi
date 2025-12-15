import { cookies } from "next/headers"
import { NextResponse } from "next/server"
import { verifySession } from "@/lib/auth"

export async function GET() {
  try {
    const cookieStore = await cookies()
    const token = cookieStore.get("session")?.value
    if (!token) return NextResponse.json({ user: null }, { status: 200 })

    const payload = await verifySession(token)

    return NextResponse.json({
      user: {
        id: payload.sub,
        name: payload.name,
        badge: payload.badge,
        department: payload.department,
        role: payload.role,
        username: payload.username,
      },
    })
  } catch (e) {
    return NextResponse.json({ user: null }, { status: 200 })
  }
}
