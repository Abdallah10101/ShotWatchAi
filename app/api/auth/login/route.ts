import { NextResponse } from "next/server"
import { createSession } from "@/lib/auth"

export async function POST(req: Request) {
  try {
    const { username, password } = await req.json()
    if (!username || !password) {
      return NextResponse.json({ error: "Missing credentials" }, { status: 400 })
    }

    // Demo/local-only auth: no database, fixed credentials.
    const demoUsername = "officer"
    const demoPassword = "dubai2025"
    const name = "Security Officer Ahmed"
    const badge = "SEC-2024-001"
    const department = "Emergency Response Unit"
    const role: "admin" | "officer" = "admin"

    if (username !== demoUsername || password !== demoPassword) {
      return NextResponse.json({ error: "Invalid credentials" }, { status: 401 })
    }

    const token = await createSession({
      sub: "local-demo-user",
      username: demoUsername,
      role,
      name,
      badge,
      department,
    })

    const res = NextResponse.json({
      user: {
        id: "local-demo-user",
        name,
        badge,
        department,
        role,
        username: demoUsername,
      },
    })

    res.cookies.set("session", token, {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "lax",
      path: "/",
      maxAge: 60 * 60 * 24 * 7,
    })

    return res
  } catch (e) {
    console.error("[auth/login] error", e)
    return NextResponse.json({ error: "Server error" }, { status: 500 })
  }
}
