import { NextResponse } from "next/server"

// With local/demo auth we don't actually need to seed anything in a DB.
// This route just returns the configured demo user so you can verify it.

export async function POST() {
  const username = process.env.DEMO_ADMIN_USERNAME || "officer"
  const name = process.env.DEMO_ADMIN_NAME || "Security Officer Ahmed"

  return NextResponse.json({
    created: false,
    user: {
      id: "local-demo-user",
      username,
      name,
    },
  })
}
