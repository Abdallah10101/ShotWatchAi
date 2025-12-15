import { SignJWT, jwtVerify } from "jose"

const JWT_SECRET = process.env.JWT_SECRET
if (!JWT_SECRET) {
  throw new Error("JWT_SECRET is not set in environment")
}

const secretKey = new TextEncoder().encode(JWT_SECRET)

export type JWTPayload = {
  sub: string
  username: string
  role: "admin" | "officer"
  name: string
  badge: string
  department: string
}

export async function createSession(payload: JWTPayload) {
  const token = await new SignJWT(payload)
    .setProtectedHeader({ alg: "HS256" })
    .setIssuedAt()
    .setExpirationTime("7d")
    .sign(secretKey)
  return token
}

export async function verifySession(token: string) {
  const { payload } = await jwtVerify<JWTPayload>(token, secretKey, {
    algorithms: ["HS256"],
  })
  return payload
}
