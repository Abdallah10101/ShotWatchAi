import mongoose, { Schema, models, model } from "mongoose"

export type UserRole = "admin" | "officer"

export interface IUser {
  username: string
  passwordHash: string
  name: string
  badge: string
  department: string
  role: UserRole
}

const UserSchema = new Schema<IUser>(
  {
    username: { type: String, required: true, unique: true, index: true },
    passwordHash: { type: String, required: true },
    name: { type: String, required: true },
    badge: { type: String, required: true },
    department: { type: String, required: true },
    role: { type: String, enum: ["admin", "officer"], default: "officer" },
  },
  { timestamps: true }
)

export const User = models.User || model<IUser>("User", UserSchema)
