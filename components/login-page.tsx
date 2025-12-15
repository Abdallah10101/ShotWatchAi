"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Shield, Eye, EyeOff, Lock, User } from "lucide-react"
import { useAuth } from "../contexts/auth-context"

export function LoginPage() {
  const [credentials, setCredentials] = useState({ username: "", password: "" })
  const [showPassword, setShowPassword] = useState(false)
  const [error, setError] = useState("")
  const { login, isLoading } = useAuth()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError("")

    const success = await login(credentials)
    if (!success) {
      setError("Invalid credentials. Please try again.")
    }
  }

  return (
    <div className="relative min-h-screen overflow-hidden bg-[#030917] text-white">
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute -top-48 right-4 h-96 w-96 rounded-full bg-[#1f3b73] opacity-50 blur-[140px]" />
        <div className="absolute bottom-0 left-12 h-80 w-80 rounded-full bg-[#0b8fd0] opacity-40 blur-[180px]" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_rgba(255,255,255,0.05),_transparent_60%)]" />
      </div>
      <div className="relative z-10 flex min-h-screen items-center justify-center px-4 py-12">
        <div className="w-full max-w-lg rounded-[32px] border border-white/10 bg-gradient-to-b from-[#141f3a]/90 via-[#0f1a31]/95 to-[#0a1323]/95 p-10 shadow-[0_35px_65px_rgba(5,10,30,0.75)] backdrop-blur-2xl">
          <div className="relative flex flex-col items-center text-center">
            <div className="mb-6 flex h-20 w-20 items-center justify-center rounded-3xl border border-white/15 bg-white/5">
              <Shield className="h-10 w-10 text-[#f5c75e]" />
            </div>
            <p className="text-xs font-semibold uppercase tracking-[0.5em] text-[#f5c75e]/80">Secure Access Portal</p>
            <h1 className="mt-4 text-3xl font-semibold tracking-tight text-white">GunShot Detection System</h1>
            <p className="text-base text-white/70">AI-Powered Surveillance</p>
          </div>

          <form onSubmit={handleSubmit} className="mt-10 space-y-6">
            <div className="space-y-2">
              <Label htmlFor="username" className="text-sm uppercase tracking-[0.35em] text-white/70">
                Officer Username
              </Label>
              <div className="relative">
                <Input
                  id="username"
                  type="text"
                  placeholder="Enter your username"
                  value={credentials.username}
                  onChange={(e) => setCredentials((prev) => ({ ...prev, username: e.target.value }))}
                  required
                  className="h-12 rounded-2xl border-white/15 bg-white/5 pl-12 text-base text-white placeholder:text-white/50 focus-visible:ring-[#51f6a7]/60 focus-visible:ring-offset-0"
                />
                <User className="pointer-events-none absolute left-4 top-1/2 h-5 w-5 -translate-y-1/2 text-white/55" />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="password" className="text-sm uppercase tracking-[0.35em] text-white/70">
                Password
              </Label>
              <div className="relative">
                <Input
                  id="password"
                  type={showPassword ? "text" : "password"}
                  placeholder="Enter your password"
                  value={credentials.password}
                  onChange={(e) => setCredentials((prev) => ({ ...prev, password: e.target.value }))}
                  required
                  className="h-12 rounded-2xl border-white/15 bg-white/5 pl-12 pr-12 text-base text-white placeholder:text-white/50 focus-visible:ring-[#51f6a7]/60 focus-visible:ring-offset-0"
                />
                <Lock className="pointer-events-none absolute left-4 top-1/2 h-5 w-5 -translate-y-1/2 text-white/55" />
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="absolute right-2 top-1/2 -translate-y-1/2 rounded-xl bg-white/5 text-white/70 hover:bg-white/10"
                  onClick={() => setShowPassword(!showPassword)}
                >
                  {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </Button>
              </div>
            </div>

            {error && (
              <div className="rounded-2xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-100">
                {error}
              </div>
            )}

            <Button
              type="submit"
              className="group h-12 w-full rounded-2xl bg-[#f5c75e] text-base font-semibold uppercase tracking-wide text-[#0a1323] shadow-[0_15px_45px_rgba(245,199,94,0.4)] transition hover:-translate-y-[1px] hover:bg-[#ffd978] focus-visible:ring-[#f5c75e]/40 disabled:opacity-70"
              disabled={isLoading}
            >
              <Lock className="h-5 w-5 group-disabled:opacity-60" />
              {isLoading ? "Securing Access..." : "Secure Login"}
            </Button>
          </form>

          <div className="mt-8 border-t border-white/10 pt-6 text-center text-sm text-white/70">
            <p className="text-xs font-semibold uppercase tracking-[0.5em] text-white/60">Authorized Personnel Only</p>
            <p>All access attempts are monitored and logged</p>
          </div>

          <div className="mt-6 rounded-2xl border border-white/10 bg-white/5 p-4 text-sm text-white/80">
            <p className="text-xs font-semibold uppercase tracking-[0.5em] text-[#f5c75e]/80">Demo Credentials</p>
            <div className="mt-3 grid grid-cols-2 gap-2 text-base font-medium">
              <div className="rounded-xl bg-white/5 px-3 py-2">
                <p className="text-xs uppercase tracking-[0.4em] text-white/50">User</p>
                <p className="text-lg text-white">officer</p>
              </div>
              <div className="rounded-xl bg-white/5 px-3 py-2">
                <p className="text-xs uppercase tracking-[0.4em] text-white/50">Password</p>
                <p className="text-lg text-[#f5c75e]">dubai2025</p>
              </div>
            </div>
          </div>

          <div className="mt-8 flex justify-center">
            <div className="inline-flex items-center gap-2 rounded-full border border-emerald-400/30 bg-emerald-400/10 px-4 py-2 text-sm text-emerald-200">
              <span className="h-2 w-2 rounded-full bg-emerald-300 animate-pulse" />
              Secure Connection Established
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
