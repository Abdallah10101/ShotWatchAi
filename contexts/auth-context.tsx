"use client"

import type React from "react"
import { createContext, useContext, useState, useEffect } from "react"

interface User {
  id: string
  name: string
  badge: string
  department: string
}

interface AuthContextType {
  user: User | null
  login: (credentials: { username: string; password: string }) => Promise<boolean>
  logout: () => void
  isLoading: boolean
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    // Load session from server
    const loadSession = async () => {
      try {
        const res = await fetch("/api/auth/me", { cache: "no-store" })
        const data = await res.json()
        if (data?.user) {
          setUser({ id: data.user.id, name: data.user.name, badge: data.user.badge, department: data.user.department })
          localStorage.setItem("gunshot-system-user", JSON.stringify({ id: data.user.id, name: data.user.name, badge: data.user.badge, department: data.user.department }))
        } else {
          setUser(null)
          localStorage.removeItem("gunshot-system-user")
        }
      } catch (e) {
        // fall back to localStorage if fetch fails
        const savedUser = localStorage.getItem("gunshot-system-user")
        if (savedUser) {
          setUser(JSON.parse(savedUser))
        }
      } finally {
        setIsLoading(false)
      }
    }
    void loadSession()
  }, [])

  const login = async (credentials: { username: string; password: string }) => {
    setIsLoading(true)
    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(credentials),
      })
      if (!res.ok) {
        setIsLoading(false)
        return false
      }
      const data = await res.json()
      const userData = {
        id: data.user.id as string,
        name: data.user.name as string,
        badge: data.user.badge as string,
        department: data.user.department as string,
      }
      setUser(userData)
      localStorage.setItem("gunshot-system-user", JSON.stringify(userData))
      setIsLoading(false)
      return true
    } catch (e) {
      setIsLoading(false)
      return false
    }
  }

  const logout = () => {
    void fetch("/api/auth/logout", { method: "POST" })
    setUser(null)
    localStorage.removeItem("gunshot-system-user")
  }

  return <AuthContext.Provider value={{ user, login, logout, isLoading }}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider")
  }
  return context
}

