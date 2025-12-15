"use client"

import { useCallback, useEffect, useMemo, useState } from "react"
import { format } from "date-fns"
import {
  Shield,
  MapPin,
  Video,
  LogOut,
  AlertCircle,
  Clock3,
  CheckCircle2,
  Radio,
  AlertTriangle,
  ClipboardList,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent } from "@/components/ui/card"
import { LiveFeed } from "./live-feed"
import { IncidentLogs } from "./incident-logs"
import { IncidentHeatMap } from "./incident-heat-map"
import { useAuth } from "@/contexts/auth-context"

type DashboardView = "overview" | "live" | "logs"

type DetectionSummary = {
  id: string
  label: string
  confidence: number
  timestamp?: number
  isoTimestamp?: string
  source: string
  thumbnail: string | null
  lat?: number | null
  lng?: number | null
  accuracy?: number | null
}

type ModelStats = {
  activeAlerts: number
  todaysDetections: number
  systemStatus: string
  systemMessage: string
  aiAccuracy: number
  detectionCount: number
}

type ConfidenceStatus = "confirmed" | "dispatched" | "pending" | "false-positive"

const statusColors: Record<ConfidenceStatus, string> = {
  confirmed: "text-emerald-300 bg-emerald-500/10 border-emerald-400/30",
  dispatched: "text-sky-300 bg-sky-500/10 border-sky-400/30",
  pending: "text-amber-300 bg-amber-500/10 border-amber-400/30",
  "false-positive": "text-rose-300 bg-rose-500/10 border-rose-400/30",
}

const pingColors: Record<ConfidenceStatus, string> = {
  confirmed: "bg-emerald-400/30",
  dispatched: "bg-sky-400/30",
  pending: "bg-amber-400/30",
  "false-positive": "bg-rose-400/30",
}

function classifyConfidence(confidence = 0): ConfidenceStatus {
  if (confidence >= 0.9) return "confirmed"
  if (confidence >= 0.8) return "dispatched"
  if (confidence >= 0.65) return "pending"
  return "false-positive"
}

export function Dashboard() {
  const { user, logout } = useAuth()
  const [activeView, setActiveView] = useState<DashboardView>("overview")
  const [clock, setClock] = useState(() => new Date())
  const [stats, setStats] = useState<ModelStats>({
    activeAlerts: 0,
    todaysDetections: 0,
    systemStatus: "idle",
    systemMessage: "System ready · Monitoring events",
    aiAccuracy: 0,
    detectionCount: 0,
  })
  const [recentDetections, setRecentDetections] = useState<DetectionSummary[]>([])

  useEffect(() => {
    const timer = setInterval(() => setClock(new Date()), 1000)
    return () => clearInterval(timer)
  }, [])

  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch("/api/dashboard-stats", { cache: "no-store" })
      if (!res.ok) {
        throw new Error("Failed to load stats")
      }
      const payload = (await res.json()) as Partial<ModelStats>
      setStats((prev) => ({
        ...prev,
        ...payload,
      }))
    } catch (error) {
      console.error("[Dashboard] Unable to fetch stats", error)
    }
  }, [])

  const fetchRecentDetections = useCallback(async () => {
    try {
      const res = await fetch("/api/detections", { cache: "no-store" })
      if (!res.ok) {
        throw new Error("Failed to load detections")
      }
      const payload = await res.json()
      const detections: DetectionSummary[] = Array.isArray(payload.detections) ? payload.detections : []
      setRecentDetections(detections)
    } catch (error) {
      console.error("[Dashboard] Unable to fetch detections", error)
    }
  }, [])

  useEffect(() => {
    void fetchStats()
    const interval = setInterval(() => {
      void fetchStats()
    }, 60_000)
    return () => clearInterval(interval)
  }, [fetchStats])

  useEffect(() => {
    void fetchRecentDetections()
  }, [fetchRecentDetections])

  const detectionBreakdown = useMemo(() => {
    return recentDetections.reduce(
      (acc, detection) => {
        const status = classifyConfidence(detection.confidence)
        if (status === "confirmed") acc.confirmed += 1
        else if (status === "dispatched") acc.dispatched += 1
        else if (status === "pending") acc.pending += 1
        else acc.falsePositives += 1
        acc.total += 1
        return acc
      },
      { total: 0, confirmed: 0, dispatched: 0, pending: 0, falsePositives: 0 }
    )
  }, [recentDetections])

  const accuracyPercent = useMemo(() => {
    if (stats.aiAccuracy) {
      return Math.round(stats.aiAccuracy * 1000) / 10
    }
    if (!recentDetections.length) return 0
    const average =
      recentDetections.reduce((sum, item) => sum + (item.confidence ?? 0), 0) / recentDetections.length
    return Math.round(average * 1000) / 10
  }, [recentDetections, stats.aiAccuracy])

  const heatMapPins = useMemo(() => {
    // Convert detections to map markers
    // Use real GPS coordinates if available, otherwise use Dubai center with random offset
    return recentDetections.slice(0, 10).map((det, index) => {
      const hasLocation = det.lat !== null && det.lng !== null && det.lat !== undefined && det.lng !== undefined
      
      return {
        id: det.id,
        lat: hasLocation ? det.lat! : 25.2048 + (Math.random() - 0.5) * 0.1,
        lng: hasLocation ? det.lng! : 55.2708 + (Math.random() - 0.5) * 0.1,
        label: det.label,
        status: classifyConfidence(det.confidence),
        timestamp: det.isoTimestamp ? new Date(det.isoTimestamp).toLocaleString() : "Unknown",
        confidence: det.confidence,
      }
    })
  }, [recentDetections])

  const navItems: { id: DashboardView; label: string }[] = [
    { id: "overview", label: "Dashboard" },
    { id: "live", label: "Live Surveillance" },
    { id: "logs", label: "Incident Logs" },
  ]

const StatCard = ({
  title,
  value,
  subtitle,
  accent,
  icon,
}: {
  title: string
  value: number
  subtitle: string
  accent?: string
  icon: React.ReactNode
}) => (
  <Card className="rounded-2xl border border-[#141c31] bg-[#090f1e] text-white shadow-[0_10px_25px_rgba(0,0,0,0.35)]">
    <CardContent className="flex items-center gap-4 p-5">
      <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-white/5 text-white">{icon}</div>
      <div className="flex flex-col">
        <p className="text-[11px] uppercase tracking-[0.5em] text-white/50">{title}</p>
        <p className={`text-3xl font-semibold leading-tight ${accent ?? "text-white"}`}>{value}</p>
        <p className="text-xs text-white/60">{subtitle}</p>
      </div>
    </CardContent>
  </Card>
)

  const renderOverview = () => (
    <div className="space-y-8">
      <div className="rounded-3xl border border-white/10 bg-gradient-to-r from-[#07122b]/90 to-[#030817]/90 px-6 py-4 shadow-[0_20px_45px_rgba(0,0,0,0.6)]">
        <div className="flex flex-wrap items-center justify-between gap-4 text-sm text-white/80">
          <div className="flex items-center gap-3">
            <div className="rounded-2xl bg-[#f5c75e]/10 p-3">
              <Shield className="h-6 w-6 text-[#f5c75e]" />
            </div>
            <div>
              <p className="text-base font-semibold text-white">GunShot Detection</p>
              <p className="text-[11px] uppercase tracking-[0.4em] text-white/60">AI-Powered Surveillance System</p>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center gap-2 rounded-2xl border border-white/10 bg-white/5 px-4 py-2 text-white">
              <Clock3 className="h-4 w-4 text-[#f5c75e]" />
              {format(clock, "EEE, dd MMM · hh:mm:ss a")}
            </div>
            <div className="flex items-center gap-2 rounded-2xl border border-white/10 bg-white/5 px-4 py-2 text-white">
              <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
              Secure Link
            </div>
            <Badge variant="outline" className="border-emerald-400/40 text-emerald-200">
              Officer On Duty
            </Badge>
          </div>
        </div>
      </div>

      <div className="space-y-6 rounded-3xl border border-white/10 bg-gradient-to-b from-[#050b19] via-[#050a14] to-[#02040a] p-8 shadow-[0_35px_80px_rgba(2,4,10,0.85)]">
        <div className="flex flex-wrap items-start justify-between gap-6 border-b border-white/5 pb-6">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.55em] text-[#f5c75e]/80">
              GunShot Detection · AI Surveillance
            </p>
            <h1 className="mt-3 text-3xl font-semibold text-white">
              Welcome back, <span className="text-[#4cc9f0]">{user?.name ?? "Officer"}</span>
            </h1>
            <p className="text-sm text-gray-400">
              Monitor gunshot detection alerts and manage incident responses in real time.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <div className="rounded-2xl border border-white/10 bg-black/30 px-5 py-3">
              <p className="text-[11px] uppercase tracking-[0.4em] text-gray-500">Current Time</p>
              <p className="text-lg font-semibold text-white">{format(clock, "EEE, MMM d · hh:mm:ss a")}</p>
            </div>
            <Badge variant="outline" className="border-emerald-400/40 text-emerald-200">
              {user?.badge ?? "On Duty"}
            </Badge>
          </div>
        </div>

        <div className="flex flex-wrap gap-3">
          {navItems.map((item) => (
            <Button
              key={item.id}
              variant={item.id === activeView ? "default" : "outline"}
              className={`rounded-full border-white/20 ${item.id === activeView ? "bg-[#f5c75e] text-black" : "bg-transparent text-white"}`}
              onClick={() => setActiveView(item.id)}
            >
              {item.label}
            </Button>
          ))}
          <Button
            variant="ghost"
            className="rounded-full border border-white/20 text-white hover:bg-white/10"
            onClick={logout}
          >
            <LogOut className="mr-2 h-4 w-4" />
            Logout
          </Button>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
        <StatCard
          title="Total Alerts"
          value={detectionBreakdown.total}
          subtitle="System-wide detections"
          icon={<AlertCircle className="h-6 w-6" />}
        />
        <StatCard
          title="Pending Review"
          value={detectionBreakdown.pending}
          subtitle="Awaiting officer review"
          icon={<Clock3 className="h-6 w-6 text-amber-200" />}
          accent="text-amber-300"
        />
        <StatCard
          title="Confirmed"
          value={detectionBreakdown.confirmed}
          subtitle="Verified incidents"
          icon={<CheckCircle2 className="h-6 w-6 text-emerald-200" />}
          accent="text-emerald-300"
        />
        <StatCard
          title="Dispatched"
          value={detectionBreakdown.dispatched}
          subtitle="Units on scene"
          icon={<Radio className="h-6 w-6 text-sky-200" />}
          accent="text-sky-300"
        />
        <StatCard
          title="False Positives"
          value={detectionBreakdown.falsePositives}
          subtitle="Filtered alerts"
          icon={<AlertTriangle className="h-6 w-6 text-rose-200" />}
          accent="text-rose-300"
        />
      </div>

      <Card className="border border-emerald-500/20 bg-gradient-to-r from-[#0d251b] via-[#0a1a14] to-[#050a0f] text-white">
        <CardContent className="flex flex-wrap items-center justify-between gap-6 p-6">
          <div className="flex items-center gap-4">
            <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-emerald-500/20">
              <Shield className="h-7 w-7 text-emerald-300" />
            </div>
            <div>
              <p className="text-xs uppercase tracking-[0.5em] text-emerald-300">AI Gunshot Detection Model</p>
              <p className="text-sm text-emerald-100">{stats.systemMessage}</p>
            </div>
          </div>
          <div className="text-right">
            <p className="text-sm text-emerald-200">Model Accuracy</p>
            <p className="text-3xl font-semibold text-white">{accuracyPercent ? `${accuracyPercent}%` : "--"}</p>
            <p className="text-xs text-emerald-200/70">+2.3% this week</p>
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-5 lg:grid-cols-2">
        {[
          {
            id: "live",
            title: "Live Surveillance",
            description: "Monitor real-time camera feeds with AI gunshot detection.",
            statusLabel: "6 Active Cameras",
            statusHint: "Open",
            icon: <Video className="h-5 w-5 text-white" />,
            accent: "from-[#101c3a] via-[#0a1430] to-[#070c1d]",
            onClick: () => setActiveView("live"),
            accentText: "text-sky-200",
            iconBg: "bg-gradient-to-br from-[#3b82f6] to-[#2563eb]",
          },
          {
            id: "logs",
            title: "Incident Logs",
            description: "Review detected incidents with timestamps and AI confidence scores.",
            statusLabel: `${stats.activeAlerts} Total Alerts`,
            statusHint: "Open",
            icon: <ClipboardList className="h-5 w-5 text-white" />,
            accent: "from-[#0c241f] via-[#071813] to-[#040e0a]",
            onClick: () => setActiveView("logs"),
            accentText: "text-emerald-200",
            iconBg: "bg-gradient-to-br from-[#34d399] to-[#059669]",
          },
        ].map((card) => (
          <Card
            key={card.id}
            className={`border border-white/5 bg-gradient-to-br ${card.accent} text-white shadow-[0_20px_35px_rgba(0,0,0,0.4)]`}
          >
            <CardContent className="flex h-full flex-col gap-4 p-6">
              <div className="flex items-start justify-between">
                <div>
                  <div className={`mb-4 inline-flex h-12 w-12 items-center justify-center rounded-2xl ${card.iconBg}`}>
                    {card.icon}
                  </div>
                  <h3 className="text-2xl font-semibold">{card.title}</h3>
                  <p className="mt-1 text-sm text-white/70">{card.description}</p>
                </div>
                <Button
                  variant="ghost"
                  className="group rounded-full border border-white/10 bg-white/5 px-4 py-1 text-sm font-semibold text-white transition hover:bg-white/10"
                  onClick={card.onClick}
                >
                  Open
                  <span className="text-base text-[#f5c75e] transition group-hover:translate-x-0.5 group-hover:text-white">›</span>
                </Button>
              </div>
              <div className={`text-sm font-semibold ${card.accentText}`}>{card.statusLabel}</div>
            </CardContent>
          </Card>
        ))}
      </div>

      <Card className="border border-white/10 bg-white/5 text-white">
        <CardContent className="space-y-4 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.4em] text-gray-400">Incident Heat Map</p>
              <p className="text-sm text-gray-400">Showing {heatMapPins.length} active incidents on map</p>
            </div>
            <Button 
              className="rounded-full bg-gradient-to-r from-[#2563eb] via-[#1d4ed8] to-[#1e40af] px-5 py-2 text-sm font-semibold text-white shadow-[0_10px_25px_rgba(37,99,235,0.35)] hover:brightness-110" 
              onClick={() => setActiveView("live")}
            >
              View Full Map
            </Button>
          </div>
          <div className="h-80">
            {heatMapPins.length > 0 ? (
              <IncidentHeatMap incidents={heatMapPins} />
            ) : (
              <div className="flex h-full items-center justify-center rounded-3xl border border-white/10 bg-gradient-to-b from-[#050b12] via-[#04070e] to-[#010205] text-sm text-gray-500">
                No incidents to plot on map.
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  )

  const renderLiveView = () => (
    <div className="rounded-[32px] border border-white/10 bg-gradient-to-b from-[#071430] via-[#050f24] to-[#030814] p-6 text-white shadow-[0_35px_75px_rgba(2,6,23,0.6)]">
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/10 pb-4">
        <div>
          <p className="text-lg font-semibold text-white">GunShot Detection</p>
          <p className="text-xs uppercase tracking-[0.4em] text-white/60">Live Surveillance</p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <Button
            className="rounded-full bg-gradient-to-r from-[#2563eb] via-[#1d4ed8] to-[#1e40af] px-5 py-2 text-sm font-semibold text-white shadow-[0_10px_25px_rgba(37,99,235,0.35)] hover:brightness-110"
            onClick={() => setActiveView("overview")}
          >
            Dashboard
          </Button>
          <Button className="rounded-full bg-gradient-to-r from-[#facc15] via-[#eab308] to-[#ca8a04] px-5 py-2 text-sm font-semibold text-[#2b1a00] shadow-[0_10px_25px_rgba(234,179,8,0.35)] hover:brightness-110">
            {user?.name ?? "Officer"}
          </Button>
          <Button
            variant="ghost"
            className="rounded-full border border-white/15 bg-white/5 px-3 py-2 text-white hover:bg-white/15"
            onClick={logout}
          >
            <LogOut className="h-4 w-4" />
          </Button>
        </div>
      </div>
      <LiveFeed onViewLogs={() => setActiveView("logs")} onViewAnalytics={() => setActiveView("overview")} />
    </div>
  )

  const renderIncidentView = () => (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-[0.4em] text-gray-400">Incident Logs</p>
          <h2 className="text-3xl font-semibold text-white">Field Evidence Review</h2>
          <p className="text-sm text-gray-400">Review captured frames saved in the temp directory.</p>
        </div>
        <div className="flex flex-wrap gap-3">
          <Button
            className="rounded-full bg-gradient-to-r from-[#2563eb] via-[#1d4ed8] to-[#1e40af] px-5 py-2 text-sm font-semibold text-white shadow-[0_10px_25px_rgba(37,99,235,0.35)] hover:brightness-110"
            onClick={() => setActiveView("overview")}
          >
            ← Dashboard
          </Button>
          <Button
            className="rounded-full bg-gradient-to-r from-[#34d399] via-[#10b981] to-[#059669] px-5 py-2 text-sm font-semibold text-white shadow-[0_10px_25px_rgba(16,185,129,0.35)] hover:brightness-110"
            onClick={() => setActiveView("live")}
          >
            Live Surveillance
          </Button>
        </div>
      </div>
      <IncidentLogs seedDetections={recentDetections} onDetectionsChange={setRecentDetections} onGoLive={() => setActiveView("live")} />
    </div>
  )

  return (
    <div className="min-h-screen bg-[#020307] text-white">
      <div className="mx-auto max-w-7xl space-y-8 px-4 py-8">
        {activeView === "overview" && renderOverview()}
        {activeView === "live" && renderLiveView()}
        {activeView === "logs" && renderIncidentView()}
      </div>
    </div>
  )
}
