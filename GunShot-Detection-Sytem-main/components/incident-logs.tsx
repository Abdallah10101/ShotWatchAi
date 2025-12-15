"use client"

import { useCallback, useEffect, useMemo, useState } from "react"
import { format, formatDistanceToNow } from "date-fns"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { AlertTriangle, Camera, CheckCircle2, Eye, MapPin, RefreshCw, Search, Trash2, Video, XOctagon, ShieldAlert, Radio, ChevronDown } from "lucide-react"

type DetectionLog = {
  id: string
  label: string
  confidence: number
  source: string
  timestamp?: number
  isoTimestamp?: string
  filename?: string | null
  thumbnail: string | null
}

type DetectionStatus = "confirmed" | "dispatched" | "pending" | "false-positive"

const statusMeta: Record<
  DetectionStatus,
  { label: string; badgeClass: string; cardClass: string }
> = {
  confirmed: {
    label: "Confirmed",
    badgeClass: "border-emerald-400/50 text-emerald-200 bg-emerald-500/10",
    cardClass: "border-emerald-500/20",
  },
  dispatched: {
    label: "Dispatched",
    badgeClass: "border-sky-400/40 text-sky-200 bg-sky-500/10",
    cardClass: "border-sky-500/20",
  },
  pending: {
    label: "Pending Review",
    badgeClass: "border-amber-400/50 text-amber-200 bg-amber-500/10",
    cardClass: "border-amber-500/20",
  },
  "false-positive": {
    label: "False Positive",
    badgeClass: "border-rose-500/50 text-rose-200 bg-rose-500/10",
    cardClass: "border-rose-500/20",
  },
}

function classifyDetection(confidence = 0): DetectionStatus {
  if (confidence >= 0.9) return "confirmed"
  if (confidence >= 0.8) return "dispatched"
  if (confidence >= 0.65) return "pending"
  return "false-positive"
}

interface IncidentLogsProps {
  seedDetections?: DetectionLog[]
  onDetectionsChange?: (detections: DetectionLog[]) => void
  onGoLive?: () => void
}

export function IncidentLogs({ seedDetections = [], onDetectionsChange, onGoLive }: IncidentLogsProps) {
  const [detections, setDetections] = useState<DetectionLog[]>(seedDetections)
  const [isLoading, setIsLoading] = useState(seedDetections.length === 0)
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [selectedDetection, setSelectedDetection] = useState<DetectionLog | null>(null)
  const [query, setQuery] = useState("")
  const [selectedStatus, setSelectedStatus] = useState<DetectionStatus | "all">("all")
  const [manualStatuses, setManualStatuses] = useState<Record<string, DetectionStatus>>({})

  useEffect(() => {
    if (seedDetections.length) {
      setDetections(seedDetections)
      setIsLoading(false)
    }
  }, [seedDetections])

  const fetchDetections = useCallback(async () => {
    setIsRefreshing(true)
    setIsLoading(true)
    try {
      const res = await fetch("/api/detections", { cache: "no-store" })
      if (!res.ok) {
        throw new Error("Failed to load detections")
      }
      const payload = await res.json()
      const list: DetectionLog[] = Array.isArray(payload.detections) ? payload.detections : []
      setDetections(list)
      onDetectionsChange?.(list)
    } catch (error) {
      console.error("[IncidentLogs] Unable to fetch detections", error)
    } finally {
      setIsLoading(false)
      setIsRefreshing(false)
    }
  }, [onDetectionsChange])

  useEffect(() => {
    if (!seedDetections.length) {
      void fetchDetections()
    }
  }, [fetchDetections, seedDetections.length])

  const filteredDetections = useMemo(() => {
    const term = query.trim().toLowerCase()
    return detections.filter((entry) => {
      const matchesStatus = selectedStatus === "all" ? true : (manualStatuses[entry.id] ?? classifyDetection(entry.confidence)) === selectedStatus
      const matchesQuery =
        !term ||
        entry.label.toLowerCase().includes(term) ||
        entry.filename?.toLowerCase().includes(term) ||
        entry.source?.toLowerCase().includes(term)
      return matchesStatus && matchesQuery
    })
  }, [detections, manualStatuses, query, selectedStatus])

  const metrics = useMemo(() => {
    return detections.reduce(
      (acc, entry) => {
        const status = manualStatuses[entry.id] ?? classifyDetection(entry.confidence)
        acc.total += 1
        if (status === "confirmed") acc.confirmed += 1
        else if (status === "dispatched") acc.dispatched += 1
        else if (status === "pending") acc.pending += 1
        else acc.falsePositives += 1
        return acc
      },
      { total: 0, confirmed: 0, dispatched: 0, pending: 0, falsePositives: 0 }
    )
  }, [detections, manualStatuses])

  const removeDetection = (id: string) => {
    setDetections((prev) => prev.filter((item) => item.id !== id))
  }

  const updateStatus = (id: string, status: DetectionStatus) => {
    setManualStatuses((prev) => ({ ...prev, [id]: status }))
  }

  const statusOptions: { label: string; value: DetectionStatus | "all" }[] = [
    { label: "All Status", value: "all" },
    { label: "Pending", value: "pending" },
    { label: "Confirmed", value: "confirmed" },
    { label: "Dispatched", value: "dispatched" },
    { label: "False Positive", value: "false-positive" },
  ]

  const getActionButtons = (entry: DetectionLog, activeStatus: DetectionStatus) => [
    {
      label: "Confirm",
      onClick: () => updateStatus(entry.id, "confirmed"),
      className: "bg-emerald-500/15 text-emerald-100 hover:bg-emerald-500/25",
      icon: <CheckCircle2 className="h-3.5 w-3.5" />,
      isActive: activeStatus === "confirmed",
    },
    {
      label: "Dispatch",
      onClick: () => updateStatus(entry.id, "dispatched"),
      className: "bg-sky-500/15 text-sky-200 hover:bg-sky-500/25",
      icon: <Radio className="h-3.5 w-3.5" />,
      isActive: activeStatus === "dispatched",
    },
    {
      label: "False +",
      onClick: () => updateStatus(entry.id, "false-positive"),
      className: "bg-rose-500/15 text-rose-200 hover:bg-rose-500/25",
      icon: <XOctagon className="h-3.5 w-3.5" />,
      isActive: activeStatus === "false-positive",
    },
  ]

  const renderDetectionCard = (entry: DetectionLog) => {
    const takenAt = entry.isoTimestamp ? new Date(entry.isoTimestamp) : entry.timestamp ? new Date(entry.timestamp * 1000) : null
    const relative = takenAt ? formatDistanceToNow(takenAt, { addSuffix: true }) : "Unknown time"
    const status = manualStatuses[entry.id] ?? classifyDetection(entry.confidence)
    const statusStyles = statusMeta[status]
    const actions = getActionButtons(entry, status)
    const location = entry.source || entry.filename || "Unknown location"

    return (
      <div
        key={entry.id}
        className="rounded-3xl border border-white/5 bg-gradient-to-b from-[#0b1224] via-[#0a0f1e] to-[#050812] p-5 text-white shadow-[0_30px_55px_rgba(3,7,18,0.65)]"
      >
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="inline-flex items-center gap-3 rounded-full bg-white/5 px-4 py-2 text-xs font-semibold uppercase tracking-[0.4em] text-white/70">
            <ShieldAlert className="h-4 w-4 text-yellow-300" />
            {entry.label}
          </div>
          <div className="flex flex-wrap gap-2">
            <Button
              size="sm"
              variant="ghost"
              className="rounded-full border border-white/15 bg-white/10 px-4 text-white hover:bg-white/20"
              onClick={() => setSelectedDetection(entry)}
            >
              <Eye className="mr-2 h-4 w-4" />
              View
            </Button>
            <Button
              size="sm"
              variant="outline"
              className="rounded-full border-rose-400/40 bg-transparent text-rose-200 hover:bg-rose-500/20"
              onClick={() => removeDetection(entry.id)}
            >
              <Trash2 className="mr-2 h-4 w-4" />
              Delete
            </Button>
            <div className={`inline-flex items-center gap-2 rounded-full border px-4 py-1.5 text-xs font-semibold ${statusStyles.badgeClass}`}>
              {statusStyles.label}
            </div>
          </div>
        </div>

        <div className="mt-5 grid gap-5 lg:grid-cols-[240px_1fr]">
          <div className="overflow-hidden rounded-2xl border border-white/10 bg-black">
            {entry.thumbnail ? (
              <img src={entry.thumbnail} alt="Detection frame" className="h-full w-full object-cover" />
            ) : (
              <div className="flex h-40 items-center justify-center bg-gradient-to-b from-gray-800 to-gray-900 text-gray-500">
                <Camera className="h-12 w-12" />
              </div>
            )}
          </div>

          <div className="flex flex-col justify-between gap-4">
            <div className="space-y-3">
              <div className="flex flex-wrap items-center gap-3 text-sm text-white/70">
                <div className="flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1">
                  <MapPin className="h-3.5 w-3.5 text-sky-300" />
                  {location}
                </div>
                <div className="flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1">
                  <AlertTriangle className="h-3.5 w-3.5 text-amber-300" />
                  {relative}
                </div>
              </div>

              <p className="text-sm leading-relaxed text-white/80">
                AI detected potential gunshot with <span className="font-semibold text-white">{Math.round(entry.confidence * 100)}% confidence</span>. Awaiting officer review.
              </p>
            </div>

            <div className="flex flex-wrap gap-3 text-sm text-white/75">
              {actions.map((action) => (
                <button
                  key={action.label}
                  type="button"
                  className={`inline-flex items-center gap-2 rounded-full px-4 py-1.5 text-xs font-semibold transition ${action.className} ${
                    action.isActive ? "ring-2 ring-offset-2 ring-white/20" : ""
                  }`}
                  onClick={action.onClick}
                >
                  {action.icon}
                  {action.label}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    )
  }

  const renderModal = () => {
    if (!selectedDetection) return null
    const takenAt = selectedDetection.isoTimestamp
      ? new Date(selectedDetection.isoTimestamp)
      : selectedDetection.timestamp
        ? new Date(selectedDetection.timestamp * 1000)
        : null

    return (
      <Dialog open={!!selectedDetection} onOpenChange={() => setSelectedDetection(null)}>
        <DialogContent className="max-w-3xl border border-white/20 bg-gradient-to-b from-[#0a1428] to-[#030a14] text-white">
          <DialogHeader>
            <DialogTitle className="text-2xl font-semibold text-white">Detection Details</DialogTitle>
          </DialogHeader>
          <div className="space-y-6">
            <div className="overflow-hidden rounded-2xl border border-white/10 bg-black">
              {selectedDetection.thumbnail ? (
                <img src={selectedDetection.thumbnail} alt="Detection frame" className="w-full object-contain" />
              ) : (
                <div className="flex h-64 items-center justify-center bg-gradient-to-b from-gray-800 to-gray-900 text-gray-500">
                  <Video className="h-16 w-16" />
                </div>
              )}
            </div>
            <div className="grid gap-4 text-sm">
              <div className="flex items-center justify-between rounded-xl border border-white/10 bg-white/5 px-4 py-3">
                <span className="text-white/70">Weapon Type</span>
                <span className="font-semibold text-white">{selectedDetection.label}</span>
              </div>
              <div className="flex items-center justify-between rounded-xl border border-white/10 bg-white/5 px-4 py-3">
                <span className="text-white/70">Confidence</span>
                <span className="font-semibold text-white">{Math.round(selectedDetection.confidence * 100)}%</span>
              </div>
              <div className="flex items-center justify-between rounded-xl border border-white/10 bg-white/5 px-4 py-3">
                <span className="text-white/70">Source</span>
                <span className="font-semibold text-white">{selectedDetection.source}</span>
              </div>
              <div className="flex items-center justify-between rounded-xl border border-white/10 bg-white/5 px-4 py-3">
                <span className="text-white/70">Timestamp</span>
                <span className="font-semibold text-white">
                  {takenAt ? format(takenAt, "PPpp") : "Unknown"}
                </span>
              </div>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    )
  }

  return (
    <div className="space-y-6">
      <Card className="border border-white/10 bg-gradient-to-r from-[#0a1629] to-[#040c18] text-white">
        <CardContent className="p-6">
          <div className="mb-6 flex flex-wrap items-center justify-between gap-4">
            <div>
              <p className="text-xs uppercase tracking-[0.4em] text-white/50">Incident Logs</p>
              <h2 className="text-2xl font-semibold text-white">Live detections feed</h2>
              <p className="text-sm text-white/60">Monitor AI gunshot detections, review evidence, and update statuses.</p>
            </div>
            <div className="flex flex-wrap gap-3">
              <Button
                variant="outline"
                className="rounded-full border-white/20 bg-white/5 text-white hover:bg-white/10"
                onClick={fetchDetections}
                disabled={isRefreshing}
              >
                <RefreshCw className={`mr-2 h-4 w-4 ${isRefreshing ? "animate-spin" : ""}`} />
                Refresh
              </Button>
              {onGoLive && (
                <Button
                  className="rounded-full bg-gradient-to-r from-emerald-500 to-teal-500 px-5 py-2 text-white hover:brightness-110"
                  onClick={onGoLive}
                >
                  <Video className="mr-2 h-4 w-4" />
                  Live Cam
                </Button>
              )}
            </div>
          </div>

          <div className="grid gap-4 md:grid-cols-5">
            <Card className="border border-white/10 bg-gradient-to-br from-[#0d1b35] to-[#060f1d] text-white">
              <CardContent className="p-4">
                <p className="text-xs uppercase tracking-[0.4em] text-white/50">Total Alerts</p>
                <p className="text-3xl font-semibold text-white">{metrics.total}</p>
                <p className="text-xs text-white/60">System-wide detections</p>
              </CardContent>
            </Card>
            <Card className="border border-amber-400/20 bg-gradient-to-br from-[#2d2009] to-[#1a1305] text-white">
              <CardContent className="p-4">
                <p className="text-xs uppercase tracking-[0.4em] text-amber-300/70">Pending</p>
                <p className="text-3xl font-semibold text-amber-200">{metrics.pending}</p>
                <p className="text-xs text-amber-200/60">Awaiting officer review</p>
              </CardContent>
            </Card>
            <Card className="border border-emerald-400/20 bg-gradient-to-br from-[#0d2519] to-[#061510] text-white">
              <CardContent className="p-4">
                <p className="text-xs uppercase tracking-[0.4em] text-emerald-300/70">Confirmed</p>
                <p className="text-3xl font-semibold text-emerald-200">{metrics.confirmed}</p>
                <p className="text-xs text-emerald-200/60">Verified incidents</p>
              </CardContent>
            </Card>
            <Card className="border border-sky-400/20 bg-gradient-to-br from-[#0d1f35] to-[#061220] text-white">
              <CardContent className="p-4">
                <p className="text-xs uppercase tracking-[0.4em] text-sky-300/70">Dispatched</p>
                <p className="text-3xl font-semibold text-sky-200">{metrics.dispatched}</p>
                <p className="text-xs text-sky-200/60">Units on scene</p>
              </CardContent>
            </Card>
            <Card className="border border-rose-400/20 bg-gradient-to-br from-[#2d0d19] to-[#1a0610] text-white">
              <CardContent className="p-4">
                <p className="text-xs uppercase tracking-[0.4em] text-rose-300/70">False Positives</p>
                <p className="text-3xl font-semibold text-rose-200">{metrics.falsePositives}</p>
                <p className="text-xs text-rose-200/60">Filtered alerts</p>
              </CardContent>
            </Card>
          </div>
        </CardContent>
      </Card>

      <div className="flex flex-wrap items-center justify-between gap-4 rounded-2xl border border-white/10 bg-white/5 p-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-white/70" />
          <Input
            type="text"
            placeholder="Search by location, notes, or weapon type..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="rounded-2xl border-white/10 bg-white/5 pl-10 text-white placeholder:text-white/50"
          />
        </div>
        <div className="relative">
          <select
            value={selectedStatus}
            onChange={(event) => setSelectedStatus(event.target.value as DetectionStatus | "all")}
            className="appearance-none rounded-2xl border border-white/10 bg-white/5 px-4 py-2 pr-8 text-sm text-white focus:outline-none"
          >
            {statusOptions.map((option) => (
              <option key={option.value} value={option.value} className="bg-[#0a1428] text-white">
                {option.label}
              </option>
            ))}
          </select>
          <ChevronDown className="pointer-events-none absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 text-white/70" />
        </div>
        <div className="flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm text-white">
          <AlertTriangle className="h-4 w-4 text-amber-300" />
          {filteredDetections.length} visible results
        </div>
      </div>

      <div className="space-y-5">
        {isLoading && !detections.length ? (
          <div className="flex h-64 items-center justify-center text-sm text-white/60">Loading detections...</div>
        ) : filteredDetections.length === 0 ? (
          <div className="flex h-64 items-center justify-center text-sm text-white/60">
            {query || selectedStatus !== "all" ? "No detections match your filters." : "No detections logged yet."}
          </div>
        ) : (
          filteredDetections.map(renderDetectionCard)
        )}
      </div>

      {renderModal()}
    </div>
  )
}
