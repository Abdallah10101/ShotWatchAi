"use client"

import { useEffect, useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { AlertTriangle, MapPin, Clock, Zap, Camera, Eye } from "lucide-react"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"

interface DetectionEntry {
  id: string
  timestamp: number
  isoTimestamp: string
  label: string
  confidence: number
  quality: number | null
  filename: string | null
  source: string
  thumbnail: string | null
}

export function AlertSystem() {
  const [detections, setDetections] = useState<DetectionEntry[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let isMounted = true

    const fetchDetections = async () => {
      setIsLoading(true)
      try {
        const res = await fetch("/api/detections")
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`)
        }
        const data = await res.json()
        if (isMounted) {
          setDetections(data.detections ?? [])
          setError(null)
        }
      } catch (err) {
        console.error("Failed to load detections", err)
        if (isMounted) {
          setError("Unable to load detections")
        }
      } finally {
        if (isMounted) {
          setIsLoading(false)
        }
      }
    }

    fetchDetections()
    const interval = setInterval(fetchDetections, 5000)

    return () => {
      isMounted = false
      clearInterval(interval)
    }
  }, [])

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case "high":
        return "bg-red-500"
      case "medium":
        return "bg-yellow-500"
      case "low":
        return "bg-green-500"
      default:
        return "bg-gray-500"
    }
  }

  return (
    <Card className="h-full border border-[#171b27] bg-[#070b12] text-white">
      <CardContent className="space-y-5 p-5">
        <div className="flex items-center justify-between border-b border-white/5 pb-4">
          <div>
            <p className="text-[11px] uppercase tracking-[0.4em] text-gray-400">Active alerts</p>
            <p className="text-lg font-semibold">Incident queue</p>
          </div>
          <div className="flex items-center gap-2 rounded-full bg-red-500/10 px-3 py-1 text-xs font-semibold text-red-300">
            <AlertTriangle className="h-4 w-4" />
            {detections.length}
          </div>
        </div>

        {error && (
          <div className="border border-red-500/30 bg-red-900/20 text-red-200 rounded-xl p-3 text-sm">
            {error}
          </div>
        )}

        {isLoading && detections.length === 0 && (
          <div className="text-sm text-gray-500">Loading detections...</div>
        )}

        {!isLoading && detections.length === 0 && !error && (
          <div className="text-sm text-gray-500">No detections yet.</div>
        )}

        <div className="space-y-4 max-h-96 overflow-y-auto pr-2">
          {detections.map((detection) => (
            <div key={detection.id} className="rounded-2xl border border-white/5 bg-black/20 p-4 space-y-3">
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-2">
                  <div className={`w-2.5 h-2.5 rounded-full ${getSeverityColor("high")} animate-pulse`} />
                  <Badge variant="destructive" className="tracking-[0.3em] text-[10px]">ACTIVE</Badge>
                </div>
                <div className="flex items-center gap-1 text-xs text-gray-400">
                  <Clock className="h-3.5 w-3.5" />
                  {new Date(detection.timestamp * 1000).toLocaleTimeString()}
                </div>
              </div>

              <div className="space-y-2 text-sm">
                <div className="flex items-center gap-2 text-red-200">
                  <Zap className="h-4 w-4 text-orange-400" />
                  <span className="font-semibold text-white">
                    {detection.label} ({Math.round(detection.confidence * 100)}% confidence)
                  </span>
                </div>

                {detection.thumbnail && (
                  <div className="rounded-xl border border-red-500/20 bg-red-500/5 p-3">
                    <div className="flex items-center gap-2 text-xs uppercase tracking-[0.3em] text-red-300 mb-2">
                      <Camera className="h-3.5 w-3.5" />
                      AI Evidence
                    </div>
                    <div className="flex gap-3">
                      <Dialog>
                        <DialogTrigger asChild>
                          <div className="relative cursor-pointer overflow-hidden rounded-lg border border-white/5">
                            <img
                              src={detection.thumbnail}
                              alt="Alert evidence"
                              className="h-16 w-24 object-cover transition hover:opacity-80"
                            />
                            <div className="absolute inset-0 flex items-center justify-center bg-black/40 opacity-0 transition hover:opacity-100">
                              <Eye className="h-4 w-4 text-white" />
                            </div>
                          </div>
                        </DialogTrigger>
                        <DialogContent className="max-w-2xl">
                          <DialogHeader>
                            <DialogTitle>Alert Evidence</DialogTitle>
                          </DialogHeader>
                          <img src={detection.thumbnail} alt="Full alert evidence" className="w-full rounded-lg" />
                        </DialogContent>
                      </Dialog>
                    </div>
                  </div>
                )}
              </div>

              <Button size="sm" variant="outline" className="w-full justify-center gap-2 border-white/10 bg-white/5 text-white hover:border-red-400">
                <MapPin className="h-4 w-4" />
                Dispatch Unit
              </Button>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
