"use client"

import { useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { FileText, Clock, MapPin, Zap, Camera, Eye, User, Users } from "lucide-react"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"

interface LogEntry {
  id: string
  timestamp: Date
  location: string
  firearmType: string
  confidence: number
  status: "detected" | "verified" | "false-positive"
  officer?: string
  capturedImage?: {
    url: string
    timestamp: Date
    aiAnalysis: {
      suspectDetected: boolean
      suspectCount: number
      weaponVisible: boolean
      description: string
    }
  }
}

export function ActivityLog() {
  const [logs, setLogs] = useState<LogEntry[]>([
    {
      id: "1",
      timestamp: new Date(Date.now() - 5 * 60 * 1000),
      location: "Downtown Dubai, Sheikh Zayed Road",
      firearmType: "Handgun (9mm)",
      confidence: 94,
      status: "detected",
      capturedImage: {
        url: "/placeholder.svg?height=200&width=300",
        timestamp: new Date(Date.now() - 5 * 60 * 1000),
        aiAnalysis: {
          suspectDetected: true,
          suspectCount: 1,
          weaponVisible: true,
          description: "Male suspect, approximately 30-35 years old, dark clothing, weapon visible in right hand",
        },
      },
    },
    {
      id: "2",
      timestamp: new Date(Date.now() - 15 * 60 * 1000),
      location: "Dubai Marina, Marina Walk",
      firearmType: "Rifle (5.56mm)",
      confidence: 87,
      status: "verified",
      officer: "Officer Al-Mansouri",
      capturedImage: {
        url: "/placeholder.svg?height=200&width=300",
        timestamp: new Date(Date.now() - 15 * 60 * 1000),
        aiAnalysis: {
          suspectDetected: true,
          suspectCount: 2,
          weaponVisible: false,
          description: "Two suspects near marina entrance, weapons not clearly visible in frame",
        },
      },
    },
    {
      id: "3",
      timestamp: new Date(Date.now() - 30 * 60 * 1000),
      location: "Business Bay, Executive Tower",
      firearmType: "Shotgun (12 gauge)",
      confidence: 76,
      status: "false-positive",
      officer: "Officer Hassan",
      capturedImage: {
        url: "/placeholder.svg?height=200&width=300",
        timestamp: new Date(Date.now() - 30 * 60 * 1000),
        aiAnalysis: {
          suspectDetected: false,
          suspectCount: 0,
          weaponVisible: false,
          description: "Construction workers using pneumatic tools - false positive detection",
        },
      },
    },
    {
      id: "4",
      timestamp: new Date(Date.now() - 45 * 60 * 1000),
      location: "Jumeirah Beach Road",
      firearmType: "Handgun (9mm)",
      confidence: 91,
      status: "verified",
      officer: "Officer Khalil",
      capturedImage: {
        url: "/placeholder.svg?height=200&width=300",
        timestamp: new Date(Date.now() - 45 * 60 * 1000),
        aiAnalysis: {
          suspectDetected: true,
          suspectCount: 1,
          weaponVisible: true,
          description: "Single suspect on beach road, weapon clearly visible, suspect apprehended",
        },
      },
    },
  ])

  const getStatusColor = (status: string) => {
    switch (status) {
      case "detected":
        return "destructive"
      case "verified":
        return "default"
      case "false-positive":
        return "secondary"
      default:
        return "default"
    }
  }

  const getStatusText = (status: string) => {
    switch (status) {
      case "detected":
        return "DETECTED"
      case "verified":
        return "VERIFIED"
      case "false-positive":
        return "FALSE POSITIVE"
      default:
        return status.toUpperCase()
    }
  }

  return (
    <Card className="h-full border border-[#171b27] bg-[#05080f] text-white">
      <CardContent className="space-y-4 p-5">
        <div className="flex items-center gap-2 text-xs uppercase tracking-[0.4em] text-gray-500">
          <FileText className="h-4 w-4 text-red-300" />
          Activity log
        </div>
        <ScrollArea className="h-80">
          <div className="space-y-4">
            {logs.map((log) => (
              <div key={log.id} className="rounded-2xl border border-white/5 bg-black/30 p-4">
                <div className="mb-3 flex items-start justify-between">
                  <Badge variant={getStatusColor(log.status) as any} className="tracking-[0.3em] text-[10px]">
                    {getStatusText(log.status)}
                  </Badge>
                  <div className="flex items-center gap-1 text-xs text-gray-400">
                    <Clock className="h-3 w-3" />
                    {log.timestamp.toLocaleString()}
                  </div>
                </div>

                <div className="space-y-3 text-sm text-gray-200">
                  <div className="flex items-center gap-2">
                    <MapPin className="h-4 w-4 text-red-400" />
                    <span className="font-semibold text-white">{log.location}</span>
                  </div>

                  <div className="flex items-center gap-2 text-red-200">
                    <Zap className="h-4 w-4 text-orange-400" />
                    <span>{log.firearmType}</span>
                    <span className="text-gray-500">({log.confidence}% confidence)</span>
                  </div>

                  {log.capturedImage && (
                    <div className="rounded-xl border border-white/10 bg-white/5 p-3">
                      <div className="flex items-center gap-2 text-xs uppercase tracking-[0.3em] text-gray-400">
                        <Camera className="h-3.5 w-3.5 text-green-400" />
                        AI Captured Evidence
                      </div>
                      <div className="mt-3 flex gap-3">
                        <Dialog>
                          <DialogTrigger asChild>
                            <div className="relative h-20 w-28 cursor-pointer overflow-hidden rounded-lg border border-white/10">
                              <img
                                src={log.capturedImage.url || "/placeholder.svg"}
                                alt="Captured evidence"
                                className="h-full w-full object-cover transition hover:opacity-80"
                              />
                              <div className="absolute inset-0 flex items-center justify-center bg-black/40 opacity-0 transition hover:opacity-100">
                                <Eye className="h-4 w-4 text-white" />
                              </div>
                            </div>
                          </DialogTrigger>
                          <DialogContent className="max-w-2xl">
                            <DialogHeader>
                              <DialogTitle>Evidence Image - {log.location}</DialogTitle>
                            </DialogHeader>
                            <div className="space-y-4">
                              <img
                                src={log.capturedImage.url || "/placeholder.svg"}
                                alt="Full evidence image"
                                className="w-full rounded-lg"
                              />
                              <div className="grid grid-cols-2 gap-4 text-sm text-gray-300">
                                <div>
                                  <strong>Capture Time:</strong> {log.capturedImage.timestamp.toLocaleString()}
                                </div>
                                <div className="flex items-center gap-1">
                                  <strong>Suspects:</strong>
                                  {log.capturedImage.aiAnalysis.suspectCount > 0 ? (
                                    <span className="flex items-center gap-1">
                                      {log.capturedImage.aiAnalysis.suspectCount === 1 ? (
                                        <User className="h-3 w-3" />
                                      ) : (
                                        <Users className="h-3 w-3" />
                                      )}
                                      {log.capturedImage.aiAnalysis.suspectCount}
                                    </span>
                                  ) : (
                                    <span className="text-gray-500">None detected</span>
                                  )}
                                </div>
                              </div>
                              <div className="space-y-2 text-sm text-gray-300">
                                <strong>AI Analysis:</strong>
                                <p className="rounded bg-black/40 p-3 text-gray-200">
                                  {log.capturedImage.aiAnalysis.description}
                                </p>
                              </div>
                            </div>
                          </DialogContent>
                        </Dialog>

                        <div className="flex-1 space-y-1 text-xs text-gray-300">
                          <div className="flex items-center gap-4">
                            <span className={`flex items-center gap-1 ${log.capturedImage.aiAnalysis.suspectDetected ? "text-red-400" : "text-gray-500"}`}>
                              {log.capturedImage.aiAnalysis.suspectCount > 0 ? (
                                log.capturedImage.aiAnalysis.suspectCount === 1 ? (
                                  <User className="h-3 w-3" />
                                ) : (
                                  <Users className="h-3 w-3" />
                                )
                              ) : null}
                              {log.capturedImage.aiAnalysis.suspectCount} suspect
                              {log.capturedImage.aiAnalysis.suspectCount !== 1 ? "s" : ""}
                            </span>
                            <Badge
                              variant={log.capturedImage.aiAnalysis.weaponVisible ? "destructive" : "secondary"}
                              className="text-[10px]"
                            >
                              {log.capturedImage.aiAnalysis.weaponVisible ? "Weapon Visible" : "No Weapon"}
                            </Badge>
                          </div>
                          <p className="text-xs text-gray-400 line-clamp-2">
                            {log.capturedImage.aiAnalysis.description}
                          </p>
                        </div>
                      </div>
                    </div>
                  )}

                  {log.officer && <div className="text-xs text-gray-500">Handled by: {log.officer}</div>}
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  )
}
