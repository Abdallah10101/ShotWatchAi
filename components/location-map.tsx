"use client"

import { Card, CardContent } from "@/components/ui/card"
import { MapPin, Navigation } from "lucide-react"

export function LocationMap() {
  // In a real implementation, this would integrate with Google Maps API
  const mockLocations = [
    { id: "1", lat: 25.2048, lng: 55.2708, address: "Downtown Dubai", type: "Handgun", time: "5 min ago" },
    { id: "2", lat: 25.0657, lng: 55.1413, address: "Dubai Marina", type: "Rifle", time: "15 min ago" },
  ]

  return (
    <Card className="h-full border border-[#171b27] bg-[#05080f] text-white">
      <CardContent className="space-y-4 p-5">
        <div className="flex items-center justify-between text-xs uppercase tracking-[0.4em] text-gray-500">
          <span className="flex items-center gap-2">
            <Navigation className="h-4 w-4 text-red-300" />
            Location overview
          </span>
          <span className="rounded-full border border-white/10 px-2 py-0.5 text-[10px] tracking-[0.3em] text-gray-400">Dubai grid</span>
        </div>

        <div className="relative h-64 overflow-hidden rounded-2xl border border-white/5 bg-gradient-to-br from-[#050b12] via-[#090c14] to-[#020307]">
          <div
            className="absolute inset-0 opacity-60"
            style={{
              backgroundImage:
                "linear-gradient(90deg, rgba(255,255,255,0.06) 1px, transparent 1px), linear-gradient(0deg, rgba(255,255,255,0.06) 1px, transparent 1px)",
              backgroundSize: "80px 80px",
            }}
          />
          <div className="absolute inset-0 bg-gradient-to-b from-transparent via-transparent to-black/60" />
          <div className="absolute top-5 left-5 rounded-xl border border-white/10 bg-black/40 px-4 py-2 text-xs">
            <p className="text-gray-400">Operational zone</p>
            <p className="font-semibold text-white">Dubai, UAE</p>
          </div>
          {mockLocations.map((location, index) => (
            <div
              key={location.id}
              className="absolute -translate-x-1/2 -translate-y-1/2"
              style={{
                left: `${30 + index * 35}%`,
                top: `${35 + index * 25}%`,
              }}
            >
              <div className="relative flex flex-col items-center gap-1 text-xs">
                <div className="rounded-full border border-red-500/40 bg-red-500/10 px-3 py-1 text-red-200 shadow-lg shadow-red-500/30">
                  {location.type}
                </div>
                <div className="relative">
                  <div className="absolute inset-0 animate-ping rounded-full bg-red-500/30" />
                  <MapPin className="relative h-6 w-6 text-red-500" />
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="space-y-3">
          <p className="text-xs uppercase tracking-[0.3em] text-gray-500">Recent detections</p>
          {mockLocations.map((location) => (
            <div key={location.id} className="flex items-center justify-between rounded-xl border border-white/5 bg-black/30 px-3 py-2 text-sm">
              <div className="flex items-center gap-3">
                <div className="h-2 w-2 rounded-full bg-red-500 animate-pulse" />
                <div>
                  <p className="font-semibold text-white">{location.address}</p>
                  <p className="text-xs text-gray-500">{location.type}</p>
                </div>
              </div>
              <div className="text-xs text-gray-500">{location.time}</div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
