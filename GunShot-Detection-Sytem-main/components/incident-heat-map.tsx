"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { GoogleMap, useJsApiLoader, Marker, InfoWindow } from "@react-google-maps/api"
import { MapPin } from "lucide-react"

type IncidentMarker = {
  id: string
  lat: number
  lng: number
  label: string
  status: "confirmed" | "dispatched" | "pending" | "false-positive"
  timestamp: string
  confidence: number
}

interface IncidentHeatMapProps {
  incidents: IncidentMarker[]
}

const containerStyle = {
  width: "100%",
  height: "100%",
  borderRadius: "24px",
}

// Default center: Dubai, UAE
const defaultCenter = {
  lat: 25.2048,
  lng: 55.2708,
}

const statusColors = {
  confirmed: "#10b981", // emerald
  dispatched: "#3b82f6", // blue
  pending: "#f59e0b", // amber
  "false-positive": "#ef4444", // red
}

const mapStyles = [
  {
    elementType: "geometry",
    stylers: [{ color: "#0a1428" }],
  },
  {
    elementType: "labels.text.stroke",
    stylers: [{ color: "#0a1428" }],
  },
  {
    elementType: "labels.text.fill",
    stylers: [{ color: "#64748b" }],
  },
  {
    featureType: "administrative.locality",
    elementType: "labels.text.fill",
    stylers: [{ color: "#cbd5e1" }],
  },
  {
    featureType: "poi",
    elementType: "labels.text.fill",
    stylers: [{ color: "#64748b" }],
  },
  {
    featureType: "poi.park",
    elementType: "geometry",
    stylers: [{ color: "#0f2a1f" }],
  },
  {
    featureType: "poi.park",
    elementType: "labels.text.fill",
    stylers: [{ color: "#6b8e7d" }],
  },
  {
    featureType: "road",
    elementType: "geometry",
    stylers: [{ color: "#1e293b" }],
  },
  {
    featureType: "road",
    elementType: "geometry.stroke",
    stylers: [{ color: "#0f172a" }],
  },
  {
    featureType: "road",
    elementType: "labels.text.fill",
    stylers: [{ color: "#94a3b8" }],
  },
  {
    featureType: "road.highway",
    elementType: "geometry",
    stylers: [{ color: "#334155" }],
  },
  {
    featureType: "road.highway",
    elementType: "geometry.stroke",
    stylers: [{ color: "#1e293b" }],
  },
  {
    featureType: "road.highway",
    elementType: "labels.text.fill",
    stylers: [{ color: "#cbd5e1" }],
  },
  {
    featureType: "transit",
    elementType: "geometry",
    stylers: [{ color: "#1e293b" }],
  },
  {
    featureType: "transit.station",
    elementType: "labels.text.fill",
    stylers: [{ color: "#64748b" }],
  },
  {
    featureType: "water",
    elementType: "geometry",
    stylers: [{ color: "#0c1e35" }],
  },
  {
    featureType: "water",
    elementType: "labels.text.fill",
    stylers: [{ color: "#475569" }],
  },
  {
    featureType: "water",
    elementType: "labels.text.stroke",
    stylers: [{ color: "#0a1428" }],
  },
]

export function IncidentHeatMap({ incidents }: IncidentHeatMapProps) {
  const { isLoaded, loadError } = useJsApiLoader({
    id: "google-map-script",
    googleMapsApiKey: process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY || "",
  })

  const [map, setMap] = useState<google.maps.Map | null>(null)
  const [selectedMarker, setSelectedMarker] = useState<IncidentMarker | null>(null)

  const onLoad = useCallback((mapInstance: google.maps.Map) => {
    setMap(mapInstance)
  }, [])

  const onUnmount = useCallback(() => {
    setMap(null)
  }, [])

  useEffect(() => {
    if (map && incidents.length > 0) {
      const bounds = new window.google.maps.LatLngBounds()
      incidents.forEach((incident) => {
        bounds.extend({ lat: incident.lat, lng: incident.lng })
      })
      map.fitBounds(bounds)
    }
  }, [map, incidents])

  if (loadError) {
    return (
      <div className="flex h-full items-center justify-center rounded-3xl border border-white/10 bg-gradient-to-b from-[#050b12] via-[#04070e] to-[#010205] text-sm text-gray-400">
        Error loading Google Maps. Please check your API key.
      </div>
    )
  }

  if (!isLoaded) {
    return (
      <div className="flex h-full items-center justify-center rounded-3xl border border-white/10 bg-gradient-to-b from-[#050b12] via-[#04070e] to-[#010205] text-sm text-gray-400">
        Loading map...
      </div>
    )
  }

  return (
    <div className="relative h-full overflow-hidden rounded-3xl border border-white/10">
      <GoogleMap
        mapContainerStyle={containerStyle}
        center={defaultCenter}
        zoom={11}
        onLoad={onLoad}
        onUnmount={onUnmount}
        options={{
          styles: mapStyles,
          disableDefaultUI: true,
          zoomControl: true,
          mapTypeControl: false,
          scaleControl: false,
          streetViewControl: false,
          rotateControl: false,
          fullscreenControl: true,
        }}
      >
        {incidents.map((incident) => (
          <Marker
            key={incident.id}
            position={{ lat: incident.lat, lng: incident.lng }}
            onClick={() => setSelectedMarker(incident)}
            icon={{
              path: window.google.maps.SymbolPath.CIRCLE,
              fillColor: statusColors[incident.status],
              fillOpacity: 0.9,
              strokeColor: "#ffffff",
              strokeWeight: 2,
              scale: 10,
            }}
          />
        ))}

        {selectedMarker && (
          <InfoWindow
            position={{ lat: selectedMarker.lat, lng: selectedMarker.lng }}
            onCloseClick={() => setSelectedMarker(null)}
          >
            <div className="rounded-lg bg-[#0a1428] p-3 text-white">
              <div className="mb-2 flex items-center gap-2">
                <MapPin className="h-4 w-4 text-[#f5c75e]" />
                <p className="text-sm font-semibold">{selectedMarker.label}</p>
              </div>
              <div className="space-y-1 text-xs text-gray-300">
                <p>
                  <span className="text-gray-400">Status:</span>{" "}
                  <span
                    className="font-semibold"
                    style={{ color: statusColors[selectedMarker.status] }}
                  >
                    {selectedMarker.status.toUpperCase()}
                  </span>
                </p>
                <p>
                  <span className="text-gray-400">Confidence:</span>{" "}
                  {Math.round(selectedMarker.confidence * 100)}%
                </p>
                <p>
                  <span className="text-gray-400">Time:</span> {selectedMarker.timestamp}
                </p>
              </div>
            </div>
          </InfoWindow>
        )}
      </GoogleMap>
    </div>
  )
}

