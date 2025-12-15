// Geolocation service for getting device location

export interface GeoLocation {
  lat: number
  lng: number
  accuracy?: number
  timestamp: number
}

/**
 * Get the current device location using the browser Geolocation API
 */
export async function getCurrentLocation(): Promise<GeoLocation | null> {
  if (!navigator.geolocation) {
    console.warn("[Geolocation] Browser doesn't support geolocation")
    return null
  }

  return new Promise((resolve) => {
    navigator.geolocation.getCurrentPosition(
      (position) => {
        resolve({
          lat: position.coords.latitude,
          lng: position.coords.longitude,
          accuracy: position.coords.accuracy,
          timestamp: position.timestamp,
        })
      },
      (error) => {
        console.error("[Geolocation] Error getting location:", error)
        // Return null on error instead of rejecting
        resolve(null)
      },
      {
        enableHighAccuracy: true,
        timeout: 5000,
        maximumAge: 0,
      }
    )
  })
}

/**
 * Get a fallback location (Dubai, UAE by default)
 * You can customize this based on your deployment location
 */
export function getFallbackLocation(): GeoLocation {
  // Dubai, UAE coordinates with small random offset for testing
  return {
    lat: 25.2048 + (Math.random() - 0.5) * 0.05,
    lng: 55.2708 + (Math.random() - 0.5) * 0.05,
    accuracy: 100,
    timestamp: Date.now(),
  }
}

/**
 * Get location with fallback to default location if GPS fails
 */
export async function getLocationWithFallback(): Promise<GeoLocation> {
  const location = await getCurrentLocation()
  
  if (location) {
    console.log("[Geolocation] GPS location obtained:", location)
    return location
  }
  
  console.log("[Geolocation] Using fallback location")
  return getFallbackLocation()
}

