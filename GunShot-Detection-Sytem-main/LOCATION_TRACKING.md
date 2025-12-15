# Real-Time GPS Location Tracking

The Incident Heat Map now tracks and displays the **real GPS location** of each gunshot detection in real-time.

## How It Works

### 1. **Location Capture**
When a gunshot is detected:
- The browser requests the device's GPS coordinates using the **Geolocation API**
- Location is captured with high accuracy
- If GPS is unavailable/denied, a fallback location is used (Dubai, UAE by default)

### 2. **Data Flow**
```
Detection Occurs
    ↓
Browser captures GPS (lat, lng, accuracy)
    ↓
Sends to backend with frame
    ↓
Python saves to detections.json
    ↓
Frontend reads and displays on map
```

### 3. **Storage Format**
Each detection in `temp/detections.json` now includes:
```json
{
  "timestamp": 1764516665.578,
  "label": "AK-12",
  "confidence": 0.884,
  "quality": 1.0,
  "filename": "30th November 7-31-05 pm gunshot fired - AK-12.jpg",
  "path": "C:\\...\\temp\\30th November 7-31-05 pm gunshot fired - AK-12.jpg",
  "detection_id": 1428,
  "source": "websocket",
  "lat": 25.2048,
  "lng": 55.2708,
  "accuracy": 15.5
}
```

## Features

### ✅ Real-Time Updates
- Map updates immediately when new detections occur
- No page refresh needed
- Markers appear on map at exact GPS coordinates

### ✅ High Accuracy
- Uses device GPS for precise location
- Accuracy metadata saved (in meters)
- Shows "GPS Active" indicator in Live Surveillance

### ✅ Fallback Handling
- If GPS permission denied → uses fallback location
- If GPS unavailable → uses fallback location
- Fallback: Dubai, UAE center with small random offset for testing

### ✅ Interactive Map
- Click markers to see detection details
- Color-coded by status (Confirmed, Dispatched, Pending, False Positive)
- Shows weapon type, confidence, timestamp

## Customizing Fallback Location

Edit `lib/geolocation.ts`:

```typescript
export function getFallbackLocation(): GeoLocation {
  // Change these coordinates to your deployment location
  return {
    lat: 25.2048,  // Your latitude
    lng: 55.2708,  // Your longitude
    accuracy: 100,
    timestamp: Date.now(),
  }
}
```

## Browser Permissions

### First Use
When a detection happens, the browser will prompt:
```
"yoursite.com wants to:
Know your location"
[Block] [Allow]
```

**Click "Allow"** to enable real GPS tracking.

### Permission Management

**Chrome/Edge:**
1. Click the lock icon in address bar
2. Site Settings → Location → Allow

**Firefox:**
1. Click the shield icon
2. Permissions → Location → Allow

**Safari:**
1. Safari → Preferences → Websites
2. Location → Allow for your site

## Testing GPS Location

### On Desktop (Development)
Most browsers allow location on `localhost`:
- Chrome/Edge: Simulates a location or uses Wi-Fi positioning
- Firefox: Allows manual location selection
- You can override location in DevTools:
  - Open DevTools (F12)
  - More Tools → Sensors
  - Set custom location

### On Mobile (Production)
- Use actual device GPS
- Must be served over HTTPS (required for Geolocation API)
- More accurate than desktop

### Simulate Different Locations
In DevTools → Sensors:
```
San Francisco: 37.7749, -122.4194
New York: 40.7128, -74.0060
London: 51.5074, -0.1278
Dubai: 25.2048, 55.2708
```

## Privacy & Security

### Data Storage
- GPS coordinates are stored in `temp/detections.json` (local file)
- No external tracking services
- Data stays on your server

### User Control
- Browser always asks for permission
- Users can deny/revoke permission anytime
- Falls back gracefully if denied

### HTTPS Required
- Geolocation API requires HTTPS in production
- Works on `localhost` for development
- Self-signed certificates OK for testing

## Troubleshooting

### "GPS Active" shows but location not captured
- Check browser console for errors
- Verify GPS permission is allowed
- Try reloading the page

### All markers show at same location
- GPS permission may be denied
- System is using fallback location
- Grant location permission and test again

### Location is inaccurate
- Check the `accuracy` value in detection data
- Mobile devices have better GPS than desktops
- Wi-Fi positioning is less accurate than GPS
- Try moving to area with better GPS signal

### HTTPS error in production
```
getUserPosition() and watchPosition() no longer work on insecure origins
```
**Solution:** Deploy with HTTPS certificate or use ngrok/cloudflare tunnel

## API Reference

### `getLocationWithFallback()`
Returns GPS location with automatic fallback:
```typescript
const location = await getLocationWithFallback()
// { lat: 25.2048, lng: 55.2708, accuracy: 15, timestamp: 1764516665 }
```

### `getCurrentLocation()`
Gets raw GPS location (returns null on error):
```typescript
const location = await getCurrentLocation()
if (location) {
  console.log(`Location: ${location.lat}, ${location.lng}`)
}
```

### `getFallbackLocation()`
Returns default fallback location:
```typescript
const fallback = getFallbackLocation()
```

## Integration with Google Maps

The GPS coordinates automatically display on the Google Maps heat map:
1. Detection occurs → GPS captured
2. Saved to detections.json
3. Dashboard fetches detections
4. Map renders markers at GPS coordinates
5. Real-time updates as new detections occur

See `GOOGLE_MAPS_SETUP.md` for map configuration.

## Future Enhancements

Potential improvements:
- [ ] Reverse geocoding (show address instead of coordinates)
- [ ] Geofencing alerts (notify when detection in specific area)
- [ ] Location history tracking
- [ ] Multiple camera locations
- [ ] Clustering for many nearby detections
- [ ] Export KML/GPX for external mapping tools

## Support

For questions or issues:
1. Check browser console for errors
2. Verify GPS permissions are granted
3. Test with manual location in DevTools
4. Check `temp/detections.json` for lat/lng fields

