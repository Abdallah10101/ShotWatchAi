# GPS Location Tracking - Quick Start

Your gunshot detection system now captures **real GPS coordinates** for each detection and displays them on Google Maps in real-time!

## 🚀 How to Enable

### Step 1: Grant Browser Permission
When the first detection occurs, your browser will ask:
```
"Allow this site to access your location?"
```
Click **"Allow"** to enable GPS tracking.

### Step 2: Verify It's Working
1. Open **Live Surveillance**
2. Look for **"GPS Active"** badge at the top (green badge with navigation icon)
3. Click **"Simulate Detection"** button
4. Go to **Dashboard** → scroll to **"Incident Heat Map"**
5. You should see a marker on Google Maps at your actual location!

## 📍 What Gets Tracked

Every detection now saves:
- **Latitude** (e.g., 25.2048)
- **Longitude** (e.g., 55.2708)
- **Accuracy** (GPS precision in meters)
- **Timestamp** (when detection occurred)

## 🗺️ Viewing on Map

**Dashboard → Incident Heat Map:**
- Red markers = False Positives
- Amber markers = Pending Review
- Blue markers = Dispatched
- Green markers = Confirmed

Click any marker to see:
- Weapon type
- Confidence percentage
- Detection time

## 🔧 Customization

### Change Fallback Location
If GPS fails, edit `lib/geolocation.ts`:

```typescript
export function getFallbackLocation(): GeoLocation {
  return {
    lat: 25.2048,  // ← Your default latitude
    lng: 55.2708,  // ← Your default longitude
    accuracy: 100,
    timestamp: Date.now(),
  }
}
```

### Test with Fake GPS
**Chrome DevTools:**
1. Press `F12` → More Tools → Sensors
2. Select "Location" or enter custom coordinates
3. Test detections with different locations

## ✅ Requirements

- ✅ Browser with GPS/Location support (all modern browsers)
- ✅ HTTPS in production (or localhost for dev)
- ✅ Google Maps API key configured
- ✅ User permission granted

## 🔍 Troubleshooting

**Problem:** All markers show at same location
**Solution:** Grant location permission in browser

**Problem:** "GPS Active" not showing
**Solution:** Reload the page after granting permission

**Problem:** Location is inaccurate
**Solution:** Use mobile device (better GPS) or move to area with good signal

**Problem:** Location permission keeps asking
**Solution:** In browser settings, set permission to "Always Allow"

## 📱 Mobile vs Desktop

**Mobile (Recommended):**
- Uses actual GPS satellite
- Very accurate (5-20 meters)
- Battery usage increased slightly

**Desktop:**
- Uses Wi-Fi positioning
- Less accurate (50-500 meters)
- No GPS hardware in most desktops

## 🔐 Privacy

- GPS data stays on YOUR server only
- Stored in `temp/detections.json` locally
- No external tracking services
- Users control permission at all times

## 📊 Data Format

Check `temp/detections.json`:
```json
{
  "timestamp": 1764516665.578,
  "label": "AK-12",
  "confidence": 0.884,
  "lat": 25.2048,
  "lng": 55.2708,
  "accuracy": 15.5
}
```

## 🎯 What's Next?

1. Test a real detection (speak/play gunshot sound)
2. Watch the map update in real-time
3. Click markers to see details
4. Review all detections in Incident Logs

---

**That's it!** Your system now tracks and maps every detection with GPS precision. 🎯

For detailed documentation, see `LOCATION_TRACKING.md`.

