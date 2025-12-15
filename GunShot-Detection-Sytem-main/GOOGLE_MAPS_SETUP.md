# Google Maps API Setup Guide

The Incident Heat Map feature now uses real Google Maps to display detection locations. Follow these steps to set up your Google Maps API key.

## Step 1: Get a Google Maps API Key

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the following APIs:
   - **Maps JavaScript API**
   - **Places API** (optional, for enhanced location features)

4. Create credentials:
   - Go to **APIs & Services** → **Credentials**
   - Click **Create Credentials** → **API Key**
   - Copy your new API key

## Step 2: Restrict Your API Key (Recommended for Production)

1. Click on your API key to edit it
2. Under **Application restrictions**, select **HTTP referrers**
3. Add your allowed domains:
   ```
   http://localhost:3000/*
   https://yourdomain.com/*
   ```
4. Under **API restrictions**, select **Restrict key**
5. Choose:
   - Maps JavaScript API
   - Places API (if using)

## Step 3: Add API Key to Your Project

### Option A: Create `.env.local` file (Recommended)

Create a file named `.env.local` in the root of your project:

```bash
# Google Maps API Key
NEXT_PUBLIC_GOOGLE_MAPS_API_KEY=AIzaSy...your_actual_key_here

# Other environment variables
MONGODB_URI=your_mongodb_uri
MONGODB_DB=your_db_name
JWT_SECRET=your_jwt_secret
DEMO_ADMIN_USERNAME=officer
DEMO_ADMIN_PASSWORD=dubai2025
```

### Option B: Use existing `.env` file

Add the following line to your existing `.env` file:

```bash
NEXT_PUBLIC_GOOGLE_MAPS_API_KEY=AIzaSy...your_actual_key_here
```

## Step 4: Restart Your Development Server

```bash
npm run dev
```

## Features

The Incident Heat Map now includes:

- ✅ **Real Google Maps** integration with dark theme
- ✅ **Interactive markers** for each detection
- ✅ **Color-coded status** (Confirmed=Green, Dispatched=Blue, Pending=Amber, False Positive=Red)
- ✅ **Info windows** showing detection details (weapon type, confidence, time)
- ✅ **Auto-zoom** to fit all incident markers
- ✅ **Custom dark styling** matching your dashboard theme

## Customizing Map Center

By default, the map centers on **Dubai, UAE** (25.2048°N, 55.2708°E).

To change the default location, edit `components/incident-heat-map.tsx`:

```typescript
const defaultCenter = {
  lat: 25.2048,  // Change to your latitude
  lng: 55.2708,  // Change to your longitude
}
```

## Troubleshooting

### Map shows "For development purposes only" watermark
- This means you need to add billing to your Google Cloud project
- Google Maps requires a billing account, but provides $200/month free credit
- Go to Google Cloud Console → Billing → Add payment method

### Map doesn't load
- Check that your API key is correct in `.env.local`
- Verify that Maps JavaScript API is enabled in Google Cloud Console
- Check browser console for specific error messages
- Ensure your API key has proper restrictions (or remove restrictions for testing)

### Map shows grey tiles
- API key might be invalid or restricted
- Check that the domain is allowed in API key restrictions
- Verify billing is set up on your Google Cloud account

## Pricing

Google Maps Platform pricing:
- First $200/month: **FREE**
- After free tier: ~$7 per 1,000 map loads
- Most small to medium deployments stay within the free tier

For more information: https://mapsplatform.google.com/pricing/

## Security Note

⚠️ **Never commit your `.env.local` or `.env` file to version control!**

These files are already in `.gitignore`, but always double-check before pushing code.

