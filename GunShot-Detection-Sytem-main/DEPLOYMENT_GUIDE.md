# Backend Deployment Guide

This guide explains how to deploy your gunshot detection backend so it can be accessed from your website frontend.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the backend:**
   ```bash
   python start_server.py
   ```

3. **Update your frontend to use the backend URL:**
   - Flask API: `http://YOUR_SERVER_IP:5000`
   - WebSocket: `ws://YOUR_SERVER_IP:8765`

## Configuration

### Environment Variables

You can configure the backend using environment variables:

```bash
# Flask server configuration
export FLASK_HOST=0.0.0.0        # 0.0.0.0 allows external connections
export FLASK_PORT=5000

# WebSocket server configuration
export WS_HOST=0.0.0.0            # 0.0.0.0 allows external connections
export WS_PORT=8765
```

### Windows (PowerShell)
```powershell
$env:FLASK_HOST="0.0.0.0"
$env:FLASK_PORT="5000"
$env:WS_HOST="0.0.0.0"
$env:WS_PORT="8765"
python start_server.py
```

### Linux/Mac
```bash
export FLASK_HOST=0.0.0.0
export FLASK_PORT=5000
export WS_HOST=0.0.0.0
export WS_PORT=8765
python start_server.py
```

## Deployment Options

### Option 1: VPS/Cloud Server (Recommended)

**Requirements:**
- Ubuntu/Debian Linux server
- Python 3.8+
- Camera access (if using physical camera)
- Public IP address

**Steps:**

1. **SSH into your server:**
   ```bash
   ssh user@your-server-ip
   ```

2. **Install dependencies:**
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-venv ffmpeg
   ```

3. **Upload your backend files:**
   ```bash
   scp -r predict_audio_only.py websocket_server.py camera_stream_server.py start_server.py requirements.txt guntype_resnet50.pth user@your-server:/path/to/backend/
   ```

4. **Set up virtual environment:**
   ```bash
   cd /path/to/backend
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

5. **Run with screen/tmux (keeps running after SSH disconnect):**
   ```bash
   screen -S gunshot-backend
   export FLASK_HOST=0.0.0.0
   export FLASK_PORT=5000
   export WS_HOST=0.0.0.0
   export WS_PORT=8765
   python start_server.py
   # Press Ctrl+A then D to detach
   ```

6. **Or use systemd service (auto-start on boot):**
   Create `/etc/systemd/system/gunshot-backend.service`:
   ```ini
   [Unit]
   Description=Gunshot Detection Backend
   After=network.target

   [Service]
   Type=simple
   User=your-username
   WorkingDirectory=/path/to/backend
   Environment="FLASK_HOST=0.0.0.0"
   Environment="FLASK_PORT=5000"
   Environment="WS_HOST=0.0.0.0"
   Environment="WS_PORT=8765"
   ExecStart=/path/to/backend/venv/bin/python start_server.py
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

   Then:
   ```bash
   sudo systemctl enable gunshot-backend
   sudo systemctl start gunshot-backend
   sudo systemctl status gunshot-backend
   ```

7. **Configure firewall:**
   ```bash
   sudo ufw allow 5000/tcp  # Flask HTTP
   sudo ufw allow 8765/tcp  # WebSocket
   ```

### Option 2: Local Machine with Port Forwarding

If you want to run the backend on your local machine:

1. **Use ngrok or similar tunnel:**
   ```bash
   # Install ngrok: https://ngrok.com/
   ngrok http 5000  # For Flask
   ngrok tcp 8765   # For WebSocket
   ```

2. **Update frontend to use ngrok URLs**

### Option 3: Docker (Advanced)

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000 8765

CMD ["python", "start_server.py"]
```

Build and run:
```bash
docker build -t gunshot-backend .
docker run -d -p 5000:5000 -p 8765:8765 \
  -e FLASK_HOST=0.0.0.0 \
  -e FLASK_PORT=5000 \
  -e WS_HOST=0.0.0.0 \
  -e WS_PORT=8765 \
  --device=/dev/video0 \
  gunshot-backend
```

## Frontend Configuration

Update your frontend code to use your backend URL:

### For Camera Stream (MJPEG):
```typescript
// Replace localhost with your server IP/domain
<img
  src="http://YOUR_SERVER_IP:5000/camera-stream"
  alt="Live camera feed"
/>
```

### For WebSocket Connection:
```typescript
// Replace localhost with your server IP/domain
const ws = new WebSocket('ws://YOUR_SERVER_IP:8765');
```

### Example with Environment Variables:
```typescript
// .env file
NEXT_PUBLIC_BACKEND_URL=http://your-server-ip:5000
NEXT_PUBLIC_WS_URL=ws://your-server-ip:8765

// In your component
const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:5000';
const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8765';
```

## Security Considerations

### For Production:

1. **Restrict CORS origins:**
   Edit `camera_stream_server.py`:
   ```python
   CORS(app, resources={
       r"/*": {
           "origins": ["https://your-website.com"],  # Only allow your domain
           "methods": ["GET", "POST", "OPTIONS"],
           "allow_headers": ["Content-Type", "Authorization"]
       }
   })
   ```

2. **Use HTTPS/WSS:**
   - Set up reverse proxy (nginx) with SSL
   - Use `wss://` for WebSocket connections
   - Use `https://` for HTTP endpoints

3. **Add authentication:**
   - Implement API keys or JWT tokens
   - Add rate limiting

4. **Firewall rules:**
   - Only allow connections from your frontend IP
   - Use VPN for additional security

## Testing

1. **Test Flask endpoint:**
   ```bash
   curl http://YOUR_SERVER_IP:5000/health
   ```

2. **Test WebSocket:**
   Use a WebSocket client or browser console:
   ```javascript
   const ws = new WebSocket('ws://YOUR_SERVER_IP:8765');
   ws.onopen = () => console.log('Connected');
   ws.onmessage = (msg) => console.log('Received:', msg.data);
   ```

3. **Test from your frontend:**
   - Open browser console
   - Check for CORS errors
   - Verify connections are established

## Troubleshooting

### CORS Errors
- Make sure `flask-cors` is installed: `pip install flask-cors`
- Check that CORS is enabled in `camera_stream_server.py`
- Verify your frontend domain is allowed

### Connection Refused
- Check firewall settings
- Verify server is listening on `0.0.0.0` not `localhost`
- Check if ports are open: `netstat -tulpn | grep -E '5000|8765'`

### Camera Not Working
- If using physical camera, ensure it's accessible on the server
- Check camera permissions
- Verify camera index is correct

## API Endpoints

- `GET /camera-stream` - MJPEG video stream
- `GET /health` - Health check endpoint
- `WS /` - WebSocket server for audio processing

## Support

For issues, check:
- Server logs: `python start_server.py` output
- Flask logs: Check console output
- WebSocket logs: Check console output

