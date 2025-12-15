"use client"

import { useState, useEffect, useRef, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Activity, Camera, Clock3, Navigation, Video, VideoOff, Zap } from "lucide-react"
import { useWebSocket } from "@/lib/useWebSocket"
import { getLocationWithFallback, type GeoLocation } from "@/lib/geolocation"

// Audio processing parameters
const SAMPLE_RATE = 22050;
const CHUNK_DURATION = 2; // seconds
const CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION;
const MIN_AUDIO_RMS = 0.015; // align with backend AUDIO_MIN_RMS

// Convert Float32 to Int16 for Web Audio API
const floatTo16BitPCM = (input: Float32Array): Int16Array => {
  const output = new Int16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i]));
    output[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }
  return output;
};

const formatLogTime = (date: Date) =>
  date.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

interface LiveFeedProps {
  onViewLogs?: () => void
  onViewAnalytics?: () => void
}

type DetectionSnapshot = {
  label: string
  confidence: number
  timestamp: Date
}

export function LiveFeed({ onViewLogs, onViewAnalytics }: LiveFeedProps) {
  const [isConnected, setIsConnected] = useState(false)
  const [isCapturing, setIsCapturing] = useState(false)
  const [lastCapture, setLastCapture] = useState<Date | null>(null)
  const [detectionInfo, setDetectionInfo] = useState<DetectionSnapshot | null>(null)
  const [lastSavedInfo, setLastSavedInfo] = useState<DetectionSnapshot | null>(null)
  const [initialDetection, setInitialDetection] = useState<DetectionSnapshot | null>(null)
  const [isSimulating, setIsSimulating] = useState(false)
  const [eventLog, setEventLog] = useState<{ time: string; message: string }[]>(() => {
    const now = new Date();
    return [
      { time: formatLogTime(new Date(now.getTime() - 2 * 60 * 1000)), message: "System initialized" },
      { time: formatLogTime(new Date(now.getTime() - 60 * 1000)), message: "Camera handshake complete" },
      { time: formatLogTime(now), message: "Awaiting alerts" },
    ];
  })
  const [isVideoReady, setIsVideoReady] = useState(false)
  const [clock, setClock] = useState(() => new Date())
  
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const processorRef = useRef<ScriptProcessorNode | null>(null)
  const audioBufferRef = useRef<Float32Array[]>([])
  const isProcessingRef = useRef(false)
  const sendMessageRef = useRef<(message: any) => boolean>(() => false)

  const cameras = [
    { id: "CAM-001", name: "Downtown Dubai - Main St", status: "online" },
    { id: "CAM-002", name: "Dubai Marina - Walk", status: "online" },
    { id: "CAM-003", name: "Business Bay - Tower", status: "offline" },
  ]

  useEffect(() => {
    const timer = setInterval(() => setClock(new Date()), 1000)
    return () => clearInterval(timer)
  }, [])

  useEffect(() => {
    const loadLatestDetection = async () => {
      try {
        const res = await fetch("/api/detections", { cache: "no-store" })
        if (!res.ok) return
        const payload = await res.json()
        const latest = Array.isArray(payload?.detections) ? payload.detections[0] : null
        if (latest) {
          const ts = typeof latest.timestamp === "number" ? latest.timestamp * 1000 : Date.now()
          setInitialDetection({
            label: latest.label ?? "Gunshot",
            confidence: latest.confidence ?? 0,
            timestamp: new Date(ts),
          })
        }
      } catch (error) {
        console.error("Failed to load recent detection", error)
      }
    }
    void loadLatestDetection()
  }, [])

  // Capture and send frame to backend API endpoint
  const captureAndSendFrame = useCallback(
    async (detection: { label: string; confidence: number; timestamp: number }) => {
      const video = videoRef.current;
      if (!video) return;

      const hasFrameData = video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA;
      const width = video.videoWidth || video.clientWidth;
      const height = video.videoHeight || video.clientHeight;

      if (!hasFrameData || !width || !height) {
        console.warn("Skipping frame capture - video stream not ready yet");
        return;
      }

      // Get GPS location
      const location = await getLocationWithFallback();
      console.log("[LOCATION] Captured location for detection:", location);

      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d");

      if (!ctx) return;

      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      // Convert canvas to blob and send via POST
      canvas.toBlob(async (blob) => {
        if (!blob) {
          console.error("Failed to create blob from canvas");
          return;
        }

        const formData = new FormData();
        formData.append("frame", blob, "image/jpeg");
        formData.append("label", detection.label);
        formData.append("confidence", detection.confidence.toString());
        formData.append("timestamp", detection.timestamp.toString());
        formData.append("lat", location.lat.toString());
        formData.append("lng", location.lng.toString());
        if (location.accuracy) {
          formData.append("accuracy", location.accuracy.toString());
        }

        try {
          const response = await fetch("/api/frame-capture", {
            method: "POST",
            body: formData,
          });

          if (response.ok) {
            const result = await response.json();
            console.log("[FRAME] Frame sent successfully:", result);
            setIsCapturing(true);
            setLastCapture(new Date(detection.timestamp * 1000));
            setTimeout(() => setIsCapturing(false), 2000);
          } else {
            console.error("[FRAME] Failed to send frame:", await response.text());
          }
        } catch (error) {
          console.error("[FRAME] Error sending frame:", error);
        }
      }, "image/jpeg", 0.9);
    },
    []
  );

  // Initialize WebSocket connection
  const handleWebSocketMessage = useCallback((message: any) => {
    if (message.type === "request_frame" && videoRef.current && message.detection) {
      if (!isVideoReady) {
        console.warn("Received frame request before video was ready - waiting for next request");
        return;
      }
      const tsSeconds = message.detection.timestamp ?? Math.floor(Date.now() / 1000);
      setDetectionInfo({
        label: message.detection.label,
        confidence: message.detection.confidence,
        timestamp: new Date(tsSeconds * 1000),
      });
      setIsSimulating(false)
      captureAndSendFrame({
        label: message.detection.label,
        confidence: message.detection.confidence,
        timestamp: tsSeconds,
      });
    } else if (message.type === "image_saved" && message.detection) {
      const ts = message.detection.timestamp ? new Date(message.detection.timestamp * 1000) : new Date();
      setLastSavedInfo({
        label: message.detection.label,
        confidence: message.detection.confidence,
        timestamp: ts,
      });
      setIsSimulating(false)
    }
  }, [captureAndSendFrame, isVideoReady]);

  const { sendMessage } = useWebSocket('ws://localhost:8765', handleWebSocketMessage);

  useEffect(() => {
    sendMessageRef.current = sendMessage;
  }, [sendMessage]);

  useEffect(() => {
    if (!detectionInfo) return;
    setEventLog((prev) => [
      {
        time: formatLogTime(detectionInfo.timestamp),
        message: `${detectionInfo.label} detected (${Math.round(detectionInfo.confidence * 100)}%)`,
      },
      ...prev,
    ].slice(0, 6));
  }, [detectionInfo]);

  useEffect(() => {
    if (!lastSavedInfo) return;
    setEventLog((prev) => [
      {
        time: formatLogTime(lastSavedInfo.timestamp),
        message: `Evidence stored (${lastSavedInfo.label})`,
      },
      ...prev,
    ].slice(0, 6));
  }, [lastSavedInfo]);


  // Initialize audio processing
  useEffect(() => {
    const initAudio = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({
          sampleRate: SAMPLE_RATE
        });
        
        const source = audioContext.createMediaStreamSource(stream);
        const processor = audioContext.createScriptProcessor(4096, 1, 1);
        
        processor.onaudioprocess = (e) => {
          if (!isProcessingRef.current) {
            isProcessingRef.current = true;
            const inputData = e.inputBuffer.getChannelData(0);
            audioBufferRef.current.push(new Float32Array(inputData));
            
            // Process chunks of 2 seconds
            const totalSamples = audioBufferRef.current.reduce((acc, chunk) => acc + chunk.length, 0);
            
            if (totalSamples >= CHUNK_SIZE) {
              const chunk = new Float32Array(CHUNK_SIZE);
              let offset = 0;
              
              while (offset < CHUNK_SIZE) {
                const buffer = audioBufferRef.current[0];
                const remaining = CHUNK_SIZE - offset;
                
                if (buffer.length <= remaining) {
                  chunk.set(buffer, offset);
                  offset += buffer.length;
                  audioBufferRef.current.shift();
                } else {
                  chunk.set(buffer.subarray(0, remaining), offset);
                  audioBufferRef.current[0] = buffer.subarray(remaining);
                  offset = CHUNK_SIZE;
                }
              }
              
              // Compute RMS to ignore silence
              const rms = Math.sqrt(chunk.reduce((acc, sample) => acc + sample * sample, 0) / chunk.length);
              if (rms < MIN_AUDIO_RMS) {
                isProcessingRef.current = false;
                return;
              }

              // Send audio chunk to server
              const int16Data = floatTo16BitPCM(chunk);
              const base64Data = btoa(
                String.fromCharCode.apply(null, Array.from(new Uint8Array(int16Data.buffer)))
              );
              
              sendMessageRef.current({
                type: 'audio',
                data: base64Data,
                sampleRate: SAMPLE_RATE
              });
            }
            
            isProcessingRef.current = false;
          }
        };
        
        source.connect(processor);
        processor.connect(audioContext.destination);
        
        audioContextRef.current = audioContext;
        processorRef.current = processor;
        
        // Initialize video stream
        const videoStream = await navigator.mediaDevices.getUserMedia({ video: true });
        streamRef.current = videoStream;
        setIsConnected(true);
        if (videoRef.current) {
          const videoEl = videoRef.current;
          videoEl.srcObject = videoStream;

          const handleVideoReady = () => {
            setIsVideoReady(true);
            if (videoEl.paused) {
              videoEl.play().catch(() => {});
            }
          };

          videoEl.addEventListener("loadeddata", handleVideoReady);
          videoEl.addEventListener("loadedmetadata", handleVideoReady);
          videoEl.addEventListener("canplay", handleVideoReady);

          if (videoEl.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
            handleVideoReady();
          }

          const removeListeners = () => {
            videoEl.removeEventListener("loadeddata", handleVideoReady);
            videoEl.removeEventListener("loadedmetadata", handleVideoReady);
            videoEl.removeEventListener("canplay", handleVideoReady);
          };

          processorRef.current && (processorRef.current as any)._cleanupVideoListeners?.();
          (processorRef.current as any) = {
            ...(processorRef.current as any),
            _cleanupVideoListeners: removeListeners,
          };
        }
        
      } catch (err) {
        console.error('Error initializing audio/video:', err);
      }
    };
    
    initAudio();
    
    return () => {
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
      if (processorRef.current) {
        processorRef.current.disconnect();
        (processorRef.current as any)?._cleanupVideoListeners?.();
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, [sendMessage]);

  const primaryCamera = cameras[0]
  const currentEvent = detectionInfo ?? lastSavedInfo ?? initialDetection
  const detailTimestamp = currentEvent?.timestamp ?? null
  const formattedClock = clock.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })
  const currentConfidencePercent = currentEvent ? Math.round(currentEvent.confidence * 100) : null
  const detectionConfidenceText = currentConfidencePercent !== null ? `${currentConfidencePercent}%` : "--"
  const detectionStatusLabel = isSimulating
    ? "Simulating..."
    : detectionInfo
      ? "Pending review"
      : lastSavedInfo
        ? "Saved"
        : initialDetection
          ? "Logged"
          : "Monitoring"

  const statusChips = [
    {
      label: "AI Model Active",
      className: "border border-emerald-400/30 bg-emerald-500/10 text-emerald-100",
      icon: <Activity className="h-4 w-4" />,
    },
    {
      label: formattedClock,
      className: "border border-white/15 bg-white/5 text-white/80",
      icon: <Clock3 className="h-4 w-4" />,
    },
    {
      label: "GPS Active",
      className: "border border-sky-400/40 bg-sky-500/10 text-sky-100",
      icon: <Navigation className="h-4 w-4" />,
    },
  ]

  const recentActivities = eventLog.slice(0, 3)

  const handleStartCamera = () => {
    if (videoRef.current) {
      videoRef.current.play().catch(() => {})
    }
    if (!isConnected && streamRef.current) {
      setIsConnected(true)
    }
  }

  const handleStopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }
    setIsConnected(false)
    setIsVideoReady(false)
  }

  const handleSimulateDetection = () => {
    setIsSimulating(true)
    sendMessageRef.current({ type: "simulate_detection" })
  }

  const handleEmergencyDispatch = () => {
    console.warn("Emergency dispatch triggered")
  }

  return (
    <div className="mt-6 space-y-6 text-white">
      <video ref={videoRef} className="hidden" playsInline muted autoPlay />

      <div>
        <p className="text-xs uppercase tracking-[0.4em] text-white/50">Live Surveillance</p>
        <h2 className="text-3xl font-semibold text-white">AI Model Active</h2>
        <p className="text-sm text-white/60">Computer webcam feed · Real-time detection stream</p>
      </div>

      <div className="flex flex-wrap gap-3">
        {statusChips.map((chip) => (
          <div key={chip.label} className={`flex items-center gap-2 rounded-full px-4 py-2 text-sm font-semibold ${chip.className}`}>
            {chip.icon}
            {chip.label}
          </div>
        ))}
      </div>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_360px]">
        <div className="rounded-[32px] border border-white/10 bg-gradient-to-b from-[#0b1f3f] via-[#081532] to-[#050c1f] shadow-[0_30px_65px_rgba(2,6,23,0.55)]">
          <div className="relative overflow-hidden rounded-[28px] border border-white/5 bg-[#020a17] px-6 py-8 min-h-[420px]">
            <img
              src="http://127.0.0.1:5000/camera-stream"
              alt="Live camera feed"
              className="absolute inset-0 h-full w-full object-cover"
              onLoad={() => setIsConnected(true)}
              onError={(e) => {
                console.error("Camera stream failed", e)
                setIsConnected(false)
              }}
            />
            <div className="absolute inset-0 bg-gradient-to-b from-transparent via-transparent to-black/60" />
            <div className="absolute left-6 top-6 flex items-center gap-3 text-xs uppercase tracking-[0.3em] text-white/70">
              <div className={`flex items-center gap-2 font-semibold ${isConnected ? "text-emerald-300" : "text-red-300"}`}>
                <span className={`h-2 w-2 rounded-full ${isConnected ? "bg-emerald-300 animate-pulse" : "bg-red-400 animate-pulse"}`} />
                {isConnected ? "Live" : "Connecting"}
              </div>
              <span className="text-white/40">·</span>
              <span>1280x720 30fps</span>
            </div>
            {isCapturing && (
              <div className="absolute right-6 top-6 flex items-center gap-2 rounded-full bg-emerald-400/90 px-4 py-1 text-xs font-semibold text-black shadow-lg shadow-emerald-500/40">
                <Camera className="h-3.5 w-3.5" />
                Capturing
              </div>
            )}
            {lastCapture && (
              <div className="absolute left-6 bottom-6 rounded-md bg-black/50 px-3 py-1 text-xs text-gray-200">
                Last capture · {lastCapture.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
              </div>
            )}
            <div className="absolute right-6 bottom-6 rounded-md bg-black/60 px-3 py-1 text-xs font-mono text-gray-200">
              {formatLogTime(new Date())}
            </div>
            {!isConnected && (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-black/60 text-white/70">
                <VideoOff className="h-12 w-12" />
                <p className="text-xs uppercase tracking-[0.4em]">Calibrating camera feed</p>
              </div>
            )}
          </div>

          <div className="flex flex-wrap items-center justify-between gap-4 border-t border-white/10 px-6 py-5">
            <div>
              <p className="text-lg font-semibold text-white">{primaryCamera.name}</p>
              <p className="text-sm text-white/60">Live Feed · AI Gunshot Detection Active</p>
            </div>
            <div className="flex flex-wrap gap-3">
              {isConnected ? (
                <Button
                  type="button"
                  className="flex items-center gap-2 rounded-full bg-red-500/90 px-5 py-2 font-semibold text-white hover:bg-red-500"
                  onClick={handleStopCamera}
                >
                  <VideoOff className="h-4 w-4" />
                  Stop Camera
                </Button>
              ) : (
                <Button
                  type="button"
                  className="flex items-center gap-2 rounded-full bg-emerald-500/90 px-5 py-2 font-semibold text-[#03180d] hover:bg-emerald-400"
                  onClick={handleStartCamera}
                >
                  <Camera className="h-4 w-4" />
                  Start Camera
                </Button>
              )}
              <Button
                type="button"
                variant="outline"
                className="flex items-center gap-2 rounded-full border-yellow-400/50 bg-yellow-500/10 px-5 py-2 text-yellow-100 hover:bg-yellow-500/20"
                onClick={handleSimulateDetection}
              >
                <Zap className="h-4 w-4" />
                Simulate Detection
              </Button>
            </div>
          </div>
        </div>

        <div className="space-y-6">
          <div className="rounded-3xl border border-white/10 bg-gradient-to-b from-[#111f3c] to-[#080f1f] p-6 space-y-5">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs uppercase tracking-[0.4em] text-white/40">AI Model Status</p>
                <h3 className="text-2xl font-semibold text-white">Gunshot v2.1</h3>
              </div>
              <div className={`flex items-center gap-2 rounded-full px-4 py-1 text-sm font-semibold ${isConnected ? "bg-emerald-400/15 text-emerald-200" : "bg-red-500/20 text-red-200"}`}>
                <span className={`h-2 w-2 rounded-full ${isConnected ? "bg-emerald-300 animate-pulse" : "bg-red-400"}`} />
                {isConnected ? "Active" : "Offline"}
              </div>
            </div>

            <div className="grid gap-2 text-sm text-white/70">
              <div className="flex items-center justify-between">
                <span>Status</span>
                <span className="text-white font-semibold">{isConnected ? "Streaming" : "Idle"}</span>
              </div>
              <div className="flex items-center justify-between">
                <span>Accuracy</span>
                <span className="text-white font-semibold">{currentConfidencePercent !== null ? `${currentConfidencePercent}%` : "94.7%"}</span>
              </div>
              <div className="flex items-center justify-between">
                <span>Latency</span>
                <span className="text-white font-semibold">~150ms</span>
              </div>
              <div className="flex items-center justify-between">
                <span>Alerts</span>
                <span className="text-white font-semibold">{eventLog.length}</span>
              </div>
            </div>

            <div className="border-t border-white/10 pt-4 text-sm text-white/70">
              <p className="text-xs uppercase tracking-[0.4em] text-white/40">Event details</p>
              <div className="mt-3 space-y-2">
                <div className="flex items-center justify-between">
                  <span>Timestamp</span>
                  <span className="font-mono text-white">{detailTimestamp ? formatLogTime(detailTimestamp) : "--:--:--"}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Weapon type</span>
                  <span className="text-white">{currentEvent ? currentEvent.label : "Unknown"}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Confidence</span>
                <span className="text-white">{detectionConfidenceText}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Camera</span>
                  <span className="text-white">{primaryCamera.id}</span>
                </div>
              </div>
            </div>

            <div className="border-t border-white/10 pt-4 text-sm text-white/70">
              <p className="text-xs uppercase tracking-[0.4em] text-white/40">Recent activity</p>
              <div className="mt-3 space-y-2">
                {recentActivities.map((item, index) => (
                  <div key={`activity-${item.time}-${index}`} className="flex items-center justify-between text-xs text-white/70">
                    <span className="font-mono text-white/40">{item.time}</span>
                    <span className="ml-4 text-right">{item.message}</span>
                  </div>
                ))}
                {!recentActivities.length && <p className="text-xs text-white/40">No recent events recorded.</p>}
              </div>
            </div>
          </div>

          <div className="rounded-3xl border border-yellow-400/30 bg-gradient-to-b from-[#1f1a05] to-[#120c02] p-6 text-sm text-white/80">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs uppercase tracking-[0.4em] text-yellow-200/70">Last Detection</p>
                <h3 className="text-2xl font-semibold text-white">
                  {currentEvent ? currentEvent.label : "Awaiting detection"}
                </h3>
              </div>
              <span className="rounded-full bg-yellow-500/15 px-3 py-1 text-sm font-semibold text-yellow-200">
                {detectionConfidenceText}
              </span>
            </div>
            <div className="mt-4 grid gap-2 text-white/70">
              <div className="flex items-center justify-between">
                <span>Confidence</span>
                <span className="text-white font-semibold">{detectionConfidenceText}</span>
              </div>
              <div className="flex items-center justify-between">
                <span>Type</span>
                <span className="text-white font-semibold capitalize">{currentEvent ? currentEvent.label : "Gunshot"}</span>
              </div>
              <div className="flex items-center justify-between">
                <span>Status</span>
                <span className="text-yellow-200 font-semibold">{currentEvent ? "Pending" : "Monitoring"}</span>
              </div>
            </div>
            <Button
              type="button"
              className="mt-4 w-full rounded-xl bg-yellow-500/90 py-3 font-semibold text-[#2a1c00] shadow-[0_15px_30px_rgba(234,179,8,0.35)] hover:bg-yellow-400"
              onClick={onViewLogs}
              disabled={!currentEvent}
            >
              Review in Logs
            </Button>
          </div>

          <div className="rounded-3xl border border-white/10 bg-gradient-to-b from-[#0c182d] to-[#060c1a] p-6 space-y-4">
            <p className="text-xs uppercase tracking-[0.4em] text-white/50">Quick actions</p>
            <div className="space-y-3">
              <Button
                type="button"
                className="w-full rounded-xl border border-white/15 bg-white/10 py-3 font-semibold text-white hover:bg-white/20"
                onClick={onViewLogs}
              >
                View All Logs
              </Button>
              <Button
                type="button"
                className="w-full rounded-xl border border-white/15 bg-white/10 py-3 font-semibold text-white hover:bg-white/20"
                onClick={onViewAnalytics}
              >
                AI Analytics
              </Button>
              <Button
                type="button"
                className="w-full rounded-xl bg-gradient-to-r from-[#ff6b6b] to-[#f43f5e] py-3 font-semibold text-white shadow-[0_15px_35px_rgba(244,63,94,0.45)] hover:brightness-110"
                onClick={handleEmergencyDispatch}
              >
                <span className="mr-2 inline-flex h-2 w-2 animate-pulse rounded-full bg-white" />
                Emergency Dispatch
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
