#!/usr/bin/env python3
"""
Max Headroom Digitizer - WebSocket Server
Multi-client real-time data broker with OBS support
Version: 3.0.0
"""
import asyncio
import json
import time
import threading
import socket
import struct
import argparse
from typing import Dict, Any, Set, List
from dataclasses import dataclass, field
from collections import deque

VERSION = "3.0.0"

@dataclass
class ServerStats:
    fps: float = 0.0
    latency_ms: float = 0.0
    clients: int = 0
    frames_received: int = 0
    bytes_received: int = 0
    uptime: float = 0.0
    obs_connected: bool = False

@dataclass
class ClientData:
    blendshapes: Dict[str, float] = field(default_factory=dict)
    head_pose: Dict[str, List[float]] = field(default_factory=dict)
    landmarks: List = field(default_factory=list)
    timestamp: float = 0.0

class MaxHeadroomServer:
    """High-performance WebSocket server"""
    
    def __init__(self, host="localhost", port=30000):
        self.host = host
        self.port = port
        self.running = False
        self.clients: Set = set()
        self.current_data: ClientData = None
        self.stats = ServerStats()
        self.start_time = time.time()
        self.frame_buffer = deque(maxlen=60)
        self.last_time = time.time()
        self.fps_counter = 0
        self._thread = None
    
    async def handle_client(self, ws, path):
        self.clients.add(ws)
        print(f"[WS] Client connected from {ws.remote_address} (total: {len(self.clients)})")
        
        try:
            async for message in ws:
                try:
                    data = json.loads(message)
                    
                    if data.get("type") == "face_data":
                        self._process_face_data(data)
                    elif data.get("type") == "ping":
                        await ws.send(json.dumps({
                            "type": "pong",
                            "server_time": time.time(),
                            "client_time": data.get("client_time", 0)
                        }))
                    elif data.get("type") == "status_request":
                        await ws.send(json.dumps({
                            "type": "status",
                            "stats": {
                                "fps": self.stats.fps,
                                "latency": self.stats.latency_ms,
                                "clients": len(self.clients),
                                "frames": self.stats.frames_received,
                                "obs": self.stats.obs_connected
                            }
                        }))
                        
                except json.JSONDecodeError:
                    pass
                    
        except Exception as e:
            print(f"[WS] Error: {e}")
        finally:
            self.clients.discard(ws)
            print(f"[WS] Client disconnected (remaining: {len(self.clients)})")
    
    def _process_face_data(self, data: Dict):
        ts = data.get("timestamp", time.time())
        
        self.current_data = ClientData(
            blendshapes=data.get("blendshapes", {}),
            head_pose=data.get("head_pose", {}),
            landmarks=data.get("landmarks", []),
            timestamp=ts
        )
        
        self.frame_buffer.append(ts)
        self.stats.frames_received += 1
        
        if len(self.frame_buffer) >= 2:
            time_diff = self.frame_buffer[-1] - self.frame_buffer[0]
            if time_diff > 0:
                self.stats.fps = len(self.frame_buffer) / time_diff * self.frame_buffer[0] if self.frame_buffer[0] > 0 else 0
        
        if ts > 0:
            self.stats.latency_ms = (time.time() - ts) * 1000
    
    async def main(self):
        import websockets
        
        print(f"[Server] Max Headroom v{VERSION}")
        print(f"[Server] Starting on ws://{self.host}:{self.port}")
        
        async with websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            ping_interval=None,
            max_size=10 * 1024 * 1024,
        ):
            print(f"[Server] Ready!")
            
            while self.running:
                await asyncio.sleep(0.5)
                
                self.stats.clients = len(self.clients)
                self.stats.uptime = time.time() - self.start_time
                
                if self.stats.frames_received % 30 == 0 and self.stats.frames_received > 0:
                    print(f"[Status] FPS: {self.stats.fps:.1f} | Clients: {self.stats.clients} | Latency: {self.stats.latency_ms:.1f}ms")
    
    def run_async(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self.main())
        except KeyboardInterrupt:
            print("[Server] Stopped")
        finally:
            loop.close()
    
    def start(self):
        if self.running:
            print("[Server] Already running!")
            return
        
        self.running = True
        self._thread = threading.Thread(target=self.run_async, daemon=True)
        self._thread.start()
        print("[Server] Started")
    
    def stop(self):
        print("[Server] Stopping...")
        self.running = False
    
    def get_data(self) -> Dict[str, Any]:
        if self.current_data:
            return {
                "blendshapes": self.current_data.blendshapes,
                "head_pose": self.current_data.head_pose,
                "timestamp": self.current_data.timestamp,
            }
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "fps": self.stats.fps,
            "latency": self.stats.latency_ms,
            "clients": self.stats.clients,
            "frames": self.stats.frames_received,
            "uptime": self.stats.uptime,
            "obs": self.stats.obs_connected
        }

_server = None

def start(host="localhost", port=30000):
    global _server
    _server = MaxHeadroomServer(host, port)
    _server.start()
    return _server

def stop():
    global _server
    if _server:
        _server.stop()

def get_data():
    return _server.get_data() if _server else None

def get_stats():
    return _server.get_stats() if _server else None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=30000)
    args = parser.parse_args()
    
    print(f"Max Headroom Server v{VERSION}")
    print("Commands: start(), stop(), get_data(), get_stats()")
    
    server = start(args.host, args.port)
    
    try:
        while True:
            time.sleep(1)
            if server:
                s = server.get_stats()
                print(f"FPS: {s['fps']:.1f} | Clients: {s['clients']} | Latency: {s['latency']:.1f}ms")
    except KeyboardInterrupt:
        stop()