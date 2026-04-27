"""
Max Headroom Digitizer - WebSocket Server with OBS Support
Multi-client server with OBS integration
"""
import asyncio
import json
import time
import threading
import socket
import struct
import numpy as np
from typing import Dict, Any, Set, List
from dataclasses import dataclass, field
from collections import deque
import argparse

VERSION = "3.0.0"

@dataclass
class ServerStats:
    fps: float = 0.0
    latency_ms: float = 0.0
    clients: int = 0
    frames_received: int = 0
    bytes_received: int = 0
    uptime: float = 0.0

@dataclass
class ClientData:
    """Data from a single client."""
    blendshapes: Dict[str, float] = field(default_factory=dict)
    head_pose: Dict[str, List[float]] = field(default_factory=dict)
    landmarks: List = field(default_factory=list)
    timestamp: float = 0.0

class MaxHeadroomServer:
    """High-performance WebSocket server with OBS integration."""
    
    def __init__(self, host="localhost", port=30000):
        self.host = host
        self.port = port
        self.running = False
        
        # Client management
        self.clients: Set = set()
        self.current_client: ClientData = None
        
        # OBS integration
        self.obs_connected = False
        self.obs_port = 5556
        
        # Statistics
        self.stats = ServerStats()
        self.start_time = time.time()
        self.frame_buffer = deque(maxlen=60)
        self.last_fps_calc = time.time()
        
        # Threading
        self._thread = None
        self._obs_thread = None
    
    async def _handle_client(self, ws, path):
        """Handle WebSocket client connection."""
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
                    
                    elif data.get("type") == "broadcast":
                        await self._broadcast_to_obs(data.get("data", {}))
                    
                except json.JSONDecodeError:
                    pass
                    
        except Exception as e:
            print(f"[WS] Client error: {e}")
        finally:
            self.clients.discard(ws)
            print(f"[WS] Client disconnected (remaining: {len(self.clients)})")
    
    def _process_face_data(self, data: Dict):
        """Process face data from client."""
        self.current_client = ClientData(
            blendshapes=data.get("blendshapes", {}),
            head_pose=data.get("head_pose", {}),
            landmarks=data.get("landmarks", []),
            timestamp=data.get("timestamp", time.time())
        )
        
        self.frame_buffer.append(self.current_client.timestamp)
        self.stats.frames_received += 1
        
        # Calculate FPS
        now = time.time()
        if now - self.last_fps_calc >= 1.0:
            if len(self.frame_buffer) >= 2:
                self.stats.fps = len(self.frame_buffer) / (self.frame_buffer[-1] - self.frame_buffer[0]) * len(self.frame_buffer)
            self.last_fps_calc = now
        
        # Calculate latency
        if self.current_client.timestamp > 0:
            self.stats.latency_ms = (now - self.current_client.timestamp) * 1000
    
    async def _broadcast_to_obs(self, data: Dict):
        """Broadcast data to OBS via WebSocket."""
        if self.obs_connected and self.clients:
            msg = {"type": "face_broadcast", "data": data}
            msg_json = json.dumps(msg)
            for client in list(self.clients):
                try:
                    await client.send(msg_json)
                except:
                    pass
    
    async def _obs_receiver_loop(self):
        """Receive data from another server to forward to OBS."""
        print(f"[OBS] Starting OBS receiver on port {self.obs_port + 1}")
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("localhost", self.obs_port + 1))
            sock.listen(1)
            self.obs_connected = True
            print(f"[OBS] Receiver ready")
            
            while self.running:
                try:
                    conn, addr = sock.accept()
                    print(f"[OBS] Connected: {addr}")
                    
                    while self.running:
                        try:
                            data = conn.recv(4096)
                            if not data:
                                break
                            
                            # Process and broadcast
                            msg = json.loads(data.decode())
                            if msg.get("type") == "face_data":
                                self._process_face_data(msg)
                                
                        except:
                            break
                    
                    conn.close()
                except:
                    break
                    
        except Exception as e:
            print(f"[OBS] Error: {e}")
        finally:
            self.obs_connected = False
    
    async def _main(self):
        """Main server coroutine."""
        import websockets
        
        print(f"[Server] Max Headroom v{VERSION}")
        print(f"[Server] Starting on ws://{self.host}:{self.port}")
        
        async with websockets.serve(
            self._handle_client,
            self.host,
            self.port,
            ping_interval=None,
            max_size=10 * 1024 * 1024,
        ):
            print(f"[Server] Ready!")
            
            while self.running:
                await asyncio.sleep(0.5)
                
                # Update stats
                self.stats.clients = len(self.clients)
                self.stats.uptime = time.time() - self.start_time
                
                # Print status every second
                if self.stats.frames_received % 30 == 0 and self.stats.frames_received > 0:
                    print(f"[Status] FPS: {self.stats.fps:.1f} | Clients: {self.stats.clients} | Latency: {self.stats.latency_ms:.1f}ms | OBS: {'ON' if self.obs_connected else 'OFF'}")
    
    def _run_async(self):
        """Run server in async loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self._main())
        except KeyboardInterrupt:
            print("[Server] Stopped")
        finally:
            loop.close()
    
    def start(self):
        """Start server."""
        if self.running:
            print("[Server] Already running!")
            return
        
        self.running = True
        self._thread = threading.Thread(target=self._run_async, daemon=True)
        self._thread.start()
        
        # Start OBS receiver
        self._obs_thread = threading.Thread(target=lambda: asyncio.run(self._obs_receiver_loop()), daemon=True)
        self._obs_thread.start()
    
    def stop(self):
        """Stop server."""
        print("[Server] Stopping...")
        self.running = False
    
    def get_current_data(self) -> Dict[str, Any]:
        """Get current face data."""
        if self.current_client:
            return {
                "blendshapes": self.current_client.blendshapes,
                "head_pose": self.current_client.head_pose,
                "timestamp": self.current_client.timestamp,
            }
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            "fps": self.stats.fps,
            "latency_ms": self.stats.latency_ms,
            "clients": self.stats.clients,
            "frames": self.stats.frames_received,
            "obs_connected": self.obs_connected,
            "uptime": self.stats.uptime,
        }

# Global server instance
_server: MaxHeadroomServer = None

def start(host="localhost", port=30000) -> MaxHeadroomServer:
    """Start the server."""
    global _server
    _server = MaxHeadroomServer(host, port)
    _server.start()
    return _server

def stop():
    """Stop the server."""
    global _server
    if _server:
        _server.stop()

def get_data() -> Dict[str, Any]:
    """Get current face data."""
    return _server.get_current_data() if _server else None

def get_stats() -> Dict[str, Any]:
    """Get server stats."""
    return _server.get_stats() if _server else None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Max Headroom Server")
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
                stats = server.get_stats()
                print(f"FPS: {stats['fps']:.1f} | Clients: {stats['clients']} | Latency: {stats['latency_ms']:.1f}ms")
    except KeyboardInterrupt:
        stop()