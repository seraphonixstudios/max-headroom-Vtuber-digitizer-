#!/usr/bin/env python3
"""
Max Headroom Digitizer - WebSocket Server with OBS Support
Multi-client server with OBS integration and structured logging
Version: 3.0.0
"""
import asyncio
import json
import time
import threading
import socket
import numpy as np
from typing import Dict, Any, Set, List
from dataclasses import dataclass, field
from collections import deque
import argparse

VERSION = "3.0.0"

# Logging setup
try:
    from logging_utils import LOG, get_logger
    LOG = get_logger("Server")
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    LOG = logging.getLogger("MaxHeadroom.Server")

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
        
        LOG.info("Server initialized: %s:%d", host, port)
    
    async def _handle_client(self, ws, path=None):
        """Handle WebSocket client connection."""
        if path is None:
            path = ""
        
        self.clients.add(ws)
        remote = getattr(ws, 'remote_address', 'unknown')
        LOG.info("WS Client connected from %s (total: %d)", remote, len(self.clients))
        
        try:
            async for message in ws:
                try:
                    data = json.loads(message)
                    msg_type = data.get("type")
                    
                    if msg_type == "face_data":
                        self._process_face_data(data)
                    
                    elif msg_type == "ping":
                        await ws.send(json.dumps({
                            "type": "pong",
                            "server_time": time.time(),
                            "client_time": data.get("client_time", 0)
                        }))
                    
                    elif msg_type == "status_request":
                        await ws.send(json.dumps({
                            "type": "status",
                            "stats": {
                                "fps": self.stats.fps,
                                "latency": self.stats.latency_ms,
                                "clients": len(self.clients),
                                "frames": self.stats.frames_received,
                                "obs": self.obs_connected,
                                "uptime": time.time() - self.start_time
                            }
                        }))
                    
                    elif msg_type == "broadcast":
                        await self._broadcast_to_obs(data.get("data", {}))
                    
                except json.JSONDecodeError as e:
                    LOG.warning("WS Invalid JSON from %s: %s", remote, e)
                except Exception as e:
                    LOG.error("WS Message handling error: %s", e)
                    
        except Exception as e:
            LOG.error("WS Client error from %s: %s", remote, e)
        finally:
            self.clients.discard(ws)
            LOG.info("WS Client disconnected from %s (remaining: %d)", remote, len(self.clients))
    
    def _process_face_data(self, data: Dict):
        """Process face data from client."""
        try:
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
                    duration = self.frame_buffer[-1] - self.frame_buffer[0]
                    if duration > 0:
                        self.stats.fps = len(self.frame_buffer) / duration
                self.last_fps_calc = now
            
            # Calculate latency
            if self.current_client.timestamp > 0:
                self.stats.latency_ms = (now - self.current_client.timestamp) * 1000
                
            LOG.debug("Processed frame %d (latency: %.1fms)", 
                     self.stats.frames_received, self.stats.latency_ms)
            
        except Exception as e:
            LOG.error("Error processing face data: %s", e)
    
    async def _broadcast_to_obs(self, data: Dict):
        """Broadcast data to OBS via WebSocket."""
        if self.obs_connected and self.clients:
            msg = {"type": "face_broadcast", "data": data}
            msg_json = json.dumps(msg)
            for client in list(self.clients):
                try:
                    await client.send(msg_json)
                except Exception as e:
                    LOG.warning("Failed to broadcast to client: %s", e)
    
    async def _obs_receiver_loop(self):
        """Receive data from another server to forward to OBS."""
        obs_port = self.obs_port + 1
        LOG.info("OBS Starting receiver on port %d", obs_port)
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("localhost", obs_port))
            sock.listen(1)
            sock.settimeout(1.0)
            self.obs_connected = True
            LOG.info("OBS Receiver ready on port %d", obs_port)
            
            while self.running:
                try:
                    conn, addr = sock.accept()
                    LOG.info("OBS Connected: %s", addr)
                    
                    while self.running:
                        try:
                            data = conn.recv(4096)
                            if not data:
                                break
                            
                            msg = json.loads(data.decode())
                            if msg.get("type") == "face_data":
                                self._process_face_data(msg)
                                
                        except socket.timeout:
                            continue
                        except json.JSONDecodeError as e:
                            LOG.warning("OBS Invalid JSON: %s", e)
                        except Exception as e:
                            LOG.error("OBS Connection error: %s", e)
                            break
                    
                    conn.close()
                except socket.timeout:
                    continue
                except Exception as e:
                    LOG.error("OBS Accept error: %s", e)
                    break
                    
        except Exception as e:
            LOG.error("OBS Fatal error: %s", e)
        finally:
            self.obs_connected = False
            try:
                sock.close()
            except:
                pass
            LOG.info("OBS Receiver stopped")
    
    async def _main(self):
        """Main server coroutine."""
        import websockets
        
        LOG.info("Max Headroom Server v%s", VERSION)
        LOG.info("Starting on ws://%s:%d", self.host, self.port)
        
        async with websockets.serve(
            self._handle_client,
            self.host,
            self.port,
            ping_interval=None,
            max_size=10 * 1024 * 1024,
        ):
            LOG.info("Server Ready!")
            
            report_interval = 30
            last_report = 0
            
            while self.running:
                await asyncio.sleep(0.5)
                
                # Update stats
                self.stats.clients = len(self.clients)
                self.stats.uptime = time.time() - self.start_time
                
                # Periodic status report
                if self.stats.frames_received > 0 and (self.stats.frames_received - last_report) >= report_interval:
                    LOG.info("Status FPS:%.1f Clients:%d Latency:%.1fms OBS:%s Frames:%d",
                            self.stats.fps, self.stats.clients, self.stats.latency_ms,
                            "ON" if self.obs_connected else "OFF", self.stats.frames_received)
                    last_report = self.stats.frames_received
    
    def _run_async(self):
        """Run server in async loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self._main())
        except KeyboardInterrupt:
            LOG.info("Server interrupted by user")
        except Exception as e:
            LOG.error("Server loop error: %s", e)
        finally:
            loop.close()
            LOG.info("Server loop closed")
    
    def start(self):
        """Start server."""
        if self.running:
            LOG.warning("Server already running!")
            return
        
        self.running = True
        self.start_time = time.time()
        self._thread = threading.Thread(target=self._run_async, daemon=True)
        self._thread.start()
        
        # Start OBS receiver
        self._obs_thread = threading.Thread(target=lambda: asyncio.run(self._obs_receiver_loop()), daemon=True)
        self._obs_thread.start()
        
        LOG.info("Server started successfully")
    
    def stop(self):
        """Stop server."""
        if not self.running:
            return
        LOG.info("Server stopping...")
        self.running = False
        
        # Wait for threads to finish
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._obs_thread and self._obs_thread.is_alive():
            self._obs_thread.join(timeout=2.0)
        
        LOG.info("Server stopped. Uptime: %.1fs Frames: %d", 
                self.stats.uptime, self.stats.frames_received)
    
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
    
    LOG.info("Max Headroom Server v%s", VERSION)
    LOG.info("Commands: start(), stop(), get_data(), get_stats()")
    
    server = start(args.host, args.port)
    
    try:
        while True:
            time.sleep(1)
            if server:
                stats = server.get_stats()
                if stats['frames'] > 0:
                    LOG.info("Live FPS:%.1f Clients:%d Latency:%.1fms",
                            stats['fps'], stats['clients'], stats['latency_ms'])
    except KeyboardInterrupt:
        LOG.info("Shutting down...")
        stop()
        LOG.info("Goodbye!")