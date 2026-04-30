#!/usr/bin/env python3
"""
Max Headroom v3.1 - Pipeline Coordinator
Manages the complete data flow: Tracker -> Server -> Exports
"""
import time
import threading
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass

VERSION = "3.1.0"

try:
    from logging_utils import get_logger
    LOG = get_logger("Pipeline")
except Exception:
    import logging
    LOG = logging.getLogger("MaxHeadroom.Pipeline")

try:
    import config as cfg
except ImportError:
    cfg = None

@dataclass
class PipelineStats:
    """Pipeline performance statistics."""
    frames_processed: int = 0
    frames_sent: int = 0
    frames_dropped: int = 0
    avg_latency_ms: float = 0.0
    current_fps: float = 0.0
    pipeline_time_ms: float = 0.0
    last_update: float = 0.0

class PipelineCoordinator:
    """Coordinates all pipeline components."""
    
    def __init__(self, config: Dict = None):
        self.cfg = config or {}
        self.stats = PipelineStats()
        
        # Components
        self.tracker = None
        self.server = None
        self.exporters = {}
        
        # Threading
        self.running = False
        self._thread = None
        self._lock = threading.Lock()
        
        # Performance monitoring
        self.frame_times = []
        self.latencies = []
    
    def initialize(self):
        """Initialize all pipeline components."""
        LOG.info("Initializing pipeline v%s", VERSION)
        
        # Initialize tracker
        tracker_cfg = self.cfg.get('tracker', {})
        try:
            from tracker_v31 import MaxHeadroomTrackerV31
            self.tracker = MaxHeadroomTrackerV31(tracker_cfg)
            LOG.info("Tracker v3.1 initialized")
        except Exception as e:
            LOG.error("Failed to initialize tracker: %s", e)
            # Fallback to v3.0 tracker
            try:
                from tracker import MaxHeadroomTracker
                self.tracker = MaxHeadroomTracker()
                LOG.info("Fallback to tracker v3.0")
            except Exception as e2:
                LOG.error("Failed to initialize fallback tracker: %s", e2)
        
        # Initialize server
        server_cfg = self.cfg.get('server', {})
        try:
            from server import MaxHeadroomServer
            self.server = MaxHeadroomServer(
                host=server_cfg.get('host', 'localhost'),
                port=server_cfg.get('port', 30000)
            )
            LOG.info("Server initialized on %s:%d",
                    server_cfg.get('host', 'localhost'),
                    server_cfg.get('port', 30000))
        except Exception as e:
            LOG.error("Failed to initialize server: %s", e)
        
        # Initialize exporters
        exports_cfg = self.cfg.get('exports', {})
        
        if exports_cfg.get('blender', {}).get('enabled', False):
            try:
                from blender_export import BlenderExporter
                blender_cfg = exports_cfg['blender']
                self.exporters['blender'] = BlenderExporter(
                    host=blender_cfg.get('host', 'localhost'),
                    port=blender_cfg.get('port', 30001)
                )
                LOG.info("Blender exporter initialized")
            except Exception as e:
                LOG.warning("Blender exporter failed: %s", e)
        
        if exports_cfg.get('vts', {}).get('enabled', False):
            try:
                from vts_export import VTSExporter
                self.exporters['vts'] = VTSExporter()
                LOG.info("VTS exporter initialized")
            except Exception as e:
                LOG.warning("VTS exporter failed: %s", e)
        
        LOG.info("Pipeline initialization complete")
    
    def start(self):
        """Start the pipeline."""
        if self.running:
            LOG.warning("Pipeline already running")
            return
        
        self.running = True
        
        # Start server
        if self.server:
            self.server.start()
        
        # Start tracker in background thread
        if self.tracker:
            self._thread = threading.Thread(target=self._tracker_loop, daemon=True)
            self._thread.start()
        
        LOG.info("Pipeline started")
    
    def _tracker_loop(self):
        """Background tracker loop."""
        try:
            if hasattr(self.tracker, 'run'):
                self.tracker.run()
        except Exception as e:
            LOG.error("Tracker loop error: %s", e)
    
    def process_frame_data(self, data: Dict):
        """Process frame data through the pipeline with filter awareness."""
        start_time = time.time()
        
        with self._lock:
            self.stats.frames_processed += 1
            
            # Apply filter pipeline if tracker has filters
            filter_status = data.get('filter_status', {})
            
            # Send to server
            if self.server:
                try:
                    self.server._process_face_data(data)
                    self.stats.frames_sent += 1
                except Exception as e:
                    LOG.warning("Server processing failed: %s", e)
                    self.stats.frames_dropped += 1
            
            # Send to exporters with filter metadata
            blendshapes = data.get('blendshapes', {})
            pose = data.get('head_pose', {})
            
            for name, exporter in self.exporters.items():
                try:
                    if hasattr(exporter, 'export'):
                        exporter.export(blendshapes, pose, filter_status=filter_status)
                    elif hasattr(exporter, 'set_blendshapes'):
                        exporter.set_blendshapes(blendshapes)
                except Exception as e:
                    LOG.warning("Exporter %s failed: %s", name, e)
            
            # Update stats
            pipeline_time = (time.time() - start_time) * 1000
            self.frame_times.append(pipeline_time)
            if len(self.frame_times) > 60:
                self.frame_times.pop(0)
            
            self.stats.pipeline_time_ms = sum(self.frame_times) / len(self.frame_times)
            self.stats.last_update = time.time()
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        with self._lock:
            # Get filter status from server if available
            filter_info = {}
            if self.server and self.server.current_client:
                filter_info = self.server.current_client.filter_status
            
            return {
                "version": VERSION,
                "running": self.running,
                "frames_processed": self.stats.frames_processed,
                "frames_sent": self.stats.frames_sent,
                "frames_dropped": self.stats.frames_dropped,
                "avg_pipeline_ms": round(self.stats.pipeline_time_ms, 2),
                "current_fps": self.stats.current_fps,
                "tracker_type": type(self.tracker).__name__ if self.tracker else None,
                "server_running": self.server.running if self.server else False,
                "exporters": list(self.exporters.keys()),
                "filter_status": filter_info,
            }
    
    def stop(self):
        """Stop the pipeline."""
        LOG.info("Stopping pipeline...")
        self.running = False
        
        if self.tracker:
            self.tracker.stop()
        if self.server:
            self.server.stop()
        if self._thread:
            self._thread.join(timeout=3)
        
        LOG.info("Pipeline stopped")
    
    def health_check(self) -> Dict:
        """Perform health check on all components."""
        health = {
            "pipeline": "ok" if self.running else "stopped",
            "tracker": "ok" if self.tracker else "missing",
            "server": "ok" if (self.server and self.server.running) else "down",
            "exporters": {}
        }
        
        for name, exporter in self.exporters.items():
            try:
                if hasattr(exporter, 'connected'):
                    health["exporters"][name] = "connected" if exporter.connected else "disconnected"
                else:
                    health["exporters"][name] = "unknown"
            except:
                health["exporters"][name] = "error"
        
        return health

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Max Headroom Pipeline")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    
    # Load config
    if cfg:
        cfg.load_config(args.config)
        pipeline_cfg = cfg.get_config()
    else:
        pipeline_cfg = {}
    
    coordinator = PipelineCoordinator(pipeline_cfg)
    coordinator.initialize()
    coordinator.start()
    
    try:
        while True:
            time.sleep(5)
            stats = coordinator.get_stats()
            LOG.info("Pipeline stats: %s", json.dumps(stats, indent=2))
    except KeyboardInterrupt:
        LOG.info("Shutting down...")
    finally:
        coordinator.stop()