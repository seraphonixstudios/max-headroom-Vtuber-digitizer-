#!/usr/bin/env python3
"""
Max Headroom - Configuration Manager
Loads and validates config.json
"""
import json
import os
from typing import Dict, Any

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

class ConfigManager:
    """Manages application configuration."""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load(self, path: str = None) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        path = path or CONFIG_PATH
        
        if not os.path.exists(path):
            self._config = self._default_config()
            self.save(path)
            return self._config
        
        try:
            with open(path, 'r') as f:
                self._config = json.load(f)
            self._validate()
            return self._config
        except Exception as e:
            print(f"[Config] Error loading config: {e}")
            self._config = self._default_config()
            return self._config
    
    def get(self, key: str, default=None):
        """Get config value by dot-separated key."""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value
    
    def set(self, key: str, value: Any):
        """Set config value by dot-separated key."""
        keys = key.split('.')
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def save(self, path: str = None):
        """Save configuration to JSON file."""
        path = path or CONFIG_PATH
        try:
            with open(path, 'w') as f:
                json.dump(self._config, f, indent=2)
        except Exception as e:
            print(f"[Config] Error saving config: {e}")
    
    def _validate(self):
        """Validate and merge with defaults."""
        defaults = self._default_config()
        self._config = self._merge_dicts(defaults, self._config)
    
    def _merge_dicts(self, default: dict, override: dict) -> dict:
        """Recursively merge override into default."""
        result = default.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_dicts(result[key], value)
            else:
                result[key] = value
        return result
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "version": "3.1.0",
            "tracker": {
                "camera_index": 0,
                "target_fps": 60,
                "resolution": [640, 480],
                "smoothing": 0.75,
                "kalman_filter": True,
                "outlier_rejection": True,
                "detector": {
                    "primary": "mediapipe",
                    "fallback": "haar",
                    "face_confidence": 0.5,
                    "tracking_confidence": 0.5
                },
                "blendshapes": {
                    "calculate_3d": True,
                    "eye_gaze_enabled": True,
                    "mouth_asymmetry": True,
                    "brow_asymmetry": True
                },
                "head_pose": {
                    "enabled": True,
                    "use_solve_pnp": True,
                    "smooth_rotation": 0.8,
                    "smooth_translation": 0.8
                },
                "digital_mode": True,
                "glitch_intensity": 0.15,
                "eye_glow": False,
                "auto_reconnect": True,
                "reconnect_interval": 5.0
            },
            "server": {
                "host": "localhost",
                "port": 30000,
                "max_clients": 10,
                "auth_token": None,
                "rate_limit": {
                    "enabled": True,
                    "max_messages_per_second": 60,
                    "burst_size": 120
                }
            },
            "websocket": {
                "host": "localhost",
                "port": 30000,
                "compression": True,
                "batch_frames": False,
                "batch_interval_ms": 16
            },
            "exports": {
                "blender": {"enabled": False, "host": "localhost", "port": 30001},
                "vts": {"enabled": False, "port": 9001},
                "obs": {"enabled": False}
            },
            "recording": {
                "auto_record": False,
                "output_dir": "recordings",
                "format": "json",
                "compress": True
            },
            "performance": {
                "gpu_enabled": True,
                "frame_skip": 0,
                "threaded_processing": True,
                "max_processing_ms": 33
            },
            "logging": {
                "level": "INFO",
                "console": True,
                "file": True,
                "max_file_size_mb": 5,
                "backup_count": 5
            }
        }

# Global config instance
config = ConfigManager()

def load_config(path: str = None) -> ConfigManager:
    """Load and return global config."""
    return config.load(path)

def get(key: str, default=None):
    """Get config value."""
    if config._config is None:
        config.load()
    return config.get(key, default)

def set(key: str, value: Any):
    """Set config value."""
    if config._config is None:
        config.load()
    config.set(key, value)

if __name__ == "__main__":
    cfg = load_config()
    print(f"Config version: {get('version')}")
    print(f"Tracker FPS: {get('tracker.target_fps')}")
    print(f"Server port: {get('server.port')}")