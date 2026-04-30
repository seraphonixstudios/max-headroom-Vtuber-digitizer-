#!/usr/bin/env python3
"""
Max Headroom v3.0 - Unified Launcher
Run server, tracker, and integrations from one command
"""
import sys
import os
import argparse
import time
import threading
import signal
import subprocess

VERSION = "3.0.0"

class MaxHeadroomLauncher:
    """Unified launcher for all Max Headroom components."""
    
    def __init__(self):
        self.server_process = None
        self.tracker_process = None
        self.running = False
        
    def start_server(self, port=30000, host="localhost"):
        """Start WebSocket server."""
        print(f"[Launcher] Starting server on {host}:{port}...")
        try:
            self.server_process = subprocess.Popen(
                [sys.executable, "server.py", "--port", str(port), "--host", host],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            time.sleep(1)
            print(f"[Launcher] Server started (PID: {self.server_process.pid})")
            return True
        except Exception as e:
            print(f"[Launcher] Server failed: {e}")
            return False
    
    def start_tracker(self, test_mode=False, camera=0, glitch=0.15, ws_host="localhost", ws_port=30000, android=False):
        """Start face tracker."""
        print(f"[Launcher] Starting tracker (test={test_mode}, camera={camera}, android={android})...")
        try:
            args = [sys.executable, "tracker.py", "--ws-host", ws_host, "--ws-port", str(ws_port)]
            
            if test_mode:
                args.append("--test")
            else:
                args.extend(["--camera", str(camera)])
            
            if glitch > 0:
                args.extend(["--glitch", str(glitch)])
            
            if android:
                args.append("--android")
            
            self.tracker_process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            print(f"[Launcher] Tracker started (PID: {self.tracker_process.pid})")
            return True
        except Exception as e:
            print(f"[Launcher] Tracker failed: {e}")
            return False
    
    def stop(self):
        """Stop all processes."""
        print("[Launcher] Stopping...")
        
        if self.server_process:
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=3)
            except:
                self.server_process.kill()
        
        if self.tracker_process:
            self.tracker_process.terminate()
            try:
                self.tracker_process.wait(timeout=3)
            except:
                self.tracker_process.kill()
        
        print("[Launcher] Stopped")
    
    def run_embedded(self, component, port=30000):
        """Run component in-process."""
        if component == "server":
            from server import MaxHeadroomServer
            server = MaxHeadroomServer(port=port)
            server.start()
            
            try:
                while self.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                server.stop()
        
        elif component == "tracker":
            from tracker import MaxHeadroomTracker
            tracker = MaxHeadroomTracker()
            tracker.run()


def run_tests():
    """Run the test suite."""
    print("[Launcher] Running test suite...")
    result = subprocess.run([sys.executable, "run_tests.py"], capture_output=False)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description=f"Max Headroom v{VERSION} - Unified Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch.py --test           # Run in test mode
  python launch.py --server         # Server only
  python launch.py --tracker        # Tracker only
  python launch.py --server --tracker --test  # Both + test mode
  python launch.py --test-suite     # Run test suite
  python launch.py --quick-test     # Quick validation
        """
    )
    
    parser.add_argument("--server", action="store_true", help="Start WebSocket server")
    parser.add_argument("--tracker", action="store_true", help="Start face tracker")
    parser.add_argument("--test", action="store_true", help="Run tracker in test mode")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--port", type=int, default=30000, help="WebSocket port (default: 30000)")
    parser.add_argument("--host", default="localhost", help="Host (default: localhost)")
    parser.add_argument("--glitch", type=float, default=0.15, help="Glitch intensity (default: 0.15)")
    parser.add_argument("--android", action="store_true", help="Enable Max Headroom android filter")
    parser.add_argument("--test-suite", action="store_true", help="Run test suite")
    parser.add_argument("--quick-test", action="store_true", help="Quick validation test")
    parser.add_argument("--all-tests", action="store_true", help="Run all test suites")
    
    args = parser.parse_args()
    
    print(f"Max Headroom v{VERSION} - Unified Launcher")
    print("=" * 50)
    
    # Quick test - just validate imports and config
    if args.quick_test:
        print("\n[Quick Test] Validating system...")
        
        try:
            from tracker import Config, MaxHeadroomTracker
            from server import MaxHeadroomServer
            from obs_controller import OBSController
            from recorder import Recorder
            from blender_export import BlenderExporter
            from vts_export import VTSExporter
            from gpu_accel import GPUDetector
            from filters import FilterManager
            print("  [OK] All modules import successfully")
            
            c = Config()
            s = MaxHeadroomServer()
            print("  [OK] Core components instantiate")
            
            import numpy as np
            t = MaxHeadroomTracker()
            t.config.test_mode = True
            bs, lm, p = t.process_frame(np.zeros((480,640,3),np.uint8))
            print(f"  [OK] Tracker produces {len(bs)} blendshapes")
            
            mgr = FilterManager()
            print(f"  [OK] Filter system with {len(mgr.filters)} filters")
            
            print("\n[Quick Test] PASSED - System ready!")
            return 0
        
        except Exception as e:
            print(f"\n[Quick Test] FAILED: {e}")
            return 1
    
    # Full test suite
    if args.test_suite:
        success = run_tests()
        return 0 if success else 1
    
    # All test suites
    if args.all_tests:
        print("\n[All Tests] Running complete test suite...")
        suites = [
            ("v3.0 Core", "run_tests.py"),
            ("v3.1 Pipeline", "test_v31.py"),
            ("Filter System", "test_filters.py"),
            ("End-to-End", "test_e2e.py"),
        ]
        all_pass = True
        for name, script in suites:
            print(f"\n>>> {name} <<<")
            result = subprocess.run([sys.executable, script], capture_output=False)
            if result.returncode != 0:
                all_pass = False
                print(f"  {name} FAILED")
            else:
                print(f"  {name} PASSED")
        print("\n" + "=" * 50)
        if all_pass:
            print("ALL TESTS PASSED")
        else:
            print("SOME TESTS FAILED")
        return 0 if all_pass else 1
    
    # Interactive mode
    launcher = MaxHeadroomLauncher()
    
    # Start server if requested
    if args.server:
        if not launcher.start_server(port=args.port, host=args.host):
            print("[Launcher] Failed to start server")
            return 1
    
    # Start tracker if requested
    if args.tracker or args.test:
        if not launcher.start_tracker(
            test_mode=args.test,
            camera=args.camera,
            glitch=args.glitch,
            ws_host=args.host,
            ws_port=args.port,
            android=args.android
        ):
            print("[Launcher] Failed to start tracker")
            return 1
    
    # If nothing specified, show help
    if not args.server and not args.tracker and not args.test:
        print("\nNo components specified. Use --help for options.")
        print("Try: python launch.py --test")
        return 0
    
    # Wait for interrupt
    try:
        print("\n[Launcher] Running... Press Ctrl+C to stop")
        while launcher.running or (launcher.server_process and launcher.server_process.poll() is None) or (launcher.tracker_process and launcher.tracker_process.poll() is None):
            time.sleep(1)
    except KeyboardInterrupt:
        print()
    finally:
        launcher.stop()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())