#!/usr/bin/env python3
"""
Max Headroom - Quick Reference
Common commands and examples
"""

def main():
    print("MAX HEADROOM v3.0 - QUICK REFERENCE")
    print("=" * 50)
    
    print("\n[BASIC STARTUP]")
    print("# Terminal 1: Server")
    print("python server.py --port 30000")
    print("")
    print("# Terminal 2: Tracker (test mode)")
    print("python tracker.py --test --glitch 0.15")
    print("")
    print("# Terminal 2: Tracker (camera)")  
    print("python tracker.py --camera 0 --glitch 0.1")
    print("")
    print("# With MediaPipe")
    print("python mediapipe_tracker.py")
    
    print("\n[OPTIONS]")
    print("--camera 0           Camera index")
    print("--fps 30            Target FPS")
    print("--width 640         Frame width")
    print("--test              Test mode")
    print("--glitch 0.15       Glitch intensity")
    print("--ws-port 30000      WebSocket port")
    
    print("\n[MODULES]")
    print("# OBS Control")
    print("from obs_controller import connect, switch_scene")
    print("connect('localhost', 4455)")
    print("switch_scene('Live')")
    print("")
    print("# Blender Export")  
    print("from blender_export import BlenderExporter")
    print("exp = BlenderExporter()")
    print("exp.export(blendshapes, pose)")
    print("")
    print("# VTS Export")
    print("from vts_export import VTSExporter")
    print("vts = VTSExporter()")
    print("vts.set_blendshapes(blendshapes)")
    print("")
    print("# Recording")
    print("from recorder import start_recording, stop_recording, save_recording")
    print("start_recording('session')")
    print("session = stop_recording()")
    print("save_recording('session.mhr')")
    
    print("\n[ISSUES]")
    print("# Camera not available")
    print("python tracker.py --test")
    print("")
    print("# Port in use")
    print("python server.py --port 30001")
    
    print("\n" + "=" * 50)
    print("Full docs: README.md")

if __name__ == "__main__":
    main()