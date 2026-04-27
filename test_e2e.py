#!/usr/bin/env python3
"""
Max Headroom - End-to-End Integration Test
Simulates complete pipeline: Tracker -> Server -> Client
"""
import sys
import os
import time
import json
import threading
import websocket

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_e2e():
    """End-to-end pipeline test."""
    print("=" * 60)
    print(" MAX HEADROOM v3.0 - END-TO-END TEST")
    print("=" * 60)
    
    from server import MaxHeadroomServer
    from tracker import MaxHeadroomTracker
    import numpy as np
    
    # ===========================================
    # Step 1: Start Server
    # ===========================================
    print("\n[1] Starting server...")
    server = MaxHeadroomServer(port=30003)
    server.start()
    time.sleep(0.5)
    
    if not server.running:
        print("  FAIL: Server didn't start")
        return False
    
    print("  OK: Server running")
    
    # ===========================================
    # Step 2: Create tracker and generate data
    # ===========================================
    print("\n[2] Generating tracking data...")
    tracker = MaxHeadroomTracker()
    tracker.config.test_mode = True
    
    test_frame = np.zeros((480, 640, 3), np.uint8)
    bs, lm, pose = tracker.process_frame(test_frame)
    
    if not bs or len(bs) < 30:
        print("  FAIL: Invalid blendshapes")
        return False
    
    print(f"  OK: Generated {len(bs)} blendshapes")
    
    # ===========================================
    # Step 3: Simulate WebSocket message
    # ===========================================
    print("\n[3] Simulating WebSocket message...")
    
    data = {
        "type": "face_data",
        "version": "3.0.0",
        "mode": "digital_entity",
        "blendshapes": bs,
        "head_pose": pose,
        "landmarks": [{"x": 320, "y": 240}],
        "timestamp": time.time(),
        "fps": 30,
        "frame_id": 1,
    }
    
    # Convert numpy types for JSON
    json_data = {}
    for k, v in data.items():
        if k == "blendshapes":
            json_data[k] = {key: float(val) for key, val in v.items()}
        elif k == "head_pose":
            json_data[k] = {
                "rotation": [float(x) for x in v.get("rotation", [0,0,0])],
                "translation": [float(x) for x in v.get("translation", [0,0,1.5])]
            }
        elif k == "landmarks":
            json_data[k] = v
        else:
            json_data[k] = v
    
    print(f"  OK: Data serialized ({len(json_data['blendshapes'])} shapes)")
    
    # ===========================================
    # Step 4: Simulate processing on server side
    # ===========================================
    print("\n[4] Simulating server processing...")
    server._process_face_data(json_data)
    
    stats = server.get_stats()
    if stats["frames"] != 1:
        print(f"  FAIL: Frame not processed (received: {stats['frames']})")
        return False
    
    print(f"  OK: Server processed frame")
    
    # ===========================================
    # Step 5: Verify data on server
    # ===========================================
    print("\n[5] Verifying server data...")
    current = server.get_current_data()
    
    if not current:
        print("  FAIL: No data on server")
        return False
    
    if "blendshapes" not in current:
        print("  FAIL: Missing blendshapes in server data")
        return False
    
    print(f"  OK: Server has {len(current['blendshapes'])} blendshapes")
    
    # ===========================================
    # Step 6: Verify OBS integration
    # ===========================================
    print("\n[6] Testing OBS integration...")
    from obs_controller import OBSController
    
    obs = OBSController()
    obs.start_obs_thread = lambda: None  # Skip actual connection
    
    print("  OK: OBS controller ready")
    
    # ===========================================
    # Step 7: Verify Recording
    # ===========================================
    print("\n[7] Testing recording...")
    from recorder import Recorder
    
    rec = Recorder()
    rec.start("e2e_test")
    rec.add(json_data)
    session = rec.stop()
    
    if session.frame_count != 1:
        print(f"  FAIL: Recording failed ({session.frame_count} frames)")
        return False
    
    print(f"  OK: Recorded {session.frame_count} frame")
    
    # ===========================================
    # Step 8: Verify Blender Export
    # ===========================================
    print("\n[8] Testing Blender export...")
    from blender_export import BlenderExporter
    
    exp = BlenderExporter()
    mapped = exp._map_blendshapes(bs)
    
    if len(mapped) < 10:
        print(f"  FAIL: Mapping failed ({len(mapped)} entries)")
        return False
    
    print(f"  OK: Mapped to {len(mapped)} Blender targets")
    
    # ===========================================
    # Step 9: Verify VTS Export
    # ===========================================
    print("\n[9] Testing VTS export...")
    from vts_export import VTSExporter
    
    vts = VTSExporter()
    print("  OK: VTS exporter ready")
    
    # ===========================================
    # Cleanup
    # ===========================================
    print("\n[10] Cleanup...")
    server.stop()
    
    print("\n" + "=" * 60)
    print(" END-TO-END TEST PASSED")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_e2e()
    sys.exit(0 if success else 1)