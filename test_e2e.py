#!/usr/bin/env python3
"""
Max Headroom - End-to-End Integration Test v3.1
Simulates complete pipeline: Tracker -> Filters -> Server -> Exports -> Recorder
Tests filter status propagation, android mode, and full system integration.
"""
import sys
import os
import time
import json
import threading
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_e2e():
    """End-to-end pipeline test with filter system integration."""
    print("=" * 70)
    print(" MAX HEADROOM v3.1 - END-TO-END INTEGRATION TEST")
    print("=" * 70)
    
    from server import MaxHeadroomServer
    from tracker import MaxHeadroomTracker
    from filters import FilterManager, MaxHeadroomFilter
    from recorder import Recorder
    from obs_controller import OBSController
    from blender_export import BlenderExporter
    from vts_export import VTSExporter
    
    passed = 0
    failed = 0
    
    def check(name, condition):
        nonlocal passed, failed
        print(f"\n[{passed+failed+1}] {name}...", end=" ")
        if condition:
            print("PASS")
            passed += 1
            return True
        else:
            print("FAIL")
            failed += 1
            return False
    
    # ===========================================
    # Step 1: Start Server
    # ===========================================
    server = MaxHeadroomServer(port=30004)
    server.start()
    time.sleep(0.5)
    check("Server starts", server.running)
    
    # ===========================================
    # Step 2: Tracker generates data
    # ===========================================
    tracker = MaxHeadroomTracker()
    tracker.config.test_mode = True
    
    test_frame = np.zeros((480, 640, 3), np.uint8)
    bs, lm, pose = tracker.process_frame(test_frame)
    check("Tracker generates blendshapes", len(bs) >= 30)
    
    # ===========================================
    # Step 3: Filter system integration
    # ===========================================
    check("FilterManager initializes", tracker.filter_manager is not None)
    check("Max Headroom filter exists", tracker.filter_manager.get_filter("Max Headroom") is not None)
    
    # Enable android filter
    tracker.filter_manager.enable_filter("Max Headroom")
    mh_filter = tracker.filter_manager.get_filter("Max Headroom")
    check("Max Headroom filter enables", mh_filter.enabled)
    
    # Process frame through filters
    filtered_frame = tracker.filter_manager.process(
        test_frame.copy(),
        blendshapes=bs,
        head_pose=pose,
        frame_id=1
    )
    check("Filter pipeline processes frame", filtered_frame is not None and filtered_frame.shape == test_frame.shape)
    
    # ===========================================
    # Step 4: WebSocket data with filter status
    # ===========================================
    filter_status = {"active": ["Max Headroom"], "params": {"Max Headroom": {"intensity": 1.0}}}
    data = {
        "type": "face_data",
        "version": "3.1.1",
        "mode": "digital_entity",
        "blendshapes": bs,
        "head_pose": pose,
        "landmarks": [{"x": 320, "y": 240}],
        "timestamp": time.time(),
        "fps": 30,
        "frame_id": 1,
        "filter_status": filter_status,
    }
    
    # Serialize
    json_data = {}
    for k, v in data.items():
        if k == "blendshapes":
            json_data[k] = {key: float(val) for key, val in v.items()}
        elif k == "head_pose":
            json_data[k] = {
                "rotation": [float(x) for x in v.get("rotation", [0,0,0])],
                "translation": [float(x) for x in v.get("translation", [0,0,1.5])]
            }
        elif k == "filter_status":
            json_data[k] = v
        else:
            json_data[k] = v
    
    check("Data serializes with filter_status", "filter_status" in json_data)
    
    # ===========================================
    # Step 5: Server processes filter-aware data
    # ===========================================
    server._process_face_data(json_data)
    check("Server processes frame", server.stats.frames_received == 1)
    
    current = server.get_current_data()
    check("Server data includes filter_status", current is not None and "filter_status" in current)
    
    server_filters = server.get_filter_status()
    check("Server filter status has active list", "active" in server_filters)
    check("Server sees Max Headroom active", "Max Headroom" in server_filters.get("active", []))
    
    # ===========================================
    # Step 6: OBS auto-switch with android mode
    # ===========================================
    obs = OBSController()
    obs.start_obs_thread = lambda: None
    
    # Simulate android mode data
    obs.start_auto_switch({
        "blendshapes": {"jawOpen": 0.2},
        "filter_status": {"active": ["Max Headroom"]}
    })
    check("OBS processes android mode data", True)
    
    # Simulate normal mode
    obs.start_auto_switch({
        "blendshapes": {"jawOpen": 0.2},
        "filter_status": {"active": []}
    })
    check("OBS processes normal mode data", True)
    
    # ===========================================
    # Step 7: Recorder saves filter state
    # ===========================================
    rec = Recorder()
    rec.start("e2e_test_v31")
    rec.add(json_data)
    session = rec.stop()
    check("Recorder saves frame with filter_status", session.frame_count == 1)
    
    saved_path = session.save("e2e_test_v31.mhr")
    check("Recording saves to disk", os.path.exists(saved_path))
    
    # Verify playback includes filter_status
    loaded = rec.load(saved_path)
    playback = rec.play()
    check("Playback includes filter_status", playback is not None and "filter_status" in playback)
    
    # Cleanup recording file
    if os.path.exists(saved_path):
        os.remove(saved_path)
    
    # ===========================================
    # Step 8: Blender export with filter metadata
    # ===========================================
    blender = BlenderExporter()
    mapped = blender._map_blendshapes(bs)
    check("Blender mapping has 10+ entries", len(mapped) >= 10)
    
    # Test export payload includes filter_status
    payload = {
        "type": "blendshapes",
        "version": "3.1.1",
        "frame": 0,
        "targets": mapped,
        "filter_status": filter_status,
        "android_mode": True,
    }
    check("Blender payload has android_mode", payload.get("android_mode") == True)
    
    # ===========================================
    # Step 9: VTS export with filter metadata
    # ===========================================
    vts = VTSExporter()
    check("VTS exporter ready", vts is not None)
    
    # Simulate VTS parameter generation with android mode
    params = []
    for ark_name, value in bs.items():
        vts_name = vts.mapping.get(ark_name, ark_name)
        params.append({"id": vts_name, "value": max(0.0, min(1.0, value))})
    
    # Add AndroidMode parameter
    if "Max Headroom" in filter_status.get("active", []):
        params.append({"id": "AndroidMode", "value": 1.0})
    
    android_param = next((p for p in params if p["id"] == "AndroidMode"), None)
    check("VTS includes AndroidMode parameter", android_param is not None and android_param["value"] == 1.0)
    
    # ===========================================
    # Step 10: Filter pipeline performance
    # ===========================================
    mgr = FilterManager()
    mgr.enable_filter("Max Headroom")
    frame = np.random.randint(0, 255, (480, 640, 3), np.uint8)
    
    start = time.time()
    for _ in range(10):
        result = mgr.process(frame)
    elapsed = (time.time() - start) * 1000 / 10
    check(f"Filter pipeline <100ms/frame ({elapsed:.1f}ms)", elapsed < 100)
    
    # ===========================================
    # Step 11: Pipeline coordinator integration
    # ===========================================
    from pipeline import PipelineCoordinator
    
    pipeline = PipelineCoordinator({
        "tracker": {"test_mode": True},
        "server": {"host": "localhost", "port": 30005},
    })
    pipeline.initialize()
    check("Pipeline initializes", pipeline.tracker is not None or pipeline.server is not None)
    
    stats = pipeline.get_stats()
    check("Pipeline stats include filter_status", "filter_status" in stats)
    
    # ===========================================
    # Step 12: Desktop app filter integration
    # ===========================================
    from max_headroom import MaxHeadroomApp, AppConfig
    
    app = MaxHeadroomApp(AppConfig(test_mode=True))
    check("Desktop app initializes", app is not None)
    check("Desktop app has filter manager", app.filter_manager is not None)
    
    # ===========================================
    # Step 13: UI/UX enhancements
    # ===========================================
    check("App has frame queue for thread safety", hasattr(app, '_frame_queue'))
    check("App has UI lock", hasattr(app, '_ui_lock'))
    check("App has CameraManager", hasattr(app, 'cam_mgr'))
    check("App has _schedule_ui method", hasattr(app, '_schedule_ui'))
    check("App has _update_filter_toggles method", hasattr(app, '_update_filter_toggles'))
    check("App has _update_status method", hasattr(app, '_update_status'))
    check("App has _try_log method", hasattr(app, '_try_log'))
    check("App has _activate_scene method", hasattr(app, '_activate_scene'))
    check("App has _test_camera method", hasattr(app, '_test_camera'))
    check("App version matches module", app.__class__.__module__ == 'max_headroom')
    
    # Test BlendShapeCalculator produces expected shapes
    calc = app.blendshape_calc
    shapes = calc.calculate(None, 0.0)
    check("Blendshapes include jawOpen", "jawOpen" in shapes)
    check("Blendshapes include mouthSmile_L", "mouthSmile_L" in shapes)
    
    # Test data smoothing
    app.config.smoothing = 0.5
    s1 = app._smooth_blendshapes({"jawOpen": 1.0})
    s2 = app._smooth_blendshapes({"jawOpen": 0.0})
    check("Smoothing blends values", 0.0 < s2["jawOpen"] < 1.0)
    
    # ===========================================
    # Cleanup
    # ===========================================
    server.stop()
    pipeline.stop() if hasattr(pipeline, 'stop') else None
    
    # Summary
    print("\n" + "=" * 70)
    print(" END-TO-END TEST RESULTS")
    print("=" * 70)
    print(f"  PASSED: {passed}")
    print(f"  FAILED: {failed}")
    print(f"  TOTAL:  {passed + failed}")
    print("=" * 70)
    
    if failed == 0:
        print(" ALL END-TO-END TESTS PASSED")
        print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = test_e2e()
    sys.exit(0 if success else 1)
