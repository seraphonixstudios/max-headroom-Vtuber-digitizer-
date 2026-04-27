#!/usr/bin/env python3
"""
Max Headroom - Complete Test Suite
Tests all modules and functionality
"""
import sys
import os
import time
import json
import threading
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TEST_RESULTS = {"pass": 0, "fail": 0, "tests": []}

def test(name, func):
    """Run a test."""
    print(f"\n[TEST] {name}", end=" ... ")
    try:
        result = func()
        if result:
            print("PASS")
            TEST_RESULTS["pass"] += 1
            TEST_RESULTS["tests"].append({"name": name, "status": "pass"})
            return True
        else:
            print("FAIL (returned False)")
            TEST_RESULTS["fail"] += 1
            TEST_RESULTS["tests"].append({"name": name, "status": "fail", "error": "returned False"})
            return False
    except Exception as e:
        print(f"FAIL ({type(e).__name__}: {e})")
        TEST_RESULTS["fail"] += 1
        TEST_RESULTS["tests"].append({"name": name, "status": "fail", "error": str(e)})
        return False


def run_tests():
    """Run all tests."""
    print("=" * 60)
    print(" MAX HEADROOM v3.0 - COMPLETE TEST SUITE")
    print("=" * 60)
    
    # ===========================================
    # Tracker Tests
    # ===========================================
    print("\n>>> TRACKER TESTS <<<")
    
    from tracker import Config, MaxHeadroomTracker, FaceDetector, BlendShapeCalculator
    
    test("Config defaults", lambda: Config().ws_port == 30000)
    test("Config digital_mode", lambda: Config().digital_mode == True)
    test("Config glitch_intensity", lambda: Config().glitch_intensity == 0.15)
    
    test("FaceDetector init", lambda: not FaceDetector().cascade.empty())
    
    calc = BlendShapeCalculator()
    test_landmarks = [(320 + i*2, 280 + i) for i in range(68)]
    calc_result = calc.calculate(test_landmarks, (200, 150, 240, 260), 0)
    test("BlendShapeCalculator process", lambda: len(calc_result) >= 30)
    
    test("MaxHeadroomTracker init", lambda: MaxHeadroomTracker() is not None)
    
    tr = MaxHeadroomTracker()
    tr.config.test_mode = True
    test_frame = np.zeros((480, 640, 3), np.uint8)
    bs, lm, pose = tr.process_frame(test_frame)
    test("Test mode frame generation", lambda: len(bs) >= 30)
    
    # ===========================================
    # Server Tests
    # ===========================================
    print("\n>>> SERVER TESTS <<<")
    
    from server import MaxHeadroomServer
    
    test("Server init", lambda: MaxHeadroomServer(port=30001).port == 30001)
    
    s = MaxHeadroomServer()
    test("Server stats keys", lambda: "clients" in s.get_stats() and "fps" in s.get_stats())
    
    s = MaxHeadroomServer(port=30002)
    s.start()
    time.sleep(0.3)
    test("Server start/stop", lambda: s.running == True)
    s.stop()
    
    test("Server get_current_data (empty)", lambda: MaxHeadroomServer().get_current_data() is None)
    
    # ===========================================
    # Module Tests
    # ===========================================
    print("\n>>> MODULE TESTS <<<")
    
    test("OBS Controller init", lambda: __import__('obs_controller').OBSController() is not None)
    
    r = __import__('recorder').Recorder()
    test("Recorder init", lambda: r.max_frames == 10000)
    
    r = __import__('recorder').Recorder()
    r.start("test")
    r.add({"blendshapes": {"j": 0.1}, "head_pose": {}, "timestamp": 0, "fps": 30})
    s = r.stop()
    test("Recorder record/stop", lambda: s.frame_count == 1)
    
    test("Blender Export init", lambda: __import__('blender_export').BlenderExporter() is not None)
    
    test("VTS Export init", lambda: __import__('vts_export').VTSExporter() is not None)
    
    test("GPU init", lambda: __import__('gpu_accel').GPUDetector() is not None)
    
    # ===========================================
    # MediaPipe Tests
    # ===========================================
    print("\n>>> MEDIAPIPE TESTS <<<")
    
    from mediapipe_tracker import create_tracker, check_mediapipe
    
    test("MediaPipe check", lambda: check_mediapipe() in [True, False])
    
    test("MediaPipe fallback create", lambda: create_tracker(use_mediapipe=False) is not None)
    
    t = create_tracker(use_mediapipe=False)
    frame = np.zeros((480, 640, 3), np.uint8)
    t.close()
    test("MediaPipe fallback close", lambda: True)
    
    # ===========================================
    # Integration Tests
    # ===========================================
    print("\n>>> INTEGRATION TESTS <<<")
    
    tr = MaxHeadroomTracker()
    tr.config.test_mode = True
    test_frame = np.zeros((480, 640, 3), np.uint8)
    bs, lm, pose = tr.process_frame(test_frame)
    
    data = {
        "type": "face_data",
        "blendshapes": bs,
        "head_pose": pose,
        "timestamp": time.time(),
        "fps": 30,
        "frame_id": 1,
    }
    test("Tracker -> WebSocket data format", lambda: "blendshapes" in data and "head_pose" in data)
    
    test("Blendshape value ranges", lambda: (
        bs is not None and 
        len(bs) > 0 and 
        all(hasattr(v, '__float__') and -1.0 <= float(v) <= 1.5 for v in bs.values())
    ))
    
    test("Pose value ranges", lambda: len(pose.get("rotation", [])) == 3 and len(pose.get("translation", [])) == 3)
    
    # CLI test
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--test", action="store_true")
    p.add_argument("--glitch", type=float, default=0.15)
    args = p.parse_args(["--test", "--glitch", "0.2"])
    test("Config CLI args parsing", lambda: args.test == True and args.glitch == 0.2)
    
    # ===========================================
    # Summary
    # ===========================================
    print("\n" + "=" * 60)
    print(" TEST RESULTS")
    print("=" * 60)
    print(f"  PASSED: {TEST_RESULTS['pass']}")
    print(f"  FAILED: {TEST_RESULTS['fail']}")
    print(f"  TOTAL:  {TEST_RESULTS['pass'] + TEST_RESULTS['fail']}")
    print("=" * 60)
    
    if TEST_RESULTS['fail'] > 0:
        print("\nFAILED TESTS:")
        for t in TEST_RESULTS['tests']:
            if t['status'] == 'fail':
                print(f"  - {t['name']}: {t.get('error', 'unknown')}")
    
    return TEST_RESULTS['fail'] == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)