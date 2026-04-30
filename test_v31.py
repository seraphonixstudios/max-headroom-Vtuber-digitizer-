#!/usr/bin/env python3
"""
Max Headroom v3.1 - Integration Test Suite
Tests new pipeline components
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TEST_RESULTS = {"pass": 0, "fail": 0}

def test(name, func):
    print(f"[TEST] {name}", end=" ... ")
    try:
        if func():
            print("PASS")
            TEST_RESULTS["pass"] += 1
            return True
        else:
            print("FAIL")
            TEST_RESULTS["fail"] += 1
            return False
    except Exception as e:
        print(f"FAIL ({e})")
        TEST_RESULTS["fail"] += 1
        return False

def run_tests():
    print("=" * 60)
    print(" MAX HEADROOM v3.1 - INTEGRATION TESTS")
    print("=" * 60)
    
    # Config tests
    print("\n>>> CONFIG TESTS <<<")
    
    test("Config loads", lambda: __import__('config').load_config() is not None)
    test("Config get value", lambda: __import__('config').get('version') is not None)
    test("Config nested get", lambda: __import__('config').get('tracker.target_fps', 0) > 0)
    
    # Advanced tracker components
    print("\n>>> TRACKER v3.1 TESTS <<<")
    
    from tracker_v31 import AdvancedBlendShapeCalculator, AdvancedHeadPoseEstimator, KalmanFilter
    
    def test_kalman():
        kf = KalmanFilter()
        v1 = kf.update(0.5)
        v2 = kf.update(0.6)
        return v1 != 0 and v2 != 0
    test("Kalman filter", test_kalman)
    
    def test_calc_init():
        calc = AdvancedBlendShapeCalculator(use_kalman=True)
        return len(calc.ARKIT_BLENDSHAPES) == 51
    test("BlendShapeCalculator 51 shapes", test_calc_init)
    
    def test_calc_process():
        calc = AdvancedBlendShapeCalculator()
        lm = [(320 + i*2, 280 + i) for i in range(68)]
        result = calc.calculate(lm, (480, 640))
        return result is not None and len(result) >= 50
    test("BlendShapeCalculator process", test_calc_process)
    
    test("HeadPoseEstimator init", lambda: AdvancedHeadPoseEstimator(smooth_rot=0.8, smooth_trans=0.8) is not None)
    
    def test_pose_estimate():
        pose_est = AdvancedHeadPoseEstimator()
        lm = [(320 + i*2, 280 + i) for i in range(68)]
        pose = pose_est.estimate(lm, (480, 640))
        return pose is not None and len(pose.get('rotation', [])) == 3
    test("HeadPoseEstimator 3D pose", test_pose_estimate)
    
    # Face detector
    print("\n>>> DETECTOR TESTS <<<")
    
    from tracker_v31 import AdvancedFaceDetector
    
    test("FaceDetector init", lambda: AdvancedFaceDetector({'primary': 'haar'}) is not None)
    
    def test_detector_type():
        det = AdvancedFaceDetector({'primary': 'haar'})
        return det.detector_type == 'haar'
    test("FaceDetector Haar fallback", test_detector_type)
    
    def test_landmarks():
        det = AdvancedFaceDetector({'primary': 'haar'})
        lm = det._generate_landmarks(100, 100, 200, 200)
        return len(lm) >= 68
    test("FaceDetector generate landmarks", test_landmarks)
    
    # Pipeline tests
    print("\n>>> PIPELINE TESTS <<<")
    
    from pipeline import PipelineCoordinator, PipelineStats
    
    test("PipelineStats init", lambda: PipelineStats() is not None)
    test("PipelineCoordinator init", lambda: PipelineCoordinator({}).stats is not None)
    
    def test_health():
        coord = PipelineCoordinator({})
        health = coord.health_check()
        return health is not None and 'pipeline' in health
    test("Pipeline health check", test_health)
    
    # Performance
    print("\n>>> PERFORMANCE TESTS <<<")
    
    def test_speed():
        calc = AdvancedBlendShapeCalculator()
        lm = [(320 + i*2, 280 + i) for i in range(68)]
        start = time.time()
        for _ in range(100):
            calc.calculate(lm, (480, 640))
        return time.time() - start < 1.0
    test("Blendshape calculation speed", test_speed)
    
    # Data format
    print("\n>>> DATA FORMAT TESTS <<<")
    
    def test_data_packet():
        calc = AdvancedBlendShapeCalculator()
        lm = [(320 + i*2, 280 + i) for i in range(68)]
        bs = calc.calculate(lm, (480, 640))
        data = {
            "type": "face_data",
            "version": "3.1.0",
            "blendshapes": bs,
            "head_pose": {"rotation": [0,0,0], "translation": [0,0,1]},
            "timestamp": time.time(),
            "frame_id": 1
        }
        return data["version"] == "3.1.0" and len(data["blendshapes"]) >= 50
    test("v3.1 data packet", test_data_packet)
    
    # Summary
    print("\n" + "=" * 60)
    print(" v3.1 TEST RESULTS")
    print("=" * 60)
    print(f"  PASSED: {TEST_RESULTS['pass']}")
    print(f"  FAILED: {TEST_RESULTS['fail']}")
    print(f"  TOTAL:  {TEST_RESULTS['pass'] + TEST_RESULTS['fail']}")
    print("=" * 60)
    
    return TEST_RESULTS["fail"] == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)