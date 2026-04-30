#!/usr/bin/env python3
"""
Max Headroom - Filter System Tests
Tests all Snapchat/WhatsApp level filters
"""
import sys
import os
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
    print(" FILTER SYSTEM TESTS")
    print("=" * 60)
    
    from filters import (FilterManager, SkinSmoothingFilter, BackgroundFilter,
                        AROverlayFilter, FaceMorphFilter, ColorGradingFilter)
    
    # Filter Manager
    print("\n>>> FILTER MANAGER <<<")
    
    test("Manager init", lambda: FilterManager() is not None)
    
    def test_manager_filters():
        mgr = FilterManager()
        return len(mgr.filters) >= 5
    test("Manager has 5+ filters", test_manager_filters)
    
    def test_manager_status():
        mgr = FilterManager()
        status = mgr.get_all_status()
        return len(status) >= 5 and all("name" in s for s in status)
    test("Manager status", test_manager_status)
    
    def test_manager_process():
        mgr = FilterManager()
        frame = np.zeros((480, 640, 3), np.uint8)
        result = mgr.process(frame)
        return result.shape == frame.shape
    test("Manager process frame", test_manager_process)
    
    def test_manager_toggle():
        mgr = FilterManager()
        initial = mgr.get_filter("Skin Smoothing").enabled
        mgr.toggle_filter("Skin Smoothing")
        return mgr.get_filter("Skin Smoothing").enabled != initial
    test("Manager toggle", test_manager_toggle)
    
    def test_manager_params():
        mgr = FilterManager()
        mgr.set_filter_param("Skin Smoothing", "strength", 0.8)
        return mgr.get_filter("Skin Smoothing").get_param("strength") == 0.8
    test("Manager set params", test_manager_params)
    
    # Skin Smoothing
    print("\n>>> SKIN SMOOTHING <<<")
    
    test("Skin filter init", lambda: SkinSmoothingFilter() is not None)
    
    def test_skin_process():
        filt = SkinSmoothingFilter()
        filt.enable()
        filt.set_param("strength", 0.5)
        frame = np.random.randint(0, 255, (480, 640, 3), np.uint8)
        result = filt.process(frame)
        return result.shape == frame.shape and result.dtype == np.uint8
    test("Skin smoothing process", test_skin_process)
    
    def test_skin_mask():
        filt = SkinSmoothingFilter()
        frame = np.zeros((100, 100, 3), np.uint8)
        frame[40:60, 40:60] = [120, 80, 200]  # Skin tone area
        mask = filt._create_skin_mask(frame)
        return mask.shape == (100, 100) and mask.max() > 0
    test("Skin mask generation", test_skin_mask)
    
    # Background
    print("\n>>> BACKGROUND FILTER <<<")
    
    test("Background init", lambda: BackgroundFilter() is not None)
    
    def test_background_blur():
        filt = BackgroundFilter()
        filt.enable()
        filt.set_param("mode", "blur")
        frame = np.random.randint(0, 255, (480, 640, 3), np.uint8)
        result = filt.process(frame)
        return result.shape == frame.shape
    test("Background blur", test_background_blur)
    
    def test_background_color():
        filt = BackgroundFilter()
        filt.enable()
        filt.set_param("mode", "color")
        filt.set_param("replacement_color", (0, 0, 255))
        frame = np.random.randint(0, 255, (100, 100, 3), np.uint8)
        result = filt.process(frame)
        return result.shape == frame.shape
    test("Background color", test_background_color)
    
    # AR Overlay
    print("\n>>> AR OVERLAY <<<")
    
    test("AR init", lambda: AROverlayFilter() is not None)
    
    def test_ar_glasses():
        filt = AROverlayFilter()
        filt.enable()
        filt.add_sticker("glasses", color=(0, 255, 255))
        frame = np.zeros((480, 640, 3), np.uint8)
        landmarks = [(320 + i*3, 280 + i) for i in range(68)]
        result = filt.process(frame, {"landmarks": landmarks})
        return result.shape == frame.shape
    test("AR glasses sticker", test_ar_glasses)
    
    def test_ar_crown():
        filt = AROverlayFilter()
        filt.enable()
        filt.clear_stickers()
        filt.add_sticker("crown")
        frame = np.zeros((480, 640, 3), np.uint8)
        landmarks = [(320 + i*3, 280 + i) for i in range(68)]
        result = filt.process(frame, {"landmarks": landmarks})
        return result.shape == frame.shape
    test("AR crown sticker", test_ar_crown)
    
    # Face Morph
    print("\n>>> FACE MORPH <<<")
    
    test("Morph init", lambda: FaceMorphFilter() is not None)
    
    def test_morph_slimming():
        filt = FaceMorphFilter()
        filt.enable()
        filt.set_param("slimming", 0.3)
        frame = np.random.randint(0, 255, (480, 640, 3), np.uint8)
        landmarks = [(320 + i*3, 280 + i) for i in range(68)]
        result = filt.process(frame, {"landmarks": landmarks})
        return result.shape == frame.shape
    test("Face slimming", test_morph_slimming)
    
    # Color Grading
    print("\n>>> COLOR GRADING <<<")
    
    test("Color init", lambda: ColorGradingFilter() is not None)
    
    def test_color_preset():
        filt = ColorGradingFilter()
        filt.enable()
        frame = np.random.randint(0, 255, (100, 100, 3), np.uint8)
        
        for preset in ["warm", "cool", "cyberpunk", "vintage", "noir", "matrix"]:
            filt.set_preset(preset)
            result = filt.process(frame)
            if result.shape != frame.shape:
                return False
        return True
    test("All color presets", test_color_preset)
    
    def test_color_contrast():
        filt = ColorGradingFilter()
        filt.enable()
        filt.set_param("contrast", 1.5)
        filt.set_param("brightness", 10)
        frame = np.random.randint(0, 255, (100, 100, 3), np.uint8)
        result = filt.process(frame)
        return result.shape == frame.shape
    test("Contrast/brightness", test_color_contrast)
    
    def test_vignette():
        filt = ColorGradingFilter()
        filt.enable()
        filt.set_param("vignette", 0.5)
        frame = np.random.randint(0, 255, (100, 100, 3), np.uint8)
        result = filt.process(frame)
        return result.shape == frame.shape
    test("Vignette effect", test_vignette)
    
    # Performance
    print("\n>>> PERFORMANCE <<<")
    
    def test_filter_speed():
        mgr = FilterManager()
        frame = np.random.randint(0, 255, (480, 640, 3), np.uint8)
        landmarks = [(320 + i*3, 280 + i) for i in range(68)]
        
        import time
        start = time.time()
        for _ in range(30):
            mgr.process(frame, landmarks=landmarks)
        elapsed = time.time() - start
        
        print(f" ({elapsed*1000/30:.1f}ms/frame)", end="")
        return elapsed < 3.0  # Should be under 100ms per frame
    test("Filter pipeline speed", test_filter_speed)
    
    # Summary
    print("\n" + "=" * 60)
    print(" FILTER TEST RESULTS")
    print("=" * 60)
    print(f"  PASSED: {TEST_RESULTS['pass']}")
    print(f"  FAILED: {TEST_RESULTS['fail']}")
    print(f"  TOTAL:  {TEST_RESULTS['pass'] + TEST_RESULTS['fail']}")
    print("=" * 60)
    
    return TEST_RESULTS["fail"] == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)