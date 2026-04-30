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
                        AROverlayFilter, FaceMorphFilter, ColorGradingFilter,
                        MaxHeadroomFilter)
    
    # Filter Manager
    print("\n>>> FILTER MANAGER <<<")
    
    test("Manager init", lambda: FilterManager() is not None)
    
    def test_manager_filters():
        mgr = FilterManager()
        return len(mgr.filters) >= 6
    test("Manager has 6+ filters", test_manager_filters)
    
    def test_manager_status():
        mgr = FilterManager()
        status = mgr.get_all_status()
        return len(status) >= 6 and all("name" in s for s in status)
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
    
    # Max Headroom Android Filter
    print("\n>>> MAX HEADROOM ANDROID FILTER <<<")
    
    test("MH filter init", lambda: MaxHeadroomFilter() is not None)
    
    def test_mh_process():
        filt = MaxHeadroomFilter()
        filt.enable()
        frame = np.random.randint(0, 255, (480, 640, 3), np.uint8)
        result = filt.process(frame)
        return result.shape == frame.shape and result.dtype == np.uint8
    test("MH filter process", test_mh_process)
    
    def test_mh_monochrome():
        filt = MaxHeadroomFilter()
        filt.enable()
        filt.set_param("monochrome", True)
        frame = np.full((100, 100, 3), [128, 64, 200], dtype=np.uint8)
        result = filt.process(frame)
        # Should be mostly cyan/blue tinted
        return result.shape == frame.shape and np.mean(result[:, :, 0]) > np.mean(result[:, :, 2])
    test("MH monochrome cyan", test_mh_monochrome)
    
    def test_mh_scanlines():
        filt = MaxHeadroomFilter()
        filt.enable()
        filt.set_param("scanlines", True)
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = filt.process(frame)
        # Scanlines should create dark horizontal bands
        row_means = [np.mean(result[y, :, :]) for y in range(0, 100, 4)]
        return max(row_means) > min(row_means) + 5
    test("MH scanlines effect", test_mh_scanlines)
    
    def test_mh_pixelation():
        filt = MaxHeadroomFilter()
        filt.enable()
        filt.set_param("pixelate", True)
        filt.set_param("pixelate_scale", 0.1)
        frame = np.random.randint(0, 255, (100, 100, 3), np.uint8)
        result = filt.process(frame)
        # Pixelation should reduce variance
        return result.shape == frame.shape and np.var(result) < np.var(frame) * 1.5
    test("MH pixelation", test_mh_pixelation)
    
    def test_mh_sharpen():
        filt = MaxHeadroomFilter()
        filt.enable()
        filt.set_param("sharpen", True)
        frame = np.random.randint(0, 255, (100, 100, 3), np.uint8)
        result = filt.process(frame)
        return result.shape == frame.shape
    test("MH edge sharpen", test_mh_sharpen)
    
    def test_mh_vignette():
        filt = MaxHeadroomFilter()
        filt.enable()
        filt.set_param("vignette", True)
        frame = np.full((100, 100, 3), 200, dtype=np.uint8)
        result = filt.process(frame)
        # Center should be brighter than corners
        center = np.mean(result[40:60, 40:60])
        corner = np.mean(result[0:10, 0:10])
        return center > corner
    test("MH vignette", test_mh_vignette)
    
    def test_mh_grid():
        filt = MaxHeadroomFilter()
        filt.enable()
        filt.set_param("grid", True)
        frame = np.zeros((100, 100, 3), np.uint8)
        result = filt.process(frame)
        # Grid should add cyan lines
        return np.mean(result) > 0
    test("MH geometric grid", test_mh_grid)
    
    def test_mh_data_overlay():
        filt = MaxHeadroomFilter()
        filt.enable()
        filt.set_param("data_overlay", True)
        frame = np.zeros((100, 100, 3), np.uint8)
        result = filt.process(frame)
        # Data overlay should add text (non-zero in top/bottom bars)
        return np.mean(result[0:20, :]) > 0 or np.mean(result[80:100, :]) > 0
    test("MH data overlay", test_mh_data_overlay)
    
    def test_mh_intensity_cycle():
        filt = MaxHeadroomFilter()
        levels = [0.0, 0.3, 0.6, 1.0]
        for _ in range(6):
            level = filt.cycle_intensity()
            if level not in levels:
                return False
        return True
    test("MH intensity cycle", test_mh_intensity_cycle)
    
    def test_mh_chromatic():
        filt = MaxHeadroomFilter()
        filt.enable()
        filt.set_param("chromatic", True)
        filt.set_param("chromatic_probability", 1.0)  # Force it
        frame = np.random.randint(0, 255, (100, 100, 3), np.uint8)
        result = filt.process(frame)
        return result.shape == frame.shape
    test("MH chromatic aberration", test_mh_chromatic)
    
    def test_mh_stutter():
        filt = MaxHeadroomFilter()
        filt.enable()
        filt.set_param("stutter", True)
        filt.set_param("stutter_probability", 1.0)  # Force stutter
        frame = np.random.randint(0, 255, (100, 100, 3), np.uint8)
        result1 = filt.process(frame)
        result2 = filt.process(frame + 50)
        # Second frame should be stuttered (repeat of result1)
        return np.array_equal(result2, result1)
    test("MH temporal stutter", test_mh_stutter)
    
    def test_mh_glitch_blocks():
        filt = MaxHeadroomFilter()
        filt.enable()
        filt.set_param("glitch_blocks", True)
        filt.set_param("glitch_block_probability", 1.0)  # Force glitch
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = filt.process(frame)
        # Should differ from uniform frame
        return not np.all(result == 128)
    test("MH glitch blocks", test_mh_glitch_blocks)
    
    def test_mh_manager_integration():
        mgr = FilterManager()
        filt = mgr.get_filter("Max Headroom")
        if not filt:
            return False
        mgr.enable_filter("Max Headroom")
        frame = np.random.randint(0, 255, (100, 100, 3), np.uint8)
        result = mgr.process(frame)
        return result.shape == frame.shape
    test("MH manager integration", test_mh_manager_integration)
    
    def test_mh_performance():
        filt = MaxHeadroomFilter()
        filt.enable()
        filt.set_param("intensity", 1.0)
        frame = np.random.randint(0, 255, (480, 640, 3), np.uint8)
        
        import time
        start = time.time()
        for _ in range(10):
            result = filt.process(frame)
        elapsed = time.time() - start
        
        print(f" ({elapsed*1000/10:.1f}ms/frame)", end="")
        return elapsed < 2.0  # Under 200ms per frame
    test("MH filter performance", test_mh_performance)
    
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