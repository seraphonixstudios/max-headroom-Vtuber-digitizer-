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
    from filters.mocap_viz import MoCapVizFilter
    
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
    
    # ===========================================
    # MO-CAP VIZ FILTER v3.4
    # ===========================================
    print("\n>>> MO-CAP VIZ FILTER <<<")
    
    _mocap_landmarks = [(160 + i*5, 140 + i) for i in range(68)]
    _mocap_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    test("MoCap init", lambda: MoCapVizFilter() is not None)
    
    def test_mocap_disabled():
        filt = MoCapVizFilter()
        result = filt.process(_mocap_frame.copy(), {"landmarks": _mocap_landmarks})
        return np.array_equal(result, _mocap_frame)
    test("MoCap disabled passes through", test_mocap_disabled)
    
    def test_mocap_wireframe():
        filt = MoCapVizFilter()
        filt.enable()
        filt.set_param("wireframe", True)
        filt.set_param("tracking_points", False)
        filt.set_param("pose_axes", False)
        filt.set_param("skeleton", False)
        filt.set_param("labels", False)
        result = filt.process(_mocap_frame.copy(), {"landmarks": _mocap_landmarks})
        return result.shape == _mocap_frame.shape and not np.array_equal(result, _mocap_frame)
    test("MoCap wireframe overlay", test_mocap_wireframe)
    
    def test_mocap_tracking_points():
        filt = MoCapVizFilter()
        filt.enable()
        filt.set_param("wireframe", False)
        filt.set_param("tracking_points", True)
        filt.set_param("pose_axes", False)
        result = filt.process(_mocap_frame.copy(), {"landmarks": _mocap_landmarks})
        return result.shape == _mocap_frame.shape and np.max(result) > 0
    test("MoCap tracking points", test_mocap_tracking_points)
    
    def test_mocap_pose_axes():
        filt = MoCapVizFilter()
        filt.enable()
        filt.set_param("wireframe", False)
        filt.set_param("tracking_points", False)
        filt.set_param("pose_axes", True)
        head_pose = {"rotation": [15.0, -10.0, 5.0], "translation": [0, 0, 100]}
        result = filt.process(_mocap_frame.copy(), {
            "landmarks": _mocap_landmarks,
            "head_pose": head_pose,
        })
        return result.shape == _mocap_frame.shape
    test("MoCap pose axes", test_mocap_pose_axes)
    
    def test_mocap_skeleton():
        filt = MoCapVizFilter()
        filt.enable()
        filt.set_param("wireframe", False)
        filt.set_param("tracking_points", False)
        filt.set_param("pose_axes", False)
        filt.set_param("skeleton", True)
        result = filt.process(_mocap_frame.copy(), {"landmarks": _mocap_landmarks})
        return result.shape == _mocap_frame.shape and not np.array_equal(result, _mocap_frame)
    test("MoCap skeleton overlay", test_mocap_skeleton)
    
    def test_mocap_labels():
        filt = MoCapVizFilter()
        filt.enable()
        filt.set_param("wireframe", False)
        filt.set_param("tracking_points", False)
        filt.set_param("pose_axes", False)
        filt.set_param("skeleton", False)
        filt.set_param("labels", True)
        result = filt.process(_mocap_frame.copy(), {"landmarks": _mocap_landmarks})
        return result.shape == _mocap_frame.shape
    test("MoCap landmark labels", test_mocap_labels)
    
    def test_mocap_all_features():
        filt = MoCapVizFilter()
        filt.enable()
        filt.set_param("wireframe", True)
        filt.set_param("tracking_points", True)
        filt.set_param("pose_axes", True)
        filt.set_param("skeleton", True)
        filt.set_param("labels", True)
        head_pose = {"rotation": [10.0, -5.0, 2.0], "translation": [0, 0, 100]}
        result = filt.process(_mocap_frame.copy(), {
            "landmarks": _mocap_landmarks,
            "head_pose": head_pose,
        })
        return result.shape == _mocap_frame.shape and np.max(result) > 0
    test("MoCap all features combined", test_mocap_all_features)
    
    def test_mocap_style_presets():
        filt = MoCapVizFilter()
        for style in ("tech", "neon", "dark", "minimal"):
            filt.set_style(style)
            if filt.params["style"] != style:
                return False
        return True
    test("MoCap style presets", test_mocap_style_presets)
    
    def test_mocap_intensity_scales():
        filt = MoCapVizFilter()
        filt.enable()
        filt.set_param("intensity", 0.0)
        r0 = filt.process(_mocap_frame.copy(), {"landmarks": _mocap_landmarks})
        filt.set_param("intensity", 1.0)
        r1 = filt.process(_mocap_frame.copy(), {"landmarks": _mocap_landmarks})
        # Intensity 0 = pass-through
        return np.array_equal(r0, _mocap_frame) and not np.array_equal(r1, _mocap_frame)
    test("MoCap intensity zero = passthrough", test_mocap_intensity_scales)
    
    def test_mocap_no_landmarks():
        filt = MoCapVizFilter()
        filt.enable()
        result = filt.process(_mocap_frame.copy(), {})
        return np.array_equal(result, _mocap_frame)
    test("MoCap no landmarks = passthrough", test_mocap_no_landmarks)
    
    # ===========================================
    # MAX HEADROOM v3.4 DIGITAL GRAPHICS MODES
    # ===========================================
    print("\n>>> MAX HEADROOM v3.4 DIGITAL GRAPHICS <<<")
    
    def test_mh_cel_shading():
        filt = MaxHeadroomFilter()
        filt.enable()
        filt.set_param("intensity", 1.0)
        # Disable other effects
        filt.set_param("monochrome", False)
        filt.set_param("scanlines", False)
        filt.set_param("film_grain", False)
        filt.set_param("glitch_blocks", False)
        filt.set_param("grid", False)
        filt.set_param("data_overlay", False)
        filt.set_param("vignette", False)
        filt.set_param("posterize", False)
        # Enable cel-shading
        filt.set_param("cel_shading", True)
        filt.set_param("cel_shading_k", 8)
        filt.set_param("cel_edge_style", "canny")
        frame = np.random.randint(0, 255, (100, 100, 3), np.uint8)
        result = filt.process(frame)
        return result.shape == frame.shape and result.dtype == np.uint8
    test("MH cel-shading (canny)", test_mh_cel_shading)
    
    def test_mh_cel_shading_sobel():
        filt = MaxHeadroomFilter()
        filt.enable()
        filt.set_param("intensity", 1.0)
        filt.set_param("monochrome", False)
        filt.set_param("scanlines", False)
        filt.set_param("film_grain", False)
        filt.set_param("glitch_blocks", False)
        filt.set_param("grid", False)
        filt.set_param("data_overlay", False)
        filt.set_param("vignette", False)
        filt.set_param("posterize", False)
        filt.set_param("cel_shading", True)
        filt.set_param("cel_edge_style", "sobel")
        frame = np.random.randint(0, 255, (100, 100, 3), np.uint8)
        result = filt.process(frame)
        return result.shape == frame.shape
    test("MH cel-shading (sobel)", test_mh_cel_shading_sobel)
    
    def test_mh_bloom():
        filt = MaxHeadroomFilter()
        filt.enable()
        filt.set_param("intensity", 1.0)
        filt.set_param("monochrome", False)
        filt.set_param("scanlines", False)
        filt.set_param("film_grain", False)
        filt.set_param("glitch_blocks", False)
        filt.set_param("grid", False)
        filt.set_param("data_overlay", False)
        filt.set_param("vignette", False)
        filt.set_param("posterize", False)
        filt.set_param("bloom", True)
        filt.set_param("bloom_threshold", 0.6)
        filt.set_param("bloom_intensity", 0.5)
        frame = np.random.randint(0, 255, (100, 100, 3), np.uint8)
        result = filt.process(frame)
        return result.shape == frame.shape
    test("MH bloom glow", test_mh_bloom)
    
    def test_mh_bloom_tinted():
        filt = MaxHeadroomFilter()
        filt.enable()
        filt.set_param("intensity", 1.0)
        filt.set_param("monochrome", False)
        filt.set_param("scanlines", False)
        filt.set_param("film_grain", False)
        filt.set_param("glitch_blocks", False)
        filt.set_param("grid", False)
        filt.set_param("data_overlay", False)
        filt.set_param("vignette", False)
        filt.set_param("posterize", False)
        filt.set_param("bloom", True)
        filt.set_param("bloom_color", [0, 200, 255])
        frame = np.random.randint(0, 255, (100, 100, 3), np.uint8)
        result = filt.process(frame)
        return result.shape == frame.shape
    test("MH bloom tinted cyan", test_mh_bloom_tinted)
    
    def test_mh_comic_style():
        filt = MaxHeadroomFilter()
        filt.enable()
        filt.set_param("intensity", 1.0)
        filt.set_param("monochrome", False)
        filt.set_param("scanlines", False)
        filt.set_param("film_grain", False)
        filt.set_param("glitch_blocks", False)
        filt.set_param("grid", False)
        filt.set_param("data_overlay", False)
        filt.set_param("vignette", False)
        filt.set_param("posterize", False)
        filt.set_param("comic_style", True)
        filt.set_param("comic_dots", 0.2)
        frame = np.random.randint(0, 255, (100, 100, 3), np.uint8)
        result = filt.process(frame)
        return result.shape == frame.shape
    test("MH comic style", test_mh_comic_style)
    
    def test_mh_neon_edges():
        filt = MaxHeadroomFilter()
        filt.enable()
        filt.set_param("intensity", 1.0)
        filt.set_param("monochrome", False)
        filt.set_param("scanlines", False)
        filt.set_param("film_grain", False)
        filt.set_param("glitch_blocks", False)
        filt.set_param("grid", False)
        filt.set_param("data_overlay", False)
        filt.set_param("vignette", False)
        filt.set_param("posterize", False)
        filt.set_param("neon_edges", True)
        filt.set_param("neon_edge_color", [255, 100, 255])
        frame = np.random.randint(0, 255, (100, 100, 3), np.uint8)
        result = filt.process(frame)
        return result.shape == frame.shape
    test("MH neon edges", test_mh_neon_edges)
    
    def test_mh_ink_edges():
        filt = MaxHeadroomFilter()
        filt.enable()
        filt.set_param("intensity", 1.0)
        filt.set_param("monochrome", False)
        filt.set_param("scanlines", False)
        filt.set_param("film_grain", False)
        filt.set_param("glitch_blocks", False)
        filt.set_param("grid", False)
        filt.set_param("data_overlay", False)
        filt.set_param("vignette", False)
        filt.set_param("posterize", False)
        filt.set_param("ink_edges", True)
        frame = np.random.randint(0, 255, (100, 100, 3), np.uint8)
        result = filt.process(frame)
        return result.shape == frame.shape
    test("MH ink edges", test_mh_ink_edges)
    
    def test_mh_cel_shading_flat_colors():
        filt = MaxHeadroomFilter()
        filt.enable()
        filt.set_param("intensity", 1.0)
        filt.set_param("monochrome", False)
        filt.set_param("scanlines", False)
        filt.set_param("film_grain", False)
        filt.set_param("glitch_blocks", False)
        filt.set_param("grid", False)
        filt.set_param("data_overlay", False)
        filt.set_param("vignette", False)
        filt.set_param("posterize", False)
        filt.set_param("cel_shading", True)
        filt.set_param("cel_shading_k", 4)
        frame = np.random.randint(0, 255, (100, 100, 3), np.uint8)
        result = filt.process(frame)
        unique = len(np.unique(result.reshape(-1, 3), axis=0))
        # k=4 quantization + edge black = ~5-8 unique colors
        return unique <= 12
    test("MH cel-shading reduces to few colors", test_mh_cel_shading_flat_colors)
    
    def test_mh_all_disabled_passthrough():
        filt = MaxHeadroomFilter()
        filt.enable()
        filt.set_param("intensity", 1.0)
        filt.set_param("cel_shading", False)
        filt.set_param("bloom", False)
        filt.set_param("comic_style", False)
        filt.set_param("neon_edges", False)
        filt.set_param("ink_edges", False)
        filt.set_param("posterize", False)
        filt.set_param("monochrome", False)
        filt.set_param("scanlines", False)
        filt.set_param("film_grain", False)
        filt.set_param("glitch_blocks", False)
        filt.set_param("grid", False)
        filt.set_param("data_overlay", False)
        filt.set_param("vignette", False)
        frame = np.random.randint(0, 255, (100, 100, 3), np.uint8)
        result = filt.process(frame)
        # Without monochrome and vfx, result should still change minimally
        return result.shape == frame.shape
    test("MH v3.4 modes all off = base pipeline", test_mh_all_disabled_passthrough)
    
    # ===========================================
    # AR OVERLAY v3.4 NEW STICKERS
    # ===========================================
    print("\n>>> AR OVERLAY v3.4 NEW STICKERS <<<")
    
    _ar_landmarks = [(160 + i*5, 140 + i) for i in range(68)]
    
    def test_ar_wireframe():
        filt = AROverlayFilter()
        filt.enable()
        filt.clear_stickers()
        filt.add_sticker("wireframe", color=(0, 200, 255), alpha=0.5)
        frame = np.zeros((480, 640, 3), np.uint8)
        result = filt.process(frame, {"landmarks": _ar_landmarks})
        return result.shape == frame.shape
    test("AR wireframe sticker", test_ar_wireframe)
    
    def test_ar_wireframe_draws_lines():
        filt = AROverlayFilter()
        filt.enable()
        filt.clear_stickers()
        filt.add_sticker("wireframe", color=(0, 255, 255), alpha=1.0)
        frame = np.zeros((480, 640, 3), np.uint8)
        result = filt.process(frame, {"landmarks": _ar_landmarks})
        return np.max(result) > 0
    test("AR wireframe draws visible pixels", test_ar_wireframe_draws_lines)
    
    def test_ar_tracking_dots():
        filt = AROverlayFilter()
        filt.enable()
        filt.clear_stickers()
        filt.add_sticker("tracking_dots", color=(0, 255, 255), size=3)
        frame = np.zeros((480, 640, 3), np.uint8)
        result = filt.process(frame, {"landmarks": _ar_landmarks})
        return result.shape == frame.shape
    test("AR tracking dots sticker", test_ar_tracking_dots)
    
    def test_ar_tracking_dots_draws():
        filt = AROverlayFilter()
        filt.enable()
        filt.clear_stickers()
        filt.add_sticker("tracking_dots", color=(0, 255, 255), size=3)
        frame = np.zeros((480, 640, 3), np.uint8)
        result = filt.process(frame, {"landmarks": _ar_landmarks})
        return np.max(result) > 0
    test("AR tracking dots visible", test_ar_tracking_dots_draws)
    
    def test_ar_both_new_stickers():
        filt = AROverlayFilter()
        filt.enable()
        filt.clear_stickers()
        filt.add_sticker("wireframe", color=(0, 200, 255))
        filt.add_sticker("tracking_dots", color=(0, 255, 255))
        frame = np.zeros((480, 640, 3), np.uint8)
        result = filt.process(frame, {"landmarks": _ar_landmarks})
        return result.shape == frame.shape and np.max(result) > 0
    test("AR wireframe + tracking dots combined", test_ar_both_new_stickers)
    
    def test_ar_wireframe_disabled():
        filt = AROverlayFilter()
        filt.enable()
        filt.clear_stickers()
        filt.add_sticker("wireframe", enabled=False, color=(0, 200, 255))
        frame = np.zeros((480, 640, 3), np.uint8)
        result = filt.process(frame, {"landmarks": _ar_landmarks})
        return np.max(result) == 0
    test("AR disabled sticker = no draw", test_ar_wireframe_disabled)
    
    def test_ar_no_landmarks():
        filt = AROverlayFilter()
        filt.enable()
        filt.clear_stickers()
        filt.add_sticker("wireframe")
        frame = np.zeros((480, 640, 3), np.uint8)
        result = filt.process(frame, {})
        return np.max(result) == 0
    test("AR no landmarks = passthrough", test_ar_no_landmarks)
    
    # ===========================================
    # FILTER MANAGER INTEGRATION v3.4
    # ===========================================
    print("\n>>> FILTER MANAGER v3.4 INTEGRATION <<<")
    
    def test_manager_has_mocap():
        mgr = FilterManager()
        filt = mgr.get_filter("MoCap Viz")
        return filt is not None
    test("Manager has MoCap Viz filter", test_manager_has_mocap)
    
    def test_manager_has_seven():
        mgr = FilterManager()
        return len(mgr.filters) == 7
    test("Manager has exactly 7 filters", test_manager_has_seven)
    
    def test_mocap_manager_enable():
        mgr = FilterManager()
        mgr.enable_filter("MoCap Viz")
        filt = mgr.get_filter("MoCap Viz")
        return filt is not None and filt.enabled
    test("Manager enables MoCap Viz", test_mocap_manager_enable)
    
    def test_mocap_manager_pipeline():
        mgr = FilterManager()
        mgr.enable_filter("MoCap Viz")
        filt = mgr.get_filter("MoCap Viz")
        filt.set_param("wireframe", True)
        filt.set_param("tracking_points", True)
        frame = np.random.randint(0, 255, (100, 100, 3), np.uint8)
        landmarks = [(160 + i*5, 140 + i) for i in range(68)]
        result = mgr.process(frame, landmarks=landmarks)
        return result.shape == frame.shape
    test("MoCap Viz in filter pipeline", test_mocap_manager_pipeline)
    
    def test_manager_all_filters_enabled():
        mgr = FilterManager()
        status = mgr.get_all_status()
        names = [s["name"] for s in status]
        expected = ["Max Headroom", "Color Grading", "Skin Smoothing",
                    "Face Morph", "Background", "AR Overlay", "MoCap Viz"]
        return all(n in names for n in expected)
    test("Manager lists all 7 filters", test_manager_all_filters_enabled)
    
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