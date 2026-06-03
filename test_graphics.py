#!/usr/bin/env python3
"""
Max Headroom - SOTA Graphics Engine Tests
Tests all advanced graphics operations with quality + performance validation.
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
    print("=" * 70)
    print(" SOTA GRAPHICS ENGINE TESTS")
    print("=" * 70)
    
    from filters.graphics_engine import (
        ColorSpace, AlphaCompositor, PyramidBlend, CLAHEEnhancer,
        ColorQuantizer, Dithering, GuidedFilter, TemporalSmoother,
        FilmGrain, ChromaticAberration, ScanlineEffects,
        GPUGraphicsEngine, GraphicsBenchmark,
        CelShading, BloomEffect, StylizedEdges,
    )
    
    # Create test frames
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    frame_small = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    print("\n>>> COLOR SPACE UTILITIES <<<")
    
    test("BGR to Lab", lambda: ColorSpace.bgr_to_lab(frame).shape == frame.shape)
    test("Lab to BGR", lambda: ColorSpace.lab_to_bgr(ColorSpace.bgr_to_lab(frame)).shape == frame.shape)
    test("BGR to Luv", lambda: ColorSpace.bgr_to_luv(frame).shape == frame.shape)
    test("BGR to YCrCb", lambda: ColorSpace.bgr_to_y_cr_cb(frame).shape == frame.shape)
    
    print("\n>>> ALPHA COMPOSITING <<<")
    
    def test_alpha_composite():
        alpha = np.ones((100, 100), dtype=np.float32) * 0.5
        result = AlphaCompositor.composite(frame_small, frame_small, alpha)
        return result.shape == frame_small.shape and result.dtype == np.uint8
    test("Alpha composite basic", test_alpha_composite)
    
    def test_alpha_with_mask():
        alpha = np.ones((100, 100), dtype=np.float32) * 0.5
        mask = np.ones((100, 100), dtype=np.float32)
        mask[50:, :] = 0
        result = AlphaCompositor.composite(frame_small, frame_small, alpha, mask)
        return result.shape == frame_small.shape
    test("Alpha composite with mask", test_alpha_with_mask)
    
    def test_alpha_gamma_correct():
        # Dark overlay should not create dark edges
        bg = np.full((100, 100, 3), 128, dtype=np.uint8)
        fg = np.full((100, 100, 3), 0, dtype=np.uint8)
        alpha = np.ones((100, 100), dtype=np.float32) * 0.5
        result = AlphaCompositor.composite(bg, fg, alpha)
        # Middle value should be ~64 in linear, but gamma corrected
        return np.mean(result) < 100 and np.mean(result) > 50
    test("Gamma-correct darkening", test_alpha_gamma_correct)
    
    print("\n>>> PYRAMID BLENDING <<<")
    
    def test_pyramid_build():
        pyramid = PyramidBlend.build_laplacian_pyramid(frame_small, levels=3)
        return len(pyramid) == 4 and pyramid[0].shape == frame_small.shape
    test("Laplacian pyramid build", test_pyramid_build)
    
    def test_pyramid_collapse():
        pyramid = PyramidBlend.build_laplacian_pyramid(frame_small, levels=3)
        reconstructed = PyramidBlend.collapse_laplacian_pyramid(pyramid)
        return reconstructed.shape == frame_small.shape
    test("Pyramid collapse", test_pyramid_collapse)
    
    def test_pyramid_blend():
        mask = np.ones((100, 100), dtype=np.float32) * 0.5
        result = PyramidBlend.blend(frame_small, frame_small, mask, levels=2)
        return result.shape == frame_small.shape
    test("Multi-scale blend", test_pyramid_blend)
    
    print("\n>>> CLAHE ENHANCEMENT <<<")
    
    def test_clahe_apply():
        clahe = CLAHEEnhancer(clip_limit=2.0)
        result = clahe.apply(frame_small)
        return result.shape == frame_small.shape and result.dtype == np.uint8
    test("CLAHE Lab enhancement", test_clahe_apply)
    
    def test_clahe_contrast_increase():
        # Low contrast image should gain contrast
        low_contrast = np.full((100, 100, 3), 128, dtype=np.uint8)
        low_contrast[40:60, 40:60] = 140  # Slight variation
        clahe = CLAHEEnhancer(clip_limit=3.0)
        result = clahe.apply(low_contrast)
        return np.std(result) > np.std(low_contrast)
    test("CLAHE increases local contrast", test_clahe_contrast_increase)
    
    print("\n>>> COLOR QUANTIZATION <<<")
    
    def test_quantize():
        result = ColorQuantizer.quantize(frame_small, k=4)
        return result.shape == frame_small.shape
    test("K-means quantization", test_quantize)
    
    def test_quantize_reduces_colors():
        result = ColorQuantizer.quantize(frame_small, k=4)
        unique = len(np.unique(result.reshape(-1, 3), axis=0))
        return unique <= 4
    test("Quantize reduces to k colors", test_quantize_reduces_colors)
    
    def test_posterize():
        result = ColorQuantizer.posterize(frame_small, levels=4)
        unique = len(np.unique(result))
        return unique <= 4
    test("Posterize bit reduction", test_posterize)
    
    print("\n>>> DITHERING <<<")
    
    def test_ordered_dither():
        result = Dithering.ordered_dither(frame_small, levels=4)
        return result.shape == frame_small.shape and result.dtype == np.uint8
    test("Ordered Bayer dither", test_ordered_dither)
    
    def test_floyd_steinberg():
        result = Dithering.floyd_steinberg(frame_small, levels=4)
        return result.shape == frame_small.shape
    test("Floyd-Steinberg dither", test_floyd_steinberg)
    
    def test_noise_dither():
        result = Dithering.noise_dither(frame_small, levels=4)
        return result.shape == frame_small.shape
    test("Noise dither", test_noise_dither)
    
    def test_halftone():
        result = Dithering.halftone(frame_small, dot_size=8)
        return result.shape == frame_small.shape
    test("Halftone effect", test_halftone)
    
    print("\n>>> GUIDED FILTER <<<")
    
    def test_guided_filter():
        result = GuidedFilter.apply(frame_small, frame_small, radius=4)
        return result.shape == frame_small.shape
    test("Guided filter smoothing", test_guided_filter)
    
    def test_guided_filter_preserves_edges():
        # Image with sharp edge
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[:, :25] = 255
        smoothed = GuidedFilter.apply(img, img, radius=8)
        # Edge should still be sharp-ish
        edge_diff = np.abs(int(smoothed[25, 24, 0]) - int(smoothed[25, 26, 0]))
        return edge_diff > 100  # Significant edge preserved
    test("Guided filter preserves edges", test_guided_filter_preserves_edges)
    
    print("\n>>> TEMPORAL SMOOTHING <<<")
    
    def test_temporal_smoother():
        smoother = TemporalSmoother(alpha=0.7)
        f1 = np.full((50, 50, 3), 100, dtype=np.uint8)
        f2 = np.full((50, 50, 3), 200, dtype=np.uint8)
        r1 = smoother.apply(f1)
        r2 = smoother.apply(f2)
        return r1.shape == f1.shape and np.mean(r2) < 200 and np.mean(r2) > 100
    test("Temporal exponential smoothing", test_temporal_smoother)
    
    print("\n>>> FILM GRAIN <<<")
    
    def test_film_grain():
        result = FilmGrain.apply(frame_small, intensity=0.1)
        return result.shape == frame_small.shape and not np.array_equal(result, frame_small)
    test("Film grain adds noise", test_film_grain)
    
    def test_film_grain_luma_only():
        # Compare noise difference between luma-only and color grain
        result_luma = FilmGrain.apply(frame_small, intensity=0.3, color=False)
        result_color = FilmGrain.apply(frame_small, intensity=0.3, color=True)
        # Luma-only: B/G channels should differ by same amount as G/R
        luma_bg_diff = np.std((result_luma[:,:,0].astype(float) - result_luma[:,:,1].astype(float)))
        color_bg_diff = np.std((result_color[:,:,0].astype(float) - result_color[:,:,1].astype(float)))
        # Luma grain creates much smaller B-G differences than color grain
        return luma_bg_diff < color_bg_diff
    test("Film grain luma-only preserves color", test_film_grain_luma_only)
    
    print("\n>>> CHROMATIC ABERRATION <<<")
    
    def test_chromatic_aberration():
        result = ChromaticAberration.apply(frame_small, strength=2.0)
        return result.shape == frame_small.shape
    test("Radial chromatic aberration", test_chromatic_aberration)
    
    def test_chromatic_color_shift():
        # Gradient image - CA should create divergence at edges
        grad = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(100):
            grad[:, i, 2] = int(255 * i / 100)  # Red gradient horizontal
            grad[:, i, 0] = int(255 * (1 - i / 100))  # Blue gradient opposite
        result = ChromaticAberration.apply(grad, strength=5.0)
        # CA should change the image (different from input)
        diff = np.mean(np.abs(result.astype(float) - grad.astype(float)))
        return diff > 0.5  # Measurable change
    test("CA shifts colors at edges", test_chromatic_color_shift)
    
    print("\n>>> SCANLINE EFFECTS <<<")
    
    def test_crt_scanlines():
        result = ScanlineEffects.crt_scanlines(frame_small)
        return result.shape == frame_small.shape
    test("CRT scanlines", test_crt_scanlines)
    
    def test_rgb_phosphor():
        result = ScanlineEffects.rgb_phosphor(frame_small, strength=0.3)
        return result.shape == frame_small.shape
    test("RGB phosphor triads", test_rgb_phosphor)
    
    def test_interlace_flicker():
        result_even = ScanlineEffects.interlace_flicker(frame_small, frame_id=0)
        result_odd = ScanlineEffects.interlace_flicker(frame_small, frame_id=1)
        return not np.array_equal(result_even, result_odd)
    test("Interlace flicker alternates", test_interlace_flicker)
    
    print("\n>>> GPU ENGINE <<<")
    
    def test_gpu_engine_init():
        engine = GPUGraphicsEngine()
        return engine is not None
    test("GPU engine initializes", test_gpu_engine_init)
    
    def test_gpu_gaussian_blur():
        engine = GPUGraphicsEngine()
        result = engine.gaussian_blur(frame_small, (5, 5))
        return result.shape == frame_small.shape
    test("GPU Gaussian blur (or CPU fallback)", test_gpu_gaussian_blur)
    
    print("\n>>> PERFORMANCE BENCHMARKS <<<")
    
    def test_benchmark():
        bench = GraphicsBenchmark()
        results = bench.benchmark_all(frame_small)
        print(f"\n    Benchmarks (ms):")
        for name, ms in results.items():
            print(f"      {name}: {ms:.2f}ms")
        return len(results) >= 9 and all(v < 500 for v in results.values())
    test("All operations under 500ms", test_benchmark)
    
    print("\n>>> CEL-SHADING / TOON RENDERING <<<")
    
    def test_cel_shading_default():
        result = CelShading.apply(frame_small)
        return result.shape == frame_small.shape and result.dtype == np.uint8
    test("CelShading default", test_cel_shading_default)
    
    def test_cel_shading_flat_colors():
        result = CelShading.apply(frame_small, quantize_levels=4)
        unique_colors = len(np.unique(result.reshape(-1, 3), axis=0))
        # 4 levels per channel = 4^3 combinations, but edges add black
        return unique_colors <= 70
    test("CelShading reduces colors", test_cel_shading_flat_colors)
    
    def test_cel_shading_edges_present():
        result = CelShading.apply(frame_small, edge_threshold1=50, edge_threshold2=100)
        # Should have black edge pixels from Canny
        black_pixels = np.sum(np.all(result == 0, axis=-1))
        return black_pixels > 0
    test("CelShading draws edge outlines", test_cel_shading_edges_present)
    
    def test_cel_shading_quantized():
        result = CelShading.apply_color_quantized(frame_small, k=6, edge_style="canny")
        unique_colors = len(np.unique(result.reshape(-1, 3), axis=0))
        return result.shape == frame_small.shape and 2 <= unique_colors <= 15
    test("CelShading quantized (k=6)", test_cel_shading_quantized)
    
    def test_cel_shading_sobel_edges():
        result = CelShading.apply_color_quantized(frame_small, k=8, edge_style="sobel")
        return result.shape == frame_small.shape
    test("CelShading sobel edges", test_cel_shading_sobel_edges)
    
    def test_cel_shading_no_edges():
        result = CelShading.apply_color_quantized(frame_small, k=8, edge_style="none")
        unique = len(np.unique(result.reshape(-1, 3), axis=0))
        return unique <= 10
    test("CelShading no edges (flat poster)", test_cel_shading_no_edges)
    
    def test_cel_shading_comic():
        result = CelShading.comic_style(frame_small, k=6, dot_density=0.1)
        return result.shape == frame_small.shape and result.dtype == np.uint8
    test("CelShading comic style", test_cel_shading_comic)
    
    def test_cel_shading_comic_reduces_colors():
        result = CelShading.comic_style(frame_small, k=4, dot_density=0.2)
        unique = len(np.unique(result.reshape(-1, 3), axis=0))
        return unique <= 10
    test("Comic style reduces to <=k colors + dots", test_cel_shading_comic_reduces_colors)
    
    print("\n>>> BLOOM / GLOW EFFECTS <<<")
    
    def test_bloom_default():
        result = BloomEffect.apply(frame_small)
        return result.shape == frame_small.shape and result.dtype == np.uint8
    test("Bloom default", test_bloom_default)
    
    def test_bloom_brightens():
        bright = np.full((100, 100, 3), 200, dtype=np.uint8)
        result = BloomEffect.apply(bright, threshold=0.5, blur_radius=9, intensity=0.8)
        # Bloom should increase brightness in bright areas
        return np.mean(result) >= np.mean(bright) - 1
    test("Bloom brightens highlights", test_bloom_brightens)
    
    def test_bloom_dim_unchanged():
        dim = np.full((100, 100, 3), 30, dtype=np.uint8)
        result = BloomEffect.apply(dim, threshold=0.7, intensity=0.5)
        # Dim image should be mostly unchanged (bloom only on brights)
        return np.abs(np.mean(result.astype(float) - dim.astype(float))) < 5
    test("Bloom leaves darks unchanged", test_bloom_dim_unchanged)
    
    def test_bloom_tinted():
        result = BloomEffect.apply(frame_small, threshold=0.5, intensity=0.6,
                                   color=(0, 200, 255))
        return result.shape == frame_small.shape
    test("Bloom with tint color", test_bloom_tinted)
    
    def test_bloom_glow_edges():
        result = BloomEffect.glow_edges(frame_small, edge_intensity=0.5,
                                        glow_color=(0, 200, 255))
        return result.shape == frame_small.shape
    test("Bloom edge glow", test_bloom_glow_edges)
    
    def test_bloom_glow_edges_creates_color():
        # On a dark uniform image, edge glow should add bright pixels
        dark = np.full((100, 100, 3), 10, dtype=np.uint8)
        dark[40:60, 40:60] = 200  # Bright square creates edges
        result = BloomEffect.glow_edges(dark, edge_intensity=1.0, blur_radius=5,
                                        glow_color=(0, 255, 255))
        return np.max(result) > 50
    test("Edge glow adds visible color", test_bloom_glow_edges_creates_color)
    
    print("\n>>> STYLIZED EDGES <<<")
    
    def test_ink_edges_default():
        result = StylizedEdges.ink_edges(frame_small)
        return result.shape == frame_small.shape and result.dtype == np.uint8
    test("Ink edges default", test_ink_edges_default)
    
    def test_ink_edges_darkens():
        result = StylizedEdges.ink_edges(frame_small, strength=2.0)
        # Ink edges should add black pixels
        black_before = np.sum(np.all(frame_small == 0, axis=-1))
        black_after = np.sum(np.all(result == 0, axis=-1))
        return black_after >= black_before
    test("Ink edges add black outlines", test_ink_edges_darkens)
    
    def test_ink_edges_varying_strength():
        weak = StylizedEdges.ink_edges(frame_small, strength=0.5)
        strong = StylizedEdges.ink_edges(frame_small, strength=3.0)
        weak_black = np.sum(np.all(weak == 0, axis=-1))
        strong_black = np.sum(np.all(strong == 0, axis=-1))
        return strong_black >= weak_black
    test("Ink edge strength increases darkness", test_ink_edges_varying_strength)
    
    def test_colored_edges_default():
        result = StylizedEdges.colored_edges(frame_small)
        return result.shape == frame_small.shape
    test("Colored edges default", test_colored_edges_default)
    
    def test_colored_edges_uses_color():
        result = StylizedEdges.colored_edges(frame_small, edge_color=(255, 0, 255))
        # Should have magenta pixels where edges exist
        magenta_pixels = np.sum(
            (result[:, :, 0] > 200) & (result[:, :, 1] < 50) & (result[:, :, 2] > 200)
        )
        return magenta_pixels > 0
    test("Colored edges uses specified color", test_colored_edges_uses_color)
    
    def test_colored_edges_soft_threshold():
        result = StylizedEdges.colored_edges(frame_small, threshold1=30, threshold2=60)
        result2 = StylizedEdges.colored_edges(frame_small, threshold1=150, threshold2=200)
        # Lower threshold = more edges
        edges_low = np.sum(np.all(result != frame_small, axis=-1))
        edges_high = np.sum(np.all(result2 != frame_small, axis=-1))
        return edges_low >= edges_high
    test("Lower threshold = more colored edges", test_colored_edges_soft_threshold)
    
    print("\n>>> PIPELINE INTEGRATION <<<")
    
    def test_filter_uses_engine():
        from filters import MaxHeadroomFilter
        filt = MaxHeadroomFilter()
        filt.enable()
        filt.set_param("intensity", 1.0)
        result = filt.process(frame_small)
        return result is not None and len(result.shape) == 3 and result.shape[2] == 3
    test("Max Headroom filter with SOTA engine", test_filter_uses_engine)
    
    def test_filter_posterization():
        from filters import MaxHeadroomFilter
        # Test posterization in isolation
        filt = MaxHeadroomFilter()
        filt.enable()
        filt.set_param("intensity", 1.0)
        filt.set_param("posterize", True)
        filt.set_param("posterize_levels", 4)
        # Disable effects that reintroduce colors
        filt.set_param("film_grain", False)
        filt.set_param("data_overlay", False)
        filt.set_param("glitch_blocks", False)
        filt.set_param("grid", False)
        filt.set_param("scanlines", False)
        filt.set_param("vignette", False)
        result = filt.process(frame_small)
        # Posterization + dithering at end of pipeline should limit colors
        unique_r = len(np.unique(result[:, :, 2]))
        unique_g = len(np.unique(result[:, :, 1]))
        unique_b = len(np.unique(result[:, :, 0]))
        # Should have fewer unique values than original (which has ~100)
        return unique_r < 80 and unique_g < 80 and unique_b < 80
    test("MH filter posterization active", test_filter_posterization)
    
    # Summary
    print("\n" + "=" * 70)
    print(" GRAPHICS ENGINE TEST RESULTS")
    print("=" * 70)
    print(f"  PASSED: {TEST_RESULTS['pass']}")
    print(f"  FAILED: {TEST_RESULTS['fail']}")
    print(f"  TOTAL:  {TEST_RESULTS['pass'] + TEST_RESULTS['fail']}")
    print("=" * 70)
    
    return TEST_RESULTS["fail"] == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
