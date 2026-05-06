#!/usr/bin/env python3
"""
Test Person Segmentation and Background Removal
"""
import sys, os, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_segmentation():
    print("=" * 60)
    print(" PERSON SEGMENTATION TEST")
    print("=" * 60)
    
    from person_segmentation import PersonSegmenter
    
    passed = 0
    failed = 0
    
    def check(name, condition):
        nonlocal passed, failed
        print(f"  [{name}]...", end=" ")
        if condition:
            print("PASS")
            passed += 1
        else:
            print("FAIL")
            failed += 1
    
    # Test 1: Init
    seg = PersonSegmenter()
    check("Segmenter init", seg is not None)
    
    # Test 2: Segment frame with face
    frame = np.random.randint(0, 255, (480, 640, 3), np.uint8)
    face_rect = (200, 150, 240, 240)
    mask = seg.segment(frame, face_rect)
    check("Segment returns mask", mask is not None)
    check("Mask shape matches frame", mask.shape == (480, 640))
    check("Mask values in range", mask.min() >= 0 and mask.max() <= 1)
    check("Mask has non-zero values", mask.sum() > 0)
    
    # Test 3: Segment without face
    seg2 = PersonSegmenter()
    mask2 = seg2.segment(frame, None)
    check("Segment without face works", mask2 is not None)
    
    # Test 4: Remove background
    seg3 = PersonSegmenter()
    result = seg3.remove_background(frame, face_rect)
    check("Remove bg returns BGR", result.shape == (480, 640, 3))
    check("Green screen present", np.any(result[:, :, 1] == 255))
    
    # Test 5: Background removal filter
    from filters import BackgroundRemovalFilter, FilterMode
    filt = BackgroundRemovalFilter(FilterMode.OFF)
    check("Filter init", filt is not None)
    
    filt.enable()
    out = filt.process(frame, {"face_rect": face_rect})
    check("Filter process works", out is not None)
    check("Filter output correct shape", out.shape == frame.shape)
    
    # Test 6: Mode switching
    filt.set_mode("blur")
    check("Blur mode", filt.params["mode"] == "blur")
    filt.set_mode("remove")
    check("Remove mode", filt.params["mode"] == "remove")
    filt.set_mode("color")
    check("Color mode", filt.params["mode"] == "color")
    
    print()
    print(f"Results: {passed}/{passed+failed} passed")
    return failed == 0

if __name__ == "__main__":
    ok = test_segmentation()
    sys.exit(0 if ok else 1)
