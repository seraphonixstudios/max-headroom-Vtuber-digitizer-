#!/usr/bin/env python3
"""
Max Headroom v3.0 - End-to-End Integration Test
Executes a full path:
- Start server
- Push synthetic WebSocket messages as if from tracker
- Validate status responses from server
- Teardown
"""
import time
import json
import subprocess
import sys
import math
import socket
try:
    import websocket
except Exception:
    websocket = None

PORT = 30100
HOST = "localhost"

ARKIT_BLENDSHAPES_32 = [
    "browDown_L","browDown_R","browUp_L","browUp_R",
    "cheekPuff","cheekSquint_L","cheekSquint_R","eyeBlink_L",
    "eyeBlink_R","eyeLookDown_L","eyeLookDown_R","eyeLookUp_L",
    "eyeLookUp_R","eyeSquint_L","eyeSquint_R","jawForward",
    "jawLeft","jawOpen","jawRight","mouthClose","mouthDimple_L",
    "mouthDimple_R","mouthFunnel","mouthLeft","mouthPucker",
    "mouthRight","mouthSmile_L","mouthSmile_R","mouthUpperUp_L",
    "mouthUpperUp_R","noseSneer_L","noseSneer_R"
]

def synth_blendshapes():
    vals = {}
    for i, name in enumerate(ARKIT_BLENDSHAPES_32):
        v = 0.5 * (math.sin(i * 0.6) + 1.0)  # range 0..1
        vals[name] = max(0.0, min(1.0, v))
    return vals

def synth_head_pose():
    t = time.time()
    return {
        "rotation": [float(5.0 * math.sin(t * 0.7)), float(8.0 * math.sin(t * 0.5)), float(0.0)],
        "translation": [float(0.0), float(0.0), float(1.5 + 0.05 * math.sin(t))],
    }

def run_server():
    # Run server in-process for easier introspection
    from server import MaxHeadroomServer
    s = MaxHeadroomServer(host=HOST, port=PORT)
    s.start()
    return s

def test_end_to_end():
    if websocket is None:
        print("WebSocket client not installed. Install with: pip install websocket-client")
        return False

    print("Starting end-to-end test...")
    srv = run_server()
    time.sleep(1.0)

    try:
        # Wait briefly for server readiness
        time.sleep(0.5)
        ws = websocket.create_connection(f"ws://{HOST}:{PORT}")
        # Send synthetic face_data
        data = {
            "type": "face_data",
            "blendshapes": synth_blendshapes(),
            "head_pose": synth_head_pose(),
            "landmarks": [],
            "timestamp": time.time(),
            "fps": 30,
            "frame_id": 1,
        }
        ws.send(json.dumps(data))
        # Request status
        ws.send(json.dumps({"type": "status_request"}))
        resp = ws.recv()
        # Basic validation
        ok = False
        try:
            j = json.loads(resp)
            if isinstance(j, dict) and ("stats" in j or j.get("type") == "status"):
                ok = True
        except Exception:
            ok = False
        ws.close()
        return ok
    finally:
        try:
            srv.stop()
        except Exception:
            pass

if __name__ == "__main__":
    ok = test_end_to_end()
    print("END-TO-END TEST RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
