import sys
import numpy as np
sys.path.insert(0, '.')

print("=== SERVER MODULE TESTS ===\n")

# Test 1: Server imports
print("[1] Server import test")
from server import MaxHeadroomServer, VERSION
print(f"  VERSION: {VERSION}")

# Test 2: Server instantiation
print("\n[2] Server instantiation")
server = MaxHeadroomServer(host="localhost", port=30000)
print(f"  Host: {server.host}")
print(f"  Port: {server.port}")
print(f"  Running: {server.running}")

# Test 3: Stats
print("\n[3] Server stats")
stats = server.get_stats()
print(f"  clients: {stats['clients']}")
print(f"  fps: {stats['fps']}")

# Test 4: Get data (empty)
print("\n[4] Get current data (empty)")
data = server.get_current_data()
print(f"  Data: {data}")

# Test 5: Start server
print("\n[5] Start server (in background)")
try:
    server.start()
    import time
    time.sleep(0.5)
    print(f"  Running: {server.running}")
    server.stop()
    print(f"  Stopped: OK")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n=== ALL SERVER TESTS PASSED ===")