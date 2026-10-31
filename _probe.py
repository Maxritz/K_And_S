import sys, inspect, re
# Avoid circular import: run from inside mcp dir like the tests do
sys.path.insert(0, "C:/Users/rr/K_And_S/mcp")
import importlib
server = importlib.import_module("server")

src = open("C:/Users/rr/K_And_S/mcp/server.py", encoding="utf-8").read()
print("current server preport_dir:", "preport_dir" in src)

dp = server.DiscoverParams
print("DiscoverParams fields:", list(dp.model_fields.keys()))

i = src.find('name == "discover_kernels"')
print(src[i:i + 900])
