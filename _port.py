import sys, asyncio, os
_KANDS_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_KANDS_ROOT, 'mcp'))
import server as s
from server import AutonomousPortParams, _load_json, _ARCH_MAP_PATH

arch_map = _load_json(_ARCH_MAP_PATH)
params = AutonomousPortParams(
    repo_path='/workspace/chats/07ce5bca12d680f57df70775/tiny-vllm',
    source_arch='sm_80', target_arch='gfx1201',
    target_backend='hip', target_language='cxx',
    project_type='llm', target_os='linux',
    mode='strict', force_refresh=True,
    timeout_seconds=120
)
r = asyncio.run(s._autonomous_port(params, arch_map))

# Save full report
fd = os.path.join(os.path.dirname(_KANDS_ROOT), 'findings', 'tiny-vllm-cpu-windows')
os.makedirs(fd, exist_ok=True)
import datetime as dt
ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
report_file = os.path.join(fd, f"porting-report-{params.target_language}-{params.target_backend}-{ts}.md")
with open(report_file, 'w') as f:
    f.write(r)

# Print summary
print("=== REPORT SAVED ===")
print(report_file)
print()

# Show key sections
for section in ["Phase 0", "Phase 7", "Self-Check", "Target arch", "Source arch", "Kernels found", "CUDA constructs", "Tensor cores", "WMMA", "Phase 8"]:
    for line in r.split('\n'):
        if section in line:
            print(line.strip()[:120])

print()
print("=== Self-Check Status ===")
for line in r.split('\n'):
    if line.startswith('##') and 'Self-Check' in line:
        print(line)
    if '✅' in line and ('Rule' in line or 'Phase 7' in line or 'Phase 8' in line):
        print(line.strip())
