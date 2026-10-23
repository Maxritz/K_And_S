import sys, asyncio, os
sys.path.insert(0, 'H:/K_and_S/mcp')
import server as s
from server import AutonomousPortParams, _load_json, _ARCH_MAP_PATH

arch_map = _load_json(_ARCH_MAP_PATH)
params = AutonomousPortParams(
    repo_path='H:/ports/portingtests/kindi',
    source_arch='auto', target_arch='auto',
    target_backend='hip', target_language='cxx',
    project_type='llm', target_os='windows',
    mode='strict', force_refresh=True
)
r = asyncio.run(s._autonomous_port(params, arch_map))

# Save full report
fd = os.path.join(params.repo_path, 'findings')
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
