#!\usr\bin\env python
"""Final comprehensive test of K_and_S bundle.

The bundle root is derived from this file's location, not a hardcoded drive
letter: the previous version pointed at ``h:\K_and_S`` and could only ever run
on the machine it was written on.
"""
import sys, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MCP = os.path.join(ROOT, "mcp")
sys.path.insert(0, MCP)

errors = []

# 1. Validate all JSON files
json_files = [
    os.path.join(ROOT, "ARCH_MAP.json"),
    os.path.join(ROOT, "MODEL_MAP.json"),
    os.path.join(ROOT, "TK_TOOLS.json"),
    os.path.join(ROOT, "CROSS_PORT.json"),
    os.path.join(ROOT, "AI_FRAMEWORKS.json"),
    os.path.join(ROOT, "MODEL_FORMATS.json"),
    os.path.join(ROOT, "PORTING_PATTERNS.json"),
    os.path.join(ROOT, "PERF_BENCHMARKS.json"),
    os.path.join(MCP, "state", "goal.json"),
    os.path.join(MCP, "state", "perf_history.json"),
    os.path.join(MCP, "claude_desktop_config.json"),
    os.path.join(ROOT, "BACKEND_REQUIREMENTS.json"),
]
for f in json_files:
    fname = os.path.basename(f)
    try:
        json.loads(open(f).read())
    except Exception as e:
        errors.append(f'JSON {fname}: {e}')

# 2. Validate Python files
import ast
for f in [os.path.join(MCP, n) for n in
          ("server.py", "reference_search.py", "backend_profile.py", "api_map.py",
           "arch_facts.py", "preport_docs.py", "porting_map.py", "isa_tools.py")]:
    try:
        ast.parse(open(f).read())
    except Exception as e:
        errors.append(f'PY {os.path.basename(f)}: {e}')

# 3. Validate TOML
try:
    import tomllib
    with open(os.path.join(MCP, "pyproject.toml"), 'rb') as f:
        tomllib.load(f)
except Exception as e:
    errors.append(f'TOML pyproject.toml: {e}')

# 4. Test server functionality
print("=== Server Test ===")
import server as s
tools = s._tools()
resources = s._list_resources()
print(f"  Tools: {len(tools)}")
print(f"  Resources: {len(resources)}")

# Check tool names
tool_names = [t.name for t in tools]
expected_tools = {
    'discover_kernels', 'check_arch_compatibility', 'query_kernel_features',
    'compare_archs', 'generate_porting_report', 'generate_tracing_plan',
    'diff_perf_traces', 'query_toolchain', 'detect_toolchain', 'search_online',
    'estimate_performance', 'compare_to_target', 'search_arxiv', 'search_github',
    'search_references', 'record_performance', 'suggest_optimizations',
    'export_contribution', 'set_goal', 'update_goal', 'check_guardrail',
    'prepare_pull_request', 'autonomous_port', 'verify_numerical_equivalence',
    'reject_common_pattern', 'radical_experiment_plan', 'write_preport_analysis',
    'postport_recheck'
}
missing = expected_tools - set(tool_names)
if missing:
    errors.append(f'Missing tools: {missing}')
else:
    print(f'  All {len(tool_names)} tools present OK')

# Check resources
print("  Resources:")
for r in resources:
    uri = r.uri if hasattr(r, 'uri') else r['uri']
    print(f"    {uri}")

# Test key resource reads
for uri in ['kands:\\perf-benchmarks', 'kands:\\status', 'kands:\\goal',
            'kands:\\architectures', 'kands:\\setup-guide', 'kands:\\porting-patterns',
            'kands:\\radical-experiments']:
    try:
        text = s._read_resource(uri)
    except Exception as e:
        errors.append(f'{uri}: {e}')

# Check key features
perf = json.loads(s._read_resource('kands:\\perf-benchmarks'))
print(f'\n  Perf benchmarks: {len(perf.get("hardware_theoretical", {}))} archs')
print(f'  Roofline models: {len(perf.get("roofline_models", {}).get("models", {}))} arches')
print(f'  Has meteor_lake roofline: {"meteor_lake" in perf.get("roofline_models", {}).get("models", {})}')

patterns = json.loads(s._read_resource('kands:\\porting-patterns'))
print(f'  Sync patterns: {"sync_patterns" in patterns}')
sync = patterns.get('sync_patterns', {})
print(f'  TDR limits defined: {len(sync.get("tdr_limits_ms", {}))} platforms')

arch = json.loads(s._read_resource('kands:\\architectures'))
archs = arch.get('entries', {})
print(f'  Architectures: {len(archs)} (meteor_lake: {"meteor_lake" in archs})')

ai_fw = json.loads(s._read_resource('kands:\\ai-frameworks'))
fw = ai_fw.get('frameworks', {})
print(f'  AI frameworks: {len(fw)} (tilelang: {"tilelang" in fw}, hipkittens: {"hipkittens" in fw})')

# Check tilelang + hipkittens present
for fw_name in ['tilelang', 'hipkittens', 'triton_language', 'executorch', 'torchinductor']:
    if fw_name not in fw:
        errors.append(f'Missing framework: {fw_name}')

# 5. Check bundle status
status = s._bundle_status()
print(f'\n  Bundle version: {status["version"]}')
print(f'  Current date: {status["current_date"]}')

# 6. Check template file exists
template = os.path.join(MCP, "templates", "breadcrumb.hpp")
if os.path.exists(template):
    print(f'  breadcrumb.hpp: {os.path.getsize(template)} bytes')
else:
    errors.append('Missing breadcrumb.hpp template')

print(f'\n=== RESULT: {"PASS - all OK!" if not errors else f"FAIL - {len(errors)} errors"} ===')
for e in errors:
    print(f'  ERROR: {e}')
sys.exit(1 if errors else 0)
