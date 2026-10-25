#!/usr/bin/env python
"""Final comprehensive test of K_and_S bundle."""
import sys, json, os
sys.path.insert(0, r'h:\K_and_S\mcp')

errors = []

# 1. Validate all JSON files
json_files = [
    r'h:\K_and_S\ARCH_MAP.json',
    r'h:\K_and_S\MODEL_MAP.json',
    r'h:\K_and_S\TK_TOOLS.json',
    r'h:\K_and_S\CROSS_PORT.json',
    r'h:\K_and_S\AI_FRAMEWORKS.json',
    r'h:\K_and_S\MODEL_FORMATS.json',
    r'h:\K_and_S\PORTING_PATTERNS.json',
    r'h:\K_and_S\PERF_BENCHMARKS.json',
    r'h:\K_and_S\mcp\state\goal.json',
    r'h:\K_and_S\mcp\state\perf_history.json',
    r'h:\K_and_S\mcp\claude_desktop_config.json',
    r'h:\K_and_S\BACKEND_REQUIREMENTS.json',
]
for f in json_files:
    fname = os.path.basename(f)
    try:
        json.loads(open(f).read())
    except Exception as e:
        errors.append(f'JSON {fname}: {e}')

# 2. Validate Python files
import ast
for f in [r'h:\K_and_S\mcp\server.py', r'h:\K_and_S\mcp\reference_search.py',
          r'h:\K_and_S\mcp\backend_profile.py', r'h:\K_and_S\mcp\api_map.py']:
    try:
        ast.parse(open(f).read())
    except Exception as e:
        errors.append(f'PY {os.path.basename(f)}: {e}')

# 3. Validate TOML
try:
    import tomllib
    with open(r'h:\K_and_S\mcp/pyproject.toml', 'rb') as f:
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
    'reject_common_pattern', 'radical_experiment_plan'
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
for uri in ['kands://perf-benchmarks', 'kands://status', 'kands://goal',
            'kands://architectures', 'kands://setup-guide', 'kands://porting-patterns',
            'kands://radical-experiments']:
    try:
        text = s._read_resource(uri)
    except Exception as e:
        errors.append(f'{uri}: {e}')

# Check key features
perf = json.loads(s._read_resource('kands://perf-benchmarks'))
print(f'\n  Perf benchmarks: {len(perf.get("hardware_theoretical", {}))} archs')
print(f'  Roofline models: {len(perf.get("roofline_models", {}).get("models", {}))} arches')
print(f'  Has meteor_lake roofline: {"meteor_lake" in perf.get("roofline_models", {}).get("models", {})}')

patterns = json.loads(s._read_resource('kands://porting-patterns'))
print(f'  Sync patterns: {"sync_patterns" in patterns}')
sync = patterns.get('sync_patterns', {})
print(f'  TDR limits defined: {len(sync.get("tdr_limits_ms", {}))} platforms')

arch = json.loads(s._read_resource('kands://architectures'))
archs = arch.get('entries', {})
print(f'  Architectures: {len(archs)} (meteor_lake: {"meteor_lake" in archs})')

ai_fw = json.loads(s._read_resource('kands://ai-frameworks'))
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
template = r'h:\K_and_S\mcp\templates\breadcrumb.hpp'
if os.path.exists(template):
    print(f'  breadcrumb.hpp: {os.path.getsize(template)} bytes')
else:
    errors.append('Missing breadcrumb.hpp template')

print(f'\n=== RESULT: {"PASS - all OK!" if not errors else f"FAIL - {len(errors)} errors"} ===')
for e in errors:
    print(f'  ERROR: {e}')
sys.exit(1 if errors else 0)