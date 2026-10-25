#!/usr/bin/env python
"""Validate the K_and_S bundle after all edits."""
import json, ast, sys, os

errors = []

# Check JSON files
json_files = [
    r'h:\K_and_S\ARCH_MAP.json',
    r'h:\K_and_S\PERF_BENCHMARKS.json',
    r'h:\K_and_S\PORTING_PATTERNS.json',
    r'h:\K_and_S\TK_TOOLS.json',
    r'h:\K_and_S\AI_FRAMEWORKS.json',
    r'h:\K_and_S\MODEL_FORMATS.json',
    r'h:\K_and_S\CROSS_PORT.json',
    r'h:\K_and_S\MODEL_MAP.json',
    r'h:\K_and_S\BACKEND_REQUIREMENTS.json',
]
for f in json_files:
    fname = os.path.basename(f)
    try:
        data = json.loads(open(f).read())
    except Exception as e:
        errors.append(f'JSON {fname}: {e}')

# Backend readiness profiles must each declare a ReBAR path with a fallback.
try:
    reqs = json.loads(open(r'h:\K_and_S\BACKEND_REQUIREMENTS.json').read())
    for name, spec in reqs.get('backends', {}).items():
        rebar = spec.get('rebar', {})
        if 'supported' not in rebar or not rebar.get('fallback'):
            errors.append(f'BACKEND_REQUIREMENTS.json: {name} has no rebar.supported/fallback')
        if not spec.get('requirements'):
            errors.append(f'BACKEND_REQUIREMENTS.json: {name} has no requirements')
except Exception as e:
    errors.append(f'BACKEND_REQUIREMENTS.json structure: {e}')

# Check server.py
try:
    ast.parse(open(r'h:\K_and_S\mcp\server.py').read())
except Exception as e:
    errors.append(f'server.py: {e}')

# Import and test server
sys.path.insert(0, r'h:\K_and_S\mcp')
try:
    import server as s
    tools = s._tools()
    resources = s._list_resources()
    print(f'Tools: {len(tools)}')
    print(f'Resources: {len(resources)}')

    # Test key resources
    for uri in ['kands://perf-benchmarks', 'kands://status', 'kands://setup-guide', 'kands://goal', 'kands://porting-patterns']:
        try:
            text = s._read_resource(uri)
            print(f'  {uri}: OK ({len(text)} chars)')
        except Exception as e:
            errors.append(f'{uri}: {e}')

    # Verify roofline_models
    perf = json.loads(s._read_resource('kands://perf-benchmarks'))
    roofline = perf.get('roofline_models', {}).get('models', {})
    print(f'Roofline models: {len(roofline)} entries')
    print(f'Has meteor_lake roofline: {"meteor_lake" in roofline}')

    # Verify sync_patterns
    patterns = json.loads(s._read_resource('kands://porting-patterns'))
    print(f'Has sync_patterns: {"sync_patterns" in patterns}')

    # Verify meteor_lake in architectures
    arch = json.loads(s._read_resource('kands://architectures'))
    entries = arch.get('entries', {})
    print(f'Architectures: {len(entries)} entries')
    print(f'Has meteor_lake: {"meteor_lake" in entries}')

    # Verify verify_numerical_equivalence tool exists
    tool_names = [t.name for t in tools]
    print(f'Has verify_numerical_equivalence: {"verify_numerical_equivalence" in tool_names}')

    # Verify the backend readiness step is wired into discovery.
    print(f'Has backend_profile module: {hasattr(s, "backend_profile")}')
    if not hasattr(s, 'backend_profile'):
        errors.append('server.py: backend_profile not imported')
    disc = [t for t in tools if t.name == 'discover_kernels'][0]
    disc_schema = (getattr(disc, 'inputSchema', None)
                   or getattr(disc, 'input_schema', None) or {})
    disc_props = disc_schema.get('properties', {}) if isinstance(disc_schema, dict) else {}
    print(f'  discover target_backend in schema: {"target_backend" in disc_props}')
    if 'target_backend' not in disc_props:
        errors.append('discover_kernels: no target_backend parameter')

    # Verify autonomous_port has security params.
    # The SDK model may expose the field as `inputSchema` (wire name) or
    # `input_schema` (python attribute), so read both rather than assuming one.
    ap_tool = [t for t in tools if t.name == 'autonomous_port'][0]
    ap_schema = (getattr(ap_tool, 'inputSchema', None)
                 or getattr(ap_tool, 'input_schema', None) or {})
    if isinstance(ap_schema, dict) and ap_schema.get('properties'):
        ap_props = ap_schema.get('properties', {})
    else:
        ap_props = {}
        errors.append('autonomous_port: no readable input schema on the Tool model')
    print(f'autonomous_port security params:')
    print(f'  max_files in schema: {"max_files" in ap_props}')
    print(f'  timeout_seconds in schema: {"timeout_seconds" in ap_props}')

except Exception as e:
    errors.append(f'Server import/test: {e}')

# Check reference_search.py
try:
    ast.parse(open(r'h:\K_and_S\mcp\reference_search.py').read())
except Exception as e:
    errors.append(f'reference_search.py: {e}')

# Check pyproject.toml
try:
    import tomllib
    with open(r'h:\K_and_S\mcp/pyproject.toml', 'rb') as f:
        tomllib.load(f)
except Exception as e:
    errors.append(f'pyproject.toml: {e}')

print()
if errors:
    print(f'ERRORS: {len(errors)}')
    for e in errors:
        print(f'  - {e}')
    sys.exit(1)
else:
    print('ALL FILES VALID -- Bundle ready!')