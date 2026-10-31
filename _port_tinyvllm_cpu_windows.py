import sys, asyncio, os, importlib.util, datetime as dt, types, pathlib, importlib, json
_KARDS_ROOT = os.path.dirname(os.path.abspath(__file__)).replace('/', '/')
mcp_server_path = os.path.join(_KARDS_ROOT, 'mcp', 'server.py').replace('/', '/')

_repo_root = pathlib.Path(mcp_server_path).resolve().parent.parent

# Pre-register the repo's mcp/*.py sibling modules (excluding tests and server itself) so
# server.py's 'from X import ...' resolves against this repo, not any installed package.
_sibling_names = []
_mcp_dir = os.path.join(_KARDS_ROOT, 'mcp')
if not os.path.isdir(_mcp_dir):
    raise SystemExit('mcp/ dir not found at: ' + _mcp_dir)
for _entry in sorted(os.listdir(_mcp_dir)):
    if not _entry.endswith('.py'):
        continue
    _name = _entry[:-3]
    if _name in ('__init__', '_final_test', '_smoke_register', '_test_gpu', '_validate_bundle', 'server'):
        continue
    _path = os.path.join(_mcp_dir, _entry).replace('/', '/')
    if _name not in sys.modules:
        _spec_sibling = importlib.util.spec_from_file_location(_name, _path)
        _sibling_mod = importlib.util.module_from_spec(_spec_sibling)
        sys.modules[_name] = _sibling_mod
        try:
            _spec_sibling.loader.exec_module(_sibling_mod)
        except Exception as _e:
            print(f'WARNING: preloaded sibling {_name} exec failed: {type(_e).__name__} {str(_e)[:200]}', file=sys.stderr)
            if _name in sys.modules:
                del sys.modules[_name]

# server.py uses 'from mcp.server import Server'. We want that to mean the installed mcp
# distribution, so load it with repo root removed from sys.path AND with a synthetic mcp.server
# already present so the 'from mcp.server import Server' takes from the installed package.
_old_path = list(sys.path)
sys.path = [p for p in sys.path if pathlib.Path(p).resolve() != _repo_root]

# IMPORTANT: do not pre-seed 'mcp.server' here; let the import system resolve it to the
# installed package first. Pre-seeding it with the file-backed module causes the circular
# import error we saw.
print('loading server.py from:', mcp_server_path, file=sys.stderr)
_spec = importlib.util.spec_from_file_location("mcp.server", mcp_server_path)
smod = importlib.util.module_from_spec(_spec)
try:
    _spec.loader.exec_module(smod)
except Exception as e:
    print('server.py load failed:', type(e).__name__, e, file=sys.stderr)
    raise
sys.path = _old_path

assert hasattr(smod, '_autonomous_port'), 'server.py did not expose _autonomous_port'
print('server.py loaded OK; mcp siblings preloaded:', _sibling_names, file=sys.stderr)

# The loaded server.py expects repo mcp/*.py siblings to remain importable by short name.
# Re-add repo root so later calls inside smod (e.g. helpers) can still import them.
sys.path.insert(0, str(_repo_root))

arch_map = smod._load_json(smod._ARCH_MAP_PATH)

params = smod.AutonomousPortParams(
    repo_path=_KARDS_ROOT + '/tiny-vllm',
    source_arch='sm_80', target_arch='gfx1201',
    target_backend='hip', target_language='cxx',
    project_type='llm', target_os='windows',
    mode='strict', force_refresh=True,
    timeout_seconds=120,
    break_self_check=False
)

r = asyncio.run(smod._autonomous_port(params, arch_map))

out_dir = os.path.join(_repo_root, 'findings', 'tiny-vllm-cpu-windows')
os.makedirs(out_dir, exist_ok=True)
ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
report_file = os.path.join(out_dir, f"porting-report-{ts}-cpu-windows.md")
with open(report_file, 'w') as f:
    f.write(r)

print("=== REPORT SAVED ===")
print(report_file)
print()
for section in ["Phase 0", "Source Inventory", "Target Feasibility", "Should We Port", "Effort Estimate", "Self-Check", "Phase 8"]:
    for line in r.split('\n'):
        if section in line:
            print(line.strip()[:140])
print()
print("=== Self-Check Status ===")
for line in r.split('\n'):
    if 'Self-Check' in line:
        print(line.strip()[:180])
for line in r.split('\n'):
    if '⛔' in line or 'exceptionally' in line:
        print(line.strip()[:220])
print()
print('=== Full Section Headers ===')
for line in r.split('\n'):
    if line.startswith('## '):
        print(line.strip()[:140])
print()
print('=== Target tree: how report treats g++ as host compiler on windows ===')
for line in r.split('\n'):
    if 'g++' in line or 'cl.exe' in line or 'host.cpp' in line or 'Host compile' in line or 'tgt_os' in line:
        print(line.strip()[:160])
print()
print('=== CPU boundary language in report ===')
for line in r.split('\n'):
    if 'CPU' in line or 'cpu' in line or 'BF16' in line or 'bf16' in line or 'numeric' in line or 'blas' in line or 'BLAS' in line:
        print(line.strip()[:160])
print()
print('=== tiny-vllm-specific rows (kernels, model, tokenizer, reference) ===')
for line in r.split('\n'):
    if 'tiny-vllm' in line or 'model.safetensors' in line or 'tokenizer' in line or 'reference' in line or 'safetensors' in line:
        print(line.strip()[:180])
print()
print('=== harness inspect: what smod._load_json does with _ARCH_MAP_PATH ===')
print('ARCH_MAP_PATH:', smod._ARCH_MAP_PATH)
print('ARCH_MAP exists:', os.path.exists(smod._ARCH_MAP_PATH))
print('ARCH_MAP keys:', list(smod._load_json(smod._ARCH_MAP_PATH).keys())[:12])
print('ARCH_MAP entries:', list(smod._load_json(smod._ARCH_MAP_PATH).get('entries', {}).keys())[:12])
print('TK_TOOLS_PATH:', smod._TK_TOOLS_PATH if hasattr(smod,'_TK_TOOLS_PATH') else 'n/a')
print('AI_FRAMEWORKS_PATH:', smod._AI_FRAMEWORKS_PATH if hasattr(smod,'_AI_FRAMEWORKS_PATH') else 'n/a')
_ai = smod._load_json(smod._AI_FRAMEWORKS_PATH) if hasattr(smod,'_AI_FRAMEWORKS_PATH') else {}
print('AI_FRAMEWORKS top-level keys:', list(_ai.keys())[:12])
for _k in _ai:
    _vv = _ai[_k]
    if isinstance(_vv, dict) and 'tiny_vllm' in _vv:
        print('AI_FRAMEWORKS has tiny_vllm under key:', _k)
        print('AI_FRAMEWORKS tiny_vllm cpu_port_notes snippet:', str(_vv['tiny_vllm'].get('cpu_port_notes','n/a'))[:400])
print('AI_FRAMEWORKS tiny_vllm cpu_port_notes snippet (alt):', str(_ai.get('tiny_vllm',{}).get('cpu_port_notes','n/a'))[:300] if _ai else 'n/a')

print()
print('=== tiny_vllm AI_FRAMEWORKS cpu_port_notes FULL ===')
for _k in smod._load_json(smod._AI_FRAMEWORKS_PATH):
    _vv = smod._load_json(smod._AI_FRAMEWORKS_PATH)[_k]
    if isinstance(_vv, dict) and 'tiny_vllm' in _vv:
        print(json.dumps(_vv['tiny_vllm'].get('cpu_port_notes',{}), indent=2))
        break
print()
print('=== tiny_vllm AI_FRAMEWORKS file_structure keys ===')
for _k in smod._load_json(smod._AI_FRAMEWORKS_PATH):
    _vv = smod._load_json(smod._AI_FRAMEWORKS_PATH)[_k]
    if isinstance(_vv, dict) and 'tiny_vllm' in _vv:
        print('tiny_vllm keys:', list(_vv['tiny_vllm'].keys()))
        fs = _vv['tiny_vllm'].get('file_structure',[])
        if not fs:
            print('file_structure is empty in AI_FRAMEWORKS -> listing real tiny-vllm tree:')
            for root, dirs, files in os.walk(os.path.join(_repo_root, 'tiny-vllm')):
                rel = os.path.relpath(root, os.path.join(_repo_root, 'tiny-vllm'))
                if rel == '.':
                    continue
                print(' ', rel)
                for f in sorted(files):
                    print('   ', f)
        else:
            print(json.dumps(fs, indent=2))
        break
print()
print('=== tiny_vllm AI_FRAMEWORKS build object keys ===')
for _k in smod._load_json(smod._AI_FRAMEWORKS_PATH):
    _vv = smod._load_json(smod._AI_FRAMEWORKS_PATH)[_k]
    if isinstance(_vv, dict) and 'tiny_vllm' in _vv:
        print('tiny_vllm keys:', list(_vv['tiny_vllm'].keys()))
        print(json.dumps(dict((k, _vv['tiny_vllm'][k]) for k in ('build',) if k in _vv['tiny_vllm']), indent=2))
        break
print()
print('=== tiny_vllm AI_FRAMEWORKS model object ===')
for _k in smod._load_json(smod._AI_FRAMEWORKS_PATH):
    _vv = smod._load_json(smod._AI_FRAMEWORKS_PATH)[_k]
    if isinstance(_vv, dict) and 'tiny_vllm' in _vv:
        print(json.dumps(dict((k, _vv['tiny_vllm'][k]) for k in ('model','architecture','compute') if k in _vv['tiny_vllm']), indent=2))
        break
print()
print('=== tiny_vllm AI_FRAMEWORKS files object ===')
for _k in smod._load_json(smod._AI_FRAMEWORKS_PATH):
    _vv = smod._load_json(smod._AI_FRAMEWORKS_PATH)[_k]
    if isinstance(_vv, dict) and 'tiny_vllm' in _vv:
        print(json.dumps(dict((k, _vv['tiny_vllm'][k]) for k in ('files',) if k in _vv['tiny_vllm']), indent=2))
        break
print()
print('=== tiny_vllm AI_FRAMEWORKS description/features/repo/endpoint ===')
for _k in smod._load_json(smod._AI_FRAMEWORKS_PATH):
    _vv = smod._load_json(smod._AI_FRAMEWORKS_PATH)[_k]
    if isinstance(_vv, dict) and 'tiny_vllm' in _vv:
        print('repo_url:', _vv['tiny_vllm'].get('repo_url'))
        print('endpoint:', _vv['tiny_vllm'].get('endpoint'))
        print('description:', str(_vv['tiny_vllm'].get('description',''))[:400])
        print('features:', _vv['tiny_vllm'].get('features'))
        break
