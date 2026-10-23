import os, shutil, stat, subprocess

base = "H:/ports/portingtests"
target = os.path.join(base, "kindi")

if os.path.exists(target):
    shutil.rmtree(target)

r = subprocess.run(
    ["git", "clone", "--depth", "1", "https://github.com/ahtx/kindi.git", target],
    capture_output=True, text=True, timeout=120
)
print("Clone rc:", r.returncode, r.stderr[:200] if r.stderr else "")

if r.returncode == 0:
    files = sorted([f for f in os.listdir(target) if not f.startswith('.')])
    print("Root:", files)

    # Find CUDA source files
    cu_files = []
    for root, dirs, fs in os.walk(target):
        dirs[:] = [d for d in dirs if d not in ('.git',)]
        for f in fs:
            if f.endswith(('.cu', '.c', '.cpp', '.py')):
                cu_files.append(os.path.relpath(os.path.join(root, f), target))
    print("Source files:", cu_files)
