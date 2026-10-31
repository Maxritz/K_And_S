"""Surgical backslash repair for mcp/server.py.

The file suffered a corruption pass that swapped forward slashes for
backslashes (paths, mime types, prose separators). Legitimate escape
sequences (\\n, \\t, \\\\, raw regex strings, line continuations) must survive.

Strategy:
  1. Fix known double-backslash URL corruption (kands:\/ -> kands://).
  2. Protect raw string literals with placeholders.
  3. Protect valid escape sequences with placeholders.
  4. Replace every remaining backslash with a forward slash.
  5. Restore placeholders.

Idempotent: running twice is a no-op the second time.
"""
import os
import re

os.chdir("C:/Users/rr/K_And_S")
path = "mcp/server.py"

with open(path, "rb") as f:
    data = f.read()

total_bs = data.count(b"\\")
print(f"backslash bytes at start: {total_bs}")

# ---------------------------------------------------------------- step 0
# Known '//'-corrupted-to-'\\' sequences (must run BEFORE \/ protection)
data = data.replace(b"kands:\\\\", b"kands://")

# ---------------------------------------------------------------- protect
protected = []


def protect(pattern, text, flags=0):
    """Replace every regex match with a unique placeholder; return new text."""
    counter = {"n": 0}

    def _repl(m):
        idx = counter["n"]
        counter["n"] += 1
        protected.append(m.group(0))
        return b"\x00KDP%d\x00" % idx

    return re.subn(pattern, _repl, text, flags=flags)


# raw triple-quoted strings (r'''...''' / r"""...""")
data, n1 = protect(
    rb"(?<![A-Za-z0-9_])[rR](?:[bB]|[fF])?('''|\"\"\")(?:\\.|.)*?\1",
    data,
    flags=re.DOTALL,
)

# single-line raw strings (r'...' / r"..." incl. rb, rf prefixes)
data, n2 = protect(
    rb"(?<![A-Za-z0-9_])[rR](?:[bB]|[fF])?(['\"])(?:\\.|(?!\1)[^\\\n])*\1",
    data,
)
print(f"protected raw strings: triple={n1} single={n2}")

# escaped backslash (\\) -- Windows paths, regex in normal strings
data, n3 = protect(rb"\\\\", data)

# \xHH
data, n4 = protect(rb"\\x[0-9a-fA-F]{2}", data)

# \N{...}
data, n5 = protect(rb"\\N\{[^}\n]*\}", data)

# \uXXXX / \UXXXXXXXX
data, n6 = protect(rb"\\u[0-9a-fA-F]{4}", data)
data, n7 = protect(rb"\\U[0-9a-fA-F]{8}", data)

# line continuations (backslash at EOL)
data, n8 = protect(rb"\\\r?\n", data)

# simple escapes: \a \b \f \n \r \t \v \0-\7 \' \"
data, n9 = protect(rb"\\[abfnrtv0-7'\"]", data)

print(
    f"protected escapes: backslash-backslash={n3} x={n4} N={n5} u={n6} "
    f"U={n7} continuation={n8} simple={n9}"
)

# ---------------------------------------------------------------- replace
data = data.replace(b"\\", b"/")

# ---------------------------------------------------------------- restore
for i in range(len(protected) - 1, -1, -1):
    data = data.replace(b"\x00KDP%d\x00" % i, protected[i])

with open(path, "wb") as f:
    f.write(data)

print(f"backslash bytes at end: {data.count(b'\\')}")

# ---------------------------------------------------------------- report
lines = data.split(b"\n")
remaining = [(i, ln) for i, ln in enumerate(lines, 1) if b"\\" in ln]
print(f"lines still containing backslash: {len(remaining)}")
for i, ln in remaining[:50]:
    print(f"  {i}: {ln.decode('utf-8', 'replace')[:140]}")
