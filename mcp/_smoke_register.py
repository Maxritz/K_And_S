"""Smoke-test the K_and_S MCP server: does it import, which mcp API path is taken, and
does every resource it advertises actually read?

The resource URIs are taken from the server's own list rather than hardcoded. The
previous version hardcoded two invented URIs ("kands://arch-map", "kands://lessons"),
which never existed and produced two false FAILs while the server was fine.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server  # noqa: E402

t = server.mcp_types
print("mcp_types module     :", getattr(t, "__name__", t))

srv = server.server
rh = getattr(srv, "request_handlers", None)
has_old = hasattr(srv, "add_request_handler")
print("server API           :",
      "add_request_handler (mcp < 1.x)" if has_old else "request_handlers (mcp >= 1.x)")
if rh is not None:
    print("registered handlers  :",
          sorted(getattr(k, "__name__", str(k)) for k in rh))
print("create_init_options  :", hasattr(srv, "create_initialization_options"))

res = server._list_resources()
uris = [str(getattr(r, "uri", "")) for r in res]
print("resources            :", len(uris))
print("   e.g.              :", ", ".join(uris[:4]))

# the test that would have caught the wrong URIs: read every advertised resource
bad = []
biggest = ("", 0)
for u in uris:
    try:
        n = len(server._read_resource(u))
    except Exception as e:  # noqa: BLE001
        bad.append((u, str(e)))
        continue
    biggest = max(biggest, (u, n), key=lambda x: x[1])
print("all resources read   :", "yes" if not bad else "NO -> %s" % bad[:3])
print("largest read         : %s -> %d chars" % biggest)

# a read through a non-str URI: mcp can pass a URL object that does not compare to str
class FakeUrl:
    def __str__(self):
        return uris[0] if uris else ""


if uris:
    try:
        print("non-str URI read     : %d chars (str() coercion works)"
              % len(server._read_resource(FakeUrl())))
    except Exception as e:  # noqa: BLE001
        print("non-str URI read     : FAILED %s" % e)

print("tools                :", len(server._tools()))
print("RESULT               :", "OK" if not bad else "PROBLEMS")
