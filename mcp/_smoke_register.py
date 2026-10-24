"""Smoke-test the mcp registration fix: does the server import, and are the four request
handlers bound under the API this mcp build actually uses?"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server  # noqa: E402

t = server.mcp_types
print("mcp_types module     :", getattr(t, "__name__", t))
print("types has AnyUrl     :", hasattr(t, "AnyUrl"))

srv = server.server
rh = getattr(srv, "request_handlers", None)
print("server API           :", "request_handlers (mcp >= 1.x)" if rh else "add_request_handler (mcp < 1.x)")
if rh is not None:
    print("registered handlers  :", sorted(getattr(k, "__name__", str(k)) for k in rh))

for label, fn in (("resources", server._list_resources),):
    try:
        r = fn()
        print("%-20s : %d" % (label, len(r)))
    except Exception as e:  # noqa: BLE001
        print("%-20s : FAILED %s" % (label, e))

for uri in ("kands://porting-patterns", "kands://arch-map", "kands://lessons"):
    try:
        got = server._read_resource(uri)
        print("read %-26s -> %d chars" % (uri, len(got)))
    except Exception as e:  # noqa: BLE001
        print("read %-26s -> FAILED %s" % (uri, e))

# the AnyUrl path: an object that is not a str but stringifies to the uri
class FakeUrl:
    def __str__(self):
        return "kands://porting-patterns"


try:
    got = server._read_resource(FakeUrl())
    print("read AnyUrl-like object    -> %d chars  (str() coercion works)" % len(got))
except Exception as e:  # noqa: BLE001
    print("read AnyUrl-like object    -> FAILED %s" % e)
