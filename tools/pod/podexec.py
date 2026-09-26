#!/usr/bin/env python3
"""Run a bash script on the Runpod pod through the Jupyter kernel API.

Usage: podexec.py [--timeout S] < script.sh     (script read from stdin)
       podexec.py --put LOCAL REMOTE             (upload a text/binary file)
       podexec.py --get REMOTE LOCAL             (download a file)
"""
import json, os, ssl, sys, time, uuid, base64, urllib.request

POD = os.environ.get("POD_ID", "de3m2w1c58cwx1")
TOKEN = os.environ["POD_JUPYTER_TOKEN"]  # the pod's JUPYTER_PASSWORD (Runpod console)
BASE = f"https://{POD}-8888.proxy.runpod.net"
CA = "/root/.ccr/ca-bundle.crt"
PROXY = __import__("urllib.parse").parse.urlparse(os.environ.get("HTTPS_PROXY", "http://127.0.0.1:3128"))
KFILE = os.path.join(os.path.dirname(__file__), ".kernel_id")


def api(method, path, body=None):
    req = urllib.request.Request(BASE + path, method=method,
                                 headers={"Authorization": f"token {TOKEN}",
                                          "Content-Type": "application/json", "User-Agent": "curl/8.5.0"},
                                 data=json.dumps(body).encode() if body is not None else None)
    ctx = ssl.create_default_context(cafile=CA)
    with urllib.request.urlopen(req, context=ctx, timeout=60) as r:
        t = r.read()
        return json.loads(t) if t else None


def kernel_id():
    kid = open(KFILE).read().strip() if os.path.exists(KFILE) else None
    alive = {k["id"] for k in api("GET", "/api/kernels")}
    if kid not in alive:
        kid = api("POST", "/api/kernels", {"name": "python3"})["id"]
        open(KFILE, "w").write(kid)
    return kid


def run_code(code, timeout):
    import websocket
    kid = kernel_id()
    ws = websocket.create_connection(
        BASE.replace("https", "wss") + f"/api/kernels/{kid}/channels?token={TOKEN}",
        http_proxy_host=PROXY.hostname, http_proxy_port=PROXY.port, proxy_type="http",
        sslopt={"ca_certs": CA}, timeout=30, header=["User-Agent: curl/8.5.0"])
    mid = uuid.uuid4().hex
    ws.send(json.dumps({"header": {"msg_id": mid, "username": "c", "session": uuid.uuid4().hex,
                                   "msg_type": "execute_request", "version": "5.3"},
                        "parent_header": {}, "metadata": {}, "channel": "shell",
                        "content": {"code": code, "silent": False, "store_history": False,
                                    "user_expressions": {}, "allow_stdin": False, "stop_on_error": True}}))
    end = time.time() + timeout
    status = "timeout"
    while time.time() < end:
        try:
            m = json.loads(ws.recv())
        except websocket.WebSocketTimeoutException:
            continue
        if m.get("parent_header", {}).get("msg_id") != mid:
            continue
        t, c = m["msg_type"], m["content"]
        if t == "stream":
            sys.stdout.write(c["text"]); sys.stdout.flush()
        elif t in ("execute_result", "display_data"):
            print(c["data"].get("text/plain", ""))
        elif t == "error":
            print("\n".join(c["traceback"]))
        elif t == "execute_reply":
            status = c["status"]
            if status == "ok" and c.get("user_expressions") is not None:
                pass
        elif t == "status" and c["execution_state"] == "idle" and status != "timeout":
            break
        if t == "execute_reply":
            break
    ws.close()
    return status


BASH_WRAPPER = r'''
import subprocess, sys
_p = subprocess.Popen(["bash", "-lc", {script!r}], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for _l in _p.stdout:
    print(_l, end="", flush=True)
_p.wait()
print(f"\n[exit {{_p.returncode}}]")
'''

if __name__ == "__main__":
    args = sys.argv[1:]
    timeout = 1800
    if args[:1] == ["--timeout"]:
        timeout = int(args[1]); args = args[2:]
    if args[:1] == ["--put"]:
        local, remote = args[1], args[2]
        data = base64.b64encode(open(local, "rb").read()).decode()
        code = (f"import base64,os; os.makedirs(os.path.dirname({remote!r}), exist_ok=True); "
                f"open({remote!r},'wb').write(base64.b64decode({data!r})); print('put ok', {remote!r})")
        sys.exit(0 if run_code(code, timeout) == "ok" else 1)
    if args[:1] == ["--get"]:
        remote, local = args[1], args[2]
        import contextlib, io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            st = run_code(f"import base64; print('B64:' + base64.b64encode(open({remote!r},'rb').read()).decode())", timeout)
        data = "".join(l[4:] for l in buf.getvalue().splitlines() if l.startswith("B64:"))
        open(local, "wb").write(base64.b64decode(data))
        print("get ok", local, len(base64.b64decode(data)))
        sys.exit(0 if st == "ok" else 1)
    script = sys.stdin.read()
    st = run_code(BASH_WRAPPER.format(script=script), timeout)
    if st != "ok":
        print(f"[kernel status: {st}]")
