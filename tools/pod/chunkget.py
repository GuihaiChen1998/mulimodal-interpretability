import sys, io, contextlib, base64, os
sys.path.insert(0, os.path.dirname(__file__))
import podexec
remote, local = sys.argv[1], sys.argv[2]
CH = 256 * 1024
out = open(local, "wb"); off = 0
while True:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        podexec.run_code(f"import base64;f=open({remote!r},'rb');f.seek({off});print('B64:'+base64.b64encode(f.read({CH})).decode())", 300)
    data = base64.b64decode("".join(l[4:] for l in buf.getvalue().splitlines() if l.startswith("B64:")))
    out.write(data); off += len(data)
    if len(data) < CH: break
out.close(); print("bytes", off)
