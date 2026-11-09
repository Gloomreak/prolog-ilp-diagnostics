import json, subprocess, sys

p = subprocess.Popen(
    ["swipl", "-q", "-g", "ilp_engine:server", "-t", "halt", "prolog/ilp_engine.pl"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    text=True, encoding="utf-8", bufsize=1,
)
for req in ({"command": "ping"}, {"command": "ping"}):
    p.stdin.write(json.dumps(req) + "\n")
    p.stdin.flush()
    line = p.stdout.readline()
    print("resp:", line.strip())
p.stdin.close()
p.wait(timeout=10)
print("rc", p.returncode, "stderr", p.stderr.read()[:500])
