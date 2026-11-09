import tomllib
from core.project_loader import parse_lock_text

s = '[[package]]\nname = "numpy"\nversion = "2.0.0"\n'
print("repr", repr(s))
print("toml", tomllib.loads(s))
print("lock", parse_lock_text("poetry.lock", s))
