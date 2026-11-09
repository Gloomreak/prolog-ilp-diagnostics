from core.project_loader import parse_lock_text, project_from_files, _pick, _LOCK_NAMES

files = {
    "requirements.txt": "numpy==2.0\n",
    "poetry.lock": '[[package]]\nname = "numpy"\nversion = "2.0.0"\n',
}
print("pick", _pick(files, _LOCK_NAMES))
print("lock", parse_lock_text("poetry.lock", files["poetry.lock"]))
print(project_from_files(files, "p").project)
