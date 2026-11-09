"""Запуск SWI-Prolog как отдельного процесса и обмен JSON.

Python записывает запрос во временный файл, запускает

    swipl -q -g main -t halt <ilp_engine.pl> -- request.json response.json

(аргументы — отдельными элементами списка, без shell) и читает ответ. Рабочий
каталог процесса — временная папка, а в аргументах остаются короткие ASCII-имена
файлов: так не возникает проблем с кириллицей в пути пользователя Windows.
"""
from __future__ import annotations

import atexit
import json
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from .settings import ENGINE_FILE, swipl_path_setting, timeout_setting


class PrologError(Exception):
    """Базовая ошибка взаимодействия с SWI-Prolog (сообщение — на русском)."""


class PrologNotFoundError(PrologError):
    pass


class PrologTimeoutError(PrologError):
    pass


class PrologEngineError(PrologError):
    pass


@dataclass
class EngineInfo:
    path: str
    version_text: str
    version: tuple[int, int, int] | None

    @property
    def is_recommended(self) -> bool:
        return self.version is not None and self.version[0] >= 9


_WINDOWS_CANDIDATES = [
    r"C:\Program Files\swipl\bin\swipl.exe",
    r"C:\Program Files (x86)\swipl\bin\swipl.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\swipl\bin\swipl.exe"),
]


def find_swipl(custom: str | None = None) -> str | None:
    """Ищет swipl: явный путь -> SWIPL_PATH -> PATH -> типовые каталоги Windows."""
    for candidate in (custom, swipl_path_setting()):
        if candidate:
            p = Path(candidate)
            if p.is_dir():                      # указали папку установки или bin
                for sub in ("swipl.exe", "swipl", "bin/swipl.exe", "bin/swipl"):
                    if (p / sub).is_file():
                        return str(p / sub)
            if p.is_file():
                return str(p)
            found = shutil.which(candidate)
            if found:
                return found
            return None                         # явный путь указан, но неверен
    found = shutil.which("swipl")
    if found:
        return found
    if os.name == "nt":
        for c in _WINDOWS_CANDIDATES:
            if os.path.isfile(c):
                return c
    return None


def _popen_flags() -> dict:
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return {}


def check_swipl(custom: str | None = None, timeout: int = 20) -> EngineInfo:
    path = find_swipl(custom)
    if not path:
        raise PrologNotFoundError(
            "SWI-Prolog не найден. Установите его (https://www.swi-prolog.org/Download.html), "
            "добавьте папку bin в PATH или укажите полный путь к swipl в боковой панели "
            "либо в переменной SWIPL_PATH (см. README).")
    try:
        proc = subprocess.run([path, "--version"], capture_output=True, timeout=timeout,
                              stdin=subprocess.DEVNULL, **_popen_flags())
    except FileNotFoundError:
        raise PrologNotFoundError(f"Файл не найден или не запускается: {path}") from None
    except subprocess.TimeoutExpired:
        raise PrologTimeoutError("swipl --version не ответил вовремя") from None
    except OSError as exc:
        raise PrologError(f"Не удалось запустить {path}: {exc}") from None
    text = (proc.stdout or b"").decode("utf-8", "replace").strip()
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", text)
    version = tuple(int(x) for x in m.groups()) if m else None
    if proc.returncode != 0 or not text:
        raise PrologError(f"{path} не похож на SWI-Prolog (код возврата {proc.returncode}).")
    return EngineInfo(path=path, version_text=text, version=version)


class _ServerUnavailable(Exception):
    """Постоянный процесс Prolog недоступен; допустим одноразовый fallback."""


class PrologServer:
    """Долгоживущий процесс SWI-Prolog с обменом JSON-строками.

    Один процесс на пару (swipl, engine_file). Это убирает запуск swipl
    на каждый запрос — заметно для пакетной диагностики и серии кликов.
    """

    def __init__(self, swipl: str, engine_file: Path, timeout: int):
        self.swipl = swipl
        self.engine_file = Path(engine_file)
        self.timeout = timeout
        self._lock = threading.Lock()
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=40)
        self._proc: subprocess.Popen | None = None
        self._out_thread: threading.Thread | None = None
        self._err_thread: threading.Thread | None = None
        self._start()

    # --- жизненный цикл
    def _start(self) -> None:
        cmd = [self.swipl, "-q", "-g", "ilp_engine:server", "-t", "halt",
               str(self.engine_file)]
        try:
            self._proc = subprocess.Popen(
                cmd, cwd=str(self.engine_file.parent.parent),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                **_popen_flags())
        except OSError as exc:
            raise _ServerUnavailable(f"не удалось запустить постоянный процесс: {exc}") from None
        self._out_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._err_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._out_thread.start()
        self._err_thread.start()

    def _read_stdout(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            self._lines.put(None)
            return
        try:
            for line in proc.stdout:
                self._lines.put(line)
        finally:
            self._lines.put(None)

    def _read_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for line in proc.stderr:
            text = line.rstrip()
            if text:
                self._stderr.append(text)

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def stderr_tail(self) -> str:
        return "\n".join(self._stderr)

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
            proc.wait(timeout=2)
        except Exception:  # noqa: BLE001
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass

    # --- запрос
    def request(self, request: dict) -> dict:
        with self._lock:
            if not self.alive:
                raise _ServerUnavailable("постоянный процесс Prolog завершился")
            payload = json.dumps(request, ensure_ascii=True)
            try:
                assert self._proc is not None and self._proc.stdin is not None
                self._proc.stdin.write(payload + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                self.close()
                raise _ServerUnavailable(f"канал к Prolog закрыт: {exc}") from None
            try:
                line = self._lines.get(timeout=self.timeout)
            except queue.Empty:
                self.close()
                raise PrologTimeoutError(
                    f"SWI-Prolog не уложился в {self.timeout} с (постоянный процесс). "
                    "Уменьшите размер задачи или увеличьте ILP_TIMEOUT_SECONDS.") from None
            if line is None:
                tail = self.stderr_tail()
                self.close()
                raise _ServerUnavailable(
                    "постоянный процесс Prolog завершился без ответа"
                    + (f":\n{tail}" if tail else ""))
            try:
                response = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PrologEngineError(
                    f"Ответ постоянного процесса Prolog не является JSON: {exc}") from None
            if not isinstance(response, dict):
                raise PrologEngineError("Ответ постоянного процесса Prolog имеет неожиданный формат.")
            return response


_SERVERS: dict[tuple[str, str, int, int], PrologServer] = {}
_SERVERS_LOCK = threading.Lock()


def _persistent_enabled() -> bool:
    return os.environ.get("ILP_PERSISTENT", "1").strip().lower() not in {"0", "false", "no", "off"}


def _get_server(info: EngineInfo, engine_file: Path, timeout: int) -> PrologServer:
    key = (info.path, str(engine_file.resolve()), int(timeout),
           engine_file.stat().st_mtime_ns)
    with _SERVERS_LOCK:
        server = _SERVERS.get(key)
        if server is None or not server.alive:
            if server is not None:
                server.close()
            server = PrologServer(info.path, engine_file, timeout)
            _SERVERS[key] = server
        return server


@atexit.register
def _close_servers() -> None:
    with _SERVERS_LOCK:
        for server in list(_SERVERS.values()):
            server.close()
        _SERVERS.clear()


class PrologEngine:
    """Обёртка над процессом swipl. Каждый запрос — отдельный изолированный запуск."""

    def __init__(self, swipl_path: str | None = None, timeout: int | None = None,
                 engine_file: Path | None = None):
        self._custom = swipl_path
        self.timeout = timeout or timeout_setting()
        self.engine_file = Path(engine_file or ENGINE_FILE)
        self._info: EngineInfo | None = None

    # --- информация
    def info(self) -> EngineInfo:
        if self._info is None:
            self._info = check_swipl(self._custom)
        return self._info

    def ping(self) -> dict:
        return self._run({"command": "ping"})

    # --- команды
    def train(self, projects: list[dict], params: dict) -> dict:
        """Обучение правил. projects — ТОЛЬКО обучающая часть."""
        req = {"command": "train", "params": params, "projects": _prolog_projects(projects)}
        return self._run(req)

    def diagnose(self, rules: list[dict], projects: list[dict], max_proofs: int = 3,
                 classes: list[str] | None = None) -> dict:
        """Применение выученных правил к проектам (с доказательствами)."""
        params: dict = {"max_proofs": max_proofs}
        if classes:
            params["classes"] = classes
        req = {"command": "diagnose", "params": params, "rules": _prolog_rules(rules),
               "projects": _prolog_projects(projects)}
        return self._run(req)

    # --- низкий уровень
    def _run(self, request: dict) -> dict:
        info = self.info()
        if not self.engine_file.is_file():
            raise PrologEngineError(f"Не найден файл логического ядра: {self.engine_file}")
        if _persistent_enabled():
            try:
                server = _get_server(info, self.engine_file, self.timeout)
                response = server.request(request)
            except _ServerUnavailable as exc:
                # Постоянный процесс не поднялся/умер — не роняем приложение,
                # выполняем запрос старым одноразовым способом.
                response = self._run_oneshot(info, request)
                response["_server_error"] = str(exc)
                return response
            if response.get("status") != "ok":
                err = response.get("error") or {}
                raise PrologEngineError(
                    f"Логическое ядро сообщило об ошибке: {err.get('message', 'неизвестная ошибка')}")
            response["_stderr"] = server.stderr_tail()
            return response
        return self._run_oneshot(info, request)

    def _run_oneshot(self, info: EngineInfo, request: dict) -> dict:
        tmp = tempfile.mkdtemp(prefix="ilp_")
        try:
            req_path = Path(tmp) / "request.json"
            resp_path = Path(tmp) / "response.json"
            # ensure_ascii=True: файл запроса чисто ASCII, кодировка не может подвести
            req_path.write_text(json.dumps(request, ensure_ascii=True), encoding="ascii")
            cmd = [info.path, "-q", "-g", "main", "-t", "halt", str(self.engine_file),
                   "--", "request.json", "response.json"]
            try:
                proc = subprocess.run(cmd, cwd=tmp, capture_output=True, timeout=self.timeout,
                                      stdin=subprocess.DEVNULL, **_popen_flags())
            except subprocess.TimeoutExpired:
                raise PrologTimeoutError(
                    f"SWI-Prolog не уложился в {self.timeout} с. Уменьшите max_body_length "
                    "или размер набора, либо увеличьте ILP_TIMEOUT_SECONDS.") from None
            except FileNotFoundError:
                raise PrologNotFoundError(f"Не удалось запустить {info.path}") from None
            except OSError as exc:
                raise PrologError(f"Ошибка запуска SWI-Prolog: {exc}") from None
            stderr = (proc.stderr or b"").decode("utf-8", "replace").strip()
            if not resp_path.is_file() or resp_path.stat().st_size == 0:
                raise PrologEngineError(
                    "SWI-Prolog не вернул ответ (код возврата "
                    f"{proc.returncode}). Сообщения процесса:\n{stderr or '(пусто)'}")
            try:
                response = json.loads(resp_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise PrologEngineError(
                    f"Ответ SWI-Prolog не является корректным JSON: {exc}\n{stderr}") from None
            if not isinstance(response, dict):
                raise PrologEngineError("Ответ SWI-Prolog имеет неожиданный формат.")
            if response.get("status") != "ok":
                err = response.get("error") or {}
                raise PrologEngineError(
                    f"Логическое ядро сообщило об ошибке: {err.get('message', 'неизвестная ошибка')}")
            response["_stderr"] = stderr
            return response
        finally:
            if not os.environ.get("ILP_KEEP_TEMP"):
                shutil.rmtree(tmp, ignore_errors=True)


def _prolog_projects(projects: list[dict]) -> list[dict]:
    """Только то, что нужно ядру: id, labels, facts (null/[] — оба дают отсутствие фактов)."""
    out = []
    for p in projects:
        item = {"id": p["id"], "facts": p["facts"]}
        if p.get("labels") is not None:
            item["labels"] = p["labels"]
        out.append(item)
    return out


def _prolog_rules(rules: list[dict]) -> list[dict]:
    out = []
    for rule in rules:
        body = []
        for lit in rule["body"]:
            if lit.get("neq"):
                body.append({"neq": True, "type": lit["type"],
                             "left": lit["left"], "right": lit["right"]})
                continue
            item = {"predicate": lit["predicate"]}
            if "const" in lit:
                item["const"] = lit["const"]
            else:
                item["var"] = lit["var"]
            if lit.get("negated"):
                item["negated"] = True
            body.append(item)
        out.append({"target": rule["target"], "body": body})
    return out
