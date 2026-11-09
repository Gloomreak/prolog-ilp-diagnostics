"""Дымовой тест интерфейса через streamlit.testing (пропускается без streamlit)."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")
PAGES = ["Обзор", "Данные", "Обучение ILP", "Диагностика", "Пакетная диагностика",
         "Эксперимент", "Экспорт / О проекте"]


def test_every_page_renders_without_exception():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    for page in PAGES:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, page


def test_demo_training_model_import_and_diagnosis_flow():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.sidebar.radio[0].set_value("Данные").run()
    at.button(key="load_demo_dataset").click().run()
    assert not at.exception
    assert len(at.session_state["dataset"]["projects"]) == 500

    at.sidebar.radio[0].set_value("Обучение ILP").run()
    at.button(key="train_model").click().run()
    assert not at.exception
    model_json = at.session_state["model"]
    assert len(model_json["model"]["rules"]) == 4

    from core.exports import model_to_json
    at.file_uploader(key="model_upload").upload(
        "model.json", model_to_json(model_json).encode("utf-8")).run()
    at.button(key="apply_model").click().run()
    assert not at.exception
    assert len(at.session_state["model"]["model"]["rules"]) == 4

    at.sidebar.radio[0].set_value("Диагностика").run()
    at.button(key="run_diagnosis").click().run()
    assert not at.exception
    statuses = at.session_state["diagnosis"]["parsed"]["classes"]
    assert statuses["missing_dependency"]["status"] == "detected"
    assert statuses["missing_environment_variable"]["status"] == "not_detected"

    mode_radio = at.radio[0]
    mode_radio.set_value(mode_radio.options[-1]).run()
    at.file_uploader(key="project_files").upload(
        "requirements.txt", b"requests>=2").run()
    assert not at.exception
