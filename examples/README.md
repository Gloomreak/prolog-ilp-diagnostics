# Примеры экспорта

Артефакты в этой папке (`model.json`, `learned_rules.pl`, `diagnosis.json/csv`,
`experiment.json`, `comparison.csv`) сформированы сквозным сценарием проекта
с установленным настоящим SWI-Prolog. Версия Python-библиотек и SWI-Prolog,
использованных при генерации, указана в `model.json` и результатах эксперимента.
Время обучения относится к этой машине и не является гарантией для другого окружения.

Чтобы обновить примеры после изменения кода:

```powershell
python scripts/make_examples.py
```

Для нестандартного расположения SWI-Prolog передайте путь явно:

```powershell
python scripts/make_examples.py --swipl "C:\Program Files\swipl\bin\swipl.exe"
```
