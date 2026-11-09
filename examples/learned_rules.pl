% Правила, выученные ILP-движком (SWI-Prolog).
% Создано (UTC): 2026-11-09T22:35:23+00:00
% Параметры: max_body_length=3, min_support=2, min_precision=0.8, max_rules_per_class=5
% Обучающих проектов: 400
% Правила применяются к фактам BK (requires_package/2, package_absent/2, ...).
% Пример: ?- requires_package(p1, pandas), package_absent(p1, pandas), has_error(p1, E).

:- discontiguous has_error/2.
:- dynamic requires_package/2, package_absent/2, version_mismatch/2,
           requires_config_key/2, config_key_absent/2,
           requires_env_variable/2, env_variable_absent/2,
           uses_build_tool/2, uses_runtime/2.

% missing_dependency: TP=128 FP=0 FN=0 precision=1.0 recall=1.0 f1=1.0
has_error(Project, missing_dependency) :-
    requires_package(Project, Package),
    package_absent(Project, Package).

% incompatible_version: TP=126 FP=0 FN=0 precision=1.0 recall=1.0 f1=1.0
has_error(Project, incompatible_version) :-
    requires_package(Project, Package),
    version_mismatch(Project, Package).

% missing_config_key: TP=132 FP=0 FN=0 precision=1.0 recall=1.0 f1=1.0
has_error(Project, missing_config_key) :-
    requires_config_key(Project, Key),
    config_key_absent(Project, Key).

% missing_environment_variable: TP=107 FP=0 FN=0 precision=1.0 recall=1.0 f1=1.0
has_error(Project, missing_environment_variable) :-
    requires_env_variable(Project, Variable),
    env_variable_absent(Project, Variable).
