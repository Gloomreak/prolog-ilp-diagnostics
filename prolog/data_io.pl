/*  data_io.pl — схема данных и обмен JSON между Python и SWI-Prolog.

    Модуль НЕ содержит алгоритма обучения. Он:
      * описывает словарь фоновых предикатов (BK) и классы ошибок;
      * читает JSON-запрос и превращает его в внутренние термы Prolog;
      * превращает результаты движка в JSON-ответ.

    Внутренние термы:
      project(Id, Labels, Facts)   Facts = [fact(Predicate, Value), ...]
      rule(Class, Body)            Body  = [Predicate-VarIndex, ...]
*/

:- module(data_io,
          [ error_class/1,
            fact_field/2,
            bk_predicate/2,
            read_request/2,
            write_response/2,
            read_json_stream/2,
            write_json_stream/2,
            request_command/2,
            request_projects/2,
            request_rules/2,
            request_options/2,
            train_response/2,
            diagnose_response/2,
            ping_response/1,
            error_response/2,
            text_atom/2
          ]).

:- encoding(utf8).

% library(json) появилась в SWI-Prolog 9.3; в более ранних версиях JSON
% лежит в library(http/json). Подключаем то, что доступно.
:- if(exists_source(library(json))).
:- use_module(library(json)).
:- else.
:- use_module(library(http/json)).
:- endif.

:- use_module(library(lists)).
:- use_module(library(apply)).

% ---------------------------------------------------------------------------
% Схема: классы ошибок и словарь фоновых предикатов
% ---------------------------------------------------------------------------

error_class(missing_dependency).
error_class(incompatible_version).
error_class(missing_config_key).
error_class(missing_environment_variable).

% bk_predicate(?Predicate, ?ArgType) — разрешённый словарь тела правила
% (языковое ограничение, language bias). Тип нужен, чтобы переменные
% объектов разных типов (пакет, ключ, ...) не смешивались.
bk_predicate(requires_package,       package).
bk_predicate(package_absent,         package).
bk_predicate(version_mismatch,       package).
bk_predicate(requires_config_key,    config_key).
bk_predicate(config_key_absent,      config_key).
bk_predicate(requires_env_variable,  env_variable).
bk_predicate(env_variable_absent,    env_variable).
bk_predicate(uses_build_tool,        build_tool).
bk_predicate(uses_runtime,           runtime).

% fact_field(?JsonField, ?Predicate) — какое поле facts даёт какие факты BK.
fact_field(required_packages,      requires_package).
fact_field(absent_packages,        package_absent).
fact_field(version_mismatches,     version_mismatch).
fact_field(required_config_keys,   requires_config_key).
fact_field(absent_config_keys,     config_key_absent).
fact_field(required_env_variables, requires_env_variable).
fact_field(absent_env_variables,   env_variable_absent).
fact_field(build_tools,            uses_build_tool).
fact_field(runtimes,               uses_runtime).

% ---------------------------------------------------------------------------
% Чтение и запись файлов
% ---------------------------------------------------------------------------

read_request(File, Request) :-
    setup_call_cleanup(
        open(File, read, In, [encoding(utf8)]),
        json_read_dict(In, Request),
        close(In)).

write_response(File, Response) :-
    setup_call_cleanup(
        open(File, write, Out, [encoding(utf8)]),
        ( json_write_dict(Out, Response, [width(0)]), nl(Out) ),
        close(Out)).

% Обмен по одному JSON-объекту на строку — для постоянного процесса Prolog.
read_json_stream(Stream, Dict) :-
    json_read_dict(Stream, Dict).

write_json_stream(Stream, Dict) :-
    json_write_dict(Stream, Dict, [width(0)]),
    nl(Stream),
    flush_output(Stream).

% text_atom(+Text, -Atom): строка/атом/число -> атом. Идентификаторы всегда
% атомы, даже если выглядят как числа ('123').
text_atom(X, A) :-
    (   atom(X)   -> A = X
    ;   string(X) -> string_to_atom(X, A)
    ;   number(X) -> format(atom(A), '~w', [X])
    ).

% ---------------------------------------------------------------------------
% Запрос -> внутренние термы
% ---------------------------------------------------------------------------

request_command(Request, Command) :-
    get_dict(command, Request, C),
    text_atom(C, Command).

request_projects(Request, Projects) :-
    (   get_dict(projects, Request, Dicts), is_list(Dicts)
    ->  maplist(project_from_dict, Dicts, Projects)
    ;   Projects = []
    ).

project_from_dict(Dict, project(Id, Labels, Facts)) :-
    get_dict(id, Dict, IdText),
    text_atom(IdText, Id),
    (   get_dict(labels, Dict, Ls), is_list(Ls)
    ->  maplist(text_atom, Ls, Labels)
    ;   Labels = []
    ),
    (   get_dict(facts, Dict, FactsDict), is_dict(FactsDict)
    ->  findall(fact(Pred, Val), dict_fact(FactsDict, Pred, Val), Facts)
    ;   Facts = []
    ).

% Поле отсутствует или равно null -> «неизвестно»: фактов нет.
% Пустой список [] -> «известно, что пусто»: фактов тоже нет, но это
% осознанное состояние, различаемое на стороне Python.
dict_fact(FactsDict, Pred, Val) :-
    fact_field(Field, Pred),
    get_dict(Field, FactsDict, Values),
    is_list(Values),
    member(V, Values),
    text_atom(V, Val).

request_rules(Request, Rules) :-
    (   get_dict(rules, Request, Dicts), is_list(Dicts)
    ->  maplist(rule_from_dict, Dicts, Rules)
    ;   Rules = []
    ).

rule_from_dict(Dict, rule(Class, Body)) :-
    get_dict(target, Dict, T),
    text_atom(T, Class),
    get_dict(body, Dict, LitDicts),
    maplist(literal_from_dict, LitDicts, Body).

literal_from_dict(Dict, Lit) :-
    (   get_dict(neq, Dict, true)
    ->  get_dict(type, Dict, T0), text_atom(T0, Type),
        get_dict(left, Dict, N1), get_dict(right, Dict, N2),
        Lit = neq(Type, N1, N2)
    ;   get_dict(predicate, Dict, P0), text_atom(P0, Pred),
        (   get_dict(const, Dict, C0)
        ->  text_atom(C0, V), Arg = const(V)
        ;   get_dict(var, Dict, N), Arg = N
        ),
        (   get_dict(negated, Dict, true)
        ->  Lit = neg(Pred-Arg)
        ;   Lit = Pred-Arg
        )
    ).

option_key(max_body_length).
option_key(min_support).
option_key(min_precision).
option_key(max_rules_per_class).
option_key(max_candidates).
option_key(max_proofs).
option_key(top_rejected).
option_key(smoothing).
option_key(mdl_penalty).
option_key(max_constants).

request_options(Request, Options) :-
    (   get_dict(params, Request, P), is_dict(P)
    ->  true
    ;   dict_create(P, params, [])
    ),
    findall(Opt,
            ( option_key(Key),
              get_dict(Key, P, Val),
              number(Val),
              Opt =.. [Key, Val] ),
            Opts1),
    (   get_dict(classes, P, Cs0), is_list(Cs0)
    ->  maplist(text_atom, Cs0, Cs),
        Opts2 = [classes(Cs)|Opts1]
    ;   Opts2 = Opts1
    ),
    (   get_dict(body_predicates, P, Ps0), is_list(Ps0)
    ->  maplist(text_atom, Ps0, Ps),
        Options = [body_predicates(Ps)|Opts2]
    ;   Options = Opts2
    ).

% ---------------------------------------------------------------------------
% Результаты -> JSON-словари
% ---------------------------------------------------------------------------

% Любой пользовательский текст (id, имена пакетов) отдаём СТРОКОЙ, чтобы
% идентификатор вида «true»/«null» не превратился в JSON-литерал.
to_str(X, S) :- atom_string(X, S).

train_response(result(Rules, ClassInfos, Stats, Warnings, log(Levels, Rejected)),
               Response) :-
    maplist(rule_dict, Rules, RuleDicts),
    maplist(class_info_pair, ClassInfos, ClassPairs),
    dict_create(ClassesDict, classes, ClassPairs),
    dict_create(StatsDict, stats, Stats),
    maplist(warning_dict, Warnings, WarnDicts),
    maplist(level_dict, Levels, LevelDicts),
    maplist(rejected_dict, Rejected, RejDicts),
    dict_create(Model, model, [rules-RuleDicts, classes-ClassesDict]),
    dict_create(LogDict, log, [levels-LevelDicts, rejected-RejDicts]),
    dict_create(Response, response,
                [ status-ok, model-Model, stats-StatsDict,
                  log-LogDict, warnings-WarnDicts ]).

rule_dict(rule(Class, Lits, Text, Metrics), Dict) :-
    maplist(literal_dict, Lits, LitDicts),
    dict_create(Dict, rule,
                [ target-Class, rule-Text, body-LitDicts | Metrics ]).

literal_dict(lit(Pred, N, Name), Dict) :-
    integer(N),
    dict_create(Dict, literal,
                [ predicate-Pred, var-N, args-['Project', Name] ]).
literal_dict(lit(Pred, const(V), Name), Dict) :-
    dict_create(Dict, literal,
                [ predicate-Pred, const-V, args-['Project', Name] ]).
literal_dict(neg_lit(Pred, N, Name), Dict) :-
    integer(N),
    dict_create(Dict, literal,
                [ predicate-Pred, var-N, negated-true,
                  args-['Project', Name] ]).
literal_dict(neg_lit(Pred, const(V), Name), Dict) :-
    dict_create(Dict, literal,
                [ predicate-Pred, const-V, negated-true,
                  args-['Project', Name] ]).
literal_dict(neq_lit(Type, N1, N2, Name1, Name2), Dict) :-
    dict_create(Dict, literal,
                [ neq-true, type-Type, left-N1, right-N2,
                  args-[Name1, Name2] ]).

class_info_pair(class_info(Class, Pairs), Class-Dict) :-
    dict_create(Dict, class_info, Pairs).

warning_dict(warning(Code, Detail), Dict) :-
    dict_create(Dict, warning, [code-Code, detail-Detail]).

level_dict(level(L, Generated, Tested, Extended), Dict) :-
    dict_create(Dict, level,
                [ length-L, generated-Generated,
                  tested-Tested, extended-Extended ]).

rejected_dict(rejected(Class, Text, TP, FP, P, R, F1, Reasons), Dict) :-
    dict_create(Dict, rejected,
                [ target-Class, rule-Text, tp-TP, fp-FP,
                  precision-P, recall-R, f1-F1, reasons-Reasons ]).

diagnose_response(Results, Response) :-
    maplist(project_result_dict, Results, Dicts),
    dict_create(Response, response, [status-ok, results-Dicts]).

project_result_dict(project_result(Id, ClassResults), Dict) :-
    maplist(class_result_pair, ClassResults, Pairs),
    dict_create(ClassesDict, classes, Pairs),
    to_str(Id, IdS),
    dict_create(Dict, project_result, [id-IdS, classes-ClassesDict]).

class_result_pair(class_result(Class, Status, Matches), Class-Dict) :-
    maplist(match_dict, Matches, MatchDicts),
    dict_create(Dict, class_result, [status-Status, matches-MatchDicts]).

match_dict(match(Index, Proofs), Dict) :-
    maplist(proof_dict, Proofs, ProofDicts),
    dict_create(Dict, match, [rule_index-Index, proofs-ProofDicts]).

proof_dict(proof(Bindings, Facts), Dict) :-
    maplist(binding_pair, Bindings, Pairs),
    dict_create(BindingsDict, bindings, Pairs),
    maplist(fact_dict, Facts, FactDicts),
    dict_create(Dict, proof, [bindings-BindingsDict, facts-FactDicts]).

binding_pair(Name-Value, Name-S) :- to_str(Value, S).

fact_dict(fact(Pred, Id, Value), Dict) :-
    to_str(Id, IdS),
    to_str(Value, VS),
    dict_create(Dict, fact, [predicate-Pred, args-[IdS, VS]]).

ping_response(Response) :-
    current_prolog_flag(version, V),
    dict_create(Response, response, [status-ok, swi_version-V]).

error_response(E, Response) :-
    format(string(Msg), '~p', [E]),
    dict_create(Error, error, [code-engine_error, message-Msg]),
    dict_create(Response, response, [status-error, error-Error]).
