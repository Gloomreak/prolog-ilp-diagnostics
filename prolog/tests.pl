/*  tests.pl — модульные тесты логического ядра (plunit).

    Запуск из корня проекта:
        swipl -g run_tests -t halt prolog/tests.pl
    Код возврата 0 — все тесты пройдены.
*/

:- encoding(utf8).
:- use_module(library(plunit)).
:- use_module(library(lists)).
:- use_module(ilp_engine).
:- use_module(data_io).

% ---------------------------------------------------------------------------
% Тестовые наборы (маленькие, собраны вручную)
% ---------------------------------------------------------------------------

proj(Id, Labels, Facts, project(Id, Labels, Facts)).

% Набор A: missing_dependency <=> пакет и требуется, и отсутствует.
% a4, a6, a7 — контрпримеры: пакет отсутствует, но НЕ требуется.
dataset_a([P1, P2, P3, P4, P5, P6, P7, P8]) :-
    proj(a1, [missing_dependency],
         [fact(requires_package, numpy), fact(requires_package, pandas),
          fact(package_absent, pandas), fact(uses_runtime, python)], P1),
    proj(a2, [missing_dependency],
         [fact(requires_package, flask), fact(package_absent, flask),
          fact(uses_build_tool, pip)], P2),
    proj(a3, [missing_dependency],
         [fact(requires_package, django), fact(requires_package, requests),
          fact(package_absent, requests), fact(uses_runtime, python)], P3),
    proj(a4, [],
         [fact(requires_package, numpy), fact(package_absent, scipy),
          fact(uses_runtime, python)], P4),
    proj(a5, [],
         [fact(requires_package, pandas), fact(requires_package, numpy),
          fact(uses_build_tool, pip)], P5),
    proj(a6, [],
         [fact(requires_package, flask), fact(package_absent, pandas),
          fact(uses_runtime, node)], P6),
    proj(a7, [],
         [fact(requires_package, requests), fact(package_absent, django),
          fact(uses_runtime, python)], P7),
    proj(a8, [missing_dependency],
         [fact(requires_package, torch), fact(package_absent, torch),
          fact(uses_build_tool, pip)], P8).

% Набор B: те же факты, но метки missing_dependency сняты.
dataset_b(Projects) :-
    dataset_a(A),
    findall(project(Id, [], Facts), member(project(Id, _, Facts), A), Projects).

% Набор C: класс missing_config_key задаётся связью «ключ требуется И отсутствует».
dataset_c([P1, P2, P3, P4, P5, P6]) :-
    proj(c1, [missing_config_key],
         [fact(requires_config_key, 'DATABASE_URL'),
          fact(config_key_absent, 'DATABASE_URL')], P1),
    proj(c2, [missing_config_key],
         [fact(requires_config_key, 'SECRET_KEY'),
          fact(requires_config_key, 'DEBUG'),
          fact(config_key_absent, 'SECRET_KEY')], P2),
    proj(c3, [missing_config_key],
         [fact(requires_config_key, 'PORT'), fact(config_key_absent, 'PORT')], P3),
    proj(c4, [],
         [fact(requires_config_key, 'PORT'), fact(config_key_absent, 'DEBUG')], P4),
    proj(c5, [],
         [fact(requires_config_key, 'DEBUG'), fact(config_key_absent, 'HOST')], P5),
    proj(c6, [],
         [fact(requires_config_key, 'HOST')], P6).

% Набор D: метки не связаны с фактами — надёжного правила быть не должно.
dataset_d([P1, P2, P3, P4, P5, P6, P7, P8]) :-
    proj(d1, [missing_dependency], [fact(uses_runtime, python)], P1),
    proj(d2, [],                   [fact(uses_runtime, python)], P2),
    proj(d3, [missing_dependency], [fact(uses_runtime, node)],   P3),
    proj(d4, [],                   [fact(uses_runtime, node)],   P4),
    proj(d5, [],                   [fact(uses_runtime, python)], P5),
    proj(d6, [],                   [fact(uses_runtime, node)],   P6),
    proj(d7, [],                   [fact(uses_runtime, python)], P7),
    proj(d8, [],                   [fact(uses_runtime, node)],   P8).

% Набор E: ошибка зависит от контекста. Положительные проекты используют
% poetry; отрицательные с теми же «пакетными» фактами — pip, поэтому
% правило длины 2 имеет FP, а правило с константой poetry — нет.
dataset_e([P1, P2, P3, P4, P5, P6, P7, P8]) :-
    proj(e1, [missing_dependency],
         [fact(requires_package, a), fact(package_absent, a),
          fact(uses_build_tool, poetry)], P1),
    proj(e2, [missing_dependency],
         [fact(requires_package, b), fact(package_absent, b),
          fact(uses_build_tool, poetry)], P2),
    proj(e3, [missing_dependency],
         [fact(requires_package, c), fact(package_absent, c),
          fact(uses_build_tool, poetry)], P3),
    proj(e4, [],
         [fact(requires_package, a), fact(package_absent, a),
          fact(uses_build_tool, make)], P4),
    proj(e5, [],
         [fact(requires_package, b), fact(package_absent, b),
          fact(uses_build_tool, pip)], P5),
    proj(e6, [],
         [fact(requires_package, c), fact(uses_build_tool, poetry)], P6),
    proj(e7, [],
         [fact(requires_package, d), fact(uses_build_tool, poetry)], P7),
    proj(e8, [],
         [fact(package_absent, z), fact(uses_build_tool, poetry)], P8).

% Набор F: положительные проекты без runtime-факта, отрицательные — с
% uses_runtime(python). Правило должно выучить отрицательный литерал.
dataset_f([P1, P2, P3, P4, P5, P6]) :-
    proj(f1, [missing_dependency],
         [fact(requires_package, a), fact(package_absent, a)], P1),
    proj(f2, [missing_dependency],
         [fact(requires_package, b), fact(package_absent, b)], P2),
    proj(f3, [missing_dependency],
         [fact(requires_package, c), fact(package_absent, c)], P3),
    proj(f4, [],
         [fact(requires_package, a), fact(package_absent, a),
          fact(uses_runtime, python)], P4),
    proj(f5, [],
         [fact(requires_package, b), fact(package_absent, b),
          fact(uses_runtime, python)], P5),
    proj(f6, [],
         [fact(requires_package, c), fact(package_absent, c),
          fact(uses_runtime, python)], P6).

% ---------------------------------------------------------------------------
% Вспомогательные предикаты тестов
% ---------------------------------------------------------------------------

rules_of_class(Class, Rules, ClassRules) :-
    findall(R, ( member(R, Rules), R = rule(Class, _, _, _) ), ClassRules).

rule_predicates(rule(_, Lits, _, _), Preds) :-
    findall(P, member(lit(P, _, _), Lits), Ps),
    msort(Ps, Preds).

metric(rule(_, _, _, Metrics), Name, Value) :-
    Pair =.. [-, Name, Value],
    memberchk(Pair, Metrics).

class_status(ClassInfos, Class, Status) :-
    memberchk(class_info(Class, Info), ClassInfos),
    memberchk(status-Status, Info).

% ---------------------------------------------------------------------------
:- begin_tests(learning).

test(learns_join_rule_not_single_literal, [nondet]) :-
    dataset_a(Ps),
    train_model(Ps, [], result(Rules, Infos, _, _, _)),
    rules_of_class(missing_dependency, Rules, [Rule|_]),
    rule_predicates(Rule, Preds),
    assertion(Preds == [package_absent, requires_package]),
    metric(Rule, tp, TP), metric(Rule, fp, FP), metric(Rule, fn, FN),
    assertion(TP == 4), assertion(FP == 0), assertion(FN == 0),
    class_status(Infos, missing_dependency, Status),
    assertion(Status == ok).

test(rule_variable_is_shared_between_literals, [nondet]) :-
    dataset_a(Ps),
    train_model(Ps, [], result(Rules, _, _, _, _)),
    rules_of_class(missing_dependency, Rules, [rule(_, Lits, Text, _)|_]),
    findall(Name, member(lit(_, _, Name), Lits), Names),
    sort(Names, Unique),
    assertion(Unique == ['Package']),
    assertion(sub_atom(Text, 0, _, _, 'has_error(Project, missing_dependency) :-')).

test(changing_examples_changes_the_model, [nondet]) :-
    dataset_a(A), dataset_b(B),
    train_model(A, [], result(RulesA, InfosA, _, _, _)),
    train_model(B, [], result(RulesB, InfosB, _, _, _)),
    rules_of_class(missing_dependency, RulesA, RA),
    rules_of_class(missing_dependency, RulesB, RB),
    assertion(RA \== []),
    assertion(RB == []),
    class_status(InfosA, missing_dependency, SA),
    class_status(InfosB, missing_dependency, SB),
    assertion(SA == ok),
    assertion(SB == no_positive_examples).

test(learns_rule_for_another_class_from_its_own_examples, [nondet]) :-
    dataset_c(Ps),
    train_model(Ps, [], result(Rules, _, _, _, _)),
    rules_of_class(missing_config_key, Rules, [Rule|_]),
    rule_predicates(Rule, Preds),
    assertion(Preds == [config_key_absent, requires_config_key]),
    rules_of_class(missing_dependency, Rules, RD),
    assertion(RD == []).

test(no_reliable_rule_for_unrelated_labels, [nondet]) :-
    dataset_d(Ps),
    train_model(Ps, [], result(Rules, Infos, _, Warnings, _)),
    rules_of_class(missing_dependency, Rules, RD),
    assertion(RD == []),
    class_status(Infos, missing_dependency, Status),
    assertion(Status == no_rule_found),
    assertion(memberchk(warning(no_rule_found, missing_dependency), Warnings)).

test(rejected_candidates_have_reasons, [nondet]) :-
    dataset_a(Ps),
    train_model(Ps, [], result(_, _, _, _, log(_, Rejected))),
    assertion(Rejected \== []),
    forall(member(rejected(_, _, _, _, _, _, _, Reasons), Rejected),
           assertion(Reasons \== [])),
    % одиночный package_absent(Project, Package) отклонён из-за точности
    assertion(( member(rejected(missing_dependency, Text, _, _, _, _, _, Rs), Rejected),
                sub_atom(Text, _, _, _, 'package_absent'),
                \+ sub_atom(Text, _, _, _, 'requires_package'),
                memberchk(low_precision, Rs) )).

test(thresholds_are_respected, [nondet]) :-
    dataset_a(Ps),
    train_model(Ps, [min_support(5)], result(Rules, Infos, _, _, _)),
    % TP = 4 < 5 -> правил быть не может
    rules_of_class(missing_dependency, Rules, RD),
    assertion(RD == []),
    class_status(Infos, missing_dependency, S),
    assertion(S == no_rule_found).

test(max_body_length_one_cannot_express_the_join, [nondet]) :-
    dataset_a(Ps),
    train_model(Ps, [max_body_length(1)], result(Rules, _, _, _, _)),
    rules_of_class(missing_dependency, Rules, RD),
    assertion(RD == []).

test(max_candidates_limit_gives_warning, [nondet]) :-
    dataset_a(Ps),
    train_model(Ps, [max_candidates(2)], result(_, _, Stats, Warnings, _)),
    assertion(memberchk(candidates_tested-2, Stats)),
    assertion(memberchk(truncated-true, Stats)),
    assertion(memberchk(warning(max_candidates_reached, 2), Warnings)).

test(stats_are_consistent, [nondet]) :-
    dataset_a(Ps),
    train_model(Ps, [], result(_, _, Stats, _, log(Levels, _))),
    memberchk(candidates_generated-G, Stats),
    memberchk(candidates_tested-T, Stats),
    memberchk(train_size-N, Stats),
    assertion(G >= T), assertion(T > 0), assertion(N == 8),
    Levels = [level(1, 9, 9, _)|_].

test(max_rules_per_class_is_respected, [nondet]) :-
    dataset_a(Ps),
    train_model(Ps, [max_rules_per_class(1)], result(Rules, _, _, _, _)),
    rules_of_class(missing_dependency, Rules, RD),
    length(RD, Len),
    assertion(Len =< 1).

test(canonical_body_merges_isomorphic_bodies, [nondet]) :-
    canonical_body([package_absent-2, requires_package-2], C1),
    canonical_body([package_absent-1, requires_package-1], C2),
    canonical_body([requires_package-1, package_absent-1], C3),
    assertion(C1 == C2), assertion(C2 == C3),
    canonical_body([requires_package-1, package_absent-2], D1),
    canonical_body([requires_package-2, package_absent-1], D2),
    assertion(D1 == D2),
    assertion(C1 \== D1).

test(refinement_removes_duplicates, [nondet]) :-
    ilp_engine:refine_level([[requires_package-1]],
                            [requires_package, package_absent], Next),
    % {req1,req2}, {req1,abs1}, {req1,abs2}, neg(req1), neg(abs1)
    length(Next, Len),
    assertion(Len == 5).

test(rule_text_is_prolog, [nondet]) :-
    rule_text(missing_dependency,
              [requires_package-1, package_absent-1], Text),
    assertion(Text == 'has_error(Project, missing_dependency) :-\n    requires_package(Project, Package),\n    package_absent(Project, Package).').

test(learns_constant_context_rule, [nondet]) :-
    dataset_e(Ps),
    train_model(Ps, [min_support(2), min_precision(0.8), max_body_length(3)],
                result(Rules, _, _, _, _)),
    rules_of_class(missing_dependency, Rules, [Rule|_]),
    rule_predicates(Rule, Preds),
    assertion(Preds == [package_absent, requires_package, uses_build_tool]),
    metric(Rule, tp, TP), metric(Rule, fp, FP),
    assertion(TP == 3), assertion(FP == 0),
    Rule = rule(_, _, Text, _),
    assertion(sub_atom(Text, _, _, _, 'uses_build_tool(Project, poetry)')).

test(learns_negated_constant_literal, [nondet]) :-
    dataset_f(Ps),
    train_model(Ps, [min_support(2), min_precision(0.8), max_body_length(3)],
                result(Rules, _, _, _, _)),
    rules_of_class(missing_dependency, Rules, [Rule|_]),
    metric(Rule, tp, TP), metric(Rule, fp, FP),
    assertion(TP == 3), assertion(FP == 0),
    Rule = rule(_, _, Text, _),
    assertion(sub_atom(Text, _, _, _, '\\+ uses_runtime(Project, python)')).

test(neq_literal_is_supported_by_covers, [nondet]) :-
    Body = [requires_package-1, package_absent-1,
            requires_package-2, neq(package, 1, 2)],
    P1 = project(n1, [], [fact(requires_package, a), fact(requires_package, b),
                          fact(package_absent, a)]),
    P2 = project(n2, [], [fact(requires_package, a), fact(package_absent, a)]),
    ilp_engine:clear_kb,
    ilp_engine:load_project(P1),
    assertion(ilp_engine:covers(Body, n1)),
    ilp_engine:clear_kb,
    ilp_engine:load_project(P2),
    assertion(\+ ilp_engine:covers(Body, n2)),
    % две разные переменные одного типа канонизируются и не склеиваются
    canonical_body(Body, C1),
    canonical_body([requires_package-2, package_absent-2,
                    requires_package-1, neq(package, 2, 1)], C2),
    assertion(C1 == C2).

:- end_tests(learning).

% ---------------------------------------------------------------------------
:- begin_tests(diagnosis).

learned_rules(Rules) :-
    dataset_a(Ps),
    train_model(Ps, [], result(Learned, _, _, _, _)),
    findall(rule(C, Body),
            ( member(rule(C, Lits, _, _), Learned),
              findall(P-N, member(lit(P, N, _), Lits), Body) ),
            Rules).

test(detects_missing_dependency_with_explanation, [nondet]) :-
    learned_rules(Rules),
    NewProject = project(new1, [],
                         [ fact(requires_package, scikit),
                           fact(requires_package, numpy),
                           fact(package_absent, scikit) ]),
    diagnose_projects(Rules, [NewProject], [], [project_result(new1, CRs)]),
    memberchk(class_result(missing_dependency, Status, Matches), CRs),
    assertion(Status == detected),
    Matches = [match(_, [proof(Bindings, Facts)|_])|_],
    assertion(memberchk('Package'-scikit, Bindings)),
    assertion(memberchk(fact(requires_package, new1, scikit), Facts)),
    assertion(memberchk(fact(package_absent, new1, scikit), Facts)).

test(absent_but_not_required_is_not_an_error, [nondet]) :-
    learned_rules(Rules),
    NewProject = project(new2, [],
                         [ fact(requires_package, numpy),
                           fact(package_absent, scipy) ]),
    diagnose_projects(Rules, [NewProject], [], [project_result(new2, CRs)]),
    memberchk(class_result(missing_dependency, Status, _), CRs),
    assertion(Status == not_detected).

test(unknown_is_not_absent, [nondet]) :-
    % у проекта нет фактов package_absent вообще («неизвестно») -> не срабатывает
    learned_rules(Rules),
    NewProject = project(new3, [], [ fact(requires_package, numpy) ]),
    diagnose_projects(Rules, [NewProject], [], [project_result(new3, CRs)]),
    memberchk(class_result(missing_dependency, Status, _), CRs),
    assertion(Status == not_detected).

test(class_without_rules_is_reported_as_such, [nondet]) :-
    learned_rules(Rules),
    NewProject = project(new4, [], [ fact(requires_package, numpy) ]),
    diagnose_projects(Rules, [NewProject], [], [project_result(new4, CRs)]),
    memberchk(class_result(incompatible_version, Status, Matches), CRs),
    assertion(Status == no_rules),
    assertion(Matches == []).

test(max_proofs_zero_gives_only_the_verdict, [nondet]) :-
    learned_rules(Rules),
    NewProject = project(new5, [],
                         [ fact(requires_package, x), fact(package_absent, x) ]),
    diagnose_projects(Rules, [NewProject], [max_proofs(0)], [project_result(new5, CRs)]),
    memberchk(class_result(missing_dependency, detected, [match(_, Proofs)|_]), CRs),
    assertion(Proofs == []).

test(invalid_rule_is_rejected, [nondet]) :-
    catch(( diagnose_projects([rule(missing_dependency, [no_such_predicate-1])],
                              [], [], _),
            fail ),
          error(domain_error(literal, _), _), true).

:- end_tests(diagnosis).

% ---------------------------------------------------------------------------
% Сквозной тест протокола JSON: запрос -> файл -> движок -> файл -> ответ
% ---------------------------------------------------------------------------
:- begin_tests(json_protocol).

write_text_file(File, Text) :-
    setup_call_cleanup(open(File, write, S, [encoding(utf8)]),
                       write(S, Text),
                       close(S)).

delete_test_file(File) :-
    ( exists_file(File) -> delete_file(File) ; true ).

test_file_base(Prefix, Base) :-
    working_directory(Cwd, Cwd),
    directory_file_path(Cwd, '_tmp', TempDir),
    make_directory_path(TempDir),
    flag(plunit_file_counter, Id, Id + 1),
    format(atom(Name), 'plunit_~w_~d', [Prefix, Id]),
    directory_file_path(TempDir, Name, Base).

test(ping, [nondet]) :-
    test_file_base(ping, Base),
    atom_concat(Base, '.json', Req),
    atom_concat(Base, '_resp.json', Resp),
    delete_test_file(Req),
    delete_test_file(Resp),
    write_text_file(Req, '{"command": "ping"}'),
    setup_call_cleanup(
        true,
        ( run_request(Req, Resp),
          read_request(Resp, Dict),
          get_dict(status, Dict, S),
          assertion(text_atom(S, ok)) ),
        ( delete_test_file(Req), delete_test_file(Resp) )).

test(train_via_json, [nondet]) :-
    test_file_base(train, Base),
    atom_concat(Base, '.json', Req),
    atom_concat(Base, '_resp.json', Resp),
    Json = '{"command":"train","params":{"min_support":2,"min_precision":0.8,"max_body_length":3},"projects":[{"id":"x1","labels":["missing_dependency"],"facts":{"required_packages":["a","b"],"absent_packages":["b"],"version_mismatches":[],"build_tools":null}},{"id":"x2","labels":["missing_dependency"],"facts":{"required_packages":["c"],"absent_packages":["c"]}},{"id":"x3","labels":[],"facts":{"required_packages":["a"],"absent_packages":["z"]}},{"id":"x4","labels":[],"facts":{"required_packages":["c"],"absent_packages":["a"]}}]}',
    delete_test_file(Req),
    delete_test_file(Resp),
    write_text_file(Req, Json),
    setup_call_cleanup(
        true,
        ( run_request(Req, Resp),
          read_request(Resp, Dict),
          get_dict(status, Dict, S),
          assertion(text_atom(S, ok)),
          get_dict(model, Dict, Model),
          get_dict(rules, Model, Rules),
          assertion(Rules \== []),
          Rules = [R|_],
          get_dict(precision, R, P),
          assertion(P =:= 1) ),
        ( delete_test_file(Req), delete_test_file(Resp) )).

test(bad_request_gives_error_response, [nondet]) :-
    test_file_base(bad_request, Base),
    atom_concat(Base, '.json', Req),
    atom_concat(Base, '_resp.json', Resp),
    delete_test_file(Req),
    delete_test_file(Resp),
    write_text_file(Req, '{"command": "nonsense"}'),
    setup_call_cleanup(
        true,
        ( run_request(Req, Resp),
          read_request(Resp, Dict),
          get_dict(status, Dict, S),
          assertion(text_atom(S, error)) ),
        ( delete_test_file(Req), delete_test_file(Resp) )).

:- end_tests(json_protocol).
