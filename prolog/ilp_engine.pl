/*  ilp_engine.pl — ограниченный движок индуктивного логического
    программирования (ILP) для диагностики ошибок конфигурации.

    ЧТО ДЕЛАЕТ ДВИЖОК
    -----------------
    Дано: фоновые знания BK (факты о проектах), положительные E+ и
    отрицательные E- примеры целевого отношения has_error(Project, Class).
    Найти: набор Horn-правил вида

        has_error(Project, Class) :- Lit1, Lit2, ..., LitK.

    которые покрывают E+ и по возможности не покрывают E-.

    В коде НЕТ готовых диагностических правил. Есть только
      * словарь допустимых литералов тела (data_io:bk_predicate/2) —
        это языковое ограничение (language bias), и
      * общий алгоритм поиска, который строит кандидаты из этого словаря,
        проверяет их логическим выводом Prolog на обучающих примерах и
        отбирает по метрикам.

    АЛГОРИТМ (сверху вниз, по уровням — длина тела 1..max_body_length)
    -----------------------------------------------------------------
    1. Уровень 1: все однолитеральные тела.
    2. Каждое тело проверяется: логический вывод (covers/2) определяет, какие
       обучающие проекты оно покрывает. Для каждого класса считаются
       TP, FP, FN, precision, recall, F1.
    3. Отсечение: покрытие конъюнкции может только УМЕНЬШАТЬСЯ при добавлении
       литерала (монотонность). Поэтому тело, у которого поддержка (TP) ни для
       одного класса не достигает min_support, не уточняется дальше.
    4. Уточнение: к каждому прошедшему тело добавляется один литерал —
       с уже использованной переменной того же типа или с новой.
       Изоморфные тела (отличающиеся только именами переменных или порядком
       литералов) сводятся к каноническому виду и удаляются как дубликаты.
    5. Отбор по каждому классу: пороги min_support / min_precision, затем
       жадное покрытие положительных примеров (до max_rules_per_class).
    6. Упрощение: удаление литералов, не влияющих на покрытие обучающей
       выборки, и правил, чьи положительные примеры уже покрыты другими.

    Тело правила хранится как отсортированный список Pred-N, где N — номер
    переменной внутри ТИПА (пакет, ключ, переменная окружения, ...).
    Переменная Project общая для всех литералов и в списке не хранится.
*/

:- module(ilp_engine,
          [ train_model/3,          % +Projects, +Options, -Result
            diagnose_projects/4,    % +Rules, +Projects, +Options, -Results
            canonical_body/2,       % +Body, -CanonicalBody
            rule_text/3,            % +Class, +Body, -Text
            run_request/2,          % +RequestFile, +ResponseFile
            server/0,               % постоянный процесс: JSON-строки stdin/stdout
            main/0
          ]).

:- encoding(utf8).

:- use_module(library(lists)).
:- use_module(library(apply)).
:- use_module(library(ordsets)).
:- use_module(library(solution_sequences)).
:- use_module(data_io).

% Фоновые знания: факты о проектах. Загружаются на время одного запуска.
:- dynamic requires_package/2, package_absent/2, version_mismatch/2,
           requires_config_key/2, config_key_absent/2,
           requires_env_variable/2, env_variable_absent/2,
           uses_build_tool/2, uses_runtime/2.

% Контекст обучения: обучающие идентификаторы и положительные примеры классов.
:- dynamic ctx_train_ids/1, ctx_positives/2.

type_base_name(package,      'Package').
type_base_name(config_key,   'Key').
type_base_name(env_variable, 'Variable').
type_base_name(build_tool,   'Tool').
type_base_name(runtime,      'Runtime').

% Порядок литералов только для ОТОБРАЖЕНИЯ правила человеку.
display_order([ requires_package, requires_config_key, requires_env_variable,
                package_absent, config_key_absent, env_variable_absent,
                version_mismatch, uses_build_tool, uses_runtime ]).

% Предикаты, для которых разрешены константы в теле. Константы имеют смысл
% прежде всего для контекстных признаков: uses_runtime(Project, python),
% uses_build_tool(Project, poetry). Для «пакетных» предикатов константа
% вида requires_package(Project, numpy) слишком специфична и ведёт к переобучению.
constant_predicate(uses_build_tool).
constant_predicate(uses_runtime).

% Внутренние формы литерала тела:
%   Pred-N              положительный литерал с переменной N
%   Pred-const(Value)   положительный литерал с константой
%   neg(Lit)            отрицательный литерал (\+)
%   neq(Type,N1,N2)     неравенство переменных одного типа (N1 \= N2)
%
% Переменные нумеруются внутри типа (package, config_key, ...). Константы
% не участвуют в переименовании переменных.
literal_positive(Pred-N) :- integer(N), bk_predicate(Pred, _).
literal_positive(Pred-const(_)) :- bk_predicate(Pred, _).
literal_negative(neg(_)).
literal_neq(neq(_, _, _)).

positive_var(Pred-N, Type, N) :-
    integer(N), bk_predicate(Pred, Type).
positive_const(Pred-const(_), Type) :-
    bk_predicate(Pred, Type).

literal_type(Pred-_, Type) :- bk_predicate(Pred, Type).
literal_type(neg(Inner), Type) :- literal_type(Inner, Type).
literal_type(neq(Type, _, _), Type) :- type_base_name(Type, _).

literal_var(Type, Pred-N, N) :- integer(N), bk_predicate(Pred, Type).
literal_var(Type, neg(Inner), N) :- literal_var(Type, Inner, N).
literal_var(Type, neq(Type, N1, _), N1).
literal_var(Type, neq(Type, _, N2), N2).

inner_literal(neg(Inner), Inner).
inner_literal(Inner, Inner) :- \+ Inner = neg(_).

% Кандидаты-константы: только реально встречающиеся в BK значения, не более
% Max за предикат (иначе поиск взрывается на больших наборах).
constant_candidates(Pred, Values) :-
    constant_predicate(Pred),
    findall(V, ( current_predicate(Pred/2), call(Pred, _, V) ), Vs0),
    sort(Vs0, Vs),
    constant_limit(Max),
    take(Max, Vs, Values).

constant_limit(Max) :- nb_getval(ilp_max_constants, Max), !.
constant_limit(20).

% ===========================================================================
% Небольшие утилиты
% ===========================================================================

opt(Options, Name, Default, Value) :-
    Opt =.. [Name, V0],
    (   memberchk(Opt, Options) -> Value = V0 ; Value = Default ).

clamp(X, Lo, Hi, Y) :- Y is max(Lo, min(Hi, X)).

round4(X, Y) :- Y is round(X * 10000) / 10000.0.

ratio(_, 0, 0.0) :- !.
ratio(A, B, R) :- R is float(A) / B.

f_measure(P, R, F) :-
    S is P + R,
    (   S =:= 0 -> F = 0.0 ; F is 2.0 * P * R / S ).

% Сглаженная (m-оценка) точность: (TP + m*prior) / (TP + FP + m), где
% prior — доля положительных примеров класса в train. В отличие от сырой
% TP/(TP+FP), она не позволяет правилу с TP=2 и FP=0 выглядеть идеальным.
hyp_prior(Class, Prior) :-
    ctx_positives(Class, Pos),
    ctx_train_ids(Ids),
    length(Pos, NPos),
    length(Ids, N),
    ratio(NPos, N, Prior).

clamp_prob(X, Y) :- Y is max(0.000001, min(0.999999, X)).

smoothed_precision(Class, TP, FP, PSm) :-
    hyp_prior(Class, Prior0),
    clamp_prob(Prior0, Prior),
    nb_getval(ilp_smoothing, M),
    Num is TP + M * Prior,
    Den is TP + FP + M,
    ratio(Num, Den, PSm).

% MDL/FOIL-подобная оценка: выигрыш логарифма правдоподобия относительно
% априорной доли класса минус штраф за длину тела. Больше — лучше.
% Используется для ранжирования правил, а не для вывода метрик.
rule_quality(Class, TP, FP, Len, Score) :-
    hyp_prior(Class, Prior0),
    clamp_prob(Prior0, Prior),
    smoothed_precision(Class, TP, FP, PS0),
    clamp_prob(PS0, PS),
    nb_getval(ilp_mdl_penalty, Penalty),
    (   TP > 0
    ->  TPart is TP * log(PS / Prior) / log(2)
    ;   TPart = 0.0
    ),
    (   FP > 0
    ->  FPart is FP * log((1.0 - PS) / (1.0 - Prior)) / log(2)
    ;   FPart = 0.0
    ),
    Score is TPart + FPart - Len * Penalty.

max_or_zero([], 0) :- !.
max_or_zero(L, M) :- max_list(L, M).

take(N, L, R) :-
    (   length(L, Len), Len =< N
    ->  R = L
    ;   length(R, N), append(R, _, L)
    ).

delete_eq([], _, []).
delete_eq([H|T], X, R) :-
    (   H == X -> R = R1 ; R = [H|R1] ),
    delete_eq(T, X, R1).

union_hits([], U, U).
union_hits([hyp(_, _, _, _, _, _, _, _, Hit)|T], U0, U) :-
    ord_union(U0, Hit, U1),
    union_hits(T, U1, U).

known_class(C) :- error_class(C).
known_predicate(P) :- bk_predicate(P, _).

% ===========================================================================
% Фоновые знания
% ===========================================================================

clear_kb :-
    forall(bk_predicate(Pred, _),
           (   functor(Goal, Pred, 2),
               retractall(Goal)
           )).

% Загрузка фактов одного проекта в BK. Все факты привязаны к Id проекта.
% Метки (labels) в BK НЕ попадают.
load_project(project(Id, _Labels, Facts)) :-
    forall(( member(fact(Pred, Arg), Facts), bk_predicate(Pred, _) ),
           (   Goal =.. [Pred, Id, Arg],
               (   call(Goal) -> true ; assertz(Goal) )
           )).

init_context(Projects, Classes, TrainIds) :-
    clear_kb,
    retractall(ctx_train_ids(_)),
    retractall(ctx_positives(_, _)),
    forall(member(P, Projects), load_project(P)),
    findall(Id, member(project(Id, _, _), Projects), Ids0),
    sort(Ids0, TrainIds),
    assertz(ctx_train_ids(TrainIds)),
    % E+ класса C: проекты с меткой C; E-: все остальные обучающие проекты.
    forall(member(C, Classes),
           (   findall(Id, ( member(project(Id, Ls, _), Projects),
                             memberchk(C, Ls) ), Pos0),
               sort(Pos0, Pos),
               assertz(ctx_positives(C, Pos))
           )).

% ===========================================================================
% Язык гипотез: тела, канонизация, уточнение
% ===========================================================================

% body_types(+Body, -Types): типы, встречающиеся в теле.
body_types(Body, Types) :-
    findall(T, ( member(L, Body), literal_type(L, T) ), Ts),
    sort(Ts, Types).

% type_vars(+Body, +Type, -Ns): все номера переменных данного типа,
% включая переменные внутри neg/neq. Нужно для канонизации.
type_vars(Body, Type, Ns) :-
    findall(N, ( member(L, Body), literal_var(Type, L, N) ), Ns0),
    sort(Ns0, Ns).

% positive_type_vars: переменные, которые уже связаны положительными
% литералами. Только по ним безопасно строить отрицания и неравенства.
positive_type_vars(Body, Type, Ns) :-
    findall(N, ( member(L, Body), positive_var(L, Type, N) ), Ns0),
    sort(Ns0, Ns).

% perm_map(+Body, +Types, -Map): все способы перенумеровать переменные
% внутри каждого типа. Map = [Type-Old-New, ...]
perm_map(_, [], []).
perm_map(Body, [Type|Types], Map) :-
    type_vars(Body, Type, Ns),
    (   Ns == []
    ->  M1 = []
    ;   length(Ns, K),
        numlist(1, K, Seq),
        permutation(Seq, Perm),
        pair_map(Type, Ns, Perm, M1)
    ),
    perm_map(Body, Types, M2),
    append(M1, M2, Map).

pair_map(_, [], [], []).
pair_map(Type, [N|Ns], [M|Ms], [Type-N-M|R]) :-
    pair_map(Type, Ns, Ms, R).

relabel_lit(Pred-N, Map, Pred-M) :-
    integer(N), !, bk_predicate(Pred, _),
    memberchk(_Type-N-M, Map).
relabel_lit(Pred-const(V), _, Pred-const(V)) :-
    !, bk_predicate(Pred, _).
relabel_lit(neg(Inner), Map, neg(Out)) :-
    !, relabel_lit(Inner, Map, Out).
relabel_lit(neq(Type, N1, N2), Map, neq(Type, M1, M2)) :-
    memberchk(Type-N1-A, Map),
    memberchk(Type-N2-B, Map),
    (   A @=< B
    ->  M1 = A, M2 = B
    ;   M1 = B, M2 = A
    ).

relabel_body([], _, []).
relabel_body([L|Ls], Map, [R|Rs]) :-
    relabel_lit(L, Map, R),
    relabel_body(Ls, Map, Rs).

% canonical_body(+Body, -Canon): представитель класса изоморфных тел.
% Два тела изоморфны, если одно получается из другого перестановкой литералов
% и переименованием переменных внутри типа. Берётся минимальный в
% стандартном порядке термов вариант.
canonical_body(Body0, Canon) :-
    sort(Body0, Body),
    body_types(Body, Types),
    findall(Sorted,
            (   perm_map(Body, Types, Map),
                relabel_body(Body, Map, Relabeled),
                sort(Relabeled, Sorted)
            ),
            All),
    msort(All, [Canon|_]).

type_var_count(Body, Type, K) :-
    type_vars(Body, Type, Ns),
    max_or_zero(Ns, K).

% refine(+Body, +Preds, -NewBody): добавить к телу один допустимый литерал.
% Разрешены:
%   * положительный литерал с существующей/новой переменной;
%   * положительный литерал с константой (constant_predicate/1);
%   * отрицательный литерал с уже связанной переменной или константой;
%   * неравенство двух уже связанных переменных одного типа.
refine(Body, Preds, NewBody) :-
    candidate_literal(Body, Preds, Lit),
    \+ memberchk(Lit, Body),
    canonical_body([Lit|Body], NewBody).

candidate_literal(Body, Preds, Lit) :-
    member(Pred, Preds),
    bk_predicate(Pred, Type),
    type_var_count(Body, Type, K),
    K1 is K + 1,
    between(1, K1, N),
    Lit = Pred-N.
candidate_literal(_, Preds, Lit) :-
    member(Pred, Preds),
    bk_predicate(Pred, _),
    constant_candidates(Pred, Values),
    member(V, Values),
    Lit = Pred-const(V).
candidate_literal(Body, Preds, neg(Pred-N)) :-
    member(Pred, Preds),
    bk_predicate(Pred, Type),
    positive_type_vars(Body, Type, Ns),
    member(N, Ns).
candidate_literal(_, Preds, neg(Pred-const(V))) :-
    member(Pred, Preds),
    bk_predicate(Pred, _),
    constant_candidates(Pred, Values),
    member(V, Values).
candidate_literal(Body, Preds, neq(Type, N1, N2)) :-
    member(Pred, Preds),
    bk_predicate(Pred, Type),
    positive_type_vars(Body, Type, Ns),
    select(N1, Ns, Rest),
    member(N2, Rest),
    N1 @< N2.

refine_level(Bodies, Preds, Next) :-
    findall(NB, ( member(B, Bodies), refine(B, Preds, NB) ), Raw),
    sort(Raw, Next).               % sort/2 удаляет дубликаты

% ===========================================================================
% Логический вывод: покрытие примеров
% ===========================================================================

% materialise(+Body, +Id, -Goals, -Env): превращает абстрактное тело в
% конкретные цели Prolog для проекта Id. Env = [Type-N-Var, ...] хранит
% соответствие номеров переменных настоящим логическим переменным.
% Положительные литералы выполняются первыми: они связывают переменные,
% после чего безопасно вычислять отрицания и неравенства.
materialise(Body, Id, Goals, Env) :-
    execution_order(Body, Ordered),
    materialise_(Ordered, Id, [], Goals, Env).

execution_order(Body, Ordered) :-
    include(literal_positive, Body, Pos),
    include(literal_negative, Body, Neg),
    include(literal_neq, Body, Neq),
    append([Pos, Neg, Neq], Ordered).

materialise_([], _, Env, [], Env).
materialise_([Lit|T], Id, Env0, [G|Gs], Env) :-
    lit_goal(Lit, Id, Env0, G, Env1),
    materialise_(T, Id, Env1, Gs, Env).

lit_goal(Pred-N, Id, Env0, Goal, Env) :-
    integer(N), !,
    bk_predicate(Pred, Type),
    (   memberchk(Type-N-V, Env0)
    ->  Env = Env0
    ;   Env = [Type-N-V|Env0]
    ),
    Goal =.. [Pred, Id, V].
lit_goal(Pred-const(V), Id, Env, Goal, Env) :-
    !, bk_predicate(Pred, _),
    Goal =.. [Pred, Id, V].
lit_goal(neg(Inner), Id, Env, Goal, Env) :-
    !,
    inner_goal(Inner, Id, Env, InnerGoal),
    Goal = (\+ InnerGoal).
lit_goal(neq(Type, N1, N2), _, Env, Goal, Env) :-
    memberchk(Type-N1-V1, Env),
    memberchk(Type-N2-V2, Env),
    Goal = (V1 \= V2).

inner_goal(Pred-N, Id, Env, Goal) :-
    integer(N),
    bk_predicate(Pred, Type),
    memberchk(Type-N-V, Env),
    Goal =.. [Pred, Id, V].
inner_goal(Pred-const(V), Id, _, Goal) :-
    bk_predicate(Pred, _),
    Goal =.. [Pred, Id, V].

prove_all([]).
prove_all([G|Gs]) :-
    call(G),
    prove_all(Gs).

% covers(+Body, +Id): правило с этим телом покрывает проект Id, если тело
% доказуемо из BK этого проекта.
covers(Body, Id) :-
    materialise(Body, Id, Goals, _),
    prove_all(Goals),
    !.

covered_set(Body, Covered) :-
    ctx_train_ids(Ids),
    findall(Id, ( member(Id, Ids), covers(Body, Id) ), Covered).

% explanation_facts(+Body, +Id, +Env, -Facts): только положительные факты BK,
% которые подтвердили тело. Отрицательные литералы и \= не являются фактами
% и в объяснение не попадают.
explanation_facts(Body, Id, Env, Facts) :-
    findall(fact(Pred, Id, V),
            (   member(L, Body),
                literal_positive(L),
                L = Pred-_,
                positive_value(L, Env, V)
            ),
            Facts).

positive_value(Pred-N, Env, V) :-
    integer(N),
    bk_predicate(Pred, Type),
    memberchk(Type-N-V, Env).
positive_value(_Pred-const(V), _, V).

% ===========================================================================
% Оценка гипотез
% ===========================================================================

% hyp(Class, Body, TP, FP, FN, Precision, Recall, F1, HitIds)
score_hyp(Class, Body, Covered,
          hyp(Class, Body, TP, FP, FN, P, R, F1, Hit)) :-
    ctx_positives(Class, Pos),
    ord_intersection(Covered, Pos, Hit),
    length(Hit, TP),
    length(Covered, NCov),
    FP is NCov - TP,
    length(Pos, NPos),
    FN is NPos - TP,
    ratio(TP, NCov, P),
    ratio(TP, NPos, R),
    f_measure(P, R, F1).

% evaluate_body(+Body, -Hyps, -MaxTP): покрытие считается один раз
% (оно не зависит от класса), затем оценивается для каждого класса.
% Гипотезы с TP = 0 не сохраняются: они бесполезны.
evaluate_body(Body, Hyps, MaxTP) :-
    covered_set(Body, Covered),
    findall(H,
            (   ctx_positives(C, _),
                score_hyp(C, Body, Covered, H),
                arg(3, H, TP),
                TP > 0
            ),
            Hyps),
    findall(TP, member(hyp(_, _, TP, _, _, _, _, _, _), Hyps), TPs),
    max_or_zero(TPs, MaxTP).

% ===========================================================================
% Поиск по уровням
% ===========================================================================

budget_exhausted :-
    nb_getval(ilp_tested, T),
    nb_getval(ilp_max_candidates, M),
    T >= M.

bump_tested :-
    nb_getval(ilp_tested, T0),
    T is T0 + 1,
    nb_setval(ilp_tested, T).

search(Preds, MaxLen, MinSupp, Hyps, Levels) :-
    findall([P-1], member(P, Preds), L1raw),
    sort(L1raw, Level1),
    search_level(1, MaxLen, Level1, MinSupp, Preds, [], Hyps, Levels).

search_level(L, MaxLen, Bodies, MinSupp, Preds, H0, H,
             [level(L, NGen, NTested, NExt)|Levels]) :-
    length(Bodies, NGen),
    eval_bodies(Bodies, MinSupp, H0, H1, [], Ext0, 0, NTested),
    reverse(Ext0, Ext),
    length(Ext, NExt),
    (   L < MaxLen, Ext \== [], \+ budget_exhausted,
        refine_level(Ext, Preds, Next), Next \== []
    ->  L1 is L + 1,
        search_level(L1, MaxLen, Next, MinSupp, Preds, H1, H, Levels)
    ;   H = H1,
        Levels = []
    ).

eval_bodies([], _, H, H, Ext, Ext, NT, NT).
eval_bodies([B|Bs], MinSupp, H0, H, Ext0, Ext, NT0, NT) :-
    (   budget_exhausted
    ->  nb_setval(ilp_truncated, true),     % достигнут max_candidates
        H = H0, Ext = Ext0, NT = NT0
    ;   bump_tested,
        evaluate_body(B, NewHyps, MaxTP),
        append(NewHyps, H0, H1),
        (   MaxTP >= MinSupp -> Ext1 = [B|Ext0] ; Ext1 = Ext0 ),
        NT1 is NT0 + 1,
        eval_bodies(Bs, MinSupp, H1, H, Ext1, Ext, NT1, NT)
    ).

% ===========================================================================
% Отбор, упрощение
% ===========================================================================

valid_hyp(hyp(Class, _, TP, FP, _, _, _, _, _), MinSupp, MinPrec) :-
    TP >= MinSupp,
    smoothed_precision(Class, TP, FP, PSm),
    PSm + 1.0e-9 >= MinPrec.

% greedy_select(+Valid, +Covered, +Left, +Acc, -Selected)
% Первое правило — лучшее по MDL-качеству (при равенстве: короче, затем
% больше TP). Следующие — те, что покрывают больше всего ЕЩЁ НЕ покрытых
% положительных примеров (при равенстве: MDL, короче, больше TP).
% Правило без нового покрытия не выбирается.
greedy_select(_, _, 0, Acc, Selected) :-
    !,
    reverse(Acc, Selected).
greedy_select(Valid, Covered, Left, Acc, Selected) :-
    findall(Key-H,
            (   member(H, Valid),
                H = hyp(Class, Body, TP, FP, _, _, _, _, Hit),
                ord_subtract(Hit, Covered, New),
                length(New, Gain),
                Gain > 0,
                length(Body, Len),
                rule_quality(Class, TP, FP, Len, Quality),
                NegQ is -Quality,
                NegTP is -TP,
                NegGain is -Gain,
                (   Acc == []
                ->  Key = k(NegQ, Len, NegTP, Body)
                ;   Key = k(NegGain, NegQ, Len, NegTP, Body)
                )
            ),
            Keyed),
    (   Keyed == []
    ->  reverse(Acc, Selected)
    ;   msort(Keyed, [_-Best|_]),
        Best = hyp(_, _, _, _, _, _, _, _, BestHit),
        ord_union(Covered, BestHit, Covered1),
        Left1 is Left - 1,
        greedy_select(Valid, Covered1, Left1, [Best|Acc], Selected)
    ).

% Упрощение по обучающей выборке: литерал удаляется, если покрытие
% обучающих проектов (и положительных, и отрицательных) не изменилось.
simplify_hyp(Class, Hyp0, Hyp) :-
    Hyp0 = hyp(_, Body0, _, _, _, _, _, _, _),
    covered_set(Body0, Cov),
    drop_redundant_literals(Body0, Cov, Body),
    (   Body == Body0
    ->  Hyp = Hyp0
    ;   score_hyp(Class, Body, Cov, Hyp)
    ).

drop_redundant_literals(Body, Cov, Simple) :-
    (   length(Body, Len), Len > 1,
        select(_, Body, Rest),
        canonical_body(Rest, RestC),
        covered_set(RestC, Cov)
    ->  drop_redundant_literals(RestC, Cov, Simple)
    ;   Simple = Body
    ).

dedupe_hyps(Hs, Unique) :- dedupe_(Hs, [], Unique).

dedupe_([], _, []).
dedupe_([H|T], Seen, Out) :-
    H = hyp(_, B, _, _, _, _, _, _, _),
    (   memberchk(B, Seen) -> Out = Out1 ; Out = [H|Out1] ),
    dedupe_(T, [B|Seen], Out1).

% Правило избыточно, если все его положительные примеры покрыты остальными
% выбранными правилами. Проверка идёт с конца (с наименее ценных).
prune_redundant(Rules, Kept) :-
    reverse(Rules, Rev),
    prune_(Rev, Rules, Kept).

prune_([], Kept, Kept).
prune_([R|Rs], Cur, Kept) :-
    delete_eq(Cur, R, Others),
    R = hyp(_, _, _, _, _, _, _, _, Hit),
    union_hits(Others, [], U),
    (   Others \== [], ord_subset(Hit, U)
    ->  prune_(Rs, Others, Kept)
    ;   prune_(Rs, Cur, Kept)
    ).

% ===========================================================================
% Обучение по классу и журнал отклонённых кандидатов
% ===========================================================================

learn_class(Class, AllHyps, MinSupp, MinPrec, MaxRules, Top,
            Selected, Info, Rejected, Warnings) :-
    ctx_positives(Class, Pos),
    length(Pos, NPos),
    ctx_train_ids(Ids),
    length(Ids, N),
    NNeg is N - NPos,
    findall(H, ( member(H, AllHyps), arg(1, H, Class) ), ClassHyps),
    findall(H, ( member(H, ClassHyps), valid_hyp(H, MinSupp, MinPrec) ), Valid),
    greedy_select(Valid, [], MaxRules, [], Sel0),
    maplist(simplify_hyp(Class), Sel0, Sel1),
    dedupe_hyps(Sel1, Sel2),
    prune_redundant(Sel2, Selected),
    union_hits(Selected, [], CoveredPos),
    length(CoveredPos, NCovPos),
    length(Selected, NRules),
    (   NPos =:= 0
    ->  Status = no_positive_examples,
        Warnings = [warning(no_positive_examples, Class)]
    ;   NRules =:= 0
    ->  Status = no_rule_found,
        Warnings = [warning(no_rule_found, Class)]
    ;   Status = ok,
        Warnings = []
    ),
    (   NNeg =:= 0, NPos > 0 -> Info1 = [warning-no_negative_examples] ; Info1 = [] ),
    Info = [ n_pos-NPos, n_neg-NNeg, n_rules-NRules,
             covered_positives-NCovPos, status-Status | Info1 ],
    rejected_candidates(ClassHyps, Selected, MinSupp, MinPrec, MaxRules, Top,
                        Rejected).

% Лучшие отклонённые кандидаты и причины отклонения.
% Кандидаты, эквивалентные по покрытию выбранному правилу, не показываются:
% это то же правило с лишним литералом, они только засоряют журнал.
rejected_candidates(ClassHyps, Selected, MinSupp, MinPrec, MaxRules, Top, Rejected) :-
    length(Selected, NSel),
    findall(Key-H,
            (   member(H, ClassHyps),
                H = hyp(Class, Body, TP, FP, _, _, _, _, Hit),
                \+ ( member(S, Selected), arg(2, S, SB), SB == Body ),
                \+ memberchk(hyp(_, _, TP, FP, _, _, _, _, Hit), Selected),
                length(Body, Len),
                rule_quality(Class, TP, FP, Len, Quality),
                NegQ is -Quality,
                Key = k(NegQ, Len, Body)
            ),
            Keyed),
    msort(Keyed, Sorted),
    findall(H, member(_-H, Sorted), All),
    take(Top, All, Best),
    % тексты строим только для показываемых кандидатов
    maplist(rejected_record(Selected, MinSupp, MinPrec, NSel, MaxRules), Best, Rejected).

rejected_record(Selected, MinSupp, MinPrec, NSel, MaxRules,
                hyp(Class, Body, TP, FP, _, P, R, F1, Hit),
                rejected(Class, Text, TP, FP, P4, R4, F4, Reasons)) :-
    union_hits(Selected, [], CoveredSel),
    ord_subtract(Hit, CoveredSel, New),
    length(New, Gain),
    smoothed_precision(Class, TP, FP, PSm),
    reject_reasons(TP, PSm, Gain, MinSupp, MinPrec, NSel, MaxRules, Reasons),
    display_body(Body, DBody),
    rule_text(Class, DBody, Text),
    round4(P, P4), round4(R, R4), round4(F1, F4).

reject_reasons(TP, P, Gain, MinSupp, MinPrec, NSel, MaxRules, Reasons) :-
    (   TP < MinSupp -> R1 = [low_support] ; R1 = [] ),
    (   P + 1.0e-9 < MinPrec -> R2 = [low_precision] ; R2 = [] ),
    append(R1, R2, R12),
    (   R12 == []
    ->  (   NSel >= MaxRules
        ->  Reasons = [rule_limit_reached]
        ;   Gain =:= 0
        ->  Reasons = [no_new_positives]
        ;   Reasons = [dominated]
        )
    ;   Reasons = R12
    ).

learn_classes([], _, _, _, _, _, [], [], [], []).
learn_classes([C|Cs], Hyps, MinS, MinP, MaxR, Top,
              Selected, [class_info(C, Info)|Infos], Rejected, Warnings) :-
    learn_class(C, Hyps, MinS, MinP, MaxR, Top, S1, Info, Rej1, W1),
    learn_classes(Cs, Hyps, MinS, MinP, MaxR, Top, S2, Infos, Rej2, W2),
    append(S1, S2, Selected),
    append(Rej1, Rej2, Rejected),
    append(W1, W2, Warnings).

% ===========================================================================
% Представление правил для людей
% ===========================================================================

display_body(Body, Sorted) :-
    display_order(Order),
    include(literal_positive, Body, Pos),
    include(literal_negative, Body, Neg),
    include(literal_neq, Body, Neq),
    sort_display(Pos, Order, SPos),
    sort_display(Neg, Order, SNeg),
    msort(Neq, SNeq),
    append([SPos, SNeg, SNeq], Sorted).

sort_display(Lits, Order, Sorted) :-
    findall(I-Arg-Lit,
            (   member(Lit, Lits),
                lit_pred_arg(Lit, Pred, Arg),
                nth0(I, Order, Pred)
            ),
            Keyed),
    msort(Keyed, Sorted0),
    findall(Lit, member(_-_-Lit, Sorted0), Sorted).

lit_pred_arg(Pred-N, Pred, N) :- integer(N).
lit_pred_arg(Pred-const(V), Pred, const(V)).
lit_pred_arg(neg(Inner), Pred, Arg) :- lit_pred_arg(Inner, Pred, Arg).

% Имена переменных: Package, Key, Variable... Если переменных одного типа
% несколько — с номером: Package1, Package2.
body_var_names(Body, Names) :-
    findall(Type-N, ( member(L, Body), literal_var(Type, L, N) ), V0),
    sort(V0, Vars),
    findall(Type-N-Name,
            (   member(Type-N, Vars),
                findall(M, member(Type-M, Vars), Ms),
                length(Ms, K),
                type_base_name(Type, Base),
                (   K =:= 1 -> Name = Base ; atom_concat(Base, N, Name) )
            ),
            Names).

literal_text(Names, Lit, Text) :-
    lit_pred_arg(Lit, Pred, Arg),
    bk_predicate(Pred, Type),
    arg_text(Names, Type, Arg, ArgText),
    (   Lit = neg(_) -> Prefix = '\\+ ' ; Prefix = '' ),
    format(atom(Text), '~w~w(Project, ~w)', [Prefix, Pred, ArgText]).
literal_text(Names, neq(Type, N1, N2), Text) :-
    memberchk(Type-N1-Name1, Names),
    memberchk(Type-N2-Name2, Names),
    format(atom(Text), '~w \\= ~w', [Name1, Name2]).

arg_text(Names, Type, N, Name) :-
    integer(N),
    memberchk(Type-N-Name, Names).
arg_text(_, _, const(V), Atom) :-
    format(atom(Atom), '~q', [V]).

rule_text(Class, Body, Text) :-
    body_var_names(Body, Names),
    maplist(literal_text(Names), Body, Lits),
    atomic_list_concat(Lits, ',\n    ', Joined),
    format(atom(Text), 'has_error(Project, ~w) :-\n    ~w.', [Class, Joined]).

hyp_rule(hyp(C, Body, TP, FP, FN, P, R, F1, _), rule(C, Lits, Text, Metrics)) :-
    display_body(Body, DBody),
    body_var_names(DBody, Names),
    findall(LitTerm,
            (   member(L, DBody),
                literal_term(L, Names, LitTerm)
            ),
            Lits),
    rule_text(C, DBody, Text),
    length(Body, Len),
    round4(P, P4), round4(R, R4), round4(F1, F4),
    smoothed_precision(C, TP, FP, PSm),
    round4(PSm, PSm4),
    Metrics = [ tp-TP, fp-FP, fn-FN, precision-P4, recall-R4, f1-F4,
                precision_smoothed-PSm4, length-Len ].

literal_term(Pred-N, Names, lit(Pred, N, Name)) :-
    integer(N),
    bk_predicate(Pred, Type),
    memberchk(Type-N-Name, Names).
literal_term(Pred-const(V), _, lit(Pred, const(V), V)).
literal_term(neg(Inner), Names, neg_lit(Pred, Arg, Name)) :-
    literal_term(Inner, Names, lit(Pred, Arg, Name)).
literal_term(neq(Type, N1, N2), Names, neq_lit(Type, N1, N2, Name1, Name2)) :-
    memberchk(Type-N1-Name1, Names),
    memberchk(Type-N2-Name2, Names).

% ===========================================================================
% Обучение: основной предикат
% ===========================================================================

% train_model(+Projects, +Options, -Result)
%   Projects: [project(Id, Labels, Facts), ...] — ТОЛЬКО обучающая часть.
%   Options : max_body_length(N), min_support(N), min_precision(P),
%             max_rules_per_class(N), max_candidates(N), top_rejected(N),
%             classes(List), body_predicates(List)
%   Result  : result(Rules, ClassInfos, Stats, Warnings, log(Levels, Rejected))
train_model(Projects, Options,
            result(Rules, ClassInfos, Stats, Warnings, log(Levels, Rejected))) :-
    get_time(T0),
    findall(C, error_class(C), AllClasses),
    findall(P, bk_predicate(P, _), AllPreds),
    opt(Options, max_body_length,     3,      L0),
    opt(Options, min_support,         2,      S0),
    opt(Options, min_precision,       0.8,    Pr0),
    opt(Options, max_rules_per_class, 5,      R0),
    opt(Options, max_candidates,      20000,  MC0),
    opt(Options, top_rejected,        5,      Top),
    opt(Options, smoothing,           1.0,    Sm0),
    opt(Options, mdl_penalty,         1.0,    Pen0),
    opt(Options, max_constants,       20,     Cons0),
    opt(Options, classes,             AllClasses, Classes0),
    opt(Options, body_predicates,     AllPreds,   Preds0),
    clamp(L0, 1, 4, MaxLen),
    clamp(S0, 1, 1000000, MinSupp),
    clamp(Pr0, 0.0, 1.0, MinPrec),
    clamp(R0, 1, 100, MaxRules),
    clamp(MC0, 1, 10000000, MaxCand),
    clamp(Sm0, 0.01, 100.0, Smoothing),
    clamp(Pen0, 0.0, 10000.0, MdlPenalty),
    clamp(Cons0, 0, 100, MaxConstants),
    include(known_class, Classes0, Classes),
    include(known_predicate, Preds0, Preds),
    init_context(Projects, Classes, TrainIds),
    length(TrainIds, NTrain),
    nb_setval(ilp_tested, 0),
    nb_setval(ilp_max_candidates, MaxCand),
    nb_setval(ilp_truncated, false),
    nb_setval(ilp_smoothing, Smoothing),
    nb_setval(ilp_mdl_penalty, MdlPenalty),
    nb_setval(ilp_max_constants, MaxConstants),
    search(Preds, MaxLen, MinSupp, Hyps, Levels),
    learn_classes(Classes, Hyps, MinSupp, MinPrec, MaxRules, Top,
                  Selected, ClassInfos, Rejected, Warnings1),
    maplist(hyp_rule, Selected, Rules),
    findall(Body, member(hyp(_, Body, _, _, _, _, _, _, _), Selected), Bodies),
    covered_by_any(Bodies, TrainIds, NCoveredAny),
    nb_getval(ilp_truncated, Trunc),
    (   Trunc == true
    ->  Warnings = [warning(max_candidates_reached, MaxCand)|Warnings1]
    ;   Warnings = Warnings1
    ),
    findall(G, member(level(_, G, _, _), Levels), Gs), sum_list(Gs, NGen),
    findall(T, member(level(_, _, T, _), Levels), Ts), sum_list(Ts, NTested),
    findall(E, member(level(_, _, _, E), Levels), Es), sum_list(Es, NExt),
    length(Hyps, NHyp),
    length(Classes, NClasses),
    NScored is NTested * NClasses,
    ratio(NCoveredAny, NTrain, CoveredShare),
    round4(CoveredShare, CoveredShare4),
    get_time(T1),
    Elapsed0 is T1 - T0,
    round4(Elapsed0, Elapsed),
    Stats = [ candidates_generated-NGen,
              candidates_tested-NTested,
              candidates_extended-NExt,
              hypotheses_scored-NScored,
              hypotheses_with_support-NHyp,
              train_size-NTrain,
              train_covered_share-CoveredShare4,
              elapsed_seconds-Elapsed,
              truncated-Trunc,
              max_body_length-MaxLen,
              min_support-MinSupp,
              min_precision-MinPrec,
              smoothing-Smoothing,
              mdl_penalty-MdlPenalty,
              max_constants-MaxConstants,
              max_rules_per_class-MaxRules,
              max_candidates-MaxCand ].

covered_by_any(Bodies, Ids, Count) :-
    findall(Id, ( member(Id, Ids), member(B, Bodies), covers(B, Id) ), L),
    sort(L, U),
    length(U, Count).

% ===========================================================================
% Диагностика новых проектов
% ===========================================================================

% diagnose_projects(+Rules, +Projects, +Options, -Results)
%   Rules   : [rule(Class, Body), ...] — ранее выученные правила.
%   Options : max_proofs(N) — сколько подстановок-доказательств показывать на
%             каждое правило (0 — только факт срабатывания), classes(List).
%   Results : [project_result(Id, [class_result(Class, Status, Matches)])]
%   Status  : detected | not_detected | no_rules
diagnose_projects(Rules, Projects, Options, Results) :-
    maplist(check_rule, Rules),
    findall(C, error_class(C), AllClasses),
    opt(Options, max_proofs, 3, MaxProofs),
    opt(Options, classes, AllClasses, Classes),
    maplist(diagnose_one(Rules, Classes, MaxProofs), Projects, Results).

check_rule(rule(Class, Body)) :-
    (   error_class(Class) -> true
    ;   throw(error(domain_error(error_class, Class), _))
    ),
    (   Body = [_|_] -> true
    ;   throw(error(domain_error(non_empty_body, Body), _))
    ),
    forall(member(Lit, Body), check_literal(Lit)),
    safe_rule(Body).

check_literal(Pred-N) :-
    (   bk_predicate(Pred, _), integer(N), N >= 1 -> true
    ;   throw(error(domain_error(literal, Pred-N), _))
    ).
check_literal(Pred-const(V)) :-
    (   bk_predicate(Pred, _), ( atom(V) ; number(V) ) -> true
    ;   throw(error(domain_error(literal, Pred-const(V)), _))
    ).
check_literal(neg(Inner)) :-
    (   check_inner_literal(Inner) -> true
    ;   throw(error(domain_error(literal, neg(Inner)), _))
    ).
check_literal(neq(Type, N1, N2)) :-
    (   type_base_name(Type, _), integer(N1), integer(N2),
        N1 >= 1, N2 >= 1, N1 < N2
    ->  true
    ;   throw(error(domain_error(literal, neq(Type, N1, N2)), _))
    ).

check_inner_literal(Pred-N) :-
    bk_predicate(Pred, _), integer(N), N >= 1.
check_inner_literal(Pred-const(V)) :-
    bk_predicate(Pred, _), ( atom(V) ; number(V) ).

% Безопасность: все переменные отрицаний и \= должны быть связаны
% положительными литералами тела.
safe_rule(Body) :-
    findall(Type-N, ( member(L, Body), positive_var(L, Type, N) ), Bound0),
    sort(Bound0, Bound),
    forall(( member(L, Body), requires_bound(L, Type, N) ),
           memberchk(Type-N, Bound)).

requires_bound(neg(Inner), Type, N) :-
    inner_var(Inner, Type, N).
requires_bound(neq(Type, N1, _), Type, N1).
requires_bound(neq(Type, _, N2), Type, N2).

inner_var(Pred-N, Type, N) :-
    integer(N), bk_predicate(Pred, Type).

diagnose_one(Rules, Classes, MaxProofs, Project,
             project_result(Id, ClassResults)) :-
    Project = project(Id, _, _),
    clear_kb,
    load_project(Project),
    findall(class_result(C, Status, Matches),
            (   member(C, Classes),
                class_diagnosis(C, Rules, Id, MaxProofs, Status, Matches)
            ),
            ClassResults).

class_diagnosis(C, Rules, Id, MaxProofs, Status, Matches) :-
    findall(Idx-Body, nth1(Idx, Rules, rule(C, Body)), ClassRules),
    (   ClassRules == []
    ->  Status = no_rules,
        Matches = []
    ;   findall(M,
                (   member(Idx-Body, ClassRules),
                    rule_match(Idx, Body, Id, MaxProofs, M)
                ),
                Matches),
        (   Matches == [] -> Status = not_detected ; Status = detected )
    ).

% Правило сработало, если covers/2 истинно. Подтверждение — подстановки
% переменных и конкретные факты BK, которые удовлетворили условиям.
rule_match(Idx, Body, Id, MaxProofs, match(Idx, Proofs)) :-
    covers(Body, Id),
    (   MaxProofs > 0
    ->  display_body(Body, DBody),
        body_var_names(DBody, Names),
        findall(proof(Bindings, Facts),
                limit(MaxProofs,
                      (   materialise(DBody, Id, Goals, Env),
                          prove_all(Goals),
                          env_bindings(Env, Names, Bindings0),
                          sort(Bindings0, Bindings),
                          explanation_facts(DBody, Id, Env, Facts)
                      )),
                Proofs0),
        sort(Proofs0, Proofs1),
        take(MaxProofs, Proofs1, Proofs)
    ;   Proofs = []
    ).

env_bindings([], _, []).
env_bindings([Type-N-V|T], Names, [Name-V|R]) :-
    memberchk(Type-N-Name, Names),
    env_bindings(T, Names, R).

% ===========================================================================
% Точка входа: swipl -g main -t halt ilp_engine.pl -- request.json response.json
% ===========================================================================

handle_dict(Request, Response) :-
    request_command(Request, Command),
    handle_command(Command, Request, Response).

handle_request(RequestFile, Response) :-
    read_request(RequestFile, Request),
    handle_dict(Request, Response).

handle_command(train, Request, Response) :-
    request_projects(Request, Projects),
    request_options(Request, Options),
    train_model(Projects, Options, Result),
    train_response(Result, Response).
handle_command(diagnose, Request, Response) :-
    request_projects(Request, Projects),
    request_rules(Request, Rules),
    request_options(Request, Options),
    diagnose_projects(Rules, Projects, Options, Results),
    diagnose_response(Results, Response).
handle_command(ping, _, Response) :-
    ping_response(Response).

run_request(RequestFile, ResponseFile) :-
    (   catch(handle_request(RequestFile, Response), E,
              error_response(E, Response))
    ->  true
    ;   error_response(goal_failed, Response)
    ),
    write_response(ResponseFile, Response).

% Постоянный процесс: читает по одному JSON-объекту со stdin, пишет ответ
% одной строкой в stdout и сразу сбрасывает буфер. Python держит такой процесс
% живым и не тратит время на запуск swipl для каждого запроса.
server :-
    set_stream(user_input, encoding(utf8)),
    set_stream(user_output, encoding(utf8)),
    repeat,
    (   catch(read_json_stream(user_input, Request), _, fail)
    ->  (   catch(handle_dict(Request, Response), E,
                  error_response(E, Response))
        ->  true
        ;   error_response(goal_failed, Response)
        ),
        (   catch(write_json_stream(user_output, Response), _, fail)
        ->  true
        ;   halt(3)
        ),
        fail
    ;   halt(0)
    ).

% Python передаёт два последних аргумента: файл запроса и файл ответа.
% Берём именно последние два элемента argv — это не зависит от того,
% как конкретная версия SWI-Prolog обрабатывает разделитель «--».
main :-
    current_prolog_flag(argv, Argv),
    (   append(_, [RequestFile, ResponseFile], Argv)
    ->  run_request(RequestFile, ResponseFile),
        halt(0)
    ;   format(user_error, 'ilp_engine: expected <request.json> <response.json>~n', []),
        halt(2)
    ).
