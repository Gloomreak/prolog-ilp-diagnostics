:- encoding(utf8).
:- consult('prolog/tests.pl').
:- initialization(main, main).

main :-
    dataset_f(Ps),
    ilp_engine:init_context(Ps, [missing_dependency], Ids),
    nb_setval(ilp_max_constants, 20),
    ( ilp_engine:constant_candidates(uses_runtime, Cs) -> format("consts=~w~n", [Cs]) ; format("no consts~n", []) ),
    ( ilp_engine:refine_level([[requires_package-1, package_absent-1]],
                              [requires_package, package_absent, uses_runtime], Next)
    -> format("next=~w~n", [Next]) ; format("refine failed~n", []) ),
    ( ilp_engine:train_model(Ps, [min_support(2), min_precision(0.8), max_body_length(3)],
                             Result)
    ->  Result = result(Rules, Infos, Stats, Warnings, _),
        format("rules=~w~ninfos=~w~nstats=~w~nwarn=~w~n", [Rules, Infos, Stats, Warnings])
    ; format("train failed~n", []) ),
    halt.
