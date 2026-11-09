:- encoding(utf8).
:- consult('prolog/tests.pl').
:- initialization(main, main).

main :-
    dataset_f(Ps),
    ilp_engine:init_context(Ps, [missing_dependency], _Ids),
    nb_setval(ilp_max_constants, 20),
    ( ilp_engine:canonical_body([neg(uses_runtime-const(python)), requires_package-1, package_absent-1], C)
    -> format("canon=~w~n", [C]) ; format("canon failed~n", []) ),
    ( ilp_engine:refine([requires_package-1, package_absent-1],
                        [requires_package, package_absent, uses_runtime], N)
    -> format("one=~w~n", [N]) ; format("refine failed~n", []) ),
    forall(ilp_engine:refine([requires_package-1, package_absent-1],
                             [requires_package, package_absent, uses_runtime], N2),
           format("cand=~w~n", [N2])),
    halt.
