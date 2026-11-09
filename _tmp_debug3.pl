:- encoding(utf8).
:- consult('prolog/tests.pl').
:- initialization(main, main).

main :-
    Body = [requires_package-1, package_absent-1, requires_package-2, neq(package, 1, 2)],
    P1 = project(n1, [], [fact(requires_package, a), fact(requires_package, b),
                          fact(package_absent, a)]),
    ilp_engine:clear_kb,
    ilp_engine:load_project(P1),
    ( ilp_engine:materialise(Body, n1, Goals, Env)
    -> format("goals=~w~nenv=~w~n", [Goals, Env]) ; format("materialise failed~n", []) ),
    ( catch((ilp_engine:materialise(Body,n1,G,E), ilp_engine:prove_all(G)), Err, format("err=~w~n",[Err])) -> format("proof ok~n") ; format("proof failed~n") ),
    ( ilp_engine:covers(Body, n1) -> format("covers ok~n") ; format("covers failed~n") ),
    halt.
