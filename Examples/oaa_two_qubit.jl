# ======== IMPORTS ========
# =========================

if !isdefined(Main, :QML)
    include("../Modules/QML.jl")
    using .QML
end

using Yao, YaoPlots
using Yao.EasyBuild

Φ = zero_state(2) |> chain(2, put(1=>Ry(π)))

# initial architecture
RyChain = chain(4, repeat(H, 3:4),
    control(3, 1=>Ry(π)),
    cnot(1, 2),
    control(4, 2=>Ry(π))
);

print(typeof(RyChain))

RxChain = chain(4, repeat(H, 3:4),
    control(3, 1=>Rx(π)),
    cnot(1, 2),
    control(4, 2=>Rx(π))
);
# for later expansion to RxChain/rotations
model_architecture = chain(6, subroutine(RxChain, [1:4]),
                        subroutine(RyChain, [1, 2, 5, 6])
                        )

YaoPlots.plot(model_architecture)

## set up OAA algorithm
## define R0* chain -- NOTE: I don't think this is accurate?
R0lstar = chain(4, repeat(X, 3:4),
                    cz(3, 4), # does not matter which is control/which is applied
                    repeat(X, 3:4)
                );

append_qubits!(Φ, 6);

focus!(Φ, 1:4)
Φ |> RxChain;

outcome = measure!(Φ, 1:2)
if (outcome != 0)
    Φ |> Daggered(RxChain);
    Φ |> R0lstar;
    Φ |> RxChain;
end

relax!(Φ, 1:4)

focus!(Φ, [1, 2, 5, 6])
Φ |> RyChain;

outcome = measure!(Φ, 1:2)
if (outcome != 0)
    Φ |> Daggered(RyChain);
    Φ |> R0lstar;
    Φ |> RyChain;
end

relax(Φ, [1, 2, 5, 6])

# outcome = measure!(Φ, 1:4)
# # TODO: visualize circuit
println(outcome)
# println(typeof(outcome))

