# ======== IMPORTS ========
# =========================

if !isdefined(Main, :QML)
    include("../Modules/QML.jl")
    using .QML
end

using Yao, YaoPlots

Φ = zero_state(2)

# initial architecture
RyChain = chain(4, repeat(H, 3:4),
    control(3, 1=>Ry(π)),
    cnot(1, 2),
    control(4, 2=>Ry(π))
);

# for later expansion to RxChain/rotations
model_architecture = chain(subroutine(RyChain, 1:4))

## set up OAA algorithm
## define R0* chain -- NOTE: I don't think this is accurate?
R0lstar = chain(4, repeat(X, 3:4),
                    cz(3, 4),
                    repeat(X, 3:4)
                );

append_qubits!(Φ, 2);

focus!(Φ, 1:4)
Φ |> RyChain;

outcome = measure!(Φ, 1:2)
if (outcome != bit"00")
    Φ |> Daggered(RyChain);
    Φ |> R0lstar;
    Φ |> RyChain;
end

relax!(Φ, 1:4)

# TODO: visualize circuit
