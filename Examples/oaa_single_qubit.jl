# ORIGINAL CODE FROM: https://niklaspirnay.com/blog/learningwithaa/
# ======== IMPORTS ========
# =========================

if !isdefined(Main, :QML)
    include("../Modules/QML.jl")
    using .QML
end

using Yao, YaoPlots
# using Yao.EasyBuild, YaoPlots

# one qubit circuit
Φ = zero_state(1) |> Rx(-π/4) |> Ry(-π/4)

# Define the parameterized rotations
# All of those need focus on 1st qb and 3 param qbs
RxChain = chain(4,repeat(H, 2:4), 
            control(2, 1=>Rx(π/4)),
            control(3, 1=>Rx(π/2)),
            control(4, 1=>Rx(π)),
        );
RyChain = chain(4,repeat(H, 2:4), 
            control(2, 1=>Ry(π/4)),
            control(3, 1=>Ry(π/2)),
            control(4, 1=>Ry(π)),
        );

model_architecture = chain(7, subroutine(RxChain, 1:4),
                            subroutine(RyChain, [1,5,6,7]))

YaoPlots.plot(model_architecture)

# Flip sign of all states that don't have clean ancillas
# This requires focus on the first and parameter qubits
R0lstar = chain(4, repeat(X, 2:4),
                    cz(2:3, 4),
                    repeat(X, 2:4)
                );

## OAA algorithm
# Add 6 zeros qubits for parameters
append_qubits!(Φ, 6);

# First parameterized rotation: Rx
focus!(Φ, 1:4)
Φ |> RxChain;
# Measure first qubit. For State tomography we want to force it to be 0.
outcome = measure!(Φ, 1)
if (outcome != 0)
    Φ |> Daggered(RxChain);
    Φ |> R0lstar;
    Φ |> RxChain;
end
relax!(Φ, 1:4)

# Second parameterized rotation: Ry
focus!(Φ, [1,5,6,7])
Φ |> RyChain;
# Measure first qubit. For State tomography we want to force it to be 0.
outcome = measure!(Φ, 1)
if (outcome != 0)
    Φ |> Daggered(RyChain);
    Φ |> R0lstar;
    Φ |> RyChain;
end
relax!(Φ, [1,5,6,7])