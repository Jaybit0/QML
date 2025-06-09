using Yao, YaoPlots

Φ = zero_state(1) |> Rx(-π/4) |> Ry(-π/4)

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

vizcircuit(R0lstar)