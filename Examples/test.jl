N = 3
R = 2

# define R0lstar
R0lstar = chain(
    N + R + 1,
    repeat(X, N + 2:N + R + 1),
    cz(N + 2:N + R, N + R + 1),
    repeat(X, N + 2:N + R + 1),
);