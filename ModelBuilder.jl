using Flux


struct Split 
    fs
end

function Split(fs...)
    Split(fs)
end

function (w::Split)(x::AbstractArray)
    tuple([w.fs[i](x) for i in 1:length(w.fs)])
end

Flux.@functor Split

struct Parallel
    fs
end

function Parallel(fs...)
    Parallel(fs)
end

function (w::Parallel)(x::AbstractArray)
    vcat([w.fs[i](x) for i in 1:length(w.fs)]...)
end

Flux.@functor Parallel


m = Dense(30, 20)
parallel = Split(Dense(20, 1), Dense(20, 3))
d = Chain(m, parallel)

mod2 = Chain(
        Dense(5, 5),
        Dense(5, 4)
)

mod = Chain(
    Dense(5, 20),
    Split(
        Chain(
            Dense(20, 4),
            softmax
        ),
        Dense(20, 1, NNlib.σ)
    )
)

x = rand(5, 3)
p = mod2(x)