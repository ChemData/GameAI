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

struct Flatten
end

function (f::Flatten)(x::AbstractArray)
    Flux.flatten(x)
end

Flux.@functor Flatten
