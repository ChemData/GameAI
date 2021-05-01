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

Base.getindex(X::Split, i) = X.paths[i]

struct Flatten
end

function (f::Flatten)(x::AbstractArray)
    Flux.flatten(x)
end

Flux.@functor Flatten

function finalnormalize(x::AbstractArray)
    return vcat(NNlib.σ.(x[1:1, :]), softmax(x[2:end, :]))
end

function make1D(x::AbstractArray)
    return reshape(x, :, size(x, 4))
end