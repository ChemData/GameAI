function woo(a::Dict{String, A}, b::Dict{String, B},) where {A<:AbstractArray, B<:AbstractArray}
    println("a")
end

d1 = Dict("a"=>[1,2,3], "b"=>[5,5])
d2 = Dict("a"=>[1,2,3], "b"=>["a", "b"])

woo(d1, d2)