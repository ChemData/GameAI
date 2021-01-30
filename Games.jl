
abstract type Player end
abstract type GameState end
abstract type Move end
abstract type Tile end
abstract type Card end

struct InvalidMove <: Exception
    msg::AbstractString
end

mutable struct SearchNode
    parent::Union{SearchNode, Nothing}
    children::Array{Union{Nothing, SearchNode}}
    gamestate::GameState
    moves::Array{Move}
    legalmoves::Array{Bool}
    N::Array{Int}
    W::Array{Float64}
    Q::Array{Float64}
    P::Array{Float64}
    ν::Float64
    ishead::Bool
    positionindex::Union{Int, Nothing}
    winner::Int
end

function Base.show(io::IO, node::SearchNode)
    print(io, "SearchNode($(node.N))")
end

function newnode(policymodel, gamestate::GameState, parent::Union{SearchNode, Nothing}=nothing, positionindex::Union{Int, Nothing}=nothing)
    ν, ps = policymodel(decisioninput(gamestate))[1]
    moves, legalmoves = listofmoves(gamestate)
    winner = winnerof(gamestate)
    if winner == gamestate.current_player
        ν = 1
    elseif winner > 0
        ν = 0
    elseif winner == 0
        ν = 0.5
    else
        ν = ν[1]
    end

    newnode = SearchNode(
        parent,
        Array{Union{Nothing, SearchNode}}(nothing, length(ps)),
        gamestate,
        moves,
        legalmoves,
        zeros(Int, length(ps)),
        zeros(Float64, length(ps)),
        zeros(Float64, length(ps)),
        ps,
        ν,
        false,
        positionindex,
        winner
    )
    if parent === nothing
        newnode.ishead = true
    end
    return newnode
end

"Return the exploration value of each move from the provided Node."
function explorationvalues(from::SearchNode, c_puct::Real)
    u = c_puct * from.P * sqrt(sum(from.N)) ./ (1 .+ from.N)
    # We need to make sure that illegal moves always have a lower score than legal moves. This forces them to be negative.
    return (from.Q + u) .* (2.0*from.legalmoves .- 1)
end

"Explore to a leaf node from a given node."
function explorefrom(from::SearchNode, c_puct::Real, number::Int, policymodel)
    for i in 1:number
        currentnode = from
        while true
            if currentnode.winner >= 0
                backup(currentnode, currentnode.ν, currentnode.gamestate.current_player)
                break
            end
            
            index = findmax(explorationvalues(currentnode, c_puct))[2]
            if typeof(currentnode.children[index]) == Nothing
                gamestate = executemove(currentnode.gamestate, currentnode.moves[index])
                child = newnode(policymodel, gamestate, currentnode, index)
                currentnode.children[index] = child
                backup(child, child.ν, child.gamestate.current_player)
                break
            end
            currentnode = currentnode.children[index]
        end
    end
end

"Back up the results of a newly expanded node to the head."
function backup(from::SearchNode, ν::Float64, perspectiveplayer::Int)
    if from.ishead
        return
    end
    parent = from.parent
    parent.N[from.positionindex] += 1
    if parent.gamestate.current_player == perspectiveplayer
        parent.W[from.positionindex] += ν
    else
        parent.W[from.positionindex] += (1-ν)
    end
    parent.Q[from.positionindex] = parent.W[from.positionindex] ./parent.N[from.positionindex]
    backup(from.parent, ν, perspectiveplayer)
end

"Return the probability of taking each move. Illegal moves will never be picked."
function moveprobabilities(from::SearchNode, T::Real)
    weights = (.^(from.N, 1/T)/sum(.^(from.N, 1/T))) .* from.legalmoves
    return weights/sum(weights)
end

function weightedpick(options::AbstractArray, weights::AbstractArray)
    weights = weights/sum(weights)
    randomnumber = rand()
    cumweight = 0
    for i in 1:length(weights)
        cumweight += weights[i]
        if cumweight > randomnumber
            return options[i]
        end
    end
end

"Return the seemingly best move."
function bestmove(from::SearchNode, T::Real)
    probs = moveprobabilities(from, T)
    moveindex = weightedpick(1:length(probs), moveprobabilities(from, T))
    return from.moves[moveindex], moveindex
end

"Take a specific move and then return the new head node.
Keeps all the children nodes of the new head but discards the rest of the tree.
If the new head was not a child of the previous head, will create a completely new node."
function takemoveandcleantree!(from::SearchNode, moveindex::Int; newgamestate::Union{GameState, Nothing}=nothing, policymodel=nothing, resethead::Bool=false)
    if from.children[moveindex] === nothing
        if (newgamestate === nothing) & (policymodel === nothing)
            throw(MissingChild("That move cannot be taken because the child does not exist. If you provide the new gamestate and policymodel, this would be OK."))
        end
        return newnode(policymodel, newgamestate)
    else
        newhead = from.children[moveindex]
        newhead.ishead = true
        if resethead
            newhead.parent = nothing
        end
        return newhead
    end
end

function playgame(game::GameState, players::Array{P}; display::Bool=true) where {P<:Player}
    if display
        displayboard(game)
    end
    while true
        move, moveindex = pickmove(game, players[game.current_player])
        game = executemove(game, move)
        for player in players
            updategamestate!(game, moveindex, player)
        end
        if display
            displayboard(game)
        end
        winner = winnerof(game)
        if winner == -1
            continue
        elseif winner == 0
            println("The game ends in a tie")
            return
        else
            println("Player ", winner, " has won!")
            return
        end
    end
end

"Tell a player what move was taken by a player so that it can update its tracking of the gamestate.

The GameState is provided in case you want to perform a check that it matches.
The default version of this does absolutely nothing.
"
function updategamestate!(game::GameState, moveindex::Int, player::Player)
end

"Return the depth of the node to its highest parent. An isolated node has a depth of 1."
function treedepth(node::SearchNode)
    depth = 1
    while true
        if node.parent === nothing
            return depth
        end
        depth += 1
        node = node.parent
    end
end

struct MissingChild <: Exception
    var::String
end