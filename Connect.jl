using LinearAlgebra
using Flux
include("Games.jl")

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

struct C4Move <: Move
    column::Int
    player::Int
end

struct C4GameOptions <: GameOptions
    phases::Tuple{Vararg{String}}
    winninglength::Int
end

struct C4Game <: GameState
    options::C4GameOptions
    board::Array{Int, 2}
    playersymbols::Array{String, 1}
    current_player::Int
    phase::String
end

struct C4Random <: Player
end

struct C4Human <: Player
end

 mutable struct C4NN <: Player
    models::Dict{String, Union{Chain, Function}}
    headnode::SearchNode
    c_puct::Float64
    lookaheads::Int
    temperature::Real
end

"Have an AI select a move."
function pickmove(gamestate::C4Game, player::C4Random)
    pick = rand(availablemoves(gamestate))
    return C4Move(pick, gamestate.current_player), pick
end

"Have a Human select a move."
function pickmove(gamestate::C4Game, player::C4Human)
    selectedmove = nothing
    choice = nothing
    println("Which column do you want to place in?")
    while true
        try
            choice = parse(Int, readline())
            selectedmove = C4Move(choice, gamestate.current_player)
            executemove(deepcopy(gamestate), selectedmove)
            break
        catch e
            if typeof(e) == ArgumentError
                println("That is not a valid column number.")
            elseif typeof(e) == BoundsError
                println("That is not a valid column number.")
            elseif typeof(e) == FullColumn
                println("That column is already full.")
            end
        end
    end
    return selectedmove, choice
end

"Have a Neural Network AI select a move."
function pickmove(gamestate::C4Game, player::C4NN)
    explorefrom(player.headnode, player.c_puct, player.lookaheads, player.models)
    return bestmove(player.headnode, player.temperature)
end

function updategamestate!(game::GameState, moveindex::Int, player::C4NN)
    player.headnode = takemoveandcleantree!(player.headnode, moveindex; newgamestate=game, policymodels=player.models, resethead=true)
end

function executemove(gamestate::C4Game, move::C4Move)
    if gamestate.board[1, move.column] != 0
        throw(FullColumn("Column $(move.column) is already full."))
    end
    openposition = findlast(x->x==0, gamestate.board[:, move.column])
    newstate = deepcopy(gamestate)
    newstate.board[openposition, move.column] = move.player
    newstate = changeplayer(newstate)
    return newstate
end

"Return which columns are free to place a piece in."
function availablemoves(gamestate::C4Game)
    return findall(t->t==0, gamestate.board[1, :])
end

"Return a list of all moves including invalid ones. Also return a list of which moves are legal."
function listofmoves(gamestate::C4Game)
    moves = [C4Move(i, gamestate.current_player) for i in 1:size(gamestate.board)[2]]
    legalmoves = (t->t==0).(gamestate.board[1, :])
    return moves, legalmoves
end

"Change the active player in the game."
function changeplayer(gamestate::C4Game)
    newplayer = gamestate.current_player%length(gamestate.playersymbols) + 1
    return setplayer(gamestate, newplayer)
end

"Change the active player in the game to a specific player."
function setplayer(gamestate::C4Game, newplayer::Int)
    return C4Game(
        gamestate.options,
        deepcopy(gamestate.board),
        gamestate.playersymbols,
        newplayer,
        gamestate.phase
    )
end

"Change the board in the game to a specific board."
function setboard(gamestate::C4Game, newboard::Array)
    return C4Game(
        gamestate.gameoptions,
        deepcopy(newboard),
        gamestate.playersymbols,
        gamestate.current_player,
        gamestate.phase
    )
end

function displayboard(gamestate::C4Game)
    output = ""
    for i in 1:size(gamestate.board)[1]
        output *= "\n|"
        for j = 1:size(gamestate.board)[2]
            try
                output *= "$(gamestate.playersymbols[gamestate.board[i, j]])|"
            catch e
                if typeof(e) == BoundsError
                    output *= " |"
                end
            end 
        end
    end
    output *= "\n " * join(1:1:size(gamestate.board)[2], " ")
    println(output)
end

"Return the winner (if any) of a game."
function winnerof(gamestate::C4Game)
    height, width = size(gamestate.board)
    length = gamestate.options.winninglength
    
    # Check the rows
    for rownum in 1:height
        winner = containswinner(gamestate.board[rownum, :], length)
        if winner != 0
            return winner
        end
    end

    # Check the columns
    for colnum in 1:width
        winner = containswinner(gamestate.board[:, colnum], length)
        if winner != 0
            return winner
        end
    end

    # Check diagonals
    revboard = reverse(gamestate.board, dims=1)
    min_diag = 0
    max_diag = 0
    if width >= length
        min_diag = length - height
    end
    if height >= length
        max_diag = width - length
    end
    if (width < length) & (height < length)
        max_diag = -1
    end
    for diagnum in min_diag:max_diag
        winner = containswinner(diag(gamestate.board, diagnum), length)
        winner2 = containswinner(diag(revboard, diagnum), length)
        if winner != 0
            return winner
        elseif winner2 != 0
            return winner2
        end
    end

    # Check to see if the game ends in a tie or is simply unresolved
    if !(0 in gamestate.board)
        return 0
    else
        return -1
    end
end

"Return if there is a winner in an array"
function containswinner(row::Array{Int}, neededtowin::Int)
    curnum = 0
    run = 0
    for i in row
        if i == 0
            curnum = 0
            run = 0
        else
            if curnum == i
                run += 1
            else
                curnum = i
                run = 1
            end
        end
        if run == neededtowin
            return curnum
        end
    end
    return 0
end

"Create a new game of Connect4 with an empty board."
function emptyboard(height=6, width=7, winninglength=4)
    options = C4GameOptions(("main", ), winninglength)
    return C4Game(options, zeros(Int, height, width), ["X", "O"], 1, "main")
end

"Return input arrays for the decision model."
function decisioninput(gamestate::C4Game)
    board = deepcopy(gamestate.board)
    my_spots = (x-> x==gamestate.current_player).(board)
    opponent_spots = (x-> x!=gamestate.current_player && x!=0).(board)
    return convert(Array{Int}, [Flux.flatten(my_spots)..., Flux.flatten(opponent_spots)...])
end

function newmodel(hiddenlayersize::Int)
    model = Chain(
        Dense(84, hiddenlayersize),
        Split(
            Dense(hiddenlayersize, 1, NNlib.σ),
            Chain(
                Dense(hiddenlayersize, 7),
                softmax
            )
        )
    )
    return model
end

function naivemodel(numcolumns::Int)
    function modelfunc(inputdata::AbstractArray)
        output = ones(Float64, 1, numcolumns+1)
        output[:, 1] *= 0.5
        output[:, 2:end] *= 1/numcolumns
        return ([[0.5], ones(Float64, numcolumns)/numcolumns],)
    end
    return modelfunc
end

struct FullColumn <: Exception
    var::String
end
