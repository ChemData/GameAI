using LinearAlgebra
include("Games.jl")

struct T3State <: GameState
    turn::Int
    current_player::Int
    board::Array{Int64}
end

struct T3Random <: Player end

struct T3Human <: Player end

struct T3Move <: Move
    x_pos::Int
    y_pos::Int
end

function newboard(boardsize::Int)
    return T3State(0, 1, zeros(Int, boardsize, boardsize))
end

function takemove(game::T3State, move::T3Move)
    array_copy = copy(game.board)
    if array_copy[move.x_pos, move.y_pos] != 0
        throw(InvalidMove("This position is already taken."))
    end
    array_copy[move.x_pos, move.y_pos] = game.current_player

    return T3State(game.turn+1, game.current_player%2+1, array_copy)
end

function legalmoves(game::T3State)
    output = T3Move[]
    for i = 1:size(game.board, 1)
        for j = 1:size(game.board, 2)
            if game.board[i, j] == 0
                push!(output, T3Move(i, j))
            end
        end
    end
    return output
end

"""
    displayboard(game::T3State)

Print the board state of a game.
"""
function displayboard(game::T3State)
    valmap = Dict([(0, " "), (1, "X"), (2, "O")])
    disp_array = map((x) -> valmap[x], game.board)
    println("___")
    for x = 1:size(game.board)[1]
        println(join(disp_array[x, begin:end]))
    end
    println("___")
end


"""
    winnerof(game::T3State)

Return which player, if any, has won the game.

0 indicates tie. 1 indicates player1, 2 indicates player2, -1 indicates unfinished.
"""
function winnerof(game::T3State)
    boardsize = size(game.board)[1]

    winner_1 = ones(Int, boardsize)
    winner_2 = winner_1 * 2
    for row in eachrow(game.board)
        if row == winner_1
            return 1
        elseif row == winner_2
            return 2
        end
    end
    for col in eachcol(game.board)
        if col == winner_1
            return 1
        elseif col == winner_2
            return 2
        end
    end
    
    for diagonal in [diag(game.board), diag(reverse(game.board, dims=1))]
        if diagonal == winner_1
            return 1
        elseif diagonal == winner_2
            return 2
        end
    end
    
    if ! (0 in game.board)
        return 0
    end
    return -1
end

function pickmove(game::T3State, player::T3Random)
    return rand(legalmoves(game))
end

function pickmove(game::T3State, player::T3Human)
    options = legalmoves(game)
    println("Where do you want to play? [Enter as row column]")
    while true
        row, col = split(readline())
        move = T3Move(tryparse(Int, row), tryparse(Int, col))
        if move in options
            return move
        else
            println("That is not an allowed move!")
        end
    end
end

g = newboard(3)
players = [T3Random(), T3Human()]

