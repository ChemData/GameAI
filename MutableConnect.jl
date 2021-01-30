using LinearAlgebra
include("Games.jl")


struct C4Move <: Move
    column::Int
    player::Int
end

mutable struct C4Game <: GameState
    board::Array{Int, 2}
    winninglength::Int
    playersymbols::Array{String, 1}
    current_player::Int
end

struct C4Random <: Player
end

struct C4Human <: Player
end

"Have an AI select a move."
function pickmove(gamestate::C4Game, player::C4Random)
    return C4Move(rand(availablemoves(gamestate)), gamestate.current_player)
end

"Have a Human select a move."
function pickmove(gamestate::C4Game, player::C4Human)
    selectedmove = Nothing
    println("Which column do you want to place in?")
    while true
        try
            choice = parse(Int, readline())
            selectedmove = C4Move(choice, gamestate.current_player)
            takemove!(deepcopy(gamestate), selectedmove)
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
    return selectedmove
end

function takemove!(gamestate::C4Game, move::C4Move)
    if gamestate.board[1, move.column] != 0
        throw(FullColumn("Column $(move.column) is already full."))
    end
    openposition = findlast(x->x==0, gamestate.board[:, move.column])
    gamestate.board[openposition, move.column] = move.player
    changeplayer!(gamestate)
end

"Return which columns are free to place a piece in."
function availablemoves(gamestate::C4Game)
    return findall(t->t==0, gamestate.board[1, :])
end

"Change the active player in the game."
function changeplayer!(gamestate::C4Game)
    newplayer = gamestate.current_player%length(gamestate.playersymbols) + 1
    gamestate.current_player = newplayer
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
    length = gamestate.winninglength
    
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

function emptyboard(height=6, width=7, winninglength=4)
    return C4Game(zeros(Int, height, width), winninglength, ["X", "O"], 1)
end

struct FullColumn <: Exception
    var::String
end

function testtime()
    g = emptyboard()
    players = [C4Random(), C4Random()]
    @time playmutablegame(g, players, false)
    @time playmutablegame(g, players, false)
end

testtime()