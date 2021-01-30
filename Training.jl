using Random
using HDF5
include("Games.jl")

"Play multiple test games."
function playtestgames(gamestart::GameState, players::Array{P}, number::Int, logfilepath::Union{String, Nothing}, randomize::Bool=true) where {P<:Player}
    if typeof(logfilepath) == String
        open(logfilepath, "w") do io
            write(io, "Match,Winner,Start player")
        end
    end
    
    for i in 1:number
        println("Playing game $i")
        game = deepcopy(gamestart)
        newplayers = deepcopy(players)
        playerorder = 1:length(players)
        if randomize
            playerorder = shuffle(playerorder)
            newplayers = newplayers[playerorder]
        end
        winner = playtestgame(game, newplayers)
        if winner > 0
            winner = playerorder[winner]
        end        
        if typeof(logfilepath) == String
            open(logfilepath, "a") do io
                write(io, "\n$(i),$(winner),$(playerorder[1])")
            end
        end
    end
end

"Play a test game in which multiple AIs play against eachother."
function playtestgame(game::GameState, players::Array{P}) where {P<:Player}
    while true
        move, moveindex = pickmove(game, players[game.current_player])
        game = executemove(game, move)
        for player in players
            updategamestate!(game, moveindex, player)
        end
        winner = winnerof(game)
        if winner == -1
            continue
        else
            return winner
        end
    end
end

"Play a series of games and save the data they generate for training in the future."
function playtraininggames(player::Player, number::Int, storepath::String)
    x = Dict{String, Array}()
    y_winprob = Dict{String, Array}()
    y_moveprob = Dict{String, Array}()
    for i in 1:number
        newplayer = deepcopy(player)
        try
            winner, newplayer = playtraininggame(newplayer)
            newx, newy_winprob, newy_moveprob = treetrainingdata(newplayer.headnode, player.temperature)
            dictionaryappend!(x, newx)
            dictionaryappend!(y_winprob, newy_winprob)
            dictionaryappend!(y_moveprob, newy_moveprob)
        catch e
            break
        end
    end
    h5open(storepath, "w") do fid
        for phasename in keys(x)
            g = create_group(fid, phasename)
            g["x"] = x[phasename]
            g["winprob"] = y_winprob[phasename]
            g["moveprob"] = y_moveprob[phasename]
        end
    end
end

"Play a traininggame in which a single AI takes all the moves."
function playtraininggame(player::Player)
    while true
        explorefrom(player.headnode, player.c_puct, player.lookaheads, player.model)
        _, moveindex = bestmove(player.headnode, player.temperature)
        player.headnode = takemoveandcleantree!(player.headnode, moveindex, resethead=false)
        winner = winnerof(player.headnode.gamestate)
        if winner > -1
            return winner, player
        end
    end
end

function treetrainingdata(finalnode::SearchNode, temperature::Real)
    winner = winnerof(finalnode.gamestate)
    x = Dict{String, AbstractArray}()
    y_winprob = Dict{String, AbstractArray}()
    y_moveprob = Dict{String, AbstractArray}()
    for i in 1:treedepth(finalnode)
        newx, newy_winprob, newy_moveprob = nodetrainingdata(finalnode, winner, temperature)
        dictionaryappend!(x, newx)
        dictionaryappend!(y_winprob, [newy_winprob])
        dictionaryappend!(y_moveprob, newy_moveprob)
        finalnode = finalnode.parent
    end
    return x, y_winprob, y_moveprob
end

function nodetrainingdata(node::SearchNode, winner::Int, temperature::Real)
    x = Dict(node.gamestate.phase => decisioninput(node.gamestate))
    if winner == 0
        winprob = 0.5
    elseif winner != node.gamestate.current_player
        winprob = 0
    else
        winprob = 1
    end
    y_winprob = Dict(node.gamestate.phase => winprob)
    y_moveprob = Dict(node.gamestate.phase => moveprobabilities(node, temperature))
    return x, y_winprob, y_moveprob
end

"Appends the arrays in one dictionary to the end of those in another.

The arrays of any keys found in appendfrom not in appendto are simply added to the dictionary.
"
function dictionaryappend!(appendto::Dict{String, A}, appendfrom::Dict{String, B}) where {A<:AbstractArray, B<:AbstractArray}
    for dictname in keys(appendfrom)
        if ! haskey(appendto, dictname)
            appendto[dictname] = deepcopy(appendfrom[dictname])
        else
            appendto[dictname] = vcat(appendto[dictname], appendfrom[dictname])
        end
    end
end