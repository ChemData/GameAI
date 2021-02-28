using Random
using JSON
include("Games.jl")

struct DominionCard <: Card
    name::String
    cost::Int
    value::Int
    vps::Int
    actions::Int
    draws::Int
    buys::Int
    types::Array{String}
    functions::Array{Function}
    marketnumber::Int
    decknumber::Int
    index::Int
end

struct CardSet
    cards::Array{DominionCard}
    nametocard:: Dict{String, DominionCard}
    numbertoname:: Dict{Int, String}
end

struct DominionState <: GameState
    hands::Array{Int, 2}
    decks::Array{Array{Int, 1}, 1}
    discards::Array{Int, 2}
    prediscards::Array{Int, 2}
    market::Array{Int, 1}
    actions::Array{Int}
    buys::Array{Int}
    money::Array{Int}
    current_player::Int
    current_phase::String
    cards::CardSet
end

abstract type DominionMove <: Move end

struct Buy <: DominionMove
    player::Int
    card::DominionCard
end

struct Play <: DominionMove
    player::Int
    card::DominionCard
end

struct EndPhase <: DominionMove
    player::Int
end

function copy(x::DominionState)
    output = DominionState(
        deepcopy(x.hands),
        deepcopy(x.decks),
        deepcopy(x.discards),
        deepcopy(x.prediscards),
        deepcopy(x.market),
        deepcopy(x.actions),
        deepcopy(x.buys),
        deepcopy(x.money),
        x.current_player,
        x.current_phase,
        x.cards)
    return output
end

"Create a CardSet from a list of desired cards."
function loadcards(cardstoinclude::Array{String})
    parseddata = JSON.parsefile("dominion_cards.json")
    cards = []
    nametocard = Dict{String, DominionCard}()
    numbertoname = Dict{Int, String}()
    for (number, name) in enumerate(cardstoinclude)
        cardinfo = parseddata[name]
        cards = cat(cards,
            DominionCard(
                cardinfo["name"], 
                get(cardinfo, "cost", 0),
                get(cardinfo, "value", 0),
                get(cardinfo, "vps", 0),
                get(cardinfo, "actions", 0),
                get(cardinfo, "draws", 0),
                get(cardinfo, "buys", 0),
                get(cardinfo, "types", []),
                [getfield(Main, Symbol(x)) for x in get(cardinfo, "special", [])],
                get(cardinfo, "game_amount", 10),
                get(cardinfo, "initial_deck", 0),
                number),
            dims=1)
        nametocard[name] = cards[number]
        numbertoname[number] = name
    end
    return CardSet(cards, nametocard, numbertoname)
end

struct DominionAI <:Player
end

struct DominionHuman <:Player
end

"Have an AI player pick a move."
function pickmove(gamestate::DominionState, player::DominionAI)
end

"Have a human player pick a move."
function pickmove(gamestate::DominionState, player::DominionHuman)
    currentplayer = gamestate.current_player
    if gamestate.current_phase == "play"
        println("What action card would you like to play?")
        choice = Nothing
        while true
            choice = readline()
            if choice == "pass"
                return EndPhase(currentplayer)
            end
            try
                playcard(gamestate, gamestate.cards.nametocard[choice], currentplayer)
                break
            catch e
                if typeof(e) == MissingCard
                    println("You don't have that card.")
                elseif typeof(e) == KeyError
                    println("That card doesn't exist.")
                elseif typeof(e) == WrongCardType
                    println("That is not an action card.")
                elseif typeof(e) == NoMoreActions
                    println("You have no more actions to take.")
                else
                    throw(e)
                end
            end
        end
        return Play(currentplayer, gamestate.cards.nametocard[choice])
    elseif gamestate.current_phase == "buy"
        println("What card would you like to buy?")
        choice = Nothing
        while true  
            choice = readline()
            if choice == "pass"
                return EndPhase(currentplayer)
            end
            try
                buycard(gamestate, gamestate.cards.nametocard[choice], currentplayer)
                break
            catch e
                if typeof(e) == NotEnoughMoney
                    println("You don't have enough money for that card.")
                elseif typeof(e) == NoMoreBuys
                    println("You have no more buys.")
                elseif typeof(e) == KeyError
                    println("That card doesn't exist.")
                elseif typeof(e) == EmptyPile
                    println("There are no more of that card in the market.")
                else
                    throw(e)
                end
            end
        end
        return Buy(currentplayer, gamestate.cards.nametocard[choice])
    elseif gamestate.current_phase == "endcheck"
        return EndPhase(currentplayer)

    end
end

"Buy a card for the player."
function takemove(gamestate::DominionState, move::Buy)
    return buycard(gamestate, move.card, gamestate.current_player)
end

"Play a card for the player."
function takemove(gamestate::DominionState, move::Play)
    return playcard(gamestate, move.card, gamestate.current_player) 
end

"End a phase and move to a new one."
function takemove(gamestate::DominionState, move::EndPhase)
    if gamestate.current_phase == "play"
        return changephase(laydowncoins(gamestate, gamestate.current_player), "buy")        
    elseif gamestate.current_phase == "buy"
        newstate = endturn(gamestate, gamestate.current_player)
        newstate = changephase(newstate, "endcheck")
        return newstate
    elseif gamestate.current_phase == "endcheck"
        newstate = changephase(gamestate, "play")
        newstate = changeplayer(newstate, nextplayer(newstate))
    else
        throw(UnknownPhase("there is no logic for handling the end of state $(gamestate.current_phase)."))
    end
end

"Display important information about the Dominion game."
function displayboard(gamestate::DominionState)
    output = "\nCurrent Player: $(gamestate.current_player)"
    if gamestate.current_phase == "play"
        output *= "\n\t"
        handlist = []
        for (cardindex, amount) in enumerate(gamestate.hands[gamestate.current_player, :])
            for i in 1:amount
                handlist = cat(handlist, gamestate.cards.cards[cardindex].name, dims=1)
            end
        end
        output *= join(handlist, ", ")
        output *= "\n\tActions: $(gamestate.actions[gamestate.current_player])"
    elseif gamestate.current_phase == "buy"
        output *= "\n\t"
        marketlist = []
        for (cardindex, amount) in enumerate(gamestate.market)
            card = gamestate.cards.cards[cardindex]
            displayamount = "$(amount)x $(card.name) (\$$(card.cost))"
            marketlist = cat(marketlist, displayamount, dims=1)
        end
        output *= join(marketlist, ", ")
    end
    output *= "\n\tMoney: $(gamestate.money[gamestate.current_player])"
    output *= "\n\tBuys: $(gamestate.buys[gamestate.current_player])"
    println(output)   
end

"Draw cards from the draw pile, shuffling if needed."
function drawcards(gamestate::DominionState, playernum::Int, number::Int)
    newstate = copy(gamestate)
    for i = 1:number
        if length(newstate.decks[playernum]) == 0
            newstate = shuffle(newstate, playernum)
        end
        if length(newstate.decks[playernum]) == 0
            break
        end
        card = pop!(newstate.decks[playernum])
        newstate.hands[playernum, card] += 1
    end
    return newstate
end

"Shuffle a discard pile and turn it into a draw pile."
function shuffle(gamestate::DominionState, playernum::Int)
    newstate = deepcopy(gamestate)
    if length(newstate.decks[playernum]) > 0
        throw(ArgumentError("A discard pile can only be shuffled if the draw pile is empty."))
    end
    newdraw = []
    for (cardnum, count) in enumerate(newstate.discards[playernum, :])
        if count > 0
            newdraw = cat(newdraw, ones(count)*cardnum, dims=1)
        end
    end
    shuffle!(newdraw)
    newstate.decks[playernum] = newdraw
    newstate.discards[playernum, :] *= 0
    return newstate
end 

function buycard(gamestate::DominionState, card::DominionCard, playernum::Int)
    if gamestate.buys[playernum] == 0
        throw(NoMoreBuys("player $playernum has no more buys."))
    elseif gamestate.money[playernum] < card.cost
        throw(NotEnoughMoney("player $playernum only has \$$(gamestate.money[playernum]) (costs $(card.cost))"))
    elseif gamestate.market[card.index] < 1
        throw(EmptyPile("The market has no more of that card."))
    end
    newstate = copy(gamestate)
    newstate.market[card.index] -= 1
    newstate.discards[playernum, card.index] += 1
    newstate.buys[playernum] -= 1
    newstate.money[playernum] -= card.cost
    return newstate
end

function playcard(gamestate::DominionState, card::DominionCard, playernum::Int)
    if gamestate.actions[playernum] == 0
        throw(NoMoreActions("player $playernum has no more actions to take."))
    elseif gamestate.hands[playernum, card.index] < 1
        throw(MissingCard("player $playernum does not have that card to play."))
    elseif !("action" in card.types)
        throw(WrongCardType("$(card.name) is not an action card."))
    end
    
    newstate = copy(gamestate)
    newstate.actions[playernum] -= 1
    newstate.hands[playernum, card.index] -= 1

    newstate.actions[playernum] += card.actions
    newstate.buys[playernum] += card.buys
    newstate.money[playernum] += card.value
    if card.draws > 0
        newstate = drawcards(newstate, playernum, card.draws)
    end
    for func in card.functions
        newstate = func(newstate, playernum)
    end
    newstate.prediscards[playernum, card.index] += 1
    return newstate
end

"Play down every coin card that a player has."
function laydowncoins(gamestate::DominionState, playernum::Int)
    newstate = copy(gamestate)
    for i in [1, 2, 3]
        count = newstate.hands[playernum, i]
        newstate.money[playernum] += i * count
        newstate.discards[playernum, i] += count
        newstate.hands[playernum, i] = 0
    end
    return newstate
end

"Perform clean up actions at the end of a players turn."
function endturn(gamestate::DominionState, playernum::Int)
    newstate = copy(gamestate)
    newstate.discards[playernum, :] += newstate.hands[playernum, :]
    newstate.hands[playernum, :] *= 0
    newstate.discards[playernum, :] += newstate.prediscards[playernum, :]
    newstate.prediscards[playernum, :] *= 0
    newstate = drawcards(newstate, playernum, 5)
    newstate.actions[playernum] = 1
    newstate.buys[playernum] = 1
    newstate.money[playernum] = 0
    return newstate
end

"Return the number of the next player to go."
function nextplayer(gamestate::DominionState)
    numplayers = size(gamestate.hands)[1]
    return gamestate.current_player%numplayers + 1
end

"Return a DominionState with an updated phase."
function changephase(gamestate::DominionState, newphase::String)
    newstate = DominionState(
        deepcopy(gamestate.hands),
        deepcopy(gamestate.decks),
        deepcopy(gamestate.discards),
        deepcopy(gamestate.prediscards),
        deepcopy(gamestate.market),
        deepcopy(gamestate.actions),
        deepcopy(gamestate.buys),
        deepcopy(gamestate.money),
        gamestate.current_player,
        newphase,
        gamestate.cards
    )
    return newstate
end

"Change the current player of a game."
function changeplayer(gamestate::DominionState, newplayer::Int)
    newstate = DominionState(
        deepcopy(gamestate.hands),
        deepcopy(gamestate.decks),
        deepcopy(gamestate.discards),
        deepcopy(gamestate.prediscards),
        deepcopy(gamestate.market),
        deepcopy(gamestate.actions),
        deepcopy(gamestate.buys),
        deepcopy(gamestate.money),
        newplayer,
        gamestate.current_phase,
        gamestate.cards
    )
    return newstate
end

"Determine the winner (if any) of a game."
function winnerof(gamestate::DominionState)
    # Determine if the game is in the buy phase (and so can be ended)
    if gamestate.current_phase != "endcheck"
        return -1
    end
    remainingprovinces = gamestate.market[gamestate.cards.nametocard["province"].index]
    depletedpiles = sum(gamestate.market .== 0)
    if remainingprovinces > 0 && depletedpiles < 3
        return -1
    end
    cardvpvalues = [card.vps for card in gamestate.cards.cards]
    scores = gamestate.discards * cardvpvalues
    winners = findall(x->x==maximum(scores), scores)
    if length(winners) > 1
        return 0
    else
        return winners[1]
    end
end

function newdominion(cardstoinclude::Array{String}, numberofplayers::Int)
    cards = loadcards(cardstoinclude)
    startstate = DominionState(
                    zeros(Int, numberofplayers, length(cards.cards)),
                    [Int[] for i in 1:numberofplayers],
                    transpose(hcat([[card.decknumber for card in cards.cards] for i in 1:numberofplayers]...)),
                    zeros(Int, numberofplayers, length(cards.cards)),
                    [card.marketnumber for card in cards.cards],
                    zeros(Int, numberofplayers),
                    zeros(Int, numberofplayers),
                    zeros(Int, numberofplayers),
                    1,
                    "play",
                    cards
    )
    for i in 1:numberofplayers
        startstate = endturn(startstate, i)
    end
    return startstate
end

"Allow the player to discard cards after playing a chapel."
function chapeldiscard(gamestate::DominionState)
    return copy(gamestate)
end

"Force other players to discard cards after playing a militia."
function militiaattack(gamestate::DominionState)
    return copy(gamestate)
end

struct NotEnoughMoney <: Exception
    var::String
end

struct NoMoreActions <: Exception
    var::String
end

struct NoMoreBuys <: Exception
    var::String
end

struct MissingCard <: Exception
    var::String
end 

struct WrongCardType <: Exception
    var::String
end

struct EmptyPile <: Exception
    var::String
end

d = newdominion(["copper", "silver", "gold", "estate", "duchy", "province", "smithy", "village", "market"], 1)
d.hands[1, 8] = 1
d.hands[1, 7] = 1
d.hands[1, 9] = 5
d.hands[1, 6] = 4
d.market[6] = 0
p = DominionHuman()
playgame(d, [p])
