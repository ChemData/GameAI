using Random
using HDF5
using SQLite
using JLD
using BSON: @save, @load
using Dates
using Logging
using JSON
using Formatting
include("SQLiteHelpers.jl")
include("Games.jl")

struct AITrainer
    path::String
    db::SQLite.DB
    playertype::DataType
    startstate::GameState
    defaults::Dict
end

AITrainer(path) = AITrainer(
    path,
    SQLite.DB(joinpath(path, "info.sqlite")),
    JLD.load(joinpath(path, "player.jld"))["playertype"],
    JLD.load(joinpath(path, "startstate.jld"))["startstate"],
    readparamjson(joinpath(path, "defaults.json"))
    )

"Create (if not existing) the folders/sql database to store training information in. Store the Type of Player to use and the starting state for games."
function setupnewtrainer(path::String, playertype::DataType, startstate::GameState)
    mkpath(joinpath(path, "models"))
    mkpath(joinpath(path, "training-data"))
    db = SQLite.DB(joinpath(path, "info.sqlite"))

    tables = [
        "TrainingData(
            dataid INTEGER PRIMARY KEY,
            modelsused INTEGER,
            creationdate TEXT,
            creationtime REAL,
            numberofgames INTEGER,
            cpuct REAL,
            lookaheads INTEGER,
            temperature REAL,
            FOREIGN KEY(modelsused) REFERENCES ModelSets(modelsetid)
        );", 
        "TrainingSessions(
            sessionid INTEGER PRIMARY KEY,
            startingmodels INTEGER,
            startdate TEXT,
            totaltrainingtime REAL,
            numberofdatasets INTEGER,
            numberofepochs INTEGER,
            trainingfraction REAL,
            learningrate REAL,
            momentum REAL,
            FOREIGN KEY(startingmodels) REFERENCES ModelSets(modelsetid)
        );",
        "Models(
            modelid INTEGER PRIMARY KEY,
            trainingsession INTEGER,
            epoch INTEGER,
            gamephase TEXT,
            trainingtime REAL,
            testloss REAL,
            trainingloss REAL,
            FOREIGN KEY(trainingsession) REFERENCES TrainingSessions(sessionid)
        );",
        "TrainingUses(
            session INTEGER,
            trainingdata INTEGER,
            FOREIGN KEY(session) REFERENCES TrainingSessions(sessionid),
            FOREIGN KEY(trainingdata) REFERENCES TrainingData(dataid)
        );",
        "TestGames(
            gamesetid INTEGER PRIMARY KEY,
            testdate TEXT,
            time REAL,
            numberofgames INTEGER
        );",        
        "TestGamePlayers(
            gameset INTEGER,
            player INTEGER,
            winfraction REAL,
            tiefraction REAL,
            FOREIGN KEY(gameset) REFERENCES TestGames(gamesetid),
            FOREIGN KEY(player) REFERENCES ModelSets(modelsetid) 
        );"
    ]
    for table in tables
        stmt = "CREATE TABLE IF NOT EXISTS " * table
        DBInterface.execute(db, stmt)
    end

    stmt = "CREATE TABLE IF NOT EXISTS ModelSets(
                modelsetid INTEGER PRIMARY KEY"
    columnsspecifications = ""
    foreignkeys = ""
    uniques = ""
    for (i, phase) in enumerate(startstate.options.phases)
        columnsspecifications *= ", $phase INTEGER"
        foreignkeys *= ", FOREIGN KEY($phase) REFERENCES Models(modelid)"
        if i == 1
            uniques *= "$phase"
        else
            uniques *= ", $phase"
        end
    end
    stmt = stmt * columnsspecifications * foreignkeys * ", UNIQUE(" * uniques * "));"
    DBInterface.execute(db, stmt)

    DBInterface.close!(db)

    if isfile(joinpath(path, "player.jld"))
        throw(ArgumentError("a stored playertype already exists there."))
    else
        JLD.save(joinpath(path, "player.jld"), "playertype", playertype)
    end
    if isfile(joinpath(path, "startstate.jld"))
        throw(ArgumentError("a stored startstate already exists there."))
    else
        JLD.save(joinpath(path, "startstate.jld"), "startstate", startstate)
    end
end

"Return the most recently generated datasets and their ids."
function newestdatasets(trainer::AITrainer, number::Int)
    sql = "SELECT dataid FROM DataSets OREDER BY creationdate DESC LIMIT $number"
end

"Add a new set of models. This is how to add models directly to the database without having to train them."
function addnewmodels(trainer::AITrainer, models::Dict{String, T}) where{T <: Any}
    modelids = Dict{String, Int}()
    for (gamephase, model) in models
        query = DBInterface.execute(trainer.db, "INSERT INTO Models (gamephase) VALUES (?)", [gamephase])
        newnumber = DBInterface.lastrowid(query)
        @save joinpath(trainer.path, "models", "$newnumber.bson") model
        modelids[gamephase] = newnumber
    end
    return modelids
end

"Return the modelsetid of this set of models if it is already in the table ModelSets. If it is not, add it and return its new id."
function idofmodelset(trainer::AITrainer, modelset::Dict{String, Int})
    stmt = "SELECT modelsetid FROM ModelSets WHERE"
    for phase in keys(modelset)
        stmt *= " $phase = ? AND"
    end
    stmt = stmt[1:end-4]
    ids = DataFrame(DBInterface.execute(trainer.db, stmt, collect(values(modelset))))
    if size(ids)[1] == 0
        columns = "(" * join(collect(keys(modelset)), ", ") * ")"
        qmark = "(" * join(repeat(["?"], length(modelset)), ", ") * ")"
        stmt = "INSERT INTO ModelSets $columns VALUES $qmark"
        query = DBInterface.execute(trainer.db, stmt, collect(values(modelset)))
        return Int64(DBInterface.lastrowid(query))
    else
        return Int64(ids[1, "modelsetid"])
    end
end

"Return the modelset of this id."
function modelsetfromid(trainer::AITrainer, modelsetid::Int)
    stmt = "SELECT * FROM ModelSets WHERE modelsetid=?;"
    data = DataFrame(DBInterface.execute(trainer.db, stmt, [modelsetid]))
    output = Dict{String, Int}()
    for col in names(data)
        if col != "modelsetid"
            output[col] = data[1, col]
        end
    end
    return output
end

"Generate training data and save it."
function generatetrainingdata(trainer::AITrainer, modelids::Dict{String, Int}; number::Int=100, c_puct::Real=1, lookaheads::Int=100, temperature::Real=1)
    modelsetid = idofmodelset(trainer, modelids)
    models = loadmodelset(trainer, modelids)
    headnode = newnode(models, deepcopy(trainer.startstate))
    player = trainer.playertype(models, headnode, c_puct, lookaheads, temperature)
    newid = columnmax(trainer.db, "TrainingData", "dataid", 0) + 1
    starttime = time()
    io = open(joinpath(trainer.path, "errorlog.txt"), "a+")
    logger = SimpleLogger(io)
    playtraininggames(player, number, joinpath(trainer.path, "training-data", "data$newid.hdf5"); logger)
    flush(io)
    close(io)
    stmt = "INSERT INTO TrainingData (modelsused, creationdate, creationtime, numberofgames, cpuct, lookaheads, temperature) VALUES (?, ?, ?, ?, ?, ?, ?)"
    DBInterface.execute(trainer.db, stmt, [modelsetid, Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"), time()-starttime, number, c_puct, lookaheads, temperature])
end

"Load and return a set of models. The keys of `model` are the names of game phases. The values are modelids of the models.
This will check that the phase of the loaded model matches the phase stated in `models`.
"
function loadmodelset(trainer::AITrainer, models::Dict{String, Int})
    output = Dict()
    for phase in keys(models)
        modelid = models[phase]
        @load joinpath(trainer.path, "models", "$modelid.bson") model
        storedphase = DataFrame(DBInterface.execute(trainer.db, "SELECT gamephase FROM Models WHERE modelid == ?", [modelid]))[1, "gamephase"]
        if storedphase != phase
            throw(WrongPhase("the phase of model $modelid is $storedphase but the input says it is $phase."))
        end
        output[phase] = model
    end
    return output
end

"Load and return training data."
function loadtrainingdata(trainer::AITrainer, datasetids::Union{Array, Int})
    if typeof(datasetids) <: Int
        datasetids = [datasetids]
    end
    alldata = Dict{String, Dict{String, AbstractArray}}()
    for (i, id) in enumerate(datasetids)
        opendata = h5open(joinpath(trainer.path, "training-data", "data$id.hdf5"))
        for phase in keys(opendata)
            if i == 1
                alldata[phase] = Dict{String, AbstractArray}(read(opendata[phase]))
            else
                dictionaryappend!(
                    alldata[phase],
                    Dict{String, AbstractArray}(read(opendata[phase])))
            end
        end
    end
    return alldata
end

function jointloss(yhat, y)
    yhat = yhat[1]
    return Flux.Losses.mse(yhat[1], y[[1], :]) + Flux.Losses.crossentropy(yhat[2], y[2:end, :])
end

"Train new ModelSets.

# Arguments
- `trainer::AITrainer`: the training process to train the models of.
- `startingmodels::Dict{String, Int}`: The ids of the models to train.
- `numberofdatasets::Int`: how many recent datasets to use for training.
- `learningrate::Real`: The learning rate for the Momentum optimizer.
- `momentum::Real`: The momentum for the Momentum optimizer.
- `trainingfraction::Real`: The percentage of the training data to use for training (the residual is used for testing).
- `savefrequency::Int`: How frequently models should be saved during training. 1 causes models to be saved each epoch. 2 causes them to be stored
    every other epoch. The model with the best performance on the test set is always saved. 0 causes no models (except the best performing) to be saved.
"
function trainmodelsets(trainer::AITrainer, startingmodels::Dict{String, Int}; numberofdatasets::Int, learningrate::Real,
                         momentum::Real, numberofepochs::Int, trainingfraction::Real=0.8, savefrequency::Real=Inf)
    DBInterface.execute(trainer.db, "SAVEPOINT trainingstart")
    modelsetid = idofmodelset(trainer, startingmodels)
    savedmodelpaths = Array{String, 1}()
    sessionid = columnmax(trainer.db, "TrainingSessions", "sessionid",  0) + 1
    modelset = loadmodelset(trainer, startingmodels)
    dataids = DataFrame(DBInterface.execute(trainer.db, "SELECT dataid FROM TrainingData ORDER BY creationdate desc LIMIT ?", [numberofdatasets]))[!, "dataid"]
    try
        for dataid in dataids
            DBInterface.execute(trainer.db, "INSERT INTO TrainingUses (session, trainingdata) VALUES (?, ?)", [sessionid, dataid])
        end

        data = loadtrainingdata(trainer, dataids)
        
        sessionstarttime = time()
        sessionstartdate = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
        savestmt = "
        INSERT INTO
            Models (trainingsession, epoch, gamephase, trainingtime, trainingloss, testloss)
        VALUES (?, ?, ?, ?, ?, ?)"
        for gamephase in keys(modelset)
            print("\nTraining $gamephase Model\n")
            model = modelset[gamephase]
            modeldata = data[gamephase]
            training_data, test_data = splitdata((modeldata["x"], modeldata["winprob"], modeldata["moveprob"]), trainingfraction)
            data = Flux.Data.DataLoader(training_data..., batchsize=size(training_data[1])[2])
            function totalloss(x, y1, y2)
                yhat = model(x)[1]
                return Flux.Losses.mse(yhat[1], y1) + Flux.Losses.crossentropy(yhat[2], y2) + sum(Flux.norm, params(model))
            end
            
            bestmodel_results = [sessionid, 0, gamephase, 0, Inf, Inf]
            bestmodel = model
            for i in 1:numberofepochs
                modelstarttime = time()
                Flux.train!(totalloss, params(model), data, Momentum(learningrate, momentum))
                trainingloss = totalloss(training_data...)
                testloss = totalloss(test_data...)
                
                fe = FormatExpr("\tEpoch {}/{} - Training loss: {:.3f}, Test loss: {:.3f}\r")
                print(format(fe, i, numberofepochs, trainingloss, testloss))
                if (testloss < bestmodel_results[6]) 
                    bestmodel_results = [sessionid, i, gamephase, time()-modelstarttime, trainingloss, testloss]
                    bestmodel = deepcopy(model)
                end
                if i%savefrequency == 0
                    query = DBInterface.execute(trainer.db, savestmt, [sessionid, i, gamephase, time()-modelstarttime, trainingloss, testloss])
                    modelid = DBInterface.lastrowid(query)
                    savepath = joinpath(trainer.path, "models", "$modelid.bson")
                    @save savepath model
                    push!(savedmodelpaths, savepath)
                end
            end
            if bestmodel_results[2]%savefrequency != 0
                query = DBInterface.execute(trainer.db, savestmt, bestmodel_results)
                modelid = DBInterface.lastrowid(query)
                savepath = joinpath(trainer.path, "models", "$modelid.bson")
                @save savepath model
                push!(savedmodelpaths, savepath)
            end
        end
        sql = "
        INSERT INTO 
            TrainingSessions (sessionid, startingmodels, startdate, totaltrainingtime, numberofdatasets, numberofepochs, learningrate, momentum, trainingfraction)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);" # if you can start with any set of models, how do you store that information???
        DBInterface.execute(
            trainer.db, sql, 
            [sessionid, modelsetid, sessionstartdate, time()-sessionstarttime, numberofdatasets,
            numberofepochs, learningrate, momentum, trainingfraction])
    catch e
        for filename in savedmodelpaths
            rm(filename)
        end
        DBInterface.execute(trainer.db, "ROLLBACK TO SAVEPOINT trainingstart")
        DBInterface.execute(trainer.db, "RELEASE SAVEPOINT trainingstart")
        rethrow(e)
    end
    DBInterface.execute(trainer.db, "RELEASE SAVEPOINT trainingstart")
    return sessionid
end

"Divide a set of data into training and test sets."
function splitdata(data, training_fraction)
    number = size(data[1])[end]
    trainingnumber = Int(round(number*training_fraction))
    order = randperm(number)
    training = []
    test = []
    for dataset in data
        push!(training, dataset[:, order[1:trainingnumber]])
        push!(test, dataset[:, order[trainingnumber:end]]) 
    end
    return training, test
end

"Return the ids of the model which performed best on the test set."
function bestmodels(trainer::AITrainer, phases::String; sessionid::Union{Int, Nothing}=nothing)
    if sessionid === nothing
        stmt = "
        SELECT 
            modelid 
        FROM 
            Models 
        WHERE 
            gamephase = ? AND trainingloss = (SELECT MIN(trainingloss) FROM Models);"
        modelid = DataFrame(DBInterface.execute(trainer.db, stmt, [phases]))
    else
        stmt = "
        SELECT 
            modelid 
        FROM 
            Models 
        WHERE 
            gamephase = ? AND trainingloss = (SELECT MIN(trainingloss) FROM Models WHERE trainingsession=?) AND trainingsession = ?;"
        modelid = DataFrame(DBInterface.execute(trainer.db, stmt, [phases, sessionid, sessionid]))
    end
    
    return modelid[1, "modelid"]
end

"Returm the ids of the models for multiple phases which performed best on the test sets."
function bestmodels(trainer::AITrainer, phases::Array{String, 1}; sessionid::Union{Int, Nothing}=nothing)
    return Dict((phase=>bestmodels(trainer, phase; sessionid=sessionid)) for phase in phases)
end

"Return the ids of models for each phase which performed best on the test sets."
function bestmodels(trainer::AITrainer; sessionid::Union{Int, Nothing}=nothing)
    return bestmodels(trainer, [trainer.startstate.options.phases...]; sessionid=sessionid)
end

"Play some test games."
function runtestgames(trainer::AITrainer, playersmodels::Array{Dict{String, Int}, 1}; numberofgames::Int, c_puct::Real, lookaheads::Int, temperature::Real)
    players = Array{trainer.playertype, 1}()
    setids = []
    for modelids in playersmodels
        push!(setids, idofmodelset(trainer, modelids))
        models = loadmodelset(trainer, modelids)
        headnode = newnode(models, deepcopy(trainer.startstate))
        newplayer = trainer.playertype(models, headnode, c_puct, lookaheads, temperature)
        push!(players, newplayer)
    end
    
    io = open(joinpath(trainer.path, "errorlog.txt"), "a+")
    logger = SimpleLogger(io)
    starttime = time()
    winners = playtestgames(trainer.startstate, players, numberofgames; logger)
    flush(io)
    close(io)

    stmt = "INSERT INTO TestGames (testdate, time, numberofgames) VALUES (?, ?, ?);"
    query = DBInterface.execute(trainer.db, stmt, [Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"), time()-starttime, winners["total"]])
    testgameid = DBInterface.lastrowid(query)
    for (playernumber, setid) in enumerate(setids)
        stmt = "INSERT INTO TestGamePlayers (gameset, player, winfraction, tiefraction) VALUES (?, ?, ?, ?);"
        DBInterface.execute(trainer.db, stmt, [testgameid, setid, winners[playernumber]/winners["total"], winners[0]/winners["total"]])
    end
    return winners
end

"Play multiple test games."
function playtestgames(gamestart::GameState, players::Array{P}, number::Int; logger::Union{Nothing, AbstractLogger}=nothing, randomize::Bool=true) where {P<:Player}
    output = Dict{Any, Int}(i=>0 for i in 0:length(players))
    output["total"] = 0
    print("\nPlaying Test Games\n")
    for i in 1:number
        print("\t$i/$number\r")
        game = deepcopy(gamestart)
        newplayers = deepcopy(players)
        playerorder = 1:length(players)
        if randomize
            playerorder = shuffle(playerorder)
            newplayers = newplayers[playerorder]
        end
        try
            winner = playtestgame(game, newplayers)
            if winner > 0
                winner = playerorder[winner]
            end
            output[winner] += 1
            output["total"] += 1
        catch e
            print("Failed.\r")
            if logger !== nothing
                with_logger(logger) do 
                    @info("\n\nIssue with Test Games", now(), sprint(showerror, e, backtrace()))
                end       
            end
        end
    end
    return output
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
function playtraininggames(player::Player, number::Int, storepath::String; logger::Union{Nothing, AbstractLogger}=nothing)
    x = Dict{String, Array}()
    y_winprob = Dict{String, Array}()
    y_moveprob = Dict{String, Array}()
    print("\nPlaying Training Games\n")
    for i in 1:number
        print("\t$i/$number\r")
        newplayer = deepcopy(player)
        try
            winner, newplayer = playtraininggame(newplayer)
            newx, newy_winprob, newy_moveprob = treetrainingdata(newplayer.headnode, player.temperature)
            dictionaryappend!(x, newx)
            dictionaryappend!(y_winprob, newy_winprob)
            dictionaryappend!(y_moveprob, newy_moveprob)
        catch e
            print("Failed.\r")
            if logger !== nothing
                with_logger(logger) do 
                    @info("\n\nIssue with Training Games", now(), sprint(showerror, e, backtrace()))
                end       
            end
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
        explorefrom(player.headnode, player.c_puct, player.lookaheads, player.models)
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
    finalnode = finalnode.parent
    x = Dict{String, AbstractArray}()
    y_winprob = Dict{String, AbstractArray}()
    y_moveprob = Dict{String, AbstractArray}()
    while true
        if finalnode === nothing
            break
        end
        newx, newy_winprob, newy_moveprob = nodetrainingdata(finalnode, winner, temperature)
        dictionaryappend!(x, newx)
        dictionaryappend!(y_winprob, newy_winprob)
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
    y_winprob = Dict(node.gamestate.phase => [winprob])
    y_moveprob = Dict(node.gamestate.phase => moveprobabilities(node, temperature))
    return x, y_winprob, y_moveprob
end

"""
A training cycle consists of these phases
1) Identify the most successful model set.
2) Use that model set to generate training data.
3) Train models using that data (starting with the constituent models of that set).
4) Pick out the best models from those that are newly trained and make a model set from them.
5) Play test games between the new and old model sets. Check if the new one is better.
"""

"Run a training cycle using the specified, already stored, models."
function runtrainingcycle(trainer::AITrainer, initialmodelset::Dict)
    io = open(joinpath(trainer.path, "trainingcycleoutput.txt"), "a")
    date = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM")
    write(io, "\n[$date]")
    
    # Generate new training data
    initialid = idofmodelset(trainer, initialmodelset)
    start = time()
    generatetrainingdata(trainer, initialmodelset; trainer.defaults["generatetrainingdata"]...)
    fe = FormatExpr("\n\tGenerated training data from modelset {}. [{:.2f} minutes]")
    write(io, format(fe, initialid, (time()-start)/60))
    
    # Train the models
    start = time()
    sessionid = trainmodelsets(trainer, initialmodelset; trainer.defaults["trainmodelsets"]...)
    if length(initialmodelset) == 1
        fe = FormatExpr("\n\tTrained model {}. [{:.2f} minutes]")
    else
        fe = FormatExpr("\n\tTrained models {}. [{:.2f} minutes]")
    end
    write(io, format(fe, join(values(initialmodelset), ", ", ", and "), (time()-start)/60))

    # Identify the best new models
    newmodelset = bestmodels(trainer, sessionid=sessionid)
    newmodelsetid = idofmodelset(trainer, newmodelset)
    if length(newmodelset) == 1
        fe = FormatExpr("\n\tThe best new model is {}. It is part of modelset {}")
    else
        fe = FormatExpr("\n\tThe best new models were {}. They are part of modelset {}")
    end
    write(io, format(fe, join(values(newmodelset), ", ", ", and "), newmodelsetid))

    # Play test games between the old and new model set
    start = time()
    result = runtestgames(trainer, [newmodelset, initialmodelset]; trainer.defaults["runtestgames"]...)
    wins = result[1]/result["total"]*100
    ties = result[0]/result["total"]*100
    fe = FormatExpr("\n\tModelset {} won {:.1f}% of the time vs modelset {} (and tied {:.1f}% of the time). [{:.2f} minutes]")
    write(io, format(fe, newmodelsetid, wins, initialid, ties, (time()-start)/60))

    close(io)
    return newmodelsetid, Dict("wins"=>result[1], "loses"=>result["total"]-result[0]-result[1])
end

"Run a training cycle using the specified already stored, modelset (from modelsetid)."
function runtrainingcycle(trainer::AITrainer, initialmodelsetid::Int)
    modelset = modelsetfromid(trainer, initialmodelsetid)
    return runtrainingcycle(trainer, modelset)
end

"Run multiple training cycles."
function runtrainingcycles(trainer::AITrainer, initialmodelset::Int, number::Real)
    touch(joinpath(trainer.path, "delete_to_stop_training_cycles.txt"))
    completed = 0
    modelsettouse = initialmodelset
    while completed < number
        newsetid, trainingresults = runtrainingcycle(trainer, modelsettouse)
        if trainingresults["wins"] >= 1.1*trainingresults["loses"]
            modelsettouse = newsetid
        else
            io = open(joinpath(trainer.path, "trainingcycleoutput.txt"), "a")
            write(io, "\n\tModel set $newsetid is not significantly better than $modelsettouse so it will not be carried forward.")
            close(io)
        end

        if ! ispath(joinpath(trainer.path, "delete_to_stop_training_cycles.txt"))
            break
        end
        completed += 1
    end
end

"Run multiple training cycles starting with some newly defined models."
function runtrainingcycles(trainer::AITrainer, newmodels::Dict{String, T}, number::Real) where {T <: Any}
    modelset = addnewmodels(trainer, newmodels)
    modelsetid = Int64(idofmodelset(trainer, modelset))
    runtrainingcycles(trainer, modelsetid, number)
end


"Appends the arrays in one dictionary to the end of those in another.

The arrays of any keys found in appendfrom not in appendto are simply added to the dictionary.
"
function dictionaryappend!(appendto::Dict{String, A}, appendfrom::Dict{String, B}) where {A<:AbstractArray, B<:AbstractArray}
    for dictname in keys(appendfrom)
        if ! haskey(appendto, dictname)
            appendto[dictname] = deepcopy(appendfrom[dictname])
        else
            appendto[dictname] = hcat(appendto[dictname], appendfrom[dictname])
        end
    end
end

"Return the parsed contents of a json file."
function readparamjson(path::String)
    open(path, "r") do f
        dict = JSON.parse(String(read(f)))
        output = Dict{String, Dict}()
        for (paramset, params) in dict
            output[paramset] = Dict(Symbol(x)=>params[x] for x in keys(params))
        end
        return output
    end
end

struct WrongPhase <: Exception
    var::String
end