using HDF5
include("Connect.jl")
include("Training.jl")


#setupnewtrainer("gaa", C4NN, emptyboard())
trainer = AITrainer("gaa")
#models =  Dict("main"=>newmodel(30))
#addnewmodels(trainer, models)
#generatetrainingdata(trainer, Dict("main"=> 1), 1000, 1, 100, .8)

#models = loadmodelset(trainer, Dict("main"=>1))
#player = C4NN(models, newnode(models, emptyboard()), 1, 100, .8)
#playtraininggame(player)


trainmodelsets(trainer, Dict("main"=>1), 1, 1, .2, 1, 100)

#runtestgames(trainer, [Dict("main"=>6), Dict("main"=>5)], 500, 1, 100, .6)
