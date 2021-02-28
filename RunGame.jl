using HDF5
include("Connect.jl")
include("Training.jl")

path = "C:/Users/Daniel/documents/juliaprojects/connectexperiments/thickmodel"
#setupnewtrainer(path, C4NN, emptyboard(), Dict("main"=>[84]))
trainer = AITrainer(path)
#models =  Dict("main"=>newmodel(50))
#addnewmodels(trainer, models)
#idofmodelset(trainer, Dict("main"=>1))
runtrainingcycles(trainer, 1, Inf)
