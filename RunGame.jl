using HDF5
include("Connect.jl")
include("Training.jl")

path = "C:/Users/Daniel/documents/juliaprojects/connectexperiments/thickmodel"
#setupnewtrainer(path, C4NN, emptyboard(), Dict("main"=>(Int32(84),)))
trainer = AITrainer(path)
#models =  Dict("main"=>newmodel(50)[1])
#addnewmodels(trainer, models)
#t = createmodelset(trainer, [4])
runtrainingcycles(trainer, 1, 10)

#g = newmodel(40)[1]
