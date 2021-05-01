using HDF5
include("Connect.jl")
include("Training.jl")


path = "B:/JuliaProgramming/AI2"
#setupnewtrainer(path, C4NN, emptyboard())
#cp("B:/JuliaProgramming/defaults.json", "B:/JuliaProgramming//AI2/defaults.json")
trainer = AITrainer(path)

#models =  Dict("main"=>convolutionalmodel(50))
#addnewmodels(trainer, models)
#t = createmodelset(trainer, [1])

startingmodelsetid = 1
numberofdatasets = 1


runtrainingcycles(trainer, startingmodelsetid, Inf)

#modelsetloss(trainer, 1, [1,2,3,4])