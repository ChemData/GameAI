using HDF5
include("Connect.jl")
include("Training.jl")

toexcel(df::DataFrame) = clipboard(sprint(show, "text/tab-separated-values", df))

path = "B:/JuliaProgramming/AI3"
#setupnewtrainer(path, C4NN, emptyboard())
#cp("B:/JuliaProgramming/defaults.json", "$path/defaults.json")
trainer = AITrainer(path)

#models =  Dict("main"=>convolutionalmodel(100))
#addnewmodels(trainer, models)
#t = createmodelset(trainer, [1])

startingmodelsetid = 1
numberofdatasets = 1


#runtrainingcycles(trainer, startingmodelsetid, Inf)

#modelsetloss(trainer, 1, [1,2,3,4])

z = playtournament(trainer, [1, 2, 6, 9, 19, 30, 39, 55], 500)