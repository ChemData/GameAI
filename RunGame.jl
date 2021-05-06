using HDF5
include("Connect.jl")
include("Training.jl")

toexcel(df::DataFrame) = clipboard(sprint(show, "text/tab-separated-values", df))

path = "B:/JuliaProgramming/AI4"
#setupnewtrainer(path, C4NN, emptyboard(); modelinputsizes=Dict("main"=>(84,)))
#cp("B:/JuliaProgramming/defaults.json", "$path/defaults.json")
trainer = AITrainer(path)

#models =  Dict("main"=>newmodel(100)[1])
#addnewmodels(trainer, models)
#t = createmodelset(trainer, [1])

startingmodelsetid = 1
numberofdatasets = 1


#runtrainingcycles(trainer, 100, Inf)


#t2 = AITrainer("B:/JuliaProgramming/AI4")

"""
p1 = createais(t2, [1, 26, 66]; lookaheads=150)
p2 = createais(t2, [1, 26, 66]; lookaheads=500)
pt = vcat(p1, p2)
playtournament(pt, trainer.startstate, 600)
#z=playtournament(trainer, [1,2, 100, 247], 400)
"""