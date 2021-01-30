using HDF5
include("Connect.jl")
include("Training.jl")


g = emptyboard()
m = newmodel(30)
m2 = naivemodel(7)
p1 = C4NN(m, 1, 1000, newnode(m, g), .1)
p2 = C4NN(m2, 1, 1000, newnode(m2, g), .1)
#r = playtraininggame(p2)
#playtraininggames(p2, 3, "testgames.hdf5")
#playtestgames(g, [p1, p2], 100, "gameoutput.txt")
r, w = playtraininggame(p2)
x,y,y2 = treetrainingdata(w.headnode, .1)

#r = h5open("testgames.hdf5")

