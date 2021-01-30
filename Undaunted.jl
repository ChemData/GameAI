

struct UndauntedBoard <: GameState
    cover::Array{Int64}
    adjacency::Array{Bool}
    allied_scouted::Array{Bool}
    axis_scouted::Array{Bool}
    allied_controlled::Array{Bool}
    axis_controlled::Array{Bool}
    point_values::Array{Int64}
    allied_spawns::Array{Bool}
    axis_spawns::Array{Bool}

    # Public
        # Invariant
            # Location of tiles
            # Cover of tiles
            # Spawn point locations
            # Point goals
        
        # Variable
            # Location of squads
            # Suppressed vs active
            # Scouted/controlled
            # Cards in supply
    
    # Private
        # Cards in hand
end

struct UndauntedTile <: TIle
    cover::Int64
    point_value::Int64
    allied_spawn::Bool
    axis_spawn::Bool
    allied_scouted::Bool
    axis_scouted::Bool
    allied_controlled::Bool
    axis_controlled::Bool
end
