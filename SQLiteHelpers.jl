using SQLite
using DataFrames

"Return the largest value in a column of a sqlite table. If that column has n max value return the defaultvalue."
function columnmax(db::SQLite.DB, table::String, column::String, defaultvalue=nothing)
    result = DataFrame(DBInterface.execute(db, "SELECT MAX($column) FROM $table"))[1, "MAX($column)"]
    if result === missing
        return defaultvalue
    end
    return result
end
