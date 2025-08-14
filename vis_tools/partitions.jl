using WriteVTK
using LinearAlgebra

const dp = 4
const dn = 5
const rp = 2
const rn = 3
const s = 1

const rrp = 4
const srp = 2
const sym = 1
const srn = 3
const rrn = 5

const trisector = 1
const wedge = 2

function decomposeTensor(a,b,c,d)
    D = (a+d)/2
    R = (c-b)/2
    S = sqrt( (a-d)^2 + (b+c)^2 )/2

    return (D,R,S)
end

function classifyTensorEigenvalue(D,R,S)
    if abs(D) >= abs(R) && abs(D) >= abs(S)
        if D > 0
            return dp
        else
            return dn
        end
    elseif abs(R) >= abs(D) && abs(R) >= abs(S)
        if R > 0
            return rp
        else
            return rn
        end
    else
        return s
    end
end

function classifyTensorEigenvector(R,S)
    if R == 0.0
        return sym
    elseif R > 0
        if S > abs(R)
            return srp
        else
            return rrp
        end
    else
        if S > abs(R)
            return srn
        else
            return rrn
        end
    end
end

function main()
    println("hi")

    if length(ARGS) == 0
        folder = "../output/slice"
        saveName = "../test"
        dims = (101,101)
        scale = 1 # how many extra points do we add between the actual grid points (so the quadratic interp is more accurate).
    else
        try
            folder = ARGS[1]
            saveName = ARGS[2]
            dims = (parse(Int64,ARGS[3]),parse(Int64,ARGS[4]))
            scale = parse(Int64,ARGS[5])
        catch
            println("ERROR: Format is folder savename dims[1] dims[2] scale")
            exit()
        end
    end

    val_colors::Array{Tuple{UInt8,UInt8,UInt8}} = Array{Tuple{UInt8,UInt8,UInt8}}(undef, 5)
    vec_colors::Array{Tuple{UInt8,UInt8,UInt8}} = Array{Tuple{UInt8,UInt8,UInt8}}(undef, 5)
    cp_colors::Array{Tuple{UInt8,UInt8,UInt8}} = Array{Tuple{UInt8,UInt8,UInt8}}(undef, 2)

    cp_colors[trisector] = ( 255, 255, 255 )
    cp_colors[wedge] = ( 255, 140, 198 )

    # candidate colors:
    # lime green: 195, 211, 80
    # yellow: 248, 198, 48
    # pink: 255, 140, 198 (this is the winner)
    # pale green: 133, 255, 199
    # a sort of yellow: 255, 204, 51 

    val_colors[dp] = ( 224, 142, 69 )
    val_colors[rp] = ( 155, 39, 51 )
    val_colors[dn] = ( 201, 168, 245 )
    val_colors[rn] = ( 49, 59, 142 )
    val_colors[s] = ( 234, 234, 234 )

    vec_colors[rrp] = ( 155, 39, 51 )
    vec_colors[srp] = ( 193, 153, 116 )
    vec_colors[sym] = ( 173, 219, 240 )
    vec_colors[srn] = (133, 153, 188 )
    vec_colors[rrn] = (46, 59, 142 )

    a_x = scale*(dims[1]-1)+dims[1]
    a_y = scale*(dims[2]-1)+dims[2]

    # data arrays
    img_val = Array{UInt8}(undef, (3,a_x,a_y))
    img_vec = Array{UInt8}(undef, (3,a_x,a_y))
    frobenius = Array{Float64}(undef, (a_x, a_y))
    categorical_val = Array{Float64}(undef, (a_x, a_y))
    categorical_vec = Array{Float64}(undef, (a_x, a_y))

    cp_locs::Array{Tuple{Float64,Float64}} = Array{Tuple{Float64,Float64}}(undef, 0)
    cp_frobenius::Array{Float64} = Array{Float64}(undef,0)
    cp_types::Array{Int64} = Array{Int64}(undef,0)
    cp_categorical_val::Array{Float64} = Array{Float64}(undef,0)
    cp_categorical_vec::Array{Float64} = Array{Float64}(undef,0)


    a_file = open("$folder/A.raw", "r")
    b_file = open("$folder/B.raw", "r")
    c_file = open("$folder/C.raw", "r")
    d_file = open("$folder/D.raw", "r")

    a_array = reshape( reinterpret( Float64, read(a_file) ), dims )
    b_array = reshape( reinterpret( Float64, read(b_file) ), dims )
    c_array = reshape( reinterpret( Float64, read(c_file) ), dims )
    d_array = reshape( reinterpret( Float64, read(d_file) ), dims )

    close(a_file)
    close(b_file)
    close(c_file)
    close(d_file)



    for j in 1:dims[2]-1
        for i in 1:dims[1]-1

            a1 = a_array[i,j]
            a2 = a_array[i+1,j]
            a3 = a_array[i,j+1]
            a4 = a_array[i+1,j+1]

            b1 = b_array[i,j]
            b2 = b_array[i+1,j]
            b3 = b_array[i,j+1]
            b4 = b_array[i+1,j+1]

            c1 = c_array[i,j]
            c2 = c_array[i+1,j]
            c3 = c_array[i,j+1]
            c4 = c_array[i+1,j+1]

            d1 = d_array[i,j]
            d2 = d_array[i+1,j]
            d3 = d_array[i,j+1]
            d4 = d_array[i+1,j+1]

            # extract degenerate points.
            Δ1 = (a1-d1)/2
            Δ2 = (a2-d2)/2
            Δ3 = (a3-d3)/2
            Δ4 = (a4-d4)/2

            F1 = (b1+c1)/2
            F2 = (b2+c2)/2
            F3 = (b3+c3)/2
            F4 = (b4+c4)/2

            cp_type1 = 0
            s1 = Δ1*F2-Δ2*F1
            s2 = Δ2*F3-Δ3*F2
            s3 = Δ3*F1-Δ1*F3

            if s1 > 0 && s2 > 0 && s3 > 0
                cp_type1 = wedge
            elseif s1 < 0 && s2 < 0 && s3 < 0
                cp_type1 = trisector
            end

            # bottom cell
            if cp_type1 != 0
                Mat1 = [Δ1 Δ2 Δ3 ; F1 F2 F3 ; 1 1 1]
                if det(Mat1) != 0.0
                    sol1 = (Mat1^-1) * [0 ; 0 ; 1]
                    push!(cp_locs, ((i-1+sol1[2])*(scale+1), a_y - 1 - (j-1+sol1[3])*(scale+1)))

                    a = sol1[1]*a1 + sol1[2]*a2 + sol1[3]*a3
                    b = sol1[1]*b1 + sol1[2]*b2 + sol1[3]*b3
                    c = sol1[1]*c1 + sol1[2]*c2 + sol1[3]*c3
                    d = sol1[1]*d1 + sol1[2]*d2 + sol1[3]*d3

                    D,R,S = decomposeTensor(a,b,c,d)
                    class_val = classifyTensorEigenvalue(D,R,S)
                    class_vec = classifyTensorEigenvector(R,S)

                    frobenius_ = sqrt(a^2+b^2+c^2+d^2)
                    push!(cp_frobenius, frobenius_)
                    push!(cp_types, cp_type1)
                    push!(cp_categorical_val, class_val)
                    push!(cp_categorical_vec, class_vec)
                end
            end

            # top cell

            # we have to do the orders differently to maintain orientation
            cp_type2 = 0
            s1 = Δ3*F2-Δ2*F3
            s2 = Δ2*F4-Δ4*F2
            s3 = Δ4*F3-Δ3*F4

            if s1 > 0 && s2 > 0 && s3 > 0
                cp_type2 = wedge
            elseif s1 < 0 && s2 < 0 && s3 < 0
                cp_type2 = trisector
            end

            if cp_type2 != 0
                Mat2 = [Δ2 Δ3 Δ4 ; F2 F3 F4 ; 1 1 1]
                if det(Mat2) != 0.0
                    sol2 = (Mat2^-1) * [0 ; 0 ; 1]

                    # the math works out here. We want (i-1) + (1-sol2[2]) for x and similar for y.

                    push!( cp_locs,  (( i-1+1.0-sol2[2] )*(scale+1), a_y-1 - ( j-1+1.0-sol2[1] )*(scale+1)) )

                    a = sol2[1]*a2 + sol2[2]*a3 + sol2[3]*a4
                    b = sol2[1]*b2 + sol2[2]*b3 + sol2[3]*b4
                    c = sol2[1]*c2 + sol2[2]*c3 + sol2[3]*c4
                    d = sol2[1]*d2 + sol2[2]*d3 + sol2[3]*d4

                    D,R,S = decomposeTensor(a,b,c,d)
                    class_val = classifyTensorEigenvalue(D,R,S)
                    class_vec = classifyTensorEigenvector(R,S)

                    frobenius_ = sqrt(a^2+b^2+c^2+d^2)
                    push!(cp_frobenius, frobenius_)
                    push!(cp_types, cp_type2)
                    push!(cp_categorical_val, class_val)
                    push!(cp_categorical_vec, class_vec)
                end
            end

            for cell_y in 0:1+scale
                for cell_x in 0:1+scale

                    if cell_y == 1+scale && j != dims[2]-1
                        continue
                    end

                    if cell_x == 1+scale && i != dims[1]-1
                        continue
                    end

                    if cell_x + cell_y <= (scale + 1.0)
                        t2 = cell_x / (scale+1.0)
                        t3 = cell_y / (scale+1.0)
                        t4 = 0.0

                        t1 = 1.0-t2-t3
                    else
                        t1 = 0.0
                        t2 = 1.0 - cell_y / (scale + 1.0)
                        t3 = 1.0 - cell_x / (scale + 1.0)
                        t4 = 1.0 - t2 - t3
                    end

                    a = t1*a1 + t2*a2 + t3*a3 + t4*a4
                    b = t1*b1 + t2*b2 + t3*b3 + t4*b4
                    c = t1*c1 + t2*c2 + t3*c3 + t4*c4
                    d = t1*d1 + t2*d2 + t3*d3 + t4*d4

                    D,R,S = decomposeTensor(a,b,c,d)
                    val_class = classifyTensorEigenvalue(D,R,S)
                    vec_class = classifyTensorEigenvector(R,S)

                    px_x = (i-1)*(scale+1) + cell_x + 1
                    px_y = (j-1)*(scale+1) + cell_y + 1

                    frobenius[px_x,a_y-px_y+1] = sqrt(a^2+b^2+c^2+d^2)
                    img_val[:,px_x,a_y-px_y+1] .= val_colors[val_class]
                    img_vec[:,px_x,a_y-px_y+1] .= vec_colors[vec_class]
                    categorical_val[px_x,a_y-px_y+1] = val_class
                    categorical_vec[px_x,a_y-px_y+1] = vec_class

                end # end for cell_x
            end # end for cell_y
        end # end for i
    end # end for j

    vtk_grid(saveName, 0:1:a_x-1,0:1:a_y-1,0:1:0) do vtk
        vtk["frobenius"] = frobenius
        vtk["categorical val"] = categorical_val
        vtk["categorical vec"] = categorical_vec
        vtk["color val"] = img_val
        vtk["color vec"] = img_vec
    end

    cells::Vector{MeshCell} = []

    cp_mesh::Array{Float64} = Array{Float64}(undef, (3,length(cp_locs)))
    cp_mesh_colors::Array{UInt8} = Array{UInt8}(undef, (3,length(cp_locs)))
    for i in eachindex(cp_locs)
        cp_mesh[1,i] = cp_locs[i][1]
        cp_mesh[2,i] = cp_locs[i][2]
        cp_mesh[3,i] = 0.0

        cp_mesh_colors[1,i] = cp_colors[cp_types[i]][1]
        cp_mesh_colors[2,i] = cp_colors[cp_types[i]][2]
        cp_mesh_colors[3,i] = cp_colors[cp_types[i]][3]
    end

    vtk_grid(saveName,cp_mesh,cells) do vtk
        vtk["frobenius"] = cp_frobenius
        vtk["color criticalType"] = cp_mesh_colors
        vtk["categorical val"] = cp_categorical_val
        vtk["categorical vec"] = cp_categorical_vec
    end
    
end

main()