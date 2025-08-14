module tensorField

using LinearAlgebra
using StaticArrays

using ..utils
using ..cellTopology

export TF
export TF_Sym

export loadTFFromFolder
export loadTFFromFolderSym
export getTensor
export setTensor
export getTensorsAtCell
export getDegeneracyType
export getDegeneracyTypeFull
export decomposeTensor
export recomposeTensor
export edgesMatch
export classifyTensorEigenvector
export classifyTensorEigenvalue
export classifyEdge
export classifyEdgeOuter
export cellTopologyMatches

export saveTF32
export saveTF64
export getMinAndMax
export edgesMatchSplit
export duplicate
export inErrorBounds

struct TF
    entries::Array{Float64, 3}
    dims::Tuple{Int64, Int64}
end

struct TF_Sym
    entries::Array{Float64, 3}
    dims::Tuple{Int64, Int64}
end

function inReb(x,y,reb)
    if x == 0.0 || y == 0.0
        return true
    elseif abs(x) > abs(y)
        return abs(y)/abs(x) >= reb
    else
        return abs(x)/abs(y) >= reb
    end
end

function inErrorBounds(M1, M2, aeb, reb)
    return maximum(abs.(M1-M2)) <= aeb && inReb(M1[1,1],M2[1,1],reb) && inReb(M1[1,2],M2[1,2],reb) && inReb(M1[2,1],M2[2,1],reb) && inReb(M1[2,2],M2[2,2],reb)
end

function duplicate(tf::TF)
    return TF(deepcopy(tf.entries),tf.dims)
end

function loadTFFromFolder(folder::String, dims::Tuple{Int64, Int64}, slice::Int64 = 1, singlePrecision::Bool=false)

    num_entries = dims[1]*dims[2]
    entries::Array{Float64} = Array{Float64}(undef, (4,dims[1],dims[2]))

    for (i,f) in enumerate(("A","B","C","D"))
        if !isfile("$folder/$f.raw")
            println("TFZ: file $folder/$f.raw is missing.")
            println("please check your file specification.")
            exit(1)
        end

        byte_file = open("$folder/$f.raw","r")

        if singlePrecision
            seek(byte_file, 4 * num_entries * (slice-1))
            F = Float64.(reshape( reinterpret( Float32, read(byte_file, 4 * num_entries) ), dims ))
        else
            seek(byte_file, 8 * num_entries * (slice-1))
            F = reshape( reinterpret( Float64, read(byte_file, 8 * num_entries) ), dims )
        end

        entries[i,:,:] = F
    end

    tf = TF(entries,dims)

    return tf

end

function loadTFFromFolderSym(folder::String, dims::Tuple{Int64, Int64}, slice::Int64 = 1, singlePrecision::Bool = false)
    num_entries = dims[1]*dims[2]
    entries::Array{Float64} = Array{Float64}(undef, (3,dims[1],dims[2]))

    for (i,f) in enumerate(("A","B","D"))
        if !isfile("$folder/$f.raw")
            println("TFZ: file $folder/$f.raw is missing.")
            println("please check your file specification.")
            exit(1)
        end

        byte_file = open("$folder/$f.raw","r")

        if singlePrecision
            seek(byte_file, 4 * num_entries * (slice-1))
            F = Float64.(reshape( reinterpret( Float32, read(byte_file, 4 * num_entries) ), dims ))
        else
            seek(byte_file, 8 * num_entries * (slice-1))
            F = reshape( reinterpret( Float64, read(byte_file, 8 * num_entries) ), dims )
        end

        entries[i,:,:] = F
    end

    tf = TF_Sym(entries,dims)

    return tf
end

function getMinAndMax(tf::TF)
    min_ = tf.entries[1,1,1,1]
    max_ = tf.entries[1,1,1,1]

    for i in tf.entries
        if i < min_
            min_ = i
        elseif i > max_
            max_ = i
        end
    end

    return (min_,max_)
end

function getMinAndMax(tf::TF_Sym)
    min_ = tf.entries[1,1,1,1]
    max_ = tf.entries[1,1,1,1]

    for i in tf.entries
        if i < min_
            min_ = i
        elseif i > max_
            max_ = i
        end
    end

    return (min_,max_)
end

function saveTF64(folder::String, tf::TF, suffix::String="")
    saveArray64("$folder/A$suffix.raw", tf.entries[1,:,:,:])
    saveArray64("$folder/B$suffix.raw", tf.entries[2,:,:,:])
    saveArray64("$folder/C$suffix.raw", tf.entries[3,:,:,:])
    saveArray64("$folder/D$suffix.raw", tf.entries[4,:,:,:])
end

function saveTF64(folder::String, tf::TF_Sym, suffix::String="")
    saveArray64("$folder/A$suffix.raw", tf.entries[1,:,:,:])
    saveArray64("$folder/B$suffix.raw", tf.entries[2,:,:,:])
    saveArray64("$folder/D$suffix.raw", tf.entries[3,:,:,:])
end

function saveTF32(folder::String, tf::TF, suffix::String="")
    saveArray32("$folder/A$suffix.raw", tf.entries[1,:,:,:])
    saveArray32("$folder/B$suffix.raw", tf.entries[2,:,:,:])
    saveArray32("$folder/C$suffix.raw", tf.entries[3,:,:,:])
    saveArray32("$folder/D$suffix.raw", tf.entries[4,:,:,:])
end

function saveTF32(folder::String, tf::TF_Sym, suffix::String="")
    saveArray32("$folder/A$suffix.raw", tf.entries[1,:,:,:])
    saveArray32("$folder/B$suffix.raw", tf.entries[2,:,:,:])
    saveArray32("$folder/D$suffix.raw", tf.entries[3,:,:,:])
end

function getTensor(tf::TF, x::Int64, y::Int64)
    return SMatrix{2,2,Float64}( tf.entries[1,x,y], tf.entries[3,x,y], tf.entries[2,x,y], tf.entries[4,x,y] )
end

function getTensor(tf::TF_Sym, x::Int64, y::Int64)
    return SVector{3,Float64}( tf.entries[1,x,y], tf.entries[2,x,y], tf.entries[3,x,y] )
end

function setTensor(tf::TF, x, y, tensor::FloatMatrix)
    tf.entries[1,x,y] = tensor[1,1]
    tf.entries[2,x,y] = tensor[1,2]
    tf.entries[3,x,y] = tensor[2,1]
    tf.entries[4,x,y] = tensor[2,2]
end

function setTensor(tf::TF_Sym, x, y, tensor::FloatMatrixSymmetric)
    tf.entries[1,x,y] = tensor[1]
    tf.entries[2,x,y] = tensor[2]
    tf.entries[3,x,y] = tensor[3]
end

# returns in counterclockwise orientation, consistent with getCellVertexCoords
function getTensorsAtCell(tf::TF, x::Int64, y::Int64, top::Bool)
    points = getCellVertexCoords(x,y,top)
    return ( getTensor(tf, points[1]...), getTensor(tf, points[2]...), getTensor(tf, points[3]...) )
end

function getDegeneracyType( tf::TF_Sym, x::Int64, y::Int64, top::Bool )
    points = getCellVertexCoords(x,y,top)
    tensor1 = getTensor(tf, points[1]...)
    tensor2 = getTensor(tf, points[2]...)
    tensor3 = getTensor(tf, points[3]...)

    D1_11 = tensor1[1] - tensor1[3]
    D1_21 = 2*tensor1[2]

    D2_11 = tensor2[1] - tensor2[3]
    D2_21 = 2*tensor2[2]

    D3_11 = tensor3[1] - tensor3[3]
    D3_21 = 2*tensor3[2]

    sign1 = sign(D1_11*D2_21 - D2_11*D1_21)
    sign2 = sign(D2_11*D3_21 - D3_11*D2_21)
    sign3 = sign(D3_11*D1_21 - D1_11*D3_21)

    if sign1 == 0 || sign2 == 0 || sign3 == 0
        # if we get three zeros, then give an answer according to the number of zeros.

        numZeroSign = 0
        numZeroMatrix = 0

        if isClose(D1_11, 0.0) && isClose(D1_21, 0.0)
            if isClose(tensor1[1],0.0) && isClose(tensor1[2],0.0)
                numZeroMatrix += 1
            else
                return CP_OTHER
            end

            numZeroSign += 2
        else
            if sign1 == 0
                numZeroSign += 1
            end

            if sign3 == 0
                numZeroSign += 1
            end
        end

        if isClose(D2_11,0.0) && isClose(D2_21, 0.0)
            if isClose(tensor2[1],0.0) && isClose(tensor2[2],0.0)
                numZeroMatrix += 1
            else
                return CP_OTHER
            end

            numZeroSign += 1
        elseif sign2 == 0
            numZeroSign += 1
        end

        if isClose(D3_11,0.0) && isClose(D3_21,0.0)
            if isClose(tensor3[1],0.0) && isClose(tensor3[2],0.0)
                numZeroMatrix += 1
            else
                return CP_OTHER
            end
        end

        if (numZeroMatrix == 1 && numZeroSign == 2)
            return CP_ZERO_CORNER
        elseif numZeroMatrix == 2
            return CP_ZERO_EDGE
        elseif numZeroMatrix == 3
            return CP_ZERO_FULL
        else
            return CP_OTHER
        end
    end

    if sign1 == sign2
        if sign3 == sign1
            if sign3 == 1
                return CP_WEDGE
            else
                return CP_TRISECTOR
            end
        else
            return CP_NORMAL
        end
    else
        return CP_NORMAL
    end

end

function getDegeneracyTypeFull( tf::TF_Sym, x::Int64, y::Int64, top::Bool )
    points = getCellVertexCoords(x,y,top)
    tensor1 = getTensor(tf, points[1]...)
    tensor2 = getTensor(tf, points[2]...)
    tensor3 = getTensor(tf, points[3]...)

    D1_11 = tensor1[1] - tensor1[3]
    D1_21 = 2*tensor1[2]

    D2_11 = tensor2[1] - tensor2[3]
    D2_21 = 2*tensor2[2]

    D3_11 = tensor3[1] - tensor3[3]
    D3_21 = 2*tensor3[2]

    sign1 = sign(D1_11*D2_21 - D2_11*D1_21)
    sign2 = sign(D2_11*D3_21 - D3_11*D2_21)
    sign3 = sign(D3_11*D1_21 - D1_11*D3_21)

    if sign1 == 0 || sign2 == 0 || sign3 == 0
        # ope this had to be totally redone :(

        numZeroDeviators = 0
        mat1Zero = false
        mat2Zero = false
        mat3Zero = false

        if isClose(D1_11, 0.0) && isClose(D1_21, 0.0)
            numZeroDeviators += 1
            mat1Zero = true
        end

        if isClose(D2_11,0.0) && isClose(D2_21,0.0)
            numZeroDeviators += 1
            mat2Zero = true
        end

        if isClose(D3_11,0.0) && isClose(D3_21,0.0)
            numZeroDeviators += 1
            mat3Zero = true
        end

        if numZeroDeviators == 3
            return CP_FULL
        elseif numZeroDeviators == 2
            return CP_EDGE
        elseif numZeroDeviators == 1
            if mat1Zero
                if sign2 == 0.0 && sign(D2_11*D3_11) < 0.0
                    return CP_LINE
                else
                    return CP_CORNER
                end
            elseif mat2Zero
                if sign3 == 0.0 && sign(D1_11*D3_11) < 0.0
                    return CP_LINE
                else
                    return CP_CORNER
                end
            elseif mat3Zero
                if sign1 == 0.0 && sign(D1_11*D2_11) < 0.0
                    return CP_LINE
                else
                    return CP_CORNER
                end
            end
        end

        # this is the "else" where none of the corners vanish...
        numEdgeZeros = 0
        if sign2 == 0.0 && sign(D2_11*D3_11) < 0.0
            numEdgeZeros += 1
        end

        if sign3 == 0.0 && sign(D1_11*D3_11) < 0.0
            numEdgeZeros += 1
        end

        if sign1 == 0.0 && sign(D1_11*D2_11) < 0.0
            numEdgeZeros += 1
        end

        if numEdgeZeros == 0
            return CP_NORMAL
        elseif numEdgeZeros == 1
            return CP_EDGE
        else
            return CP_LINE
        end

    end

    if sign1 == sign2
        if sign3 == sign1
            if sign3 == 1
                return CP_WEDGE
            else
                return CP_TRISECTOR
            end
        else
            return CP_NORMAL
        end
    else
        return CP_NORMAL
    end

end

function classifyCellEigenvalue( tf::TF, x::Int64, y::Int64, top::Bool, eigenvector::Bool, verbose::Bool = false)
    if top
        return cellTopology.classifyCellEigenvalue( getTensor(tf, x, y+1), getTensor(tf, x+1, y), getTensor(tf, x+1, y+1), eigenvector, verbose )
    else
        return cellTopology.classifyCellEigenvalue( getTensor(tf, x, y), getTensor(tf, x+1, y), getTensor(tf, x, y+1), eigenvector, verbose )
    end
end

function classifyCellEigenvector( tf::TF, x::Int64, y::Int64, top::Bool )
    if top
        return cellTopology.classifyCellEigenvector( getTensor(tf, x, y+1), getTensor(tf, x+1, y), getTensor(tf, x+1, y+1) )
    else
        return cellTopology.classifyCellEigenvector( getTensor(tf, x, y), getTensor(tf, x+1, y), getTensor(tf, x, y+1) )
    end
end

function getDegeneracyType( tf::TF, x::Int64, y::Int64, top::Bool )
    return getDegeneracyType( getTensorsAtCell( tf, x, y, top )... )
end

function getDegeneracyTypeFull( tf::TF, x::Int64, y::Int64, top::Bool )
    return getDegeneracyTypeFull( getTensorsAtCell( tf, x, y, top )... )
end

# we assume that the tensors are in counterclockwise direction
function getDegeneracyType(tensor1::FloatMatrix, tensor2::FloatMatrix, tensor3::FloatMatrix, verbose=false)
    # rather than explicitly computing the deviator it is faster to do it this way.
    # yes I know the readability kind of sucks but it makes kind of a huge difference.
    D1_11 = tensor1[1,1] - tensor1[2,2]
    D1_21 = tensor1[2,1] + tensor1[1,2]

    D2_11 = tensor2[1,1] - tensor2[2,2]
    D2_21 = tensor2[2,1] + tensor2[1,2]

    D3_11 = tensor3[1,1] - tensor3[2,2]
    D3_21 = tensor3[2,1] + tensor3[1,2]

    sign1 = sign(D1_11*D2_21 - D2_11*D1_21)
    sign2 = sign(D2_11*D3_21 - D3_11*D2_21)
    sign3 = sign(D3_11*D1_21 - D1_11*D3_21)

    if sign1 == 0 || sign2 == 0 || sign3 == 0
        # if we get three zeros, then give an answer according to the number of zeros.

        numZeroSign = 0
        numZeroMatrix = 0

        if isClose(D1_11, 0.0) && isClose(D1_21, 0.0)
            if isClose(tensor1[1,1],0.0) && isClose(tensor1[1,2],0.0)
                numZeroMatrix += 1
            else
                return CP_OTHER
            end

            numZeroSign += 2
        else
            if sign1 == 0
                numZeroSign += 1
            end

            if sign3 == 0
                numZeroSign += 1
            end
        end

        if isClose(D2_11,0.0) && isClose(D2_21, 0.0)
            if isClose(tensor2[1,1],0.0) && isClose(tensor2[1,2],0.0)
                numZeroMatrix += 1
            else
                return CP_OTHER
            end

            numZeroSign += 1
        elseif sign2 == 0
            numZeroSign += 1
        end

        if isClose(D3_11,0.0) && isClose(D3_21,0.0)
            if isClose(tensor3[1,1],0.0) && isClose(tensor3[1,2],0.0)
                numZeroMatrix += 1
            else
                return CP_OTHER
            end
        end

        if (numZeroMatrix == 1 && numZeroSign == 2)
            return CP_ZERO_CORNER
        elseif numZeroMatrix == 2
            return CP_ZERO_EDGE
        elseif numZeroMatrix == 3
            return CP_ZERO_FULL
        else
            return CP_OTHER
        end
    end

    if sign1 == sign2
        if sign3 == sign1
            setup = SMatrix{3,3,Float64}(( D1_11, D1_21, 1, D2_11, D2_21, 1, D3_11, D3_21, 1 ))
            targets = SMatrix{3,1,Float64}(( 0, 0, 1 ))
            mu = (setup^-1) * targets
            R0 = mu[1] * (tensor1[2,1] - tensor1[1,2]) + mu[2] * (tensor2[2,1] - tensor2[1,2]) + mu[3] * (tensor3[2,1] - tensor3[1,2])

            if sign3 == 1
                if R0 > 0
                    return CP_WEDGE_RP
                else
                    return CP_WEDGE_RN
                end
            else
                if R0 > 0
                    return CP_TRISECTOR_RP
                else
                    return CP_WEDGE_RN
                end
            end
        else
            return CP_NORMAL
        end
    else
        return CP_NORMAL
    end
end

# Classifies the circular point type based on its full class (normal/trisector/corner/onEdge/fullEdge/line/full)
function getDegeneracyTypeFull(tensor1::FloatMatrix, tensor2::FloatMatrix, tensor3::FloatMatrix)
    # rather than explicitly computing the deviator it is faster to do it this way.
    # yes I know the readability kind of sucks but it makes kind of a huge difference.
    D1_11 = tensor1[1,1] - tensor1[2,2]
    D1_21 = tensor1[2,1] + tensor1[1,2]

    D2_11 = tensor2[1,1] - tensor2[2,2]
    D2_21 = tensor2[2,1] + tensor2[1,2]

    D3_11 = tensor3[1,1] - tensor3[2,2]
    D3_21 = tensor3[2,1] + tensor3[1,2]

    sign1 = sign(D1_11*D2_21 - D2_11*D1_21)
    sign2 = sign(D2_11*D3_21 - D3_11*D2_21)
    sign3 = sign(D3_11*D1_21 - D1_11*D3_21)

    if sign1 == 0 || sign2 == 0 || sign3 == 0
        numZeroDeviators = 0
        mat1Zero = false
        mat2Zero = false
        mat3Zero = false

        if isClose(D1_11, 0.0) && isClose(D1_21, 0.0)
            numZeroDeviators += 1
            mat1Zero = true
        end

        if isClose(D2_11,0.0) && isClose(D2_21,0.0)
            numZeroDeviators += 1
            mat2Zero = true
        end

        if isClose(D3_11,0.0) && isClose(D3_21,0.0)
            numZeroDeviators += 1
            mat3Zero = true
        end

        if numZeroDeviators == 3
            return CP_FULL
        elseif numZeroDeviators == 2
            return CP_EDGE
        elseif numZeroDeviators == 1
            if mat1Zero
                if sign2 == 0.0 && sign(D2_11*D3_11) < 0.0
                    return CP_LINE
                else
                    return CP_CORNER
                end
            elseif mat2Zero
                if sign3 == 0.0 && sign(D1_11*D3_11) < 0.0
                    return CP_LINE
                else
                    return CP_CORNER
                end
            elseif mat3Zero
                if sign1 == 0.0 && sign(D1_11*D2_11) < 0.0
                    return CP_LINE
                else
                    return CP_CORNER
                end
            end
        end

        # this is the "else" where none of the corners vanish...
        numEdgeZeros = 0
        if sign2 == 0.0 && sign(D2_11*D3_11) < 0.0
            numEdgeZeros += 1
        end

        if sign3 == 0.0 && sign(D1_11*D3_11) < 0.0
            numEdgeZeros += 1
        end

        if sign1 == 0.0 && sign(D1_11*D2_11) < 0.0
            numEdgeZeros += 1
        end

        if numEdgeZeros == 0
            return CP_NORMAL
        elseif numEdgeZeros == 1
            return CP_EDGE
        else
            return CP_LINE
        end
    end

    if sign1 == sign2
        if sign3 == sign1

            setup = SMatrix{3,3,Float64}(( D1_11, D1_21, 1, D2_11, D2_21, 1, D3_11, D3_21, 1 ))
            targets = SMatrix{3,1,Float64}(( 0, 0, 1 ))
            mu = (setup^-1) * targets
            R0 = mu[1] * (tensor1[2,1] - tensor1[1,2]) + mu[2] * (tensor2[2,1] - tensor2[1,2]) + mu[3] * (tensor3[2,1] - tensor3[1,2])

            if sign3 == 1
                if R0 > 0
                    return CP_WEDGE_RP
                else
                    return CP_WEDGE_RN
                end
            else
                if R0 > 0
                    return CP_TRISECTOR
                else
                    return CP_WEDGE_RN
                end
            end
        else
            return CP_NORMAL
        end
    else
        return CP_NORMAL
    end
end

function decomposeTensor(tensor::FloatMatrix)

    y_d::Float64 = (tensor[1,1] + tensor[2,2])/2
    y_r::Float64 = (tensor[2,1] - tensor[1,2])/2
    # tensor -= [ y_d -y_r ; y_r y_d ]

    # cplx = tensor[1,1] + tensor[1,2]*im
    cplx = (tensor[1,1] - y_d) + (tensor[1,2]+y_r)*im
    y_s::Float64 = abs(cplx)
    θ::Float64 = angle(cplx)

    return (y_d, y_r, y_s, θ)
end

function recomposeTensor(y_d::AbstractFloat, y_r::AbstractFloat, y_s::AbstractFloat, θ::AbstractFloat)

    cos_ = cos(θ)
    sin_ = sin(θ)

    return SMatrix{2,2,Float64}( y_d+y_s*cos_,  y_r+y_s*sin_,  -y_r+y_s*sin_,  y_d-y_s*cos_ )

end

function decomposeTensor(T::FloatMatrixSymmetric)
    trace = (T[1] + T[3]) / 2
    cplx = (T[1] - T[3])/2 + T[2]*im
    r = abs(cplx)
    θ = angle(cplx)
    return (trace,r,θ)
end

function recomposeTensor(trace,r,θ)
    return SVector{3,Float64}( trace + r*cos(θ), r*sin(θ), trace - r*cos(θ) )
end

function classifyTensorEigenvector(yr::AbstractFloat, ys::AbstractFloat)
    if yr == 0.0
        if ys == 0.0
            return Z
        else
            return SYM
        end
    elseif yr > 0
        if isRelativelyClose(yr,ys)
            return DegenRP
        elseif ys > yr
            return SRP
        else
            return RRP
        end
    else
        if isRelativelyClose(-yr,ys)
            return DegenRN
        elseif isRelativelyGreater(ys,-yr)
            return SRN
        else
            return RRN
        end
    end
end

function classifyTensorEigenvector(tensor::FloatMatrix)
    _, yr, ys, _ = decomposeTensor(tensor)

    return classifyTensorEigenvector(yr, ys)
end

function classifyTensorEigenvalue(tensor::FloatMatrix)
    yd, yr, ys, _ = decomposeTensor(tensor)
    
    return classifyTensorEigenvalue(yd,yr,ys)
end

function classifyTensorEigenvalue(yd::AbstractFloat, yr::AbstractFloat, ys::AbstractFloat)

    if isRelativelyClose(abs(yd),abs(yr))
        if isRelativelyClose(abs(yd),ys)
            if yd == 0.0
                return Z
            elseif yd > 0
                if yr > 0
                    return DPRPS
                else
                    return DPRNS
                end
            else
                if yr > 0
                    return DNRPS
                else
                    return DNRNS
                end
            end
        elseif ys > abs(yr)
            return S
        else
            if yd > 0
                if yr > 0
                    return DPRP
                else
                    return DPRN
                end
            else
                if yr > 0
                    return DNRP
                else
                    return DNRN
                end
            end
        end
    elseif abs(yr) > abs(yd)
        if isRelativelyClose(abs(yr),ys)
            if yr > 0
                return RPS
            else
                return RNS
            end
        elseif abs(yr) > ys
            if yr > 0
                return RP
            else
                return RN
            end
        else
            return S
        end
    else
        if isRelativelyClose(abs(yd),ys)
            if yd > 0
                return DPS
            else
                return DNS
            end
        elseif abs(yd) > ys
            if yd > 0
                return DP
            else
                return DN
            end
        else
            return S
        end
    end

end

end