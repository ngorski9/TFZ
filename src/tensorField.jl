module tensorField

using LinearAlgebra
using StaticArrays

using ..utils
using ..cellTopology

export TensorField2d
export TensorField2dSymmetric

export loadTensorField2dFromFolder
export loadTensorField2dSymmetricFromFolder
export getTensor
export setTensor
export getTensorsAtCell
export getCircularPointType
export getCircularPointTypeFull
export decomposeTensor
export recomposeTensor
export decomposeTensorSymmetric
export edgesMatch
export recomposeTensorSymmetric
export classifyTensorEigenvector
export classifyTensorEigenvalue
export classifyEdge
export classifyEdgeOuter
export cellTopologyMatches

export saveTensorField32
export saveTensorField64
export saveTensorFieldSymmetric32
export saveTensorFieldSymmetric64
export getMinAndMax
export getCriticalType
export getCriticalTypeFull
export edgesMatchSplit
export duplicateTensorField2d
export inErrorBounds

struct TensorField2d
    entries::Array{Float64}
    dims::Tuple{Int64, Int64, Int64}
end

struct TensorField2dSymmetric
    entries::Array{Float64}
    dims::Tuple{Int64, Int64, Int64}
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

function duplicateTensorField2d(tf::TensorField2d)
    return TensorField2d(deepcopy(tf.entries),tf.dims)
end

function loadTensorField2dFromFolder(folder::String, dims::Tuple{Int64, Int64, Int64})

    num_entries = dims[1]*dims[2]*dims[3]
    num_bytes = filesize("$folder/row_1_col_1.dat")/num_entries

    if num_bytes == 8
        A_byte_file = open("$folder/row_1_col_1.dat", "r")
        A = reshape( reinterpret( Float64, read(A_byte_file) ), dims )
        close(A_byte_file)
    
        B_byte_file = open("$folder/row_1_col_2.dat", "r")
        B = reshape( reinterpret( Float64, read(B_byte_file) ), dims )
        close(B_byte_file)
    
        C_byte_file = open("$folder/row_2_col_1.dat", "r")
        C = reshape( reinterpret( Float64, read(C_byte_file) ), dims )
        close(C_byte_file)
    
        D_byte_file = open("$folder/row_2_col_2.dat", "r")
        D = reshape( reinterpret( Float64, read(D_byte_file) ), dims )
        close(D_byte_file)
    elseif num_bytes == 4
        println("loading from 32 bits to 64")
        A_byte_file = open("$folder/row_1_col_1.dat", "r")
        A = Array{Float64}(reshape( reinterpret( Float32, read(A_byte_file) ), dims ))
        close(A_byte_file)
    
        B_byte_file = open("$folder/row_1_col_2.dat", "r")
        B = Array{Float64}(reshape( reinterpret( Float32, read(B_byte_file) ), dims ))
        close(B_byte_file)
    
        C_byte_file = open("$folder/row_2_col_1.dat", "r")
        C = Array{Float64}(reshape( reinterpret( Float32, read(C_byte_file) ), dims ))
        close(C_byte_file)
    
        D_byte_file = open("$folder/row_2_col_2.dat", "r")
        D = Array{Float64}(reshape( reinterpret( Float32, read(D_byte_file) ), dims ))
        close(D_byte_file)
    else
        println("The file that you specified is $num_bytes bytes per point")
        println("only 32 and 64 bits are currently supported")        
        exit(1)
    end

    entries::Array{Float64} = Array{Float64}(undef, (4,dims[1],dims[2],dims[3]))
    for k in 1:dims[3]
        for j in 1:dims[2]
            for i in 1:dims[1]
                entries[1,i,j,k] = A[i,j,k]
                entries[2,i,j,k] = B[i,j,k]
                entries[3,i,j,k] = C[i,j,k]
                entries[4,i,j,k] = D[i,j,k]
            end
        end
    end

    tf = TensorField2d(entries,dims)

    return tf

end

function loadTensorField2dSymmetricFromFolder(folder::String, dims::Tuple{Int64, Int64, Int64})

    num_entries = dims[1]*dims[2]*dims[3]
    num_bytes = filesize("$folder/row_1_col_1.dat")/num_entries

    if num_bytes == 8
        A_byte_file = open("$folder/row_1_col_1.dat", "r")
        A = reshape( reinterpret( Float64, read(A_byte_file) ), dims )
        close(A_byte_file)
    
        B_byte_file = open("$folder/row_1_col_2.dat", "r")
        B = reshape( reinterpret( Float64, read(B_byte_file) ), dims )
        close(B_byte_file)
    
        D_byte_file = open("$folder/row_2_col_2.dat", "r")
        D = reshape( reinterpret( Float64, read(D_byte_file) ), dims )
        close(D_byte_file)
    elseif num_bytes == 4
        println("loading from 32 bits to 64")
        A_byte_file = open("$folder/row_1_col_1.dat", "r")
        A = Array{Float64}(reshape( reinterpret( Float32, read(A_byte_file) ), dims ))
        close(A_byte_file)
    
        B_byte_file = open("$folder/row_1_col_2.dat", "r")
        B = Array{Float64}(reshape( reinterpret( Float32, read(B_byte_file) ), dims ))
        close(B_byte_file)
    
        D_byte_file = open("$folder/row_2_col_2.dat", "r")
        D = Array{Float64}(reshape( reinterpret( Float32, read(D_byte_file) ), dims ))
        close(D_byte_file)
    else
        println("The file that you specified is $num_bytes bytes per point")
        println("only 32 and 64 bits are currently supported")        
        exit(1)
    end

    entries::Array{Float64} = Array{Float64}(undef, (3,dims[1],dims[2],dims[3]))
    for k in 1:dims[3]
        for j in 1:dims[2]
            for i in 1:dims[1]
                entries[1,i,j,k] = A[i,j,k]
                entries[2,i,j,k] = B[i,j,k]
                entries[3,i,j,k] = D[i,j,k]
            end
        end
    end

    tf = TensorField2dSymmetric(entries,dims)

    return tf

end

function getMinAndMax(tf::TensorField2d)
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

function getMinAndMax(tf::TensorField2dSymmetric)
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

function saveTensorField64(folder::String, tf::TensorField2d, suffix::String="")
    saveArray64("$folder/row_1_col_1$suffix.dat", tf.entries[1,:,:,:])
    saveArray64("$folder/row_1_col_2$suffix.dat", tf.entries[2,:,:,:])
    saveArray64("$folder/row_2_col_1$suffix.dat", tf.entries[3,:,:,:])
    saveArray64("$folder/row_2_col_2$suffix.dat", tf.entries[4,:,:,:])
end

function saveTensorFieldSymmetric64(folder::String, tf::TensorField2dSymmetric, suffix::String="")
    saveArray64("$folder/row_1_col_1$suffix.dat", tf.entries[1,:,:,:])
    saveArray64("$folder/row_1_col_2$suffix.dat", tf.entries[2,:,:,:])
    saveArray64("$folder/row_2_col_2$suffix.dat", tf.entries[3,:,:,:])
end

function saveTensorField32(folder::String, tf::TensorField2d, suffix::String="")
    saveArray32("$folder/row_1_col_1$suffix.dat", tf.entries[1,:,:,:])
    saveArray32("$folder/row_1_col_2$suffix.dat", tf.entries[2,:,:,:])
    saveArray32("$folder/row_2_col_1$suffix.dat", tf.entries[3,:,:,:])
    saveArray32("$folder/row_2_col_2$suffix.dat", tf.entries[4,:,:,:])
end

function saveTensorFieldSymmetric32(folder::String, tf::TensorField2dSymmetric, suffix::String="")
    saveArray32("$folder/row_1_col_1$suffix.dat", tf.entries[1,:,:,:])
    saveArray32("$folder/row_1_col_2$suffix.dat", tf.entries[2,:,:,:])
    saveArray32("$folder/row_2_col_2$suffix.dat", tf.entries[3,:,:,:])
end

function getTensor(tf::TensorField2d, x::Int64, y::Int64, t::Int64)
    return SMatrix{2,2,Float64}( tf.entries[1,x,y,t], tf.entries[3,x,y,t], tf.entries[2,x,y,t], tf.entries[4,x,y,t] )
end

function getTensor(tf::TensorField2dSymmetric, x::Int64, y::Int64, t::Int64)
    return SVector{3,Float64}( tf.entries[1,x,y,t], tf.entries[2,x,y,t], tf.entries[3,x,y,t] )
end

function setTensor(tf::TensorField2d, x, y, t, tensor::FloatMatrix)
    tf.entries[1,x,y,t] = tensor[1,1]
    tf.entries[2,x,y,t] = tensor[1,2]
    tf.entries[3,x,y,t] = tensor[2,1]
    tf.entries[4,x,y,t] = tensor[2,2]
end

function setTensor(tf::TensorField2dSymmetric, x, y, t, tensor::FloatMatrixSymmetric)
    tf.entries[1,x,y,t] = tensor[1]
    tf.entries[2,x,y,t] = tensor[2]
    tf.entries[3,x,y,t] = tensor[3]
end

# returns in counterclockwise orientation, consistent with getCellVertexCoords
function getTensorsAtCell(tf::TensorField2d, x::Int64, y::Int64, t::Int64, top::Bool)
    points = getCellVertexCoords(x,y,t,top)
    return ( getTensor(tf, points[1]...), getTensor(tf, points[2]...), getTensor(tf, points[3]...) )
end

function getCriticalType( tf::TensorField2dSymmetric, x::Int64, y::Int64, t::Int64, top::Bool )
    points = getCellVertexCoords(x,y,t,top)
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

function getCriticalTypeFull( tf::TensorField2dSymmetric, x::Int64, y::Int64, t::Int64, top::Bool )
    points = getCellVertexCoords(x,y,t,top)
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

function classifyCellEigenvalue( tf::TensorField2d, x::Int64, y::Int64, t::Int64, top::Bool, eigenvector::Bool, verbose::Bool = false)
    if top
        return cellTopology.classifyCellEigenvalue( getTensor(tf, x, y+1, t), getTensor(tf, x+1, y, t), getTensor(tf, x+1, y+1, t), eigenvector, verbose )
    else
        return cellTopology.classifyCellEigenvalue( getTensor(tf, x, y, t), getTensor(tf, x+1, y, t), getTensor(tf, x, y+1, t), eigenvector, verbose )
    end
end

function classifyCellEigenvalueOld( tf::TensorField2d, x::Int64, y::Int64, t::Int64, top::Bool, eigenvector::Bool, verbose::Bool = false)
    if top
        return cellTopologyOld.classifyCellEigenvalue( getTensor(tf, x, y+1, t), getTensor(tf, x+1, y, t), getTensor(tf, x+1, y+1, t), eigenvector, verbose )
    else
        return cellTopologyOld.classifyCellEigenvalue( getTensor(tf, x, y, t), getTensor(tf, x+1, y, t), getTensor(tf, x, y+1, t), eigenvector, verbose )
    end
end

function classifyCellEigenvector( tf::TensorField2d, x::Int64, y::Int64, t::Int64, top::Bool )
    if top
        return cellTopology.classifyCellEigenvector( getTensor(tf, x, y+1, t), getTensor(tf, x+1, y, t), getTensor(tf, x+1, y+1, t) )
    else
        return cellTopology.classifyCellEigenvector( getTensor(tf, x, y, t), getTensor(tf, x+1, y, t), getTensor(tf, x, y+1, t) )
    end
end

function getCircularPointType( tf::TensorField2d, x::Int64, y::Int64, t::Int64, top::Bool )
    return getCircularPointType( getTensorsAtCell( tf, x, y, t, top )... )
end

function getCircularPointTypeFull( tf::TensorField2d, x::Int64, y::Int64, t::Int64, top::Bool )
    return getCircularPointTypeFull( getTensorsAtCell( tf, x, y, t, top )... )
end

function symmetricDeviator(tensor::FloatMatrix)
    diagonal = 0.5*tr(tensor)
    off_diagonal = 0.5*(tensor[2,1]-tensor[1,2])
    return SMatrix{2,2,Float64}( tensor[1,1] - diagonal, tensor[2,1] - off_diagonal, tensor[1,2] + off_diagonal, tensor[2,2] - diagonal )
    # return tensor - 0.5*tr(tensor)*I + 0.5*(tensor[2,1]-tensor[1,2])*[ 0 1 ; -1 0 ]
end

# we assume that the tensors are in counterclockwise direction
function getCircularPointType(tensor1::FloatMatrix, tensor2::FloatMatrix, tensor3::FloatMatrix, verbose=false)
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
function getCircularPointTypeFull(tensor1::FloatMatrix, tensor2::FloatMatrix, tensor3::FloatMatrix)
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

function decomposeTensorSymmetric(T::FloatMatrixSymmetric)
    trace = (T[1] + T[3]) / 2
    cplx = (T[1] - T[3])/2 + T[2]*im
    r = abs(cplx)
    θ = angle(cplx)
    return (trace,r,θ)
end

function recomposeTensorSymmetric(trace,r,θ)
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

function edgesMatch( t11::FloatMatrix, t21::FloatMatrix, t12::FloatMatrix, t22::FloatMatrix, eb::Float64, eigenvalue::Bool, eigenvector::Bool, show::Bool=false, minCrossing::Float64 = 0.01 )
    if t11 == t12 && t21 == t22
        return true
    end

    evalClass1, evalLoc1, evecClass1, evecLoc1 = classifyEdgeOuter( t11, t21, show, minCrossing )

    if show
        println((evalClass1,evalLoc1,evecClass1,evecLoc1))
        println("**********************************************")        
    end

    evalClass2, evalLoc2, evecClass2, evecLoc2 = classifyEdgeOuter( t12, t22, show, minCrossing )

    if show
        println((evalClass2,evalLoc2,evecClass2,evecLoc2))
        println("=========================")
    end

    if length(evalClass1) > 0 && evalClass1[1] == 99 || length(evalClass2) > 0 && evalClass2[1] == 99
        return false
    end

    return (!eigenvalue || evalClass1 == evalClass2) && (!eigenvector || evecClass1 == evecClass2) && (!eigenvalue || length(evalClass1) == 0 || maximum( abs.(evalLoc1-evalLoc2) ) <= eb) && (!eigenvector || evecClass1 == 0 || maximum( abs.(evecLoc1-evecLoc2) ) <= eb )
end

# returns two bools for whether it matches for eigenvalue / eigenvector
function edgesMatchSplit( t11::FloatMatrix, t21::FloatMatrix, t12::FloatMatrix, t22::FloatMatrix, eb::Float64, minCrossing = 0.01 )

    if t11 == t12 && t21 == t22
        return [true,true]
    end

    result = [false,false]

    evalClass1, evalLoc1, evecClass1, evecLoc1 = classifyEdgeOuter( t11, t21, false, minCrossing )
    evalClass2, evalLoc2, evecClass2, evecLoc2 = classifyEdgeOuter( t12, t22, false, minCrossing )

    if length(evalClass1) > 0 && evalClass1[1] == 99 && length(evalClass2) > 0 && evalClass2[1] == 99
        println("this occurred")
        result[1] = true
    else
        result[1] = (evalClass1 == evalClass2) && (length(evalClass1) == 0 || maximum( abs.(evalLoc1-evalLoc2) ) <= eb)
        # if !result[1]
        #     if evalClass1 == evalClass2
        #         println((evalClass1, evalClass2, maximum(abs.(evalLoc1-evalLoc2))))
        #         println(length(evalClass1) == 0)
        #         println(eb)
        #         println(maximum( abs.(evalLoc1-evalLoc2) ) <= eb)
        #     else
        #         println((evalClass1, evalClass2))
        #     end
        # end
    end

    result[2] = (evecClass1 == evecClass2) && (evecClass1 == 0 || maximum( abs.(evecLoc1-evecLoc2) ) <= eb)

    return result
end

function classifyEdgeOuter( t1::FloatMatrix, t2::FloatMatrix, p=false, minCrossing = 0.01 )
    decomp1 = decomposeTensor(t1)
    decomp2 = decomposeTensor(t2)
    return classifyEdge(decomp1..., decomp2...,p, minCrossing)
end

# returns a list of numbers with the following entries (ordered)
# as one moves from t2 -> t1 (small to large interp values)
# values are -2 signifying r=-d,
#            -1 signifying r=d
#             0 signifying anisotropic stretching starts or ends being dominant
# these correspond to vertex patterns from the visualization.
# The fourth number corresponds to the bin number when using the bin method.
function classifyEdge( d1::AbstractFloat, r1::AbstractFloat, s1::AbstractFloat, θ1::AbstractFloat, d2::AbstractFloat, r2::AbstractFloat, s2::AbstractFloat, θ2::AbstractFloat,p=false, minCrossing=1e-15 )
    margin = 1e-13
    y1 = (d2-r2) / ( (d2-r2) - (d1 - r1) )
    y2 = (d2+r2) / ( (d2+r2) - (d1 + r1) )

    if p
        println(("initial y values", y1, y2))
    end

    # if -0.0001*ϵ <= y1 <= 0.0001*ϵ || -0.0001*ϵ <= y2 <= 0.0001*ϵ || 1.0-0.0001*ϵ <= y1 <= 1.0+0.0001*ϵ || 1.0-0.0001*ϵ <= y2 <= 1.0+0.0001*ϵ
    #     if p
    #         println("first degen")
    #     end
    #     return ([99], [0.0], 0, [0.0])
    # end

    if margin <= y1 <= 1-margin
        if margin <= y2 <= 1-margin
            if y1 < y2
                cross_values = [(-1, y1, 0), (-2,y2,0)]
            else
                cross_values = [(-2,y2, 0), (-1,y1,0)]
            end
        else
            cross_values = [(-1,y1,0)]
        end
    else
        if margin <= y2 <= 1-margin
            cross_values = [(-2,y2,0)]
        else
            cross_values::Vector{Tuple{Int64,Float64,Int64}} = Vector{Tuple{Int64, Float64, Int64}}(undef,0)
        end
    end

    if p
        println("cross values initial $cross_values")
    end

    u_base = s1^2
    v_base = 2*s1*s2*cos(θ1 - θ2)
    w_base = s2^2

    # we will now classify two parabolas whose values correspond to s^2-r^2 and s^2-d^2, respectively,
    # in the interval [0,1]. The behavior of their sign changes will determine if/where s dominates the other
    # two coefficients.

    # 0 - always positive
    # 1 - always negative
    # 2 - positive -> negative
    # 3 - negative -> positive
    # 4 - positive -> negative -> positive
    # 5 - negative -> positive -> negative
    intercept_categories = [-1, -1]
    c1_list = [d1, r1]
    c2_list = [d2, r2]
    
    for i in 1:2
        c1 = c1_list[i]
        c2 = c2_list[i]

        # coefficients for ut^2 + vt(1-t) + w(1-t)^2. As t->[0,1] this goes from w to u
        u = u_base - c1^2
        v = v_base - 2*c1*c2
        w = w_base - c2^2

        su = sign(u)
        sw = sign(w)

        # if su < 0.0001 * ϵ
        #     su = 0
        # end

        # if sw < 0.0001 * ϵ
        #     sw = 0
        # end

        if abs(u) < 0.00000001*ϵ
            su = 0.0
            u = 0.0
        end

        if abs(w) < 0.00000001*ϵ
            sw = 0.0
            w = 0.0
        end

        if p
            println(("vals",su,sw,u,w))
        end

        if su == 0.0 || sw == 0.0
            # Degenerate case
            if su == sw
                if v > 0
                    category = 0
                else
                    # Always 0 is the same as always negative due to how ties are broken.
                    category = 1
                end
            elseif su == 0
                if v < -margin && w > 0
                    category = 2
                elseif v > margin && w < 0
                    category = 3
                elseif w > 0
                    category = 0
                else
                    category = 1
                end
            else
                if u < 0 && v > margin
                    category = 2
                elseif u > 0 && v < -margin
                    category = 3
                elseif u > 0
                    category = 0
                else
                    category = 1
                end
            end
        else
            # Nondegenerate case
            if su != sw
                if sw > 0
                    category = 2
                else
                    category = 3
                end
            else
                sv = sign(v)
                if sv == -su && (v^2 - 4*u*w) > 0
                    if su > 0
                        category = 4
                    else
                        category = 5
                    end
                else
                    if su > 0
                        category = 0
                    else
                        category = 1
                    end
                end
            end
        end

        # Now store important markers in cross_values as a tuple
            # 1st value: 0 (to distinguish from the other crossing types)
            # 2nd value: interp value in [0,1]
            # 3rd value: 1 for d, 2 for r

        if category != 0 && category != 1
            # # scale up the values to remove numerical instability:
            # if u < 10e-4 && v < 10e-4 && w < 10e-4
            #     u *= 10e5
            #     v *= 10e5
            #     w *= 10e5
            # end

            square_root = sqrt(v^2-4*u*w)
            t1 = (2*w-v + square_root) / (2*u - 2*v + 2*w)
            t2 = (2*w-v - square_root) / (2*u - 2*v + 2*w)

            # We run into numerical precision issues during the computation so raise the precision...
            if abs(t1) > 10e12 || abs(t2) > 10e12
                return ([99], [0.0], 0, [0.0])
            end

            if p
                println("before interp: $((u,v,w))")
                println(category)
            end

            small_t = round(min(t1, t2), digits=14)
            large_t = round(max(t1, t2), digits=14)

            # Check if we find any intersection at an endpoint and adjust the category accordingly.
            # I swear every imaginable degenerate case seems to occur at some point in time >:(

            if -margin <= small_t < margin
                small_t = 0.0
            end

            if -margin <= large_t < margin
                large_t = 0.0
            end

            if 1.0 < small_t < 1.0 + margin
                small_t = 1.0
            end

            if 1.0 < large_t < 1.0 + margin
                large_t = 1.0
            end

            if 0.0 <= small_t <= margin
                if category == 2
                    category = 1
                elseif category == 3
                    category = 0
                elseif category == 4
                    category = 3
                elseif category == 5
                    category = 2
                end
            end

            if 0.0 <= large_t <= margin
                if category == 2
                    category = 1
                elseif category == 3
                    category = 0
                elseif category == 4
                    category = 3
                elseif category == 5
                    category = 2
                end
            end

            if 1.0 - margin <= small_t <= 1.0
                if category == 2
                    category = 0
                elseif category == 3
                    category = 1
                elseif category == 4
                    category = 2
                elseif category == 5
                    category = 3
                end
            end

            if 1.0 - margin <= large_t <= 1.0             
                if category == 2
                    category = 0
                elseif category == 3
                    category = 1
                elseif category == 4
                    category = 2
                elseif category == 5
                    category = 3
                end
            end

            if p
                println("t values",(small_t,large_t))
            end

            if category == 2 || category == 3

                if su == 0
                    push!(cross_values, (0, small_t, i))
                elseif sw == 0
                    push!(cross_values, (0, large_t, i))
                elseif 0 < small_t < 1
                    push!(cross_values, (0, small_t, i))
                else
                    push!(cross_values, (0, large_t, i))
                end
            elseif category == 4 || category == 5
                push!(cross_values, (0, small_t, i))
                push!(cross_values, (0, large_t, i))
            end
        end

        intercept_categories[i] = category
    end


    # from the array of cross values, generate the edge class
    sort!(cross_values, by=f(x)=(x[2],x[1],x[3]))
    eigenvalueEdgeClass::Vector{Int64} = Vector{Float64}(undef, 0)
    eigenvalueEdgeLocations::Vector{Float64} = Vector{Float64}(undef, 0)
    eigenvectorEdgeClass = 0
    eigenvectorEdgeLocations::Vector{Float64} = Vector{Float64}(undef, 0)
    
    # 1st entry - s is larger than d
    # 2nd entry - s is larger than r
    s_is_larger = [ intercept_categories[1] % 2 == 0, intercept_categories[2] % 2 == 0 ]

    if p
        println("<<<<<<<<")        
        println(s_is_larger)
        println(cross_values)
        println(intercept_categories)
        println("<<<<<<<<")
    end

    for i in eachindex(cross_values)
        c = cross_values[i]

        if p
            println(("------------->", c[2], minCrossing, 1.0-minCrossing))
        end

        if 0 <= c[2] <= minCrossing || 1.0-minCrossing <= c[2] <= 1.0 + minCrossing
            return ([99], [0.0], 0, [0.0])
        end

        if c[1] == 0

            if c[2] >= 0.0001*ϵ && c[2] <= 1-0.0001*ϵ
                if p
                    println("triggered by $c")
                end
                if (c[3] == 1 || s_is_larger[1]) && (c[3] == 2 || s_is_larger[2])
                    push!(eigenvalueEdgeClass, 0)
                    push!(eigenvalueEdgeLocations, c[2])
                end
                if c[3] == 2
                    eigenvectorEdgeClass += 1
                    push!(eigenvectorEdgeLocations, c[2])
                end
                s_is_larger[c[3]] ⊻= true
            end   

        elseif !( s_is_larger[1] && s_is_larger[2] ) && c[2] > 0.0001*ϵ && c[2] < 1.0-0.0001*ϵ && ( i == length(cross_values) || cross_values[i+1][2] != c[2] )
            # The last conditional is to handle the degenerate case where |r|=|d| crosses at the same time as |r|=s or |d|=s.
            # In that case, we ignore the swap here as it is a switch between two colors where one will be hidden anyway.
            push!(eigenvalueEdgeClass, c[1])
            push!(eigenvalueEdgeLocations, c[2])
        end

    end

    return eigenvalueEdgeClass, eigenvalueEdgeLocations, eigenvectorEdgeClass, eigenvectorEdgeLocations

end

end