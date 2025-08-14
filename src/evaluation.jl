using Statistics
using StaticArrays

using ..tensorField
using ..conicUtils
using ..cellTopology

function vertexMatching(tf1::TF, tf2::TF)

    result = MArray{Tuple{2,2}}(zeros(Int64, (2,2)))

    VAL = 1
    VEC = 2
    SAME = 1
    DIF = 2

    x,y = tf1.dims

    for j in 1:y
        for i in 1:x

            t1 = getTensor(tf1, i, j)
            t2 = getTensor(tf2, i, j)

            d1, r1, s1, _ = decomposeTensor(t1)
            d2, r2, s2, _ = decomposeTensor(t2)

            if classifyTensorEigenvalue(d1, r1, s1) == classifyTensorEigenvalue(d2, r2, s2)
                result[VAL,SAME] += 1
            else
                result[VAL,DIF] += 1
            end

            if classifyTensorEigenvector(r1, s1) == classifyTensorEigenvector(r2, s2)
                result[VEC,SAME] += 1
            else
                result[VEC,DIF] += 1
            end

        end
    end


    return result
end

# For multiple dispatch to work out with the symmetric tensor fields.
function vertexMatching(tf1::TF_Sym, tf2::TF_Sym)
    return MArray{Tuple{2,2}}(zeros(Int64, (2,2)))
end

function cellMatching(tf1::TF, tf2::TF)

    # The order follows the constants
    result = MArray{Tuple{7}}(zeros(Int64, (7,)))

    VECDPSAME = 1 # do the intneral degenerate points match or not.
    VECDPDIF = 2
    VALSAME = 3 # does the internal topology match for the eigenvalue topology
    VALDIF = 4
    VECSAME = 5 # does the internal topology match for the eigenvector partition
    VECSAMEEXCEPTDP = 6 # does the internal topology match for the eigenvector (but critical points do not match)
    VECDIF = 7

    x, y = tf1.dims
    x -= 1
    y -= 1

    for j in 1:y
        for i in 1:x                
            for k in 0:1

                c1 = getDegeneracyTypeFull(tf1, i, j, Bool(k))
                c2 = getDegeneracyTypeFull(tf2, i, j, Bool(k))

                if c1 == c2
                    result[VECDPSAME] += 1
                else
                    result[VECDPDIF] += 1
                end

                top1 = tensorField.classifyCellEigenvalue(tf1, i, j, Bool(k), true)
                top2 = tensorField.classifyCellEigenvalue(tf2, i, j, Bool(k), true)

                if top1.vertexTypesEigenvalue == top2.vertexTypesEigenvalue && cyclicMatch(top1.DPArray,top2.DPArray) && cyclicMatch(top1.DNArray,top2.DNArray) && cyclicMatch(top1.RPArray,top2.RPArray) && cyclicMatch(top1.RNArray,top2.RNArray)
                    result[VALSAME] += 1
                else
                    result[VALDIF] += 1
                end

                if top1.vertexTypesEigenvector == top2.vertexTypesEigenvector && top1.RPArrayVec == top2.RPArrayVec && top1.RNArrayVec == top2.RNArrayVec && c1 == c2
                    if c1 == c2
                        result[VECSAME] += 1
                    else
                        result[VECSAMEEXCEPTDP] += 1
                    end
                else
                    result[VECDIF] += 1
                end

            end
        end
    end


    return result

end

function cellMatching(tf1::TF_Sym, tf2::TF_Sym)

    # We have 7 entries here so that the output matches the above method...
    result = MArray{Tuple{7}}(zeros(Int64, (7,)))

    SAME = 1
    DIF = 2

    x, y = tf1.dims
    x -= 1
    y -= 1

    for j in 1:y
        for i in 1:x                
            for k in 0:1

                c1 = getDegeneracyTypeFull(tf1, i, j, Bool(k))
                c2 = getDegeneracyTypeFull(tf2, i, j, Bool(k))

                if c1 == c2
                    result[SAME] += 1
                else
                    result[DIF] += 1
                end

            end
        end
    end


    return result
end

function reconstructionQuality(tf_ground, tf_reconstructed)
    min_val, max_val = getMinAndMax(tf_ground)
    
    if min_val == max_val
        return -1.0, -1.0, -1.0
    end

    peak_signal = max_val - min_val
    mse = mean( (tf_ground.entries - tf_reconstructed.entries) .^ 2 )
    psnr = 10 * log(10, peak_signal^2 / mse)

    max_error = maximum( abs.(tf_ground.entries - tf_reconstructed.entries) )

    return psnr, mse, (max_val - min_val), max_error
end

function evaluateCompression(::Val{symmetric}, ground::String, reconstructed::String, dims::Tuple{Int64, Int64}, compressed_size::Int64 = -1) where symmetric

    if symmetric
        tf1 = loadTFFromFolderSym(ground, dims)
        tf2 = loadTFFromFolderSym(reconstructed, dims)
    else
        tf1 = loadTFFromFolder(ground, dims)
        tf2 = loadTFFromFolder(reconstructed, dims)
    end

    vertexMatching_output = vertexMatching(tf1, tf2)
    cellMatching_output = cellMatching(tf1, tf2)
    _, mse, range, max_error = reconstructionQuality(tf1, tf2)

    bitrate = compressed_size*8/(dims[1]*dims[2])

    if range != 0.0 # fail safe in case the tensor field is uniform.
        mseByRangeSquared = mse / range^2
        maxErrorByRange = max_error / range
    else
        mseByRangeSquared = -1.0
        maxErrorByRange = -1.0
    end

    return (vertexMatching_output, cellMatching_output, bitrate, mseByRangeSquared, maxErrorByRange)

end