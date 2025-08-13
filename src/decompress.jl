module decompress

using StaticArrays

using ..utils
using ..tensorField
using ..huffman

export decompress2d
export decompress2dNaive
export decompress2dSymmetric
export decompress2dSymmetricNaive
export decompress2dSymmetricNaiveWithMask

function decompress2dNaive(compressed_file, decompress_folder, output = "../output", baseCompressor = "sz3")
    try
        run(`mkdir $output/$decompress_folder`)
    catch
    end

    # Un XZ the compressed file and undo the tar

    cwd = pwd()
    cd(output)

    # run(`xz -dv $output/$compressed_file.tar.xz`)
    run(`zstd -d $compressed_file.tar.zst`)
    run(`tar xvf $compressed_file.tar`)

    cd(cwd)
    
    vals_file = open("$output/vals.bytes", "r")
    dims = reinterpret(Int64, read(vals_file, 24))
    bound = reinterpret(Float64, read(vals_file, 8))[1]
    close(vals_file)

    if baseCompressor == "sz3"
        run(`../SZ3/build/bin/sz3 -d -z $output/row_1_col_1.cmp -o $output/$decompress_folder/row_1_col_1.dat -3 $(dims[3]) $(dims[2]) $(dims[1]) -M ABS $bound`)
        run(`../SZ3/build/bin/sz3 -d -z $output/row_1_col_2.cmp -o $output/$decompress_folder/row_1_col_2.dat -3 $(dims[3]) $(dims[2]) $(dims[1]) -M ABS $bound`)
        run(`../SZ3/build/bin/sz3 -d -z $output/row_2_col_1.cmp -o $output/$decompress_folder/row_2_col_1.dat -3 $(dims[3]) $(dims[2]) $(dims[1]) -M ABS $bound`)
        run(`../SZ3/build/bin/sz3 -d -z $output/row_2_col_2.cmp -o $output/$decompress_folder/row_2_col_2.dat -3 $(dims[3]) $(dims[2]) $(dims[1]) -M ABS $bound`)
    elseif baseCompressor == "mgard"
        run(`../MGARD/build/bin/mgard-cpu -x -t d -c $output/row_1_col_1.cmp -d $output/$decompress_folder/row_1_col_1.dat -n 3 $(dims[3]) $(dims[2]) $(dims[1]) -m abs -e $bound -s $smoothness`)
        run(`../MGARD/build/bin/mgard-cpu -x -t d -c $output/row_1_col_2.cmp -d $output/$decompress_folder/row_1_col_2.dat -n 3 $(dims[3]) $(dims[2]) $(dims[1]) -m abs -e $bound -s $smoothness`)
        run(`../MGARD/build/bin/mgard-cpu -x -t d -c $output/row_2_col_1.cmp -d $output/$decompress_folder/row_2_col_1.dat -n 3 $(dims[3]) $(dims[2]) $(dims[1]) -m abs -e $bound -s $smoothness`)
        run(`../MGARD/build/bin/mgard-cpu -x -t d -c $output/row_2_col_2.cmp -d $output/$decompress_folder/row_2_col_2.dat -n 3 $(dims[3]) $(dims[2]) $(dims[1]) -m abs -e $bound -s $smoothness`)
    elseif baseCompressor == "sperr"
        run(`../SPERR/build/bin/sperr2d $output/row_1_col_1.cmp -d --ftype 64 --dims $(dims[1]) $(dims[2]) --decomp_d $output/$decompress_folder/row_1_col_1.dat --pwe $bound`)
        run(`../SPERR/build/bin/sperr2d $output/row_1_col_2.cmp -d --ftype 64 --dims $(dims[1]) $(dims[2]) --decomp_d $output/$decompress_folder/row_1_col_2.dat --pwe $bound`)
        run(`../SPERR/build/bin/sperr2d $output/row_2_col_1.cmp -d --ftype 64 --dims $(dims[1]) $(dims[2]) --decomp_d $output/$decompress_folder/row_2_col_1.dat --pwe $bound`)
        run(`../SPERR/build/bin/sperr2d $output/row_2_col_2.cmp -d --ftype 64 --dims $(dims[1]) $(dims[2]) --decomp_d $output/$decompress_folder/row_2_col_2.dat --pwe $bound`)
    elseif baseCompressor == "zfp"
        run(`../zfp/build/bin/zfp -d -z $output/row_1_col_1.cmp -o $output/$decompress_folder/row_1_col_1.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -a $bound`)
        run(`../zfp/build/bin/zfp -d -z $output/row_1_col_2.cmp -o $output/$decompress_folder/row_1_col_2.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -a $bound`)
        run(`../zfp/build/bin/zfp -d -z $output/row_2_col_1.cmp -o $output/$decompress_folder/row_2_col_1.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -a $bound`)
        run(`../zfp/build/bin/zfp -d -z $output/row_2_col_2.cmp -o $output/$decompress_folder/row_2_col_2.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -a $bound`)
    end

    remove("$output/row_1_col_1.cmp")
    remove("$output/row_1_col_2.cmp")
    remove("$output/row_2_col_1.cmp")
    remove("$output/row_2_col_2.cmp")
    remove("$output/$compressed_file.tar")
    remove("$output/vals.bytes")

    return [0.0]
end

function decompress2dSymmetricNaive(compressed_file, decompress_folder, output = "../output", baseCompressor = "sz3")
    try
        run(`mkdir $output/$decompress_folder`)
    catch
    end

    # Un XZ the compressed file and undo the tar

    cwd = pwd()
    cd(output)

    # run(`xz -dv $output/$compressed_file.tar.xz`)
    run(`zstd -d $compressed_file.tar.zst`)
    run(`tar xvf $compressed_file.tar`)

    cd(cwd)
    
    vals_file = open("$output/vals.bytes", "r")
    dims = reinterpret(Int64, read(vals_file, 24))
    bound = reinterpret(Float64, read(vals_file, 8))[1]
    close(vals_file)

    if baseCompressor == "sz3"
        run(`../SZ3/build/bin/sz3 -d -z $output/row_1_col_1.cmp -o $output/$decompress_folder/row_1_col_1.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $bound`)
        run(`../SZ3/build/bin/sz3 -d -z $output/row_1_col_2.cmp -o $output/$decompress_folder/row_1_col_2.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $bound`)
        run(`../SZ3/build/bin/sz3 -d -z $output/row_2_col_2.cmp -o $output/$decompress_folder/row_2_col_2.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $bound`)
    elseif baseCompressor == "sperr"
        run(`../SPERR/build/bin/sperr2d $output/row_1_col_1.cmp -d --ftype 64 --dims $(dims[1]) $(dims[2]) --decomp_d $output/$decompress_folder/row_1_col_1.dat --pwe $bound`)
        run(`../SPERR/build/bin/sperr2d $output/row_1_col_2.cmp -d --ftype 64 --dims $(dims[1]) $(dims[2]) --decomp_d $output/$decompress_folder/row_1_col_2.dat --pwe $bound`)
        run(`../SPERR/build/bin/sperr2d $output/row_2_col_2.cmp -d --ftype 64 --dims $(dims[1]) $(dims[2]) --decomp_d $output/$decompress_folder/row_2_col_1.dat --pwe $bound`)
    else
        println("ERROR: unrecognized base compressor $baseCompressor")
        exit(1)
    end        

    remove("$output/row_1_col_1.cmp")
    remove("$output/row_1_col_2.cmp")
    remove("$output/row_2_col_2.cmp")
    remove("$output/$compressed_file.tar")
    remove("$output/vals.bytes")

    return [0.0]
end

function decompress2dSymmetricNaiveWithMask(compressed_file, decompress_folder, output = "../output", baseCompressor = "sz3")
    try
        run(`mkdir $output/$decompress_folder`)
    catch
    end

    # Un XZ the compressed file and undo the tar

    cwd = pwd()
    cd(output)

    # run(`xz -dv $output/$compressed_file.tar.xz`)
    run(`zstd -d $compressed_file.tar.zst`)
    run(`tar xvf $compressed_file.tar`)

    cd(cwd)
    
    vals_file = open("$output/vals.bytes", "r")
    dims = reinterpret(Int64, read(vals_file, 24))
    bound = reinterpret(Float64, read(vals_file, 8))[1]
    huffmanBytes = read(vals_file)
    close(vals_file)

    if baseCompressor == "sz3"
        run(`../SZ3/build/bin/sz3 -d -z $output/row_1_col_1.cmp -o $output/$decompress_folder/row_1_col_1.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $bound`)
        run(`../SZ3/build/bin/sz3 -d -z $output/row_1_col_2.cmp -o $output/$decompress_folder/row_1_col_2.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $bound`)
        run(`../SZ3/build/bin/sz3 -d -z $output/row_2_col_2.cmp -o $output/$decompress_folder/row_2_col_2.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $bound`)
    elseif baseCompressor == "sperr"
        run(`../SPERR/build/bin/sperr2d $output/row_1_col_1.cmp -d --ftype 64 --dims $(dims[1]) $(dims[2]) --decomp_d $output/$decompress_folder/row_1_col_1.dat --pwe $bound`)
        run(`../SPERR/build/bin/sperr2d $output/row_1_col_2.cmp -d --ftype 64 --dims $(dims[1]) $(dims[2]) --decomp_d $output/$decompress_folder/row_1_col_2.dat --pwe $bound`)
        run(`../SPERR/build/bin/sperr2d $output/row_2_col_2.cmp -d --ftype 64 --dims $(dims[1]) $(dims[2]) --decomp_d $output/$decompress_folder/row_2_col_2.dat --pwe $bound`)
    else
        println("ERROR: unrecognized base compressor $baseCompressor")
        exit(1)
    end

    mask = huffmanDecode(huffmanBytes)

    if length(mask) > 0
        tf = loadTensorField2dSymmetricFromFolder("$output/$decompress_folder", (dims[1]*dims[2]*dims[3],1,1))

        for i in 1:(dims[1]*dims[2]*dims[3])
            if mask[i] == 0.0
                tf.entries[1,i,1,1] = 0.0
                tf.entries[2,i,1,1] = 0.0
                tf.entries[3,i,1,1] = 0.0
            end
        end

        saveTensorFieldSymmetric64("$output/$decompress_folder", tf)        
    end

    remove("$output/row_1_col_1.cmp")
    remove("$output/row_1_col_2.cmp")
    remove("$output/row_2_col_2.cmp")
    remove("$output/$compressed_file.tar")
    remove("$output/vals.bytes")

    return [0.0]
end

function decompress2d(compressed_file, decompress_folder, output = "../output", baseCompressor = "sz3")
    startTime = time()

    try
        run(`mkdir $output/$decompress_folder`)
    catch
    end

    # Un XZ the compressed file and undo the tar

    cwd = pwd()
    cd(output)

    # run(`xz -dv $output/$compressed_file.tar.xz`)
    run(`zstd -d $compressed_file.tar.zst`)
    zstdSplit = time()
    run(`tar xvf $compressed_file.tar`)
    tarSplit = time()

    cd(cwd)

    # Read in metadata
    vals_file = open("$output/vals.bytes")
    dims = reinterpret(Int64, read(vals_file, 24))
    aeb = reinterpret(Float64, read(vals_file, 8))[1]

    # Read in quantization bytes
    typeCodeBytesLength = reinterpret(Int64, read(vals_file, 8))[1]
    typeCodeBytes = read(vals_file, typeCodeBytesLength)
    θAndSFixBytesLength = reinterpret(Int64, read(vals_file, 8))[1]
    θAndSFixBytes = read(vals_file, θAndSFixBytesLength)
    dBytesLength = reinterpret(Int64, read(vals_file, 8))[1]
    dBytes = read(vals_file, dBytesLength)
    rBytesLength = reinterpret(Int64, read(vals_file, 8))[1]
    rBytes = read(vals_file, rBytesLength)
    sBytesLength = reinterpret(Int64, read(vals_file, 8))[1]
    sBytes = read(vals_file, sBytesLength)

    # Read in various lossless lists
    losslessθLength = reinterpret(Int64, read(vals_file, 8))[1]*8
    lossless_θ = reinterpret(Float64, read(vals_file, losslessθLength))
    losslessALength = reinterpret(Int64, read(vals_file, 8))[1]*8
    lossless_A = reinterpret(Float64, read(vals_file, losslessALength))
    losslessBLength = reinterpret(Int64, read(vals_file, 8))[1]*8
    lossless_B = reinterpret(Float64, read(vals_file, losslessBLength))
    losslessCLength = reinterpret(Int64, read(vals_file, 8))[1]*8
    lossless_C = reinterpret(Float64, read(vals_file, losslessCLength))
    losslessDLength = reinterpret(Int64, read(vals_file, 8))[1]*8
    lossless_D = reinterpret(Float64, read(vals_file, losslessDLength))

    close(vals_file)

    dims_tuple = Tuple(dims)

    # De-Huffman the codes
    typeCodesBase = huffmanDecode(typeCodeBytes)
    if length(typeCodesBase) == 0
        typeCodes = zeros(Int64, dims_tuple)
    else
        typeCodes = reshape(typeCodesBase,dims_tuple)
    end

    θAndSFixCodesBase = huffmanDecode(θAndSFixBytes)
    if length(θAndSFixCodesBase) == 0
        θAndSFixCodes = zeros(Int64, dims_tuple)
    else
        θAndSFixCodes = reshape(θAndSFixCodesBase, dims_tuple)
    end

    dCodesBase = huffmanDecode(dBytes)
    if length(dCodesBase) == 0
        dCodes = zeros(Int64, dims_tuple)
    else
        dCodes = reshape(dCodesBase, dims_tuple)
    end

    rCodesBase = huffmanDecode(rBytes)
    if length(rCodesBase) == 0
        rCodes = zeros(Int64, dims_tuple)
    else
        rCodes = reshape(rCodesBase, dims_tuple)
    end

    sCodesBase = huffmanDecode(sBytes)
    if length(sCodesBase) == 0
        sCodes = zeros(Int64, dims_tuple)
    else
        sCodes = reshape(sCodesBase, dims_tuple)
    end

    readSplit = time()

    # Decompress from SZ and load into a tensor field
    if baseCompressor == "sz3"
        run(`../SZ3/build/bin/sz3 -f -z $output/row_1_col_1.cmp -o $output/row_1_col_1.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)
        run(`../SZ3/build/bin/sz3 -f -z $output/row_1_col_2.cmp -o $output/row_1_col_2.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)
        run(`../SZ3/build/bin/sz3 -f -z $output/row_2_col_1.cmp -o $output/row_2_col_1.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)    
        run(`../SZ3/build/bin/sz3 -f -z $output/row_2_col_2.cmp -o $output/row_2_col_2.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $aeb`)
    elseif baseCompressor == "sperr"
        run(`../SPERR/build/bin/sperr2d $output/row_1_col_1.cmp -d --ftype 32 --dims $(dims[1]) $(dims[2]) --decomp_f $output/row_1_col_1.dat --pwe $aeb`)
        run(`../SPERR/build/bin/sperr2d $output/row_1_col_2.cmp -d --ftype 32 --dims $(dims[1]) $(dims[2]) --decomp_f $output/row_1_col_2.dat --pwe $aeb`)
        run(`../SPERR/build/bin/sperr2d $output/row_2_col_1.cmp -d --ftype 32 --dims $(dims[1]) $(dims[2]) --decomp_f $output/row_2_col_1.dat --pwe $aeb`)
        run(`../SPERR/build/bin/sperr2d $output/row_2_col_2.cmp -d --ftype 32 --dims $(dims[1]) $(dims[2]) --decomp_f $output/row_2_col_2.dat --pwe $aeb`)
    else
        println("ERROR: unrecognized base compressor $baseCompressor")
        exit(1)
    end

    baseDecompressorSplit = time()

    dims_tuple::Tuple{Int64, Int64, Int64} = (dims[1], dims[2], dims[3])
    tf = loadTensorField2dFromFolder("$output", dims_tuple)

    read2Split = time()

    next_lossless_full = 1
    next_lossless_d = 1
    next_lossless_r = 1
    next_lossless_s = 1
    next_lossless_θ = 1

    # Iterate through the cells and adjust accordingly.
    for t in 1:dims[3]
        for j in 1:dims[2]
            for i in 1:dims[1]

                # Make sure that a change must actually be applied in the first place.
                if !(typeCodes[i,j,t] == 0 && θAndSFixCodes[i,j,t] == 0 && dCodes[i,j,t] == 0 && rCodes[i,j,t] == 0 && sCodes[i,j,t] == 0)

                    if typeCodes[i,j,t] == 255
                        setTensor( tf, i, j, t,  SMatrix{2,2,Float64}(lossless_A[next_lossless_full], lossless_C[next_lossless_full], lossless_B[next_lossless_full], lossless_D[next_lossless_full]) )
                        next_lossless_full += 1
                    else
                        typeCode = typeCodes[i,j,t]
                        θCode::UInt8 = θAndSFixCodes[i,j,t] & (2^6-1)
                        sFix::UInt8 = ( θAndSFixCodes[i,j,t] & (2^6+2^7) ) >> 6

                        tensor = getTensor(tf, i, j, t)
                        d,r,s,θ = decomposeTensor(tensor)

                        d = d + aeb * dCodes[i,j,t] / (2^MAX_PRECISION)
                        r = r + aeb * rCodes[i,j,t] / (2^MAX_PRECISION)
                        s = s + sqrt(2) * aeb * sCodes[i,j,t] / (2^MAX_PRECISION)

                        if θCode == 2^6-1
                            θ = lossless_θ[next_lossless_θ]
                            next_lossless_θ += 1
                        else
                            θ = θ + 2pi / (2^6-1) * θCode
                        end

                        # apply the various s fixes
                        if s < -(sqrt(2)-1)*aeb
                            s += sqrt(2)*aeb
                        elseif s < 0
                            s += (sqrt(2)-1)*aeb
                        end

                        if sFix == 1
                            s -= (sqrt(2)-1)*aeb
                        elseif sFix == 2
                            s += (sqrt(2)-1)*aeb
                        end

                        # apply the swapping
                        d_sign_swap = (typeCode & (3 << 6)) >> 6
                        r_sign_swap = (typeCode & (3 << 4)) >> 4
                        d_largest_swap = (typeCode & (3 << 2)) >> 2
                        r_over_s_swap = typeCode & 3

                        # handle signs

                        if d_sign_swap == 2
                            d = 0.0
                            r = 0.0
                            s = 0.0
                        else

                            # handle signs (pt 2)

                            if r_sign_swap == 1
                                if r > 0.0
                                    r -= aeb
                                else
                                    r += aeb
                                end
                            elseif r_sign_swap == 2
                                r = 0.0
                            elseif r_sign_swap == 3
                                r = 0.0
                                s = 0.0
                            end

                            if d_sign_swap == 1
                                if d > 0.0
                                    d -= aeb
                                else
                                    d += aeb
                                end
                            end

                            d_swap_rank, r_swap_rank, s_swap_rank = rankOrder(abs(d),abs(r),s)

                            mags = MArray{Tuple{3},Float64}(0.0,0.0,0.0)
                            mags[d_swap_rank] = abs(d)
                            mags[r_swap_rank] = abs(r)
                            mags[s_swap_rank] = s

                            # handle normal swaps
                            if d_largest_swap != 0 || d_sign_swap == 1
                                if d_swap_rank == 1
                                    if d_sign_swap == 0
                                        if d_largest_swap != 2 && d_largest_swap != 3 # those codes require d to be on top!
                                            d_swap_rank = 2
                                            if s_swap_rank == 2
                                                s_swap_rank = 1
                                            else
                                                r_swap_rank = 1
                                            end
                                        end
                                    end
                                else
                                    d_swap_rank = 1
                                    if s_swap_rank == 1
                                        s_swap_rank = 2
                                    else
                                        r_swap_rank = 2
                                    end
                                end
                            end

                            if r_over_s_swap == 1
                                temp = r_swap_rank
                                r_swap_rank = s_swap_rank
                                s_swap_rank = temp
                            end

                            # handle degenerate setting

                            if d_largest_swap == 2
                                s_swap_rank = 1
                            end

                            if r_over_s_swap == 2
                                minRank = min(r_swap_rank,s_swap_rank)
                                r_swap_rank = minRank
                                s_swap_rank = minRank
                            end

                            if d_largest_swap == 3
                                r_swap_rank = 1
                            end

                            # set stuff back to their ranks
                            d = nonzeroSign(d) * mags[d_swap_rank]
                            r = nonzeroSign(r) * mags[r_swap_rank]
                            s = mags[s_swap_rank]

                        end

                        setTensor(tf, i, j, t, recomposeTensor(d, r, s, θ))

                    end # end if precision >= 8 (then else for the main reconstruction)

                end # end if not all of the base codes are 0

            end # end for
        end # end for 
    end # end for

    augmentSplit = time()

    # tf2 = loadTensorField2dFromFolder("../output/test", dims_tuple)

    # numMismatches = 0
    # for t in 1:dims[3]
    #     for j in 1:dims[2]
    #         for i in 1:dims[1]
    #             if getTensor(tf,i,j,t) != getTensor(tf2,i,j,t)
    #                 numMismatches += 1                    
    #                 println("mismatch at $((i,j,t))")
    #                 println(getTensor(tf,i,j,t))
    #                 println(decomposeTensor(getTensor(tf,i,j,t)))
    #                 println("---")
    #                 println(getTensor(tf2,i,j,t))
    #                 println(decomposeTensor(getTensor(tf2,i,j,t)))
    #                 typeCode = typeCodes[i,j,t]
    #                 d_sign_swap = (typeCode & (3 << 6)) >> 6
    #                 r_sign_swap = (typeCode & (3 << 4)) >> 4
    #                 d_largest_swap = (typeCode & (3 << 2)) >> 2
    #                 r_over_s_swap = typeCode & 3
    #                 θCode::UInt8 = θAndSFixCodes[i,j,t] & (2^6-1)
    #                 sFix::UInt8 = ( θAndSFixCodes[i,j,t] & (2^6+2^7) ) >> 6                    
    #                 println(θCode)
    #                 println(sFix)
    #                 println((d_sign_swap,r_sign_swap,d_largest_swap,r_over_s_swap))    
    #                 println((dCodes[i,j,t],rCodes[i,j,t],sCodes[i,j,t]))
    #                 println("=======================")  
    #             end
    #         end
    #     end
    # end
    # println("$numMismatches mismatches")

    # Save to file
    saveTensorField64("$output/$decompress_folder", tf)

    saveSplit = time()

    remove("$output/row_1_col_1.dat")
    remove("$output/row_1_col_2.dat")
    remove("$output/row_2_col_1.dat")
    remove("$output/row_2_col_2.dat")
    remove("$output/row_1_col_1.cmp")
    remove("$output/row_1_col_2.cmp")
    remove("$output/row_2_col_1.cmp")
    remove("$output/row_2_col_2.cmp")
    remove("$output/vals.bytes")

    endTime = time()

    return [zstdSplit - startTime, tarSplit - zstdSplit, readSplit - tarSplit, baseDecompressorSplit - readSplit, read2Split - baseDecompressorSplit, augmentSplit - read2Split, saveSplit - augmentSplit, endTime - saveSplit ]
end

function decompress2dSymmetric(compressed_file, decompress_folder, output = "../output", baseCompressor = "sz3")
    startTime = time()
    bits = 6 # this used to be a parameter
    try
        run(`mkdir $output/$decompress_folder`)
    catch
    end

    # Un XZ the compressed file and undo the tar

    cwd = pwd()
    cd(output)

    # run(`xz -dv $output/$compressed_file.tar.xz`)
    run(`zstd -d $compressed_file.tar.zst`)
    zstdSplit = time()
    run(`tar xvf $compressed_file.tar`)
    tarSplit = time()

    cd(cwd)
    
    vals_file = open("$output/vals.bytes", "r")
    dims = reinterpret(Int64, read(vals_file, 24))
    bound = reinterpret(Float64, read(vals_file, 8))[1]
    codeBytesLength = reinterpret(Int64, read(vals_file, 8))[1]
    codeBytes = read(vals_file, codeBytesLength)
    fullLosslessBytesLength = reinterpret(Int64, read(vals_file, 8))[1]
    fullLosslessBytes = read(vals_file, fullLosslessBytesLength)
    losslessValues = reinterpret(Float64, read(vals_file))
    close(vals_file)

    codesBase = huffmanDecode(codeBytes)
    dims_tuple::Tuple{Int64,Int64,Int64} = (dims[1],dims[2],dims[3])
    if length(codesBase) == 0
        codes = zeros(Int64, dims_tuple)
    else
        codes = reshape(codesBase,dims_tuple)
    end

    fullLosslessBase = huffmanDecode(fullLosslessBytes)
    if length(fullLosslessBase) == 0
        fullLossless = zeros(Int64, dims_tuple)
    else
        fullLossless = reshape(fullLosslessBase, dims_tuple)
    end

    readSplit = time()

    if baseCompressor == "sz3"
        run(`../SZ3/build/bin/sz3 -f -z $output/row_1_col_1.cmp -o $output/$decompress_folder/row_1_col_1.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $bound`)
        run(`../SZ3/build/bin/sz3 -f -z $output/row_1_col_2.cmp -o $output/$decompress_folder/row_1_col_2.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $bound`)
        run(`../SZ3/build/bin/sz3 -f -z $output/row_2_col_2.cmp -o $output/$decompress_folder/row_2_col_2.dat -3 $(dims[1]) $(dims[2]) $(dims[3]) -M ABS $bound`)
    elseif baseCompressor == "sperr"
        run(`../SPERR/build/bin/sperr2d $output/row_1_col_1.cmp -d --ftype 32 --dims $(dims[1]) $(dims[2]) --decomp_f $output/$decompress_folder/row_1_col_1.dat --pwe $bound`)
        run(`../SPERR/build/bin/sperr2d $output/row_1_col_2.cmp -d --ftype 32 --dims $(dims[1]) $(dims[2]) --decomp_f $output/$decompress_folder/row_1_col_2.dat --pwe $bound`)
        run(`../SPERR/build/bin/sperr2d $output/row_2_col_2.cmp -d --ftype 32 --dims $(dims[1]) $(dims[2]) --decomp_f $output/$decompress_folder/row_2_col_2.dat --pwe $bound`)
    else
        println("ERROR: unrecognized base compressor $baseCompressor")
        exit(1)
    end

    baseDecompressorSplit = time()

    tf = loadTensorField2dSymmetricFromFolder("$output/$decompress_folder", dims_tuple)

    read2Split = time()

    # adjust

    next_lossless = 1
    for t in 1:dims[3]
        for j in 1:dims[2]
            for i in 1:dims[1]
                if fullLossless[i,j,t] == 1
                    nextTensor = SVector{3,Float64}(losslessValues[next_lossless], losslessValues[next_lossless+1], losslessValues[next_lossless+2])
                    setTensor(tf, i, j, t, nextTensor)
                    next_lossless += 3
                elseif codes[i,j,t] == 2^bits-1
                    intermediateTensor = getTensor(tf, i, j, t)
                    trace = (intermediateTensor[1]+intermediateTensor[3])/2
                    nextTensor = SVector{3,Float64}(losslessValues[next_lossless]+trace, losslessValues[next_lossless+1], -losslessValues[next_lossless]+trace)
                    setTensor(tf, i, j, t, nextTensor)
                    next_lossless += 2
                elseif codes[i,j,t] != 0
                    trace, r, θ = decomposeTensorSymmetric( getTensor( tf, i, j, t ) )
                    θ += 2pi/(2^bits-1)*codes[i,j,t]
                    setTensor(tf, i, j, t, recomposeTensorSymmetric( trace, r, θ ))
                end
            end
        end
    end

    augmentSplit = time()

    saveTensorFieldSymmetric64("$output/$decompress_folder", tf)

    saveSplit = time()

    remove("$output/row_1_col_1.cmp")
    remove("$output/row_1_col_2.cmp")
    remove("$output/row_2_col_2.cmp")
    remove("$output/$compressed_file.tar")
    remove("$output/vals.bytes")

    endTime = time()

    return [zstdSplit - startTime, tarSplit - zstdSplit, readSplit - tarSplit, baseDecompressorSplit - readSplit, read2Split - baseDecompressorSplit, augmentSplit - read2Split, saveSplit - augmentSplit, endTime - saveSplit ]

end

end