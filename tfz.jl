using StaticArrays

include("src/utils.jl")
include("src/conicUtils.jl")
include("src/cellTopology.jl")
include("src/tensorField.jl")
include("src/huffman.jl")
include("src/decompress.jl")
include("src/compress.jl")
include("src/evaluation.jl")

using .compress
using .decompress
using .tensorField
using .utils

function combine_counts(a::Vector{Int64},b::Vector{Int64})
    if length(a) >= length(b)
        for i in eachindex(b)
            a[i] += b[i]
        end
        return a
    else
        return combine_counts(b,a)
    end
end

function error()
    println("use -h for help.")
    exit(1)
end

function help()
    println("Usage: julia tfz.jl <options>")
    println("Please see the README for more details.")
    println()
    println("Specifying Files")
    println("\t-i <path> : Folder containing tensor field files to compress.")
    println("\t-z <path> : Compressed output(s).")
    println("\t-o <path> : Decompressed output(s).")
    println("\t-2 <int> <int> : Dimensions of the tensor field.")
    println("\t-sym : Specify that the tensor field is symmetric.")
    println("\t-asym : Specify that the tensor field is asymmetric.")
    println("\t-n_slice :")
    println("\t\tIf you are compressing multiple slices stacked")
    println("\t\tin a single file, specify how many slices.")
    println("\t-slice :")
    println("\t\tIf you would only like to compress one slice out")
    println("\t\tof many slices stored in one file, specify which slice.")
    println()
    println("Compression Parameters")
    println("\t-bc <bc> : Base compressor. Choose one of SZ3 or SPERR.")
    println("\t-eb <float> :")
    println("\t\tPointwise error bound. Percentage of the function range.")
    println("\t\tDo not set this when decompressing.")
    println("\t-val :")
    println("\t\tUse only with -asym. Preserve only the topology of the")
    println("\t\teigenvalue partition.")
    println("\t-vec :")
    println("\t\tUse only with -asym. Preserve only the topology of the")
    println("\t\teigenvector partition.")
    println("\t-bcOnly : Only run the base compressor with no augmentation.")
    println()
    println("Options for Running Experiments")
    println("\t-experiment :")
    println("\t\tRun compression and decompression, and evaluate the results.")
    println("\t\tIf used, using -z and -o are optional. If either is not")
    println("\t\tspecified, temporary files will be used where needed.")
    println("\t\tmissing parameters.")
    println("\t-csv <path> :")
    println("\t\tUse with -experiment. Specify CSV file to save results. If")
    println("\t\tthe file already exists, the results will be appended to")
    println("\t\tthe end of the file.")
    println("\t-skipStatistics :")
    println("\tUse with -experimient. Skip computing statistics like number of")
    println("\t\ttopological errors and PSNR. This is designed for timing.")
    println()
    println("Other Options")
    println("\t-of <path> :")
    println("\t\tTFZ creates many temporary files during compression.")
    println("\t\tThis specifies where these files should be written to")
    println("\t\t(default CWD).")
    println("\t-verbose : Do not suppress the outputs of system calls.")
    println("\t-bcFolder : Folder that contains the base compressor (default CWD).")
    println("\t-h, -help : Display this message.")
end

function main()

    # mandatory arguments
    data_folder = ""
    specified_input = false
    slice_size = (-1,-1)
    specified_slice_size = false
    symmetric = false
    specified_sym = false # to check if symmetric was specified
    specified_asym = false # check if asymmetric was specified
    base_compressor = ""
    specified_base_compressor = false
    eb = -1.0
    specified_eb = false

    # optional arguments
    experiment = false
    num_slices = 1
    specified_num_slices = false
    slice_number = -1
    specified_slice_number = false
    naive = false
    eigenvalue = true
    eigenvector = true
    specified_eigenvalue = false
    specified_eigenvector = false
    csv = ""
    output = "."
    verbose = false
    skipStatistics = false
    intermediate_name = "compressed_output"
    specified_intermediate = false
    decompressed_name = "reconstructed"
    specified_decompressed = false
    bcFolder = "."

    # parse arguments

    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "-i"
            try
                data_folder = ARGS[i+1]
            catch
                println("TFZ: -i must be followed by path to data folder")
                error()
            end
            specified_input = true
            i += 2
        elseif ARGS[i] == "-2"
            try
                slice_size = (parse(Int64,ARGS[i+1]), parse(Int64,ARGS[i+2]))
            catch
                println("TFZ: -2 must be followed by two int numbers specifying the scalar field size.")
                error()
            end
            specified_slice_size = true
            i += 3
        elseif ARGS[i] == "-sym"
            symmetric = true
            specified_sym = true
            i += 1
        elseif ARGS[i] == "-asym"
            symmetric = false
            specified_asym = true
            i += 1
        elseif ARGS[i] == "-bc"
            try
                base_compressor = ARGS[i+1]
            catch
                println("TFZ: -bc must be followed by the name of the base compressor. None specified.")
                error()
            end
            specified_base_compressor = true
            i += 2
        elseif ARGS[i] == "-eb"
            try
                eb = parse(Float64,ARGS[i+1])
            catch
                println("TFZ: -eb must be followed by a floating point number between 0 and 1 representing the error bound.")
                error()
            end
            specified_eb = true
            i += 2
        elseif ARGS[i] == "-experiment"
            experiment = true
            i += 1
        elseif ARGS[i] == "-n_slice"
            try
                num_slices = parse(Int64,ARGS[i+1])
            catch
                println("TFZ: -n_slice must be followed by an positive integer specifying the number of slices.")
                error()
            end
            specified_num_slices = true
            i += 2
        elseif ARGS[i] == "-slice"
            try
                slice_number = parse(Int64,ARGS[i+1])
            catch
                println("TFZ: -slice must be followed by a positive integer specifying the slice to compress.")
                error()
            end
            specified_slice_number = true
            i += 2
        elseif ARGS[i] == "-bcOnly"
            naive = true
            i += 1
        elseif ARGS[i] == "-val"
            eigenvalue = true
            eigenvector = false
            specified_eigenvalue = true
            i += 1
        elseif ARGS[i] == "-vec"
            eigenvalue = false
            eigenvector = true
            specified_eigenvector = true
            i += 1
        elseif ARGS[i] == "-csv"
            csv = ARGS[i+1]
            i += 2
        elseif ARGS[i] == "-of"
            try
                output = ARGS[i+1]
            catch
                println("TFZ: -of must be followed by the name of the output folder.")
                error()
            end
            i += 2
        elseif ARGS[i] == "-verbose"
            verbose = true
            i += 1
        elseif ARGS[i] == "-skipStatistics"
            skipStatistics = true
            i += 1
        elseif ARGS[i] == "-z"
            try
                intermediate_name = ARGS[i+1]
            catch
                println("TFZ: -z must be followed by the name of the compressed file.")
                error()
            end
            specified_intermediate = true
            i += 2
        elseif ARGS[i] == "-o"
            try
                decompressed_name = ARGS[i+1]
            catch
                println("TFZ: -o must be followed by the name of the decompressed file.")
                error()
            end
            specified_decompressed = true
            i += 2
        elseif ARGS[i] == "-bcFolder"
            try
                bcFolder = ARGS[i+1]
            catch
                println("TFZ: -bcFolder must be followed by the name of the folder containing the base compressor.")
            end
            i += 2
        elseif ARGS[i] == "-h" || ARGS[i] == "-help"
            help()
            exit(0)
        else
            println("TFZ: unknown argument $(ARGS[i])")
            error()
        end
    end

    if !specified_input && !specified_intermediate
        println("TFZ: no inputs specified.")
        println("Please specify a file to compress with -i or a file to decompress with -z.")
        error()
    end

    if !specified_input && experiment
        println("TFZ: using experiment requires you to specify an input.")
        println("Please specify a file to compress using -i.")
        error()
    end

    if specified_input && !experiment && !specified_intermediate && !specified_decompressed
        println("TFZ: no outputs specified.")
        println("Please specify a compressed filename with -z, a decompressed filename with -o, or use the -experiment flag.")
        error()
    end

    if !specified_input && specified_intermediate && !specified_decompressed
        println("TFZ: no decompressed filename specified for decompression.")
        println("Please specify a decompressed filename using -o.")
        error()
    end

    if !specified_slice_size
        println("TFZ: no tensor field dimensions specified.")
        println("Please specify the size of the 2D tensor field using -2.")
        error()
    end

    if slice_size[1] <= 0 || slice_size[2] <= 0
        println("TFZ: invalid tensor field size $(slice_size).")
        error()
    end

    if !specified_sym && !specified_asym
        println("TFZ: please specify that the tensor field is symmetric with -sym or asymmetric with -asym.")
        error()
    end

    if specified_sym && specified_asym
        println("TFZ: you specified that the tensor field is both symmetric and asymmetric (using -sym and -asym).")
        println("Please choose only one.")
        error()
    end

    if !specified_base_compressor
        println("TFZ: Please specify a base compressor using -bc.")
        error()
    end

    if base_compressor != "SZ3" && base_compressor != "SPERR"
        println("TFZ: unsupported base compressor $base_compressor")
        println("Please enter either SZ3 or SPERR (case sensitive).")
        error()
    end

    if !specified_eb && specified_input
        println("TFZ: Please specify an error bound using -eb.")
        error()
    end

    if specified_eb && !specified_input
        println("TFZ: Do not specify an error bound when decompressing.")
        error()
    end

    if eb <= 0
        println("TFZ: The error bound specified with -eb must be positive.")
        error()
    end

    if eb > 1
        println("TFZ: The error bound is a percentage of the range and must be less than one.")
        error()
    end

    if num_slices <= 0
        println("TFZ: The number of slices must be positive.")
        error()
    end
    
    if specified_slice_number && slice_number <= 0
        println("TFZ: The slice number that you specify must be positive.")
        error()
    end

    if specified_num_slices && specified_slice_number
        println("TFZ: You specified both the number of slices and a slice number. Please only specify one or the other.")
        error()
    end

    if specified_eigenvector && symmetric
        println("TFZ: You used -eigenvector with -sym, which is invalid.")
        error()
    end

    if specified_eigenvalue && symmetric
        println("TFZ: You used -eigenvalue with -sym, which is invalid.")
        error()
    end

    if specified_eigenvector && specified_eigenvalue
        println("TFZ: You used -eigenvalue and -eigenvector, which is invalid.")
        println("To preserve the topology of both the eigenvector and eigenvalue partitions, use neither flag.")
        error()
    end

    if csv != "" && !experiment
        println("TFZ: -csv should only be used with -experiment")
        error()
    end

    if skipStatistics && !experiment
        println("TFZ: -skipStatistics shouold only be used with -experiment")
        error()
    end

    # don't do any evaluation unless we are doing an experiment.
    if !experiment
        skipStatistics = true
    end

    totalCompressionTime = 0.0
    totalDecompressionTime = 0.0

    totalBitrate = 0.0
    totalMSEByRangeSquared = 0.0
    overallMaxErrorByRange = 0.0
    ctv = MArray{Tuple{11}}(zeros(Float64, (11,))) # compression time vector
    dtv = MArray{Tuple{8}}(zeros(Float64, (8,))) # decompression time vector
    totalVertexMatching = MArray{Tuple{2,2}}(zeros(Int64, (2,2)))
    totalCellMatching = MArray{Tuple{7}}(zeros(Int64, (7,)))
    
    combined_visit_counts = zeros(Int64, 1)

    stdout_ = stdout
    stderr_ = stderr

    numNoRange = 0 # we use this to keep track of the number of slices with 0 range so we can omit them from psnr

    trialStart = time()

    if occursin("/",data_folder)
        dataset_name = data_folder[first(findlast("/",data_folder))+1:lastindex(data_folder)]
    else
        dataset_name = data_folder
    end

    if slice_number != -1
        dataset_name = dataset_name * " (slice $slice_number)"
        range = slice_number:slice_number
    else
        range = 1:num_slices
    end

    # compute the target
    if symmetric
        target = "SYM"
    elseif eigenvector
        if eigenvalue
            target = "BOTH"
        else
            target = "EIGENVECTOR"
        end
    else
        target = "EIGENVALUE"
    end

    if naive
        target = target * " (NAIVE)"
    end

    naiveArrays = Array{Float64, 3}[]

    if naive
        if symmetric
            array_names = ("A","B","D")
        else
            array_names = ("A","B","C","D")
        end

        for a in array_names
            byte_file = open("$data_folder/$a.raw", "r")
            reshaped = reshape( reinterpret( Float64, read(byte_file) ), (slice_size[1],slice_size[2],num_slices) )
            push!(naiveArrays, reshaped)
            close(byte_file)
        end
    end

    if num_slices != 1
        if specified_intermediate && !ispath("$output/$intermediate_name")
            try
                run(`mkdir $output/$intermediate_name`)
            catch
            end
        end

        if specified_decompressed && !ispath("$output/$decompressed_name")
            try
                run(`mkdir $output/$decompressed_name`)
            catch
            end
        end
    end

    num_digits = Int64(ceil(log(10,num_slices+1)))

    for t in range

        if num_slices != 1
            digits_str = lpad(t,num_digits,'0')
            print("processing slice $digits_str / $num_slices\r")
        end

        if !verbose
            redirect_stdout(devnull)
            redirect_stderr(devnull)
        end

        compression_start = time()

        if num_slices != 1
            if specified_intermediate
                intermediate_folder = "$output/$intermediate_name/slice_$digits_str"
            else
                intermediate_folder = "$output/$intermediate_name"
            end

            if specified_decompressed
                decompressed_folder = "$output/$decompressed_name/slice_$digits_str"
            else
                decompressed_folder = "$output/$decompressed_name"
            end
        else
            intermediate_folder = "$output/$intermediate_name"
            decompressed_folder = "$output/$decompressed_name"
        end

        if specified_input
            if naive
                removeIfExists("$output/slice")
                run(`mkdir $output/slice`)
                saveArray64("$output/slice/A.raw", naiveArrays[1][:,:,t])
                saveArray64("$output/slice/B.raw", naiveArrays[2][:,:,t])
                if symmetric
                    saveArray64("$output/slice/D.raw", naiveArrays[3][:,:,t])
                else
                    saveArray64("$output/slice/C.raw", naiveArrays[3][:,:,t])
                    saveArray64("$output/slice/D.raw", naiveArrays[4][:,:,t])
                end

                if symmetric
                    compress2dSymmetricNaiveWithMask("$output/slice", (slice_size[1],slice_size[2]), intermediate_folder, eb, output, base_compressor, bcFolder)
                else
                    compress2dNaive("$output/slice", (slice_size[1],slice_size[2]), intermediate_folder, eb, output, base_compressor, bcFolder)
                end
            else
                if symmetric
                    ctVector, slice_visit_counts = compress2dSymmetric(data_folder, (slice_size[1],slice_size[2]), intermediate_folder, eb, output, base_compressor, !skipStatistics, t, bcFolder)
                else
                    ctVector, slice_visit_counts = compress2d(data_folder, (slice_size[1],slice_size[2]), intermediate_folder, eb, output, eigenvalue, eigenvector, base_compressor, !skipStatistics, t, bcFolder)
                end

                ctv += ctVector
                if !skipStatistics
                    combined_visit_counts = combine_counts(combined_visit_counts, slice_visit_counts)
                end

            end
        end

        compression_end = time()
        ct = compression_end - compression_start

        compressed_size = filesize("$intermediate_folder.tar.zst")

        removeIfExists("$output/$intermediate_name.tar")
        decompression_start = time()

        if specified_decompressed || experiment
            if naive
                if symmetric
                    decompress2dSymmetricNaiveWithMask(intermediate_folder, decompressed_folder, output, base_compressor, bcFolder)
                else
                    decompress2dNaive(intermediate_folder, decompressed_folder, output, base_compressor, bcFolder)
                end
            else
                if symmetric
                    decompressionList = decompress2dSymmetric(intermediate_folder, decompressed_folder, output, base_compressor, bcFolder)
                else
                    decompressionList = decompress2d(intermediate_folder, decompressed_folder, output, base_compressor, bcFolder)
                end
                dtv += decompressionList
            end
        end

        decompression_end = time()
        dt = decompression_end - decompression_start

        totalCompressionTime += ct
        totalDecompressionTime += dt

        if !skipStatistics
            metrics = evaluateCompression(Val(symmetric), data_folder, decompressed_folder, (slice_size[1], slice_size[2]), compressed_size, t )
            vertexMatching, cellMatching, bitrate, mseByRangeSquared, maxErrorByRange = metrics

            if !naive
                correct = true

                if vertexMatching[1,2] != 0 || vertexMatching[2,2] != 0
                    correct = false
                end

                if cellMatching[2] != 0 || cellMatching[4] != 0 || cellMatching[6] != 0
                    correct = false
                end

                if maxErrorByRange > eb
                    correct = false
                end

                if !correct

                    redirect_stdout(stdout_)
                    println("failed on slice $t")
                    println(vertexMatching)
                    println(cellMatching)
                    println(maxErrorByRange)

                    if csv != ""
                        if isfile(csv)
                            outf = open(csv, "a")
                        else
                            outf = open(csv, "w")
                        end

                        # If running multiple trials and writing to the same file, this is
                        # helpful for knowing if/what failed.
                        write(outf, "$dataset_name $target failed on slice $t")                
                    end

                    exit(1)
                end
            end

            totalBitrate += bitrate
            overallMaxErrorByRange = max(overallMaxErrorByRange, maxErrorByRange)

            if mseByRangeSquared != -1
                totalMSEByRangeSquared += mseByRangeSquared
            else
                numNoRange += 1
            end

            totalVertexMatching += vertexMatching
            totalCellMatching += cellMatching
        else
            totalBitrate += compressed_size*8/(slice_size[1]*slice_size[2])
        end

        redirect_stdout(stdout_)
        redirect_stderr(stderr_)

    end

    trialEnd = time()
    trialTime = trialEnd - trialStart

    averageBitrate = totalBitrate
    if specified_num_slices && !specified_slice_number
        averageBitrate = totalBitrate / num_slices
    end

    if symmetric
        ratio = 192.0/averageBitrate # 192 = 64 * 3 (number of bits in a symmetric tensor)
    else
        ratio = 256.0/averageBitrate
    end

    println("compression ratio: $ratio")

    if specified_input
        println("compression time: $totalCompressionTime")
    end

    if specified_decompressed
        println("decompression time: $totalDecompressionTime")
    end

    if experiment
        println("trial time: $trialTime")
    end

    if !skipStatistics
        averageMSEByRangeSquared = totalMSEByRangeSquared
        if specified_num_slices && !specified_slice_number
            averageMSEByRangeSquared /= (num_slices - numNoRange)
        end

        psnr = -10 * log(10, averageMSEByRangeSquared)

        println("psnr: $psnr")

        if naive
            if !symmetric
                println("false vertex eigenvalue: $(totalVertexMatching[1,2])")
                println("false vertex eigenvector: $(totalVertexMatching[2,2])")
                println("false cell eigenvalue: $(totalCellMatching[4])")
                println("false cell eigenvector: $(totalCellMatching[7])")
            end

            println("false cell degenerate points: $(totalCellMatching[2])")

        end
    end

    if csv != ""
        if isfile(csv)
            outf = open(csv, "a")
        else
            outf = open(csv, "w")

            # write header :(
            write(outf, "dataset,target,base compressor,error bound,comp. ratio,comp. time,decomp. time,trial time,#vertices,#cells,")
            write(outf, "PSNR,max error,false DP,false vertices e-val,false vertices e-vec,false cells e-val,false cells e-vec,")
            write(outf, "setup time,bc time,setup time 2,proc. points time,proc. DP time,proc. cells times,queue time,total proc. time,")
            write(outf, "comp. write time,lossless comp. time,comp. clean time,lossless decomp. time,decomp. tar time,decomp. load time,")
            write(outf, "base decomp. time,read base decomp. time,decomp. augment time,decomp. save time,decomp. cleanup time, cell visit counts\n")
        end

        # compute composite data

        # compute the name of the dataset
        if last(data_folder) == '/'
            data_folder = data_folder[1:lastindex(data_folder)-1]
        end

        numPoints = slice_size[1]*slice_size[2]
        numCells = (slice_size[1]-1)*(slice_size[2]-1)*2

        combined_visit_counts_str = replace( string(combined_visit_counts), "," => "" )

        # write the data :((

        # we provide the corresponding rows of writing the header as comments.

        # write(outf, "dataset,target,base compressor,error bound,comp. ratio,comp. time,decomp. time,trial time,#vertices,#cells,")
        write(outf, "$dataset_name,$target,$base_compressor,$eb,$ratio,$totalCompressionTime,$totalDecompressionTime,$trialTime,$numPoints,$numCells,")

        if !skipStatistics
            # write(outf, "PSNR,max error,false DP,false vertices e-val,false vertices e-vec,false cells e-val,false cells e-vec,")
            write(outf, "$psnr,$overallMaxErrorByRange,$(totalCellMatching[2]),")

            if symmetric
                write(outf, ",,,,")
            else
                write(outf, "$(totalVertexMatching[1,2]),$(totalVertexMatching[2,2]),$(totalCellMatching[4]),$(totalCellMatching[7]),")
            end
        else
            write(outf, ",,,,,,,")
        end

        if !naive
            if symmetric
                ctv6 = ""
            else
                ctv6 = string(ctv[6])
            end

            # write(outf, "setup time,bc time,setup time 2,proc. points time,proc. DP time,proc. cells times,queue time,total proc. time,")
            write(outf, "$(ctv[1]),$(ctv[2]),$(ctv[3]),$(ctv[4]),$(ctv[5]),$ctv6,$(ctv[7]),$(ctv[8]),")

            # write(outf, "comp. write time,lossless comp. time,comp. clean time,lossless decomp. time,decomp. tar time,decomp. load time,")
            write(outf, "$(ctv[9]),$(ctv[10]),$(ctv[11]),$(dtv[1]),$(dtv[2]),$(dtv[3]),")

            # write(outf, "base decomp. time,read base decomp. time,decomp. augment time,decomp. save time,decomp. cleanup time, cell visit counts\n")
            write(outf, "$(dtv[4]),$(dtv[5]),$(dtv[6]),$(dtv[7]),$(dtv[8]),")
            if !skipStatistics
                write(outf, combined_visit_counts_str)
            end
        end
        write(outf,"\n")

    end

    if naive
        removeIfExists("$output/slice")
    end

    if !specified_intermediate
        removeIfExists("$intermediate_name.tar.zst")
    end

    if !specified_decompressed
        removeIfExists("$decompressed_name")
    end

end

main()
