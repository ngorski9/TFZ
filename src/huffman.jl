module huffman

using DataStructures

export huffmanEncode
export huffmanDecode

# a huffman tree will simply be an array of these
struct HuffmanNode
    elt::Int64
    left::Int64
    right::Int64
end

function makeHuffmanTree(frequencies::Dict{Int64,Int64}, symbols::Array{Int64})
    numSymbols = length(symbols)
    heapList::Array{Tuple{Int64,Int64,Int64}} = Array{Tuple{Int64,Int64,Int64}}(undef,numSymbols)

    for i in 1:numSymbols
        heapList[i] = (symbols[i],-1,frequencies[symbols[i]])
    end

    heap = BinaryHeap{Tuple{Int64,Int64,Int64}}(Base.By(last),heapList)
    
    numNodes = 2*numSymbols-1
    tree = Array{HuffmanNode}(undef,numNodes)

    iter = 1
    while iter <= numNodes
        small1 = pop!(heap)
        small2 = pop!(heap)
        if small1[2] == -1
            tree[iter] = HuffmanNode(small1[1],-1,-1)
            small1Index = iter
            iter += 1
        else
            small1Index = small1[2]
        end

        if small2[2] == -1
            tree[iter] = HuffmanNode(small2[1],-1,-1)
            small2Index = iter
            iter += 1
        else
            small2Index = small2[2]
        end

        tree[iter] = HuffmanNode(-1,small1Index,small2Index)
        push!(heap, (-1,iter,small1[3]+small2[3]))
        iter += 1
    end

    return tree
end

function getHuffmanCodes(tree::Array{HuffmanNode})
    node = length(tree)
    codeSoFar::Array{Int8} = []
    codes = Dict{Int64,Array{Int8}}()
    recursiveGetHuffmanCodes(tree, node, codes, codeSoFar)
    return codes
end

function recursiveGetHuffmanCodes(tree::Array{HuffmanNode}, node::Int64, codes::Dict{Int64,Array{Int8}}, codeSoFar::Array{Int8})
    if tree[node].left == -1 # eg if this is a leaf node
        codes[tree[node].elt] = copy(codeSoFar)
    else
        push!(codeSoFar,0)
        recursiveGetHuffmanCodes(tree, tree[node].left, codes, codeSoFar)
        pop!(codeSoFar)
        push!(codeSoFar,1)
        recursiveGetHuffmanCodes(tree, tree[node].right, codes, codeSoFar)
        pop!(codeSoFar)
    end
end

# Codes can only be integer type. Currently, must be Int64 but we'll change that if we need to.
function huffmanEncode(symbols)
    frequencies = Dict{Int64, Int64}()
    symbolTypes::Array{Int64} = Array{Int64}([])

    for s in symbols
        sI64 = Int64(s)
        if haskey(frequencies, sI64)
            frequencies[sI64] += 1
        else
            frequencies[sI64] = 1
            push!(symbolTypes, sI64)
        end
    end

    numSymbolTypes = length(symbolTypes)
    header::Array{Int64} = Array{Int64}(undef, 2*numSymbolTypes+1)
    header[1] = numSymbolTypes
    for i in 1:numSymbolTypes
        header[2*i] = symbolTypes[i]
        header[2*i+1] = frequencies[symbolTypes[i]]
    end

    headerBytes::Array{UInt8} = reinterpret(UInt8, header)

    if numSymbolTypes == 1
        return headerBytes
    end

    tree = makeHuffmanTree(frequencies, symbolTypes)
    codes = getHuffmanCodes(tree)

    output::Array{UInt8} = []
    nextOut::UInt8 = 0
    nextOutPosition = 0
    for s in symbols
        code = codes[s]
        for bit in code
            if bit == 1
                nextOut |= 1 << nextOutPosition
            end

            nextOutPosition += 1
            if nextOutPosition == 8
                push!(output, nextOut)
                nextOut = 0
                nextOutPosition = 0
            end
        end
    end
    push!(output, nextOut)

    return cat(headerBytes, output, dims=(1,1))

end

function huffmanDecode(bytes::Array{UInt8})
    numSymbolTypes = reinterpret(Int64, bytes[1:8])[1]
    header = reinterpret(Int64, bytes[9:8+16*numSymbolTypes])
    body = bytes[9+16*numSymbolTypes:length(bytes)]

    frequencies = Dict{Int64, Int64}()
    symbolTypes::Array{Int64} = Array{Int64}(undef, numSymbolTypes)
    totalLength = 0

    for i in 1:numSymbolTypes
        symbolTypes[i] = header[2*i-1]
        frequencies[header[2*i-1]] = header[2*i]
        totalLength += header[2*i]
    end

    if numSymbolTypes == 1
        return symbolTypes[1] * ones(Int64, totalLength)
    end

    tree = makeHuffmanTree(frequencies, symbolTypes)
    root = length(tree)

    out = Array{Int64}(undef, totalLength)

    currentByteNumber = 1
    currentByte = body[1]
    currentPosition = 0

    for i in 1:totalLength
        node = root
        while tree[node].left != -1 # test if it is an internal node
            if currentByte & (1 << currentPosition) == 0
                node = tree[node].left
            else
                node = tree[node].right
            end
            currentPosition += 1
            if currentPosition == 8
                currentPosition = 0
                currentByteNumber += 1
                currentByte = body[currentByteNumber]
            end
        end
        out[i] = tree[node].elt
    end

    return out

end

end