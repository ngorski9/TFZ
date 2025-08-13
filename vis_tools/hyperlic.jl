using LinearAlgebra
using DataStructures
using Random
using WriteVTK

const trisector = 2
const wedge = 1

struct TensorField
    size_x::Int64
    size_y::Int64
    a::Matrix{Float64}
    b::Matrix{Float64}
    c::Matrix{Float64}
    d::Matrix{Float64}
end

struct Tensor
    a::Float64
    b::Float64
    c::Float64
    d::Float64
end

function isZero(t::Tensor)
    return t.a == 0.0 && t.b == 0.0 && t.c == 0.0 && t.d == 0.0
end

struct Vec
    u::Float64
    v::Float64
end

function mag(u::Float64, v::Float64)
    return sqrt(u^2+v^2)
end

function inBounds(tf::TensorField, v::Vec)
    return v.u >= 1 && v.u <= tf.size_x && v.v >= 1 && v.v <= tf.size_y
end

function dot(a::Vec, b::Vec)
    return a.u*b.u + a.v*b.v
end

function Base.:+(a::Vec, b::Vec)
    return Vec(a.u+b.u,a.v+b.v)
end

function Base.:*(a::Float64, b::Vec)
    return Vec(a*b.u, a*b.v)
end

function Base.:*(a::Int64, b::Vec)
    return Vec(a*b.u, a*b.v)
end

function Base.:*(a::Vec, b::Float64)
    return Vec(a.u*b, a.v*b)
end

function Base.:*(a::Vec, b::Int64)
    return Vec(a.u*b, a.v*b)
end

function Base.:/(a::Vec, b::Float64)
    return Vec(a.u/b, a.v/b)
end

function Base.:/(a::Vec, b::Int64)
    return Vec(a.u/b, a.v/b)
end

function Base.:-(a::Vec)
    return Vec(-a.u,-a.v)
end

function extractCP(tf::TensorField, cp_array, ctypes_array, cp_frobenius_array, scale)
    # identify critical points
    for j in 1:tf.size_y-1
        for i in 1:tf.size_x-1
            for k in 0:1
                
                if k == 0
                    # bottom
                    x1, y1 = (i,j)
                    x2, y2 = (i+1,j)
                    x3, y3 = (i,j+1)
                else
                    # top
                    x1, y1 = (i,j+1)
                    x2, y2 = (i+1,j)
                    x3, y3 = (i+1,j+1)
                end

                Δ1 = tf.a[x1,y1] - tf.d[x1,y1]
                Δ2 = tf.a[x2,y2] - tf.d[x2,y2]
                Δ3 = tf.a[x3,y3] - tf.d[x3,y3]

                F1 = tf.b[x1,y1] + tf.c[x1,y1]
                F2 = tf.b[x2,y2] + tf.c[x2,y2]
                F3 = tf.b[x3,y3] + tf.c[x3,y3]

                l12 = Δ1*F2 - Δ2*F1
                l23 = Δ2*F3 - Δ3*F2
                l31 = Δ3*F1 - Δ1*F3

                cp = 0

                if l12 < 0 && l23 < 0 && l31 < 0
                    cp = trisector
                elseif l12 > 0 && l23 > 0 && l31 > 0
                    cp = wedge
                end

                if cp == 1 || cp == 2
                    quot = 1.0
                    for val in (Δ1, Δ2, Δ3, F1, F2, F3)
                        if 0 < abs(val) < quot
                            quot = abs(val)
                        end
                    end

                        Δ1 /= quot
                        Δ2 /= quot
                        Δ3 /= quot
                        F1 /= quot
                        F2 /= quot
                        F3 /= quot

                    mat = [ Δ1 Δ2 Δ3 ; F1 F2 F3 ; 1 1 1 ]

                    μ = (mat^-1) * [0 ; 0 ; 1]

                    cx = μ[1] * Float64(x1) + μ[2] * Float64(x2) + μ[3] * Float64(x3)
                    cy = μ[1] * Float64(y1) + μ[2] * Float64(y2) + μ[3] * Float64(y3)

                    t = interpolate(tf, Vec(cx,cy))

                    cx = (cx - 1)*scale + 1
                    cy = (cy - 1)*scale + 1

                    frobenius = sqrt(t.a^2 + t.b^2 + t.c^2 + t.d^2)
                    push!(cp_array, (cx,cy))
                    push!(ctypes_array, cp)
                    push!(cp_frobenius_array, frobenius)

                end

            end
        end
    end
end

function randomNoise(array)
    width, height = size(array)
    for j in 1:height
        for i in 1:width
            array[i,j] = rand()
        end
    end
end

function randomCircles(array, radius)
    width, height = size(array)

    for i in 1:width*height/radius
        centerX = Int64(round(rand() * (width-1))+1)
        centerY = Int64(round(rand() * (height-1))+1)
        intensity = rand()
        for xo in -radius:radius
            for yo in -radius:radius
                if xo^2 + yo^2 <= radius && 1 <= centerX + xo <= width && 1 <= centerY + yo <= height
                    array[centerX+xo, centerY+yo] = intensity
                end
            end
        end

    end

end

function vector(A::Tensor, dual=false, scale::Float64=0.0)

    if dual

        Δ = (A.a - A.d) / 2
        F = (A.b + A.c) / 2
        r = (A.c - A.b) / 2
        s = mag(Δ,F)
        if s == 0.0
            return Vec(0.0,0.0)
        end

        if r > 0
            mag1 = mag(F,s-Δ)
            return Vec(F/mag1, (s-Δ)/mag1)
        else
            mag1 = mag(F,-s-Δ)
            return Vec(F/mag1, (-s-Δ)/mag1)
        end

    else
        Δ = (A.a - A.d) / 2
        F = A.b
        s = mag(Δ,F)
        if s == 0.0
            return Vec(0.0,0.0)
        end
        mag2 = mag(F,s-Δ)
        return Vec(F/mag2,(s-Δ)/mag2)
    end

end

function interpolate(tf::TensorField,v::Vec)
    x = v.u
    y = v.v
    xFloor = Int64(floor(x))
    yFloor = Int64(floor(y))
    xFrac = x - floor(x)
    yFrac = y - floor(y)
    if xFrac - yFrac <= 1
        # bottom cell
        a = tf.a[xFloor + 1, yFloor] * xFrac + tf.a[xFloor, yFloor + 1] * yFrac + tf.a[xFloor, yFloor] * (1 - xFrac - yFrac)
        b = tf.b[xFloor + 1, yFloor] * xFrac + tf.b[xFloor, yFloor + 1] * yFrac + tf.b[xFloor, yFloor] * (1 - xFrac - yFrac)
        c = tf.c[xFloor + 1, yFloor] * xFrac + tf.c[xFloor, yFloor + 1] * yFrac + tf.c[xFloor, yFloor] * (1 - xFrac - yFrac)
        d = tf.d[xFloor + 1, yFloor] * xFrac + tf.d[xFloor, yFloor + 1] * yFrac + tf.d[xFloor, yFloor] * (1 - xFrac - yFrac)

        return Tensor(a,b,c,d)
    else
        # top cell
        a = tf.a[xFloor, yFloor + 1] * (1 - xFrac) + tf.a[xFloor + 1, yFloor] * (1 - yFrac) + tf.a[xFloor + 1, yFloor + 1] * (xFrac + yFrac - 1)
        a = tf.b[xFloor, yFloor + 1] * (1 - xFrac) + tf.b[xFloor + 1, yFloor] * (1 - yFrac) + tf.b[xFloor + 1, yFloor + 1] * (xFrac + yFrac - 1)
        a = tf.c[xFloor, yFloor + 1] * (1 - xFrac) + tf.c[xFloor + 1, yFloor] * (1 - yFrac) + tf.c[xFloor + 1, yFloor + 1] * (xFrac + yFrac - 1)
        a = tf.d[xFloor, yFloor + 1] * (1 - xFrac) + tf.d[xFloor + 1, yFloor] * (1 - yFrac) + tf.d[xFloor + 1, yFloor + 1] * (xFrac + yFrac - 1)

        return Tensor(a,b,c,d)
    end

end

# expensive lol
function rk4_vector(tf::TensorField, x::Vec, δ::Float64, lastVector::Vec, dual::Bool, evecScale::Float64=0.0)

    t1 = interpolate(tf, x)
    e1 = vector(t1, dual, evecScale)
    if dot(lastVector, e1) < 0
        e1 *= -1
    end
    a = δ * e1
    xa = x + 0.5a
    if !inBounds(tf, xa)
        return Vec(Inf,Inf)
    end

    t2 = interpolate(tf, xa)
    e2 = vector(t2, dual, evecScale)
    if dot(lastVector, e2) < 0
        e2 *= -1
    end
    b = δ * e2
    xb = x + 0.5b
    if !inBounds(tf, xb)
        return Vec(Inf,Inf)
    end

    t3 = interpolate(tf, xb)
    e3 = vector(t3, dual, evecScale)
    if dot(lastVector, e3) < 0
        e3 *= -1
    end
    c = δ * e3
    xc = x + c # yes this is correct. This should not be x + 0.5c
    if !inBounds(tf, xc)
        return Vec(Inf,Inf)
    end

    t4 = interpolate(tf, xc)
    e4 = vector(t4, dual, evecScale)
    if dot(lastVector, e4) < 0
        e4 *= -1
    end
    d = δ * e4

    if dot(lastVector, a+2b+2c+d) < 0
        println("something is wrong")
        println((e1,e2,e3,e4))
        println((a,b,c,d))
        println(lastVector)
        println((dot(lastVector, a), dot(lastVector,b), dot(lastVector,c), dot(lastVector,d)))
    end

    return (a + 2b + 2c + d) / 6

end

function to_pixel(x, scale)
    return (Int64(floor( (x.u-1)*scale+1 )), Int64(floor( (x.v-1)*scale+1 )))
end

function to_vec(px_x, px_y, scale)
    return 
end

function pixel_near(px,list)
    for j in -2:2
        for i in -2:2
            if (px[1]+i, px[2]+j) in list
                return true
            end
        end
    end
    return false
end

function main()
    Random.seed!(2)
    t1 = time()

    if length(ARGS) == 0
        folder = "../output/reconstructed"
        saveName = "../LIC2"
        size = (66, 108)
        scale = 3
        evecScale=0.0
        power = 1.0
    else
        try
            folder = ARGS[1]
            saveName = ARGS[2]
            size = (parse(Int64,ARGS[3]),parse(Int64,ARGS[4]))
            scale = parse(Int64,ARGS[5])
            evecScale = 0.0
            if length(ARGS) >= 6
                power = parse(Float64,ARGS[6])
            else
                power = 1.0
            end
        catch
            println("ERROR: Format is folder saveName size[1] size[2] scale (power)")
            exit()
        end
    end

    # don't think I'm gonna use this again.
    alt_folder = ""
    load_lic = ""
    save_lic = ""

    # old settings: num_steps: 60
    # max_path_tracing = 60
    # block_size = 20
    # delta: 0.1

    num_steps = 180 # Interpolation length
    max_path_tracing = 180 # kill after a certain number of steps to avoid loops
    block_size = 20
    hit_threshold = 8
    # δ = 0.01
    δ = 0.1
    asymmetric = false

    # load and set up the tensor field
    a = time()
    a_file = open("$folder/row_1_col_1.dat", "r")
    b_file = open("$folder/row_1_col_2.dat", "r")
    d_file = open("$folder/row_2_col_2.dat", "r")

    a_array = reshape( reinterpret( Float64, read(a_file) ), size )
    b_array = reshape( reinterpret( Float64, read(b_file) ), size )
    d_array = reshape( reinterpret( Float64, read(d_file) ), size )

    if asymmetric
        c_file = open("$folder/row_2_col_1.dat", "r")
        c_array = reshape( reinterpret( Float64, read(c_file) ), size )
    else
        c_array = b_array
    end

    # for i in 1:dims[2]
    #     println((a_array[1,i],b_array[1,i],c_array[1,i]))
    # end

    tf = TensorField(size[1], size[2], a_array, b_array, c_array, d_array)

    cp_locs::Array{Tuple{Float64,Float64}} = Array{Tuple{Float64,Float64}}(undef, 0)
    cp_frobenius::Array{Float64} = Array{Float64}(undef,0)
    cp_types::Array{Int64} = Array{Int64}(undef,0)

    # extract critical points
    extractCP(tf,cp_locs, cp_types, cp_frobenius,scale)

    # set up the input texture and other images

    imageSize = ( (size[1]-1)*scale, (size[2]-1)*scale )
    noise = zeros(Float64, imageSize )
    # randomNoise(noise)
    randomCircles(noise, scale-1)
    for j in 1:imageSize[2]
        for i in 2:imageSize[1]
            xo = (i-0.5)/scale+1.0
            yL = (j-0.7)/scale+1.0
            yH = (j-0.3)/scale+1.0

            if isZero(interpolate(tf, Vec(xo, yL) )) && isZero(interpolate(tf, Vec(xo,yH)))
                noise[i,j] = 0.0
            end

        end
    end

    finalImage = zeros(Float64, imageSize)
    frobenius = zeros(Float64, imageSize)
    majorEigval = zeros(Float64, imageSize)

    if load_lic != ""
        inf = open(load_lic, "r")
        inBytes = reshape(reinterpret(Float64,read(inf)),imageSize)
        for j in 1:imageSize[2]
            for i in 1:imageSize[1]
                finalImage[i,j] = inBytes[i,j]
            end
        end
    else
        # finalImage = noise
        numHits = zeros(Int64, imageSize)

        # iterate and create streamlines
        blocks_dims = ( Int64(ceil(imageSize[1] / block_size)), Int64(ceil(imageSize[2] / block_size)) )

        numSkips = 0

        for y in 1:block_size
            for x in 1:block_size
                println((x,y,numSkips))
                for by = 1:blocks_dims[2]
                    for bx in 1:blocks_dims[1]

                        px = (bx-1) * block_size + x
                        py = (by-1) * block_size + y

                        if px <= imageSize[1] && py <= imageSize[2]

                            # compute the frobenius mse at this pixel
                            center = Vec( (px + 0.5 - 1.0) / scale + 1, (py + 0.5 - 1.0) / scale + 1 )
                            tCenter = interpolate(tf, center)
                            frobenius[ px, py ] = sqrt( tCenter.a^2 + tCenter.b^2 + tCenter.c^2 + tCenter.d^2 )
                            majorEigval[px, py] = (tCenter.a+tCenter.d)/2 + sqrt( (tCenter.d-tCenter.a)^2 + (tCenter.b+tCenter.c)^2 )/2

                            # skip the LIC here if a sufficient number of hits have been hit already.
                            if numHits[px,py] >= hit_threshold
                                numSkips += 1
                                continue
                            end

                            seed = Vec( (px + rand() - 1) / scale + 1, (py + rand() - 1) / scale + 1 )

                            # compute pixels in the path around the seed point

                            pixels = Deque{Tuple{Int64, Int64}}()
                            push!(pixels, (px, py))

                            # Compute initial eigenvectors for seeding

                            tRoot = interpolate(tf, seed)
                            evecRoot = vector(tRoot,asymmetric)

                            # forward pass
                            num_forward = 0
                            xf = seed

                            vf = evecRoot
                            for step in 1:num_steps
                                vf = rk4_vector(tf, xf, δ, vf, asymmetric, evecScale)
                                xf = xf + vf
                                if !inBounds(tf, xf)
                                    break
                                end
                                push!(pixels, to_pixel(xf,scale))
                                num_forward += 1
                                # if pixel_near(xf, wedge_pixels)
                                #     break
                                # end
                            end

                            # backward pass
                            num_backward = 0
                            xb = seed
                            vb = -evecRoot
                            for step in 1:num_steps
                                vb = rk4_vector(tf, xb, δ, vb, asymmetric, evecScale)
                                xb = xb + vb
                                if !inBounds(tf, xb)
                                    break
                                end
                                pushfirst!(pixels, to_pixel(xb,scale))
                                # if pixel_near(first(pixels), wedge_pixels)
                                #     break
                                # end
                                num_backward += 1
                            end
                            
                            intensity = 0

                            for pixel in pixels
                                intensity += noise[pixel[1],pixel[2]]
                            end

                            finalImage[ px, py ] += intensity / length(pixels)
                            numHits[ px, py ] += 1
                        
                            # propagate intensities forward

                            pixels_f = collect(pixels)
                            head_f = num_forward # the number of pixels in front of the one that we are currently working with.
                            tail_f = num_backward # the number of pixels behind the one that we are currently working with.
                            tail_f_position = 1
                            intensity_f = intensity
                            steps = 0 # kill counter to avoid getting stuck in a loop

                            if num_forward < num_steps
                                at_edge = true
                            else
                                at_edge = false
                            end

                            index = 0
                            for pixel in pixels_f
                                index += 1
                                if index <= num_backward
                                    continue
                                end

                                steps += 1
                                if steps >= max_path_tracing
                                    break
                                end

                                if !at_edge
                                    vf = rk4_vector(tf, xf, δ, vf, asymmetric, evecScale)
                                    xf = xf + vf
                                    if !inBounds(tf, xf)
                                        at_edge = true
                                    end
                                end

                                if tail_f < num_steps
                                    tail_f += 1
                                else
                                    furthestTail = pixels_f[tail_f_position]
                                    intensity_f -= noise[ furthestTail[1], furthestTail[2] ]
                                    tail_f_position += 1
                                end

                                if at_edge
                                    head_f -= 1
                                else
                                    nextPixel = to_pixel(xf,scale)
                                    push!(pixels_f, nextPixel)
                                    intensity_f += noise[ nextPixel[1], nextPixel[2] ]
                                    # if pixel_near(nextPixel, wedge_pixels)
                                    #     at_edge = true
                                    # end
                                end

                                finalImage[ pixel[1], pixel[2] ] += intensity_f / ( head_f + tail_f + 1 )
                                numHits[ pixel[1], pixel[2] ] += 1
                            end

                            # propagate intensities backward

                            pixels_b = reverse(collect(pixels))
                            head_b = num_backward # the number of pixels in front of the one that we are currently working with.
                            tail_b = num_forward # the number of pixels behind the one that we are currently working with.
                            tail_b_position = 1
                            intensity_b = intensity

                            if num_backward < num_steps
                                at_edge = true
                            else
                                at_edge = false
                            end

                            index = 0
                            for pixel in pixels_b
                                index += 1
                                if index <= num_forward
                                    continue
                                end

                                steps += 1
                                if steps >= max_path_tracing
                                    break
                                end

                                if !at_edge
                                    vb = rk4_vector(tf, xb, δ, vb, asymmetric, evecScale)
                                    xb = xb + vb
                                    if !inBounds(tf, xb)
                                        at_edge = true
                                    end
                                end

                                if tail_b < num_steps
                                    tail_b += 1
                                else
                                    furthestTail = pixels_b[tail_b_position]
                                    intensity_b -= noise[ furthestTail[1], furthestTail[2] ]
                                    tail_b_position += 1
                                end

                                if at_edge
                                    head_b -= 1
                                else
                                    nextPixel = to_pixel(xb,scale)
                                    push!(pixels_b, nextPixel)
                                    intensity_b += noise[ nextPixel[1], nextPixel[2] ]
                                    # if pixel_near(nextPixel, wedge_pixels)
                                    #     at_edge = true
                                    # end
                                end

                                finalImage[ pixel[1], pixel[2] ] += intensity_b / ( head_b + tail_b + 1 )
                                numHits[ pixel[1], pixel[2] ] += 1

                            end

                        end # end if px and py are valid coordinates

                    end # end for bx in 1:blocks_dims[1]
                end # end for by in 1:blocks_dims[2]
            end # end for x in 1:block_size
        end # end for y in 1:block_size

        for j in 1:imageSize[2]
            for i in 1:imageSize[1]
                finalImage[i,j] /= numHits[i,j]
            end
        end
    end

    t2 = time()
    println(t2-t1)

    if save_lic != ""
        outf = open(save_lic, "w")
        write(outf, finalImage)
        close(outf)
    end

    # visualize
    finalImage = finalImage .^ power

    vtk_grid(saveName, 0:1:imageSize[1]-1,0:1:imageSize[2]-1,0:1:0) do vtk
        vtk["frobenius"] = frobenius
        vtk["LIC"] = finalImage
        vtk["Major Eigval"] = majorEigval
    end

    cp_colors::Array{Tuple{UInt8,UInt8,UInt8}} = Array{Tuple{UInt8,UInt8,UInt8}}(undef, 2)

    cp_colors[trisector] = ( 255, 255, 255 )
    cp_colors[wedge] = ( 255, 140, 198 )

    if length(cp_locs) > 0
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
        end
    else
        println("no degenerate points")
    end

end

main()
