module conicUtils

using ..utils

export interpolationConic
export add
export sub
export center
export discriminant
export evaluate
export quadraticFormula
export to_string
export intersectWithStandardFormLine
export gradient
export normalizedGradient
export tangentDerivative
export conicEquation
export curvature
export classifyAndReturnCenter

export HYPERBOLA
export ELLIPSE
export INTERSECTING_LINES
export PARABOLA
export PARALLEL_INNER
export PARALLEL_OUTER
export SIDEWAYS_PARABOLA
export PARALLEL_INNER_HORIZONTAL
export PARALLEL_OUTER_HORIZONTAL
export POINT
export LINE
export LINE_NO_REGION
export HORIZONTAL_LINE
export VERTICAL_LINE
export EMPTY

# we only deal with the cases that can actually come up.
const HYPERBOLA = 0
const ELLIPSE = 1
const INTERSECTING_LINES = 3
const PARABOLA = 2
const PARALLEL_INNER = 4
const PARALLEL_OUTER = 5
const SIDEWAYS_PARABOLA = 6
const PARALLEL_INNER_HORIZONTAL = 7
const PARALLEL_OUTER_HORIZONTAL = 8
const POINT = 9
const LINE = 10
const LINE_NO_REGION = 11
const HORIZONTAL_LINE = 12
const VERTICAL_LINE = 13
const EMPTY = 14 # entire domain or nothing

# This entire file assumes the interpolation paradigm where we are linearly interpolating three tensors in a triangle.
# We assume that the first one is (0,0), the second (1,0), and the third (0,1). However, the goal here is just to determine
# whether one of the ellipses that we need is actually inside of the cell, so the locations of the vertices do not actually matter.

struct conicEquation
    A::Float64
    B::Float64
    C::Float64
    D::Float64
    E::Float64
    F::Float64
end

function interpolationConic(u1::Float64, u2::Float64, u3::Float64)
    return conicEquation( (u2-u1)^2, 2*(u2-u1)*(u3-u1), (u3-u1)^2, 2*u1*(u2-u1), 2*u1*(u3-u1), u1^2 )
end

function add(eq1::conicEquation, eq2::conicEquation)
    return conicEquation( eq1.A+eq2.A, eq1.B+eq2.B, eq1.C+eq2.C, eq1.D+eq2.D, eq1.E+eq2.E, eq1.F+eq2.F )
end

function sub(eq1::conicEquation, eq2::conicEquation)
    return conicEquation( eq1.A-eq2.A, eq1.B-eq2.B, eq1.C-eq2.C, eq1.D-eq2.D, eq1.E-eq2.E, eq1.F-eq2.F )
end

function center(eq::conicEquation)
    disc = -discriminant(eq)
    return ( (eq.B*eq.E - 2*eq.C*eq.D)/disc, (eq.B*eq.D - 2*eq.A*eq.E)/disc )
end

function discriminant(eq::conicEquation)
    return eq.B^2 - 4*eq.A*eq.C
end

function evaluate(eq::conicEquation, x::Float64, y::Float64)
    return eq.A*x^2 + eq.B*x*y + eq.C*y^2 + eq.D*x + eq.E*y + eq.F
end

# returns the gradient of the conic equation as a function of x and y at the point (x,y)
function gradient(eq::conicEquation,x::Float64,y::Float64)
    return (2*eq.A*x+eq.B*y+eq.D, eq.B*x+2*eq.C*y+eq.E)
end

function normalizedGradient(eq::conicEquation,x::Float64,y::Float64)
    grad = gradient(eq,x,y)
    norm = sqrt(grad[1]^2+grad[2]^2)
    if norm == 0.0
        return (0.0,0.0)
    end
    return (grad[1]/norm, grad[2]/norm)
end

# returns a normalized vector that is tangent to the curve of eq=0 at the point (x,y)
# (note that F does not affect anything here)
function tangentDerivative(eq::conicEquation,x::Float64,y::Float64)
    grad = normalizedGradient(eq,x,y)
    return (-grad[2],grad[1])
end

function tangentDerivative(grad::Tuple{Float64,Float64})
    return (-grad[2],grad[1])
end

# returns the curvature of given conic equation at point (x,y)
function curvature(eq::conicEquation,x::Float64,y::Float64)
    Fx = 2*eq.A*x + eq.B*y + eq.D
    Fy = eq.B*x + 2*eq.C*y + eq.E
    Fxx = 2*eq.A
    Fxy = eq.B
    Fyy = 2*eq.C

    return abs( Fxx*Fy^2 - 2*Fx*Fy*Fxy + Fyy*Fx^2 ) / ( Fx^2 + Fy^2 )^1.5
end

# intersect the conic with the line ax + by + c = 0
function intersectWithStandardFormLine(eq::conicEquation, a::Float64, b::Float64, c::Float64)
    if b != 0.0
        qa = a/b
        qc = c/b

        # find x which is roots of quadratic equation formed by substituting y = (-ax-c)/b
        intersections_x = quadraticFormula( eq.A - qa*eq.B + (qa^2)*eq.C, -qc*eq.B + 2*qa*qc*eq.C + eq.D - qa*eq.E, (qc^2)*eq.C - qc*eq.E + eq.F )
        if intersections_x[1] == Inf
            return ((Inf,Inf),(Inf,Inf))
        else
            x1 = intersections_x[1]
            y1 = (-a*x1-c)/b

            if intersections_x[2] == Inf
                return ((x1,y1),(Inf,Inf))
            else
                x2 = intersections_x[2]
                y2 = (-a*x2-c)/b
                return ((x1,y1),(x2,y2))
            end
        end
    else
        qc = -c/a
        intersections_y = quadraticFormula( eq.C, eq.B*qc + eq.E, eq.A*qc*qc + eq.D*qc + eq.F )
        if intersections_y[1] == Inf
            return ((Inf,Inf),(Inf,Inf))
        elseif intersections_y[2] == Inf
            return ((qc,intersections_y[1]),(Inf,Inf))
        else
            return ((qc,intersections_y[1]),(qc,intersections_y[2]))
        end
    end
end

function quadraticFormula(a::Float64, b::Float64, c::Float64)
    a2 = a
    b2 = b
    c2 = c

    if abs(a) >= abs(b) && abs(a) >= abs(c)
        a2 = sign(a)
        b2 /= abs(a)
        c2 /= abs(a)
    elseif abs(c) >= abs(a) && abs(c) >= abs(a)
        a2 /= abs(c)
        b2 /= abs(c)
        c2 = sign(c)
    else
        a2 /= abs(b)
        b2 = sign(b)
        c2 /= abs(b)
    end

    if (b != 0.0 && abs(a)/abs(b) < ϵ) || (c != 0.0 && abs(a)/abs(c) < ϵ)
        a2 = 0.0
    end

    if (a != 0.0 && abs(b)/abs(a) < ϵ) || (c != 0.0 && abs(b)/abs(c) < ϵ)
        b2 = 0.0
    end

    if (a != 0.0 && abs(c)/abs(a) < ϵ) || (b != 0.0 && abs(c)/abs(b) < ϵ)
        c2 = 0.0
    end

    a = a2
    b = b2
    c = c2

    disc = b^2-4*a*c

    if disc != 0 && disc/((abs(a)+abs(b)+abs(c))^2) < -ϵ
        return (Inf,Inf)
    elseif disc != 0 && disc/((abs(a)+abs(b)+abs(c))^2) < ϵ
        disc = 0.0
    end

    sr = sqrt(disc)

    if a == 0.0
        if b == 0.0
            return (Inf, Inf)
        else
            return (-c/b,-c/b)
        end
    elseif a > 0.0
        return ( (-b - sr) / (2*a), (-b + sr) / (2*a) )
    else
        return ( (-b + sr) / (2*a), (-b - sr) / (2*a) )
    end

end

function to_string(eq::conicEquation,m=1.0)
    return string(m*eq.A) * "x^2 + " * string(m*eq.B) * "xy + " * string(m*eq.C) * "y^2 + " * string(m*eq.D) * "x + " * string(m*eq.E) * "y + " * string(m*eq.F)
end

function classifyAndReturnCenter(eq::conicEquation)
    disc = discriminant(eq)
    if disc < -ϵ
        center = ((2*eq.C*eq.D - eq.B*eq.E)/disc, (2*eq.A*eq.E - eq.B*eq.D)/disc)
        if isClose(evaluate(eq, center[1], center[2]),0.0)
            return (POINT, center)
        else
            return (ELLIPSE, center)
        end
    elseif disc > ϵ
        center = ((2*eq.C*eq.D - eq.B*eq.E)/disc, (2*eq.A*eq.E - eq.B*eq.D)/disc)
        if isClose(evaluate(eq, center[1], center[2]),0.0)
            return (INTERSECTING_LINES, center)
        else
            return (HYPERBOLA, center)
        end
    else
        if eq.B != 0 || eq.A != 0
            x1 = (-eq.B*eq.E - 2*eq.A*eq.D) / (4*eq.A*eq.A + eq.B^2)
            center = (x1,0.0)
            grad = gradient(eq, x1, 0.0)
            if grad[1] == 0.0 && grad[2] == 0.0
                if isClose(evaluate(eq, x1, 0.0), 0.0)
                    return (LINE_NO_REGION, center)
                elseif eq.A > 0
                    return (PARALLEL_OUTER, center)
                else
                    return (PARALLEL_INNER, center)
                end
            else
                return (PARABOLA, center)
            end
        else
            if eq.C != 0
                center = (0.0,-eq.E/(2*eq.C))
                if eq.D == 0
                    if isClose(eq.E^2 - 4*eq.C*eq.F, 0.0)
                        return (LINE_NO_REGION, center)
                    elseif eq.C > 0
                        return (PARALLEL_OUTER_HORIZONTAL, center)
                    else
                        return (PARALLEL_INNER_HORIZONTAL, center)
                    end
                else
                    return (SIDEWAYS_PARABOLA, center)
                end
            else
                if eq.D == 0.0
                    if eq.E == 0.0
                        return (EMPTY, (0.0,0.0))
                    else
                        return (VERTICAL_LINE, (0.0,-eq.F/eq.E))
                    end
                elseif eq.E == 0.0
                    return (HORIZONTAL_LINE, (-eq.F/eq.D,0.0))
                else
                    return (LINE, (-eq.F/eq.D,0.0))
                end
            end
        end
    end

end

end