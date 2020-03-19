using LinearAlgebra
using SparseArrays


function P1(r, t, X)
    """
    First step in camera projection
    """
    θ = norm(r)
    k = r / θ
    return cos(θ)*X + sin(θ)*cross(k,X) + (1 - cos(θ))*dot(k,X)*k + t
end


function P2(X)
    """
    Second step in camera projection
    """
    return - X[1:2] / X[3]
end


function JP1(r, X)
    """
    Jacobian of the first step of the projection
    """
    θ = norm(r)
    c, s = cos(θ), sin(θ)
    k = r / θ
    x, y, z = X
    kx, ky, kz = k
    d = dot(k, X)
    JP1 = spzeros(6,12)

    JP1[1, 1] = c + (1 - c)*kx^2
    JP1[1, 2] = - s*kz + (1 - c)*ky*kx
    JP1[1, 3] = s*ky + (1 - c)*kz*kx
    JP1[1, 4] = - s*x*kx + c*kx*(ky*z - kz*y) + s*kx^2*d + (1 - c)/θ*(2*kx*x + ky*y + kz*z)
    JP1[1, 5] = - s*x*ky + c*ky*(ky*z - kz*y) + s/θ*z + s*kx*ky*d + (1 - c)/θ*kx*y
    JP1[1, 6] = - s*x*kz + c*kz*(ky*z - kz*y) - s/θ*y + s*kx*kz*d + (1 - c)/θ*kx*z
    JP1[1, 7] = 1

    JP1[2, 1] = s*kz + (1 - c)*ky*kx
    JP1[2, 2] = c + (1 - c)*ky^2
    JP1[2, 3] = - s*y*kx + (1 - c)*ky*kz
    JP1[2, 4] = - s*y*kx + c*kx*(kz*x - kx*z) - s/θ*z + s*kx*ky*d + (1 - c)/θ*ky*x
    JP1[2, 5] = - s*x*ky + c*ky*(kz*x - kx*z) + s*ky^2*d + (1 - c)/θ*(kx*x + 2*ky*y + kz*z)
    JP1[2, 6] = - s*x*kz + c*kz*(kz*x - kx*z) + s/θ*x + s*kz*ky*d + (1 - c)/θ*ky*z
    JP1[2, 8] = 1

    JP1[3, 1] = - s*ky + (1 - c)*kx*kz
    JP1[3, 2] = s*kx + (1 - c)*ky*kz
    JP1[3, 3] =  c + (1 - c)*kz^2
    JP1[3, 4] = - s*z*kx + c*kx*(kx*y - ky*x) + s/θ*y + s*kx*kz*d + (1 - c)/θ*kz*x
    JP1[3, 5] = - s*z*ky + c*ky*(kx*y - ky*x) - s/θ*x + s*ky*kz*d + (1 - c)/θ*kz*y
    JP1[3, 6] = - s*z*kz + c*kz*(kx*y - ky*x) + s*kz^2*d + (1 - c)/θ*(kx*x + ky*y + 2*kz*z)
    JP1[3, 9] = 1

    JP1[4, 10] = 1
    JP1[5, 11] = 1
    JP1[6, 12] = 1
    return JP1
end


function JP2(X)
    """
    Jacobian of the second step of the projection
    """
    x, y, z = X
    JP2 = spzeros(5,6)

    JP2[1, 1] = - 1 / z
    JP2[1, 3] = x / z^2

    JP2[2, 2] = - 1 / z
    JP2[2, 3] = y / z^2

    JP2[3, 4] = 1
    JP2[4, 5] = 1
    JP2[5, 6] = 1
    return JP2
end


function JP3(X, f, k1, k2)
    """
    Jacobian of the third step of the projection
    """
    x, y = X
    JP3 = spzeros(2, 5)
    norm2 = x^2 + y^2
    norm4 = norm2^2
    r = 1 + k1*norm2 + k2*norm4

    JP3[1,1] = r*x
    JP3[1,2] = f*norm2*x
    JP3[1,3] = f*norm4*x
    JP3[1,4] = f*r + f*(2*k1*x + k2*(4*x^3 + 4*x*y^2))*x
    JP3[1,5] = f*(2*k1*y + k2*(4*y^3 + 4*y*x^2))*x

    JP3[2,1] = r*y
    JP3[2,2] = f*norm2*y
    JP3[2,3] = f*norm4*y
    JP3[2,4] = f*(2*k1*x + k2*(4*x^3 + 4*x*y^2))*y
    JP3[2,5] = f*r + f*(2*k1*y + k2*(4*y^3 + 4*y*x^2))*y
    return JP3
end
