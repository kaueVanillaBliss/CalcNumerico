# ==============================================================================
# Arquivo: eliminacao_gaussiana.jl
# Método : Eliminação Gaussiana com Pivoteamento Parcial
# ==============================================================================

using LinearAlgebra
using Printf

"""
    eliminacao_gaussiana!(U::AbstractMatrix{T}, c::AbstractVector{T}; piv_tol=nothing) where {T<:AbstractFloat}

Versão in-place da Eliminação Gaussiana com Pivoteamento Parcial. Sobrescreve `U` com a
matriz triangular superior escalonada e `c` com o vetor modificado, retornando a solução
e metadados numéricos.
"""
function eliminacao_gaussiana!(
    U::AbstractMatrix{T},
    c::AbstractVector{T};
    piv_tol::Union{Real, Nothing} = nothing
) where {T<:AbstractFloat}
    n, m = size(U)
    if n != m
        throw(DimensionMismatch("A matriz A deve ser quadrada. Dimensões recebidas: ($n, $m)."))
    end
    if length(c) != n
        throw(DimensionMismatch("O vetor b ($(length(c))) deve ter a mesma dimensão de A ($n)."))
    end

    # Tolerância numérica para detecção de pivô nulo escalada pela norma infinito da matriz
    norma_A = opnorm(U, Inf)
    tol_pivo = isnothing(piv_tol) ? eps(T) * max(one(T), norma_A) * n : T(piv_tol)

    trocas_linhas = 0
    singular = false

    # 1. Fase de Escalonamento (Eliminação Progressiva)
    @inbounds for k in 1:(n - 1)
        # Busca pelo pivô de maior módulo na coluna k (linhas k até n)
        linha_pivo = k
        max_val = abs(U[k, k])
        for i in (k + 1):n
            val = abs(U[i, k])
            if val > max_val
                max_val = val
                linha_pivo = i
            end
        end

        # Verificação de singularidade numérica
        if max_val <= tol_pivo
            singular = true
            @warn "Matriz singular ou quase singular detectada na etapa k = $k (|pivô| = $max_val)."
            break
        end

        # Permutação de linhas em U e no vetor c (se necessário)
        if linha_pivo != k
            for j in k:n
                U[k, j], U[linha_pivo, j] = U[linha_pivo, j], U[k, j]
            end
            c[k], c[linha_pivo] = c[linha_pivo], c[k]
            trocas_linhas += 1
        end

        # Eliminação abaixo do pivô U[k, k]
        pivo = U[k, k]
        for i in (k + 1):n
            m_ik = U[i, k] / pivo
            U[i, k] = zero(T) # Zera explicitamente a entrada eliminada
            for j in (k + 1):n
                U[i, j] = muladd(-m_ik, U[k, j], U[i, j])
            end
            c[i] = muladd(-m_ik, c[k], c[i])
        end
    end

    # Verifica também o último pivô U[n, n]
    if !singular && abs(U[n, n]) <= tol_pivo
        singular = true
        @warn "Matriz singular detectada no último pivô U[$n, $n] = $(U[n, n])."
    end

    x = Vector{T}(undef, n)
    if singular
        fill!(x, T(NaN))
        return (
            solucao        = x,
            U              = U,
            c              = c,
            determinante   = zero(T),
            trocas_linhas  = trocas_linhas,
            singular       = true,
            convergiu      = false
        )
    end

    # 2. Fase de Substituição Regressiva (Back-Substitution)
    @inbounds for i in n:-1:1
        soma = c[i]
        for j in (i + 1):n
            soma = muladd(-U[i, j], x[j], soma)
        end
        x[i] = soma / U[i, i]
    end

    # Cálculo do determinante: (-1)^(trocas) * prod(diag(U))
    sinal_det = iseven(trocas_linhas) ? one(T) : -one(T)
    det_A = sinal_det * prod(@views U[diagind(U)])

    return (
        solucao        = x,
        U              = U,
        c              = c,
        determinante   = det_A,
        trocas_linhas  = trocas_linhas,
        singular       = false,
        convergiu      = true
    )
end

"""
    eliminacao_gaussiana(A::AbstractMatrix, b::AbstractVector; piv_tol=nothing)

Resolve o sistema linear `Ax = b` por Eliminação Gaussiana com Pivoteamento Parcial sem
modificar `A` e `b` originais, calculando também o erro residual `||b - Ax||_∞`.

# Retorno
Uma `NamedTuple` contendo:
- `solucao`: Vetor solução aproximada `x` (`Vector{T}`).
- `erro_residuo`: Norma infinito do resíduo `||b - Ax||_∞` (`T`).
- `erro_relativo_residuo`: Resíduo relativo `||b - Ax||_∞ / ||b||_∞` (`T`).
- `determinante`: Determinante de `A` obtido pelo produto da diagonal de `U` (`T`).
- `trocas_linhas`: Número de permutações de linha realizadas (`Int`).
- `U`: Matriz triangular superior resultante (`Matrix{T}`).
- `convergiu`: `true` se o sistema é não-singular e foi resolvido com sucesso (`Bool`).
"""
function eliminacao_gaussiana(
    A::AbstractMatrix,
    b::AbstractVector;
    piv_tol::Union{Real, Nothing} = nothing
)
    T = float(promote_type(eltype(A), eltype(b)))
    U = Matrix{T}(A)
    c = Vector{T}(b)

    res = eliminacao_gaussiana!(U, c; piv_tol = piv_tol)

    if !res.convergiu
        return (
            solucao               = res.solucao,
            erro_residuo          = T(NaN),
            erro_relativo_residuo = T(NaN),
            determinante          = zero(T),
            trocas_linhas         = res.trocas_linhas,
            U                     = res.U,
            convergiu             = false
        )
    end

    # Cálculo do resíduo r = b - A*x na norma infinito (máximo absoluto)
    r = b - A * res.solucao
    erro_res = norm(r, Inf)
    norma_b = norm(b, Inf)
    erro_rel_res = iszero(norma_b) ? erro_res : erro_res / norma_b

    return (
        solucao               = res.solucao,
        erro_residuo          = T(erro_res),
        erro_relativo_residuo = T(erro_rel_res),
        determinante          = res.determinante,
        trocas_linhas         = res.trocas_linhas,
        U                     = res.U,
        convergiu             = true
    )
end

# ==============================================================================
# Arquivo: eliminacao_gaussiana.jl
# Método : Eliminação Gaussiana com Pivoteamento Parcial
# ==============================================================================

using LinearAlgebra
using Printf

"""
    eliminacao_gaussiana!(U::AbstractMatrix{T}, c::AbstractVector{T}; piv_tol=nothing) where {T<:AbstractFloat}

Versão in-place da Eliminação Gaussiana com Pivoteamento Parcial. Sobrescreve `U` com a
matriz triangular superior escalonada e `c` com o vetor modificado, retornando a solução
e metadados numéricos.
"""
function eliminacao_gaussiana!(
    U::AbstractMatrix{T},
    c::AbstractVector{T};
    piv_tol::Union{Real, Nothing} = nothing
) where {T<:AbstractFloat}
    n, m = size(U)
    if n != m
        throw(DimensionMismatch("A matriz A deve ser quadrada. Dimensões recebidas: ($n, $m)."))
    end
    if length(c) != n
        throw(DimensionMismatch("O vetor b ($(length(c))) deve ter a mesma dimensão de A ($n)."))
    end

    # Tolerância numérica para detecção de pivô nulo escalada pela norma infinito da matriz
    norma_A = opnorm(U, Inf)
    tol_pivo = isnothing(piv_tol) ? eps(T) * max(one(T), norma_A) * n : T(piv_tol)

    trocas_linhas = 0
    singular = false

    # 1. Fase de Escalonamento (Eliminação Progressiva)
    @inbounds for k in 1:(n - 1)
        # Busca pelo pivô de maior módulo na coluna k (linhas k até n)
        linha_pivo = k
        max_val = abs(U[k, k])
        for i in (k + 1):n
            val = abs(U[i, k])
            if val > max_val
                max_val = val
                linha_pivo = i
            end
        end

        # Verificação de singularidade numérica
        if max_val <= tol_pivo
            singular = true
            @warn "Matriz singular ou quase singular detectada na etapa k = $k (|pivô| = $max_val)."
            break
        end

        # Permutação de linhas em U e no vetor c (se necessário)
        if linha_pivo != k
            for j in k:n
                U[k, j], U[linha_pivo, j] = U[linha_pivo, j], U[k, j]
            end
            c[k], c[linha_pivo] = c[linha_pivo], c[k]
            trocas_linhas += 1
        end

        # Eliminação abaixo do pivô U[k, k]
        pivo = U[k, k]
        for i in (k + 1):n
            m_ik = U[i, k] / pivo
            U[i, k] = zero(T) # Zera explicitamente a entrada eliminada
            for j in (k + 1):n
                U[i, j] = muladd(-m_ik, U[k, j], U[i, j])
            end
            c[i] = muladd(-m_ik, c[k], c[i])
        end
    end

    # Verifica também o último pivô U[n, n]
    if !singular && abs(U[n, n]) <= tol_pivo
        singular = true
        @warn "Matriz singular detectada no último pivô U[$n, $n] = $(U[n, n])."
    end

    x = Vector{T}(undef, n)
    if singular
        fill!(x, T(NaN))
        return (
            solucao        = x,
            U              = U,
            c              = c,
            determinante   = zero(T),
            trocas_linhas  = trocas_linhas,
            singular       = true,
            convergiu      = false
        )
    end

    # 2. Fase de Substituição Regressiva (Back-Substitution)
    @inbounds for i in n:-1:1
        soma = c[i]
        for j in (i + 1):n
            soma = muladd(-U[i, j], x[j], soma)
        end
        x[i] = soma / U[i, i]
    end

    # Cálculo do determinante: (-1)^(trocas) * prod(diag(U))
    sinal_det = iseven(trocas_linhas) ? one(T) : -one(T)
    det_A = sinal_det * prod(@views U[diagind(U)])

    return (
        solucao        = x,
        U              = U,
        c              = c,
        determinante   = det_A,
        trocas_linhas  = trocas_linhas,
        singular       = false,
        convergiu      = true
    )
end

"""
    eliminacao_gaussiana(A::AbstractMatrix, b::AbstractVector; piv_tol=nothing)

Resolve o sistema linear `Ax = b` por Eliminação Gaussiana com Pivoteamento Parcial sem
modificar `A` e `b` originais, calculando também o erro residual `||b - Ax||_∞`.

# Retorno
Uma `NamedTuple` contendo:
- `solucao`: Vetor solução aproximada `x` (`Vector{T}`).
- `erro_residuo`: Norma infinito do resíduo `||b - Ax||_∞` (`T`).
- `erro_relativo_residuo`: Resíduo relativo `||b - Ax||_∞ / ||b||_∞` (`T`).
- `determinante`: Determinante de `A` obtido pelo produto da diagonal de `U` (`T`).
- `trocas_linhas`: Número de permutações de linha realizadas (`Int`).
- `U`: Matriz triangular superior resultante (`Matrix{T}`).
- `convergiu`: `true` se o sistema é não-singular e foi resolvido com sucesso (`Bool`).
"""
function eliminacao_gaussiana(
    A::AbstractMatrix,
    b::AbstractVector;
    piv_tol::Union{Real, Nothing} = nothing
)
    T = float(promote_type(eltype(A), eltype(b)))
    U = Matrix{T}(A)
    c = Vector{T}(b)

    res = eliminacao_gaussiana!(U, c; piv_tol = piv_tol)

    if !res.convergiu
        return (
            solucao               = res.solucao,
            erro_residuo          = T(NaN),
            erro_relativo_residuo = T(NaN),
            determinante          = zero(T),
            trocas_linhas         = res.trocas_linhas,
            U                     = res.U,
            convergiu             = false
        )
    end

    # Cálculo do resíduo r = b - A*x na norma infinito (máximo absoluto)
    r = b - A * res.solucao
    erro_res = norm(r, Inf)
    norma_b = norm(b, Inf)
    erro_rel_res = iszero(norma_b) ? erro_res : erro_res / norma_b

    return (
        solucao               = res.solucao,
        erro_residuo          = T(erro_res),
        erro_relativo_residuo = T(erro_rel_res),
        determinante          = res.determinante,
        trocas_linhas         = res.trocas_linhas,
        U                     = res.U,
        convergiu             = true
    )
end