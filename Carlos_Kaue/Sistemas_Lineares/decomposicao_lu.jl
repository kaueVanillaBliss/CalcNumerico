# ==============================================================================
# Arquivo: decomposicao_lu.jl
# Método : Decomposição LU com Pivoteamento Parcial (PA = LU)
# ==============================================================================

using LinearAlgebra
using Printf

"""
    fatorar_lu(A::AbstractMatrix; piv_tol=nothing)

Calcula a decomposição `PA = LU` com pivoteamento parcial da matriz quadrada `A`.

# Retorno
Uma `NamedTuple` contendo:
- `L`: Matriz triangular inferior unitária (`Matrix{T}`).
- `U`: Matriz triangular superior (`Matrix{T}`).
- `p`: Vetor de permutação de linhas tal que `A[p, :] == L * U` (`Vector{Int}`).
- `determinante`: Determinante de `A` (`T`).
- `trocas_linhas`: Número de permutações de linha realizadas (`Int`).
- `singular`: Booleano indicando se a matriz é numericamente singular (`Bool`).
"""
function fatorar_lu(
    A::AbstractMatrix;
    piv_tol::Union{Real, Nothing} = nothing
)
    n, m = size(A)
    if n != m
        throw(DimensionMismatch("A matriz A deve ser quadrada. Dimensões recebidas: ($n, $m)."))
    end

    T = float(eltype(A))
    LU = Matrix{T}(A)
    p = collect(1:n)

    norma_A = opnorm(LU, Inf)
    tol_pivo = isnothing(piv_tol) ? eps(T) * max(one(T), norma_A) * n : T(piv_tol)

    trocas_linhas = 0
    singular = false

    # Armazenamento compacto in-place: L (abaixo da diagonal) e U (diagonal e acima) em LU
    @inbounds for k in 1:(n - 1)
        linha_pivo = k
        max_val = abs(LU[k, k])
        for i in (k + 1):n
            val = abs(LU[i, k])
            if val > max_val
                max_val = val
                linha_pivo = i
            end
        end

        if max_val <= tol_pivo
            singular = true
            @warn "Pivô nulo ou quase nulo na coluna k = $k (|pivô| = $max_val)."
            break
        end

        # Permuta a linha inteira de LU (incluindo multiplicadores já calculados de L) e o vetor p
        if linha_pivo != k
            for j in 1:n
                LU[k, j], LU[linha_pivo, j] = LU[linha_pivo, j], LU[k, j]
            end
            p[k], p[linha_pivo] = p[linha_pivo], p[k]
            trocas_linhas += 1
        end

        pivo = LU[k, k]
        for i in (k + 1):n
            m_ik = LU[i, k] / pivo
            LU[i, k] = m_ik # Guarda o multiplicador na parte estritamente inferior (L)
            for j in (k + 1):n
                LU[i, j] = muladd(-m_ik, LU[k, j], LU[i, j])
            end
        end
    end

    if !singular && abs(LU[n, n]) <= tol_pivo
        singular = true
        @warn "Último pivô U[$n,$n] numericamente nulo."
    end

    # Extração explícita das matrizes L e U para inspeção didática e uso direto
    L = Matrix{T}(I, n, n)
    U = zeros(T, n, n)
    @inbounds for j in 1:n
        for i in 1:j
            U[i, j] = LU[i, j]
        end
        for i in (j + 1):n
            L[i, j] = LU[i, j]
        end
    end

    sinal_det = iseven(trocas_linhas) ? one(T) : -one(T)
    det_A = singular ? zero(T) : sinal_det * prod(@views U[diagind(U)])

    return (
        L             = L,
        U             = U,
        p             = p,
        determinante  = det_A,
        trocas_linhas = trocas_linhas,
        singular      = singular
    )
end

"""
    resolver_lu(L::AbstractMatrix{T}, U::AbstractMatrix{T}, p::AbstractVector{Int}, b::AbstractVector) where {T}

Resolve `Ax = b` em O(n²) utilizando os fatores `L`, `U` e o vetor de permutação `p` já calculados:
1. Substituição Direta: `L * y = b[p]`
2. Substituição Regressiva: `U * x = y`
"""
function resolver_lu(
    L::AbstractMatrix{T},
    U::AbstractMatrix{T},
    p::AbstractVector{Int},
    b::AbstractVector
) where {T}
    n = size(L, 1)
    length(b) == n || throw(DimensionMismatch("Dimensão de b incompatível com L e U."))

    TS = float(promote_type(T, eltype(b)))
    y = Vector{TS}(undef, n)
    x = Vector{TS}(undef, n)

    # 1. Substituição Direta (Forward Substitution): L * y = b[p]
    @inbounds for i in 1:n
        soma = TS(b[p[i]])
        for j in 1:(i - 1)
            soma = muladd(-L[i, j], y[j], soma)
        end
        y[i] = soma # L é unitária (L[i, i] == 1)
    end

    # 2. Substituição Regressiva (Backward Substitution): U * x = y
    @inbounds for i in n:-1:1
        soma = y[i]
        for j in (i + 1):n
            soma = muladd(-U[i, j], x[j], soma)
        end
        x[i] = soma / U[i, i]
    end

    return (solucao = x, vetor_intermediario_y = y)
end

"""
    decomposicao_lu(A::AbstractMatrix, b::AbstractVector; piv_tol=nothing)

Fatora `PA = LU` e resolve `Ax = b`, retornando a solução `x`, o vetor intermediário `y`
(`Ly = Pb`), os fatores `L`, `U`, `p` e o erro residual `||b - Ax||_∞`.
"""
function decomposicao_lu(
    A::AbstractMatrix,
    b::AbstractVector;
    piv_tol::Union{Real, Nothing} = nothing
)
    fat = fatorar_lu(A; piv_tol = piv_tol)
    T = eltype(fat.L)

    if fat.singular
        n = size(A, 1)
        return (
            solucao               = fill(T(NaN), n),
            vetor_y               = fill(T(NaN), n),
            L                     = fat.L,
            U                     = fat.U,
            p                     = fat.p,
            erro_residuo          = T(NaN),
            erro_relativo_residuo = T(NaN),
            determinante          = zero(T),
            convergiu             = false
        )
    end

    sol = resolver_lu(fat.L, fat.U, fat.p, b)

    r = b - A * sol.solucao
    erro_res = norm(r, Inf)
    norma_b = norm(b, Inf)
    erro_rel_res = iszero(norma_b) ? erro_res : erro_res / norma_b

    return (
        solucao               = sol.solucao,
        vetor_y               = sol.vetor_intermediario_y,
        L                     = fat.L,
        U                     = fat.U,
        p                     = fat.p,
        erro_residuo          = T(erro_res),
        erro_relativo_residuo = T(erro_rel_res),
        determinante          = fat.determinante,
        convergiu             = true
    )
end

A = [10.0  2.0   1.0;
      1.0  5.0   1.0;
      2.0  3.0  10.0]
b = [7.0, -8.0, 6.0]

res = decomposicao_lu(A, b)

println("=== Decomposição PA = LU ===")
println("Vetor de permutação p : ", res.p)
println("Vetor intermediário y : ", res.vetor_y)
println("Solução final x       : ", res.solucao)
@printf("Erro Residual ||r||_∞ : %.3e\n", res.erro_residuo)
@printf("Erro de Fatoração ||PA - LU||_∞ : %.3e\n", norm(A[res.p, :] - res.L * res.U, Inf))