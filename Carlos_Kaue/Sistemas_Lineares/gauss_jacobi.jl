# ==============================================================================
# Arquivo: gauss_jacobi.jl
# Método : Método Iterativo de Gauss-Jacobi
# ==============================================================================

using LinearAlgebra
using Printf

"""
    criterio_das_linhas(A::AbstractMatrix)

Calcula os fatores `α_i = (∑_{j≠i} |a_ij|) / |a_ii|` de cada linha e retorna `(fator_maximo, satisfeito)`.
Se `fator_maximo < 1`, o Método de Gauss-Jacobi tem convergência garantida para qualquer `x0`.
"""
function criterio_das_linhas(A::AbstractMatrix)
    n, m = size(A)
    n == m || throw(DimensionMismatch("A matriz deve ser quadrada."))
    T = float(eltype(A))
    alpha_max = zero(T)

    @inbounds for i in 1:n
        aii = abs(T(A[i, i]))
        iszero(aii) && return (fator_maximo = T(Inf), satisfeito = false)
        soma = zero(T)
        for j in 1:n
            if j != i
                soma += abs(T(A[i, j]))
            end
        end
        alpha_i = soma / aii
        if alpha_i > alpha_max
            alpha_max = alpha_i
        end
    end

    return (fator_maximo = alpha_max, satisfeito = alpha_max < one(T))
end

"""
    gauss_jacobi(A::AbstractMatrix, b::AbstractVector; x0=nothing, criterio::Symbol=:combinado, atol=1e-8, rtol=1e-8, ftol=1e-8, maxiter::Int=1000)

Resolve o sistema linear `Ax = b` pelo Método Iterativo de Gauss-Jacobi.

# Argumentos Opcionais (Keywords)
- `x0`: Vetor de aproximação inicial (padrão: vetor nulo `zeros(n)`).
- `criterio::Symbol=:combinado`: Critério de parada na norma infinito `||.||_∞`:
    - `:absoluto`  -> Para quando `||x^{(k)} - x^{(k-1)}||_∞ <= atol`.
    - `:relativo`  -> Para quando `||x^{(k)} - x^{(k-1)}||_∞ / ||x^{(k)}||_∞ <= rtol`.
    - `:residuo`   -> Para quando `||b - A*x^{(k)}||_∞ <= ftol`.
    - `:maxiter`   -> Executa exatamente `maxiter` iterações.
    - `:combinado` -> Para quando o erro relativo/absoluto no domínio ou no resíduo for atingido.
- `atol::Real=1e-8`: Tolerância para o erro absoluto `||x^{(k)} - x^{(k-1)}||_∞`.
- `rtol::Real=1e-8`: Tolerância para o erro relativo `||x^{(k)} - x^{(k-1)}||_∞ / ||x^{(k)}||_∞`.
- `ftol::Real=1e-8`: Tolerância para o erro residual `||b - A*x^{(k)}||_∞`.
- `maxiter::Int=1000`: Número máximo de iterações.

# Retorno
Uma `NamedTuple` contendo:
`(solucao, iteracoes, maxiter, erro_absoluto, erro_relativo, erro_residuo, fator_linhas, criterio_usado, motivo_parada, convergiu)`
"""
function gauss_jacobi(
    A::AbstractMatrix,
    b::AbstractVector;
    x0::Union{AbstractVector, Nothing} = nothing,
    criterio::Symbol = :combinado,
    atol::Real = 1e-8,
    rtol::Real = 1e-8,
    ftol::Real = 1e-8,
    maxiter::Int = 1000
)
    if !(criterio === :absoluto || criterio === :relativo ||
         criterio === :residuo  || criterio === :maxiter  || criterio === :combinado)
        throw(ArgumentError(
            "Critério inválido: :$criterio. Use :absoluto, :relativo, :residuo, :maxiter ou :combinado."
        ))
    end

    n, m = size(A)
    n == m || throw(DimensionMismatch("A matriz A deve ser quadrada ($n x$m)."))
    length(b) == n || throw(DimensionMismatch("O vetor b deve ter tamanho $n."))
    maxiter < 1 && throw(ArgumentError("maxiter deve ser >= 1."))

    T = float(promote_type(eltype(A), eltype(b)))
    tol_a, tol_r, tol_f = T(atol), T(rtol), T(ftol)

    # Verificação de zeros na diagonal principal
    @inbounds for i in 1:n
        if iszero(A[i, i])
            throw(DomainError(i, "Elemento diagonal A[$i,$i] é zero. Permute as linhas antes de aplicar Gauss-Jacobi."))
        end
    end

    # Diagnóstico do Critério das Linhas
    crit_linhas = criterio_das_linhas(A)
    if !crit_linhas.satisfeito
        @warn "Critério das Linhas não satisfeito (α = $(crit_linhas.fator_maximo) >= 1). A convergência não é garantida."
    end

    # Pré-alocação de vetores para zero alocações dentro do loop while
    x_atu  = isnothing(x0) ? zeros(T, n) : Vector{T}(x0)
    x_novo = similar(x_atu)
    res_vec = similar(x_atu)

    erro_abs = T(Inf)
    erro_rel = T(Inf)

    # Cálculo do resíduo inicial r = b - A*x0
    mul!(res_vec, A, x_atu)
    @inbounds for i in 1:n
        res_vec[i] = b[i] - res_vec[i]
    end
    erro_res = norm(res_vec, Inf)

    if iszero(erro_res)
        return (solucao = x_atu, iteracoes = 0, maxiter = maxiter,
                erro_absoluto = zero(T), erro_relativo = zero(T), erro_residuo = zero(T),
                fator_linhas = crit_linhas.fator_maximo, criterio_usado = criterio,
                motivo_parada = :raiz_exata, convergiu = true)
    end

    iter = 0
    convergiu = false
    motivo_parada = :maxiter_atingido

    while iter < maxiter
        iter += 1
        erro_abs = zero(T)
        norma_x_novo = zero(T)

        # Passo de Jacobi: usa exclusivamente x_atu para calcular todo x_novo
        @inbounds for i in 1:n
            soma = T(b[i])
            for j in 1:n
                if j != i
                    soma = muladd(-T(A[i, j]), x_atu[j], soma)
                end
            end
            xi = soma / T(A[i, i])
            x_novo[i] = xi

            # Acumula ||x_novo - x_atu||_∞ e ||x_novo||_∞ no mesmo laço
            diff_i = abs(xi - x_atu[i])
            if diff_i > erro_abs
                erro_abs = diff_i
            end
            abs_xi = abs(xi)
            if abs_xi > norma_x_novo
                norma_x_novo = abs_xi
            end
        end

        if isnan(erro_abs) || isinf(erro_abs)
            motivo_parada = :divergencia_numerica
            @warn "Divergência numérica detectada na iteração $iter."
            break
        end

        erro_rel = iszero(norma_x_novo) ? T(Inf) : erro_abs / norma_x_novo

        # Atualiza x_atu in-place
        copyto!(x_atu, x_novo)

        # Calcula o novo resíduo ||b - A*x_atu||_∞ sem alocar memória
        erro_res = zero(T)
        @inbounds for i in 1:n
            soma_ax = zero(T)
            for j in 1:n
                soma_ax = muladd(T(A[i, j]), x_atu[j], soma_ax)
            end
            ri = abs(T(b[i]) - soma_ax)
            if ri > erro_res
                erro_res = ri
            end
        end

        if iszero(erro_res)
            convergiu = true
            motivo_parada = :raiz_exata
            break
        end

        # Avaliação do critério de parada selecionado
        if criterio === :absoluto
            convergiu = erro_abs <= tol_a
        elseif criterio === :relativo
            convergiu = erro_rel <= tol_r
        elseif criterio === :residuo
            convergiu = erro_res <= tol_f
        elseif criterio === :combinado
            convergiu = (erro_abs <= tol_a + tol_r * norma_x_novo) || (erro_res <= tol_f)
        elseif criterio === :maxiter
            convergiu = (iter == maxiter)
        end

        if convergiu
            motivo_parada = ifelse(criterio === :maxiter, :maxiter_atingido, :tolerancia_atingida)
            break
        end
    end

    if !convergiu && motivo_parada === :maxiter_atingido
        @warn "Gauss-Jacobi atingiu maxiter = $maxiter sem satisfazer o critério :$criterio."
    end

    return (
        solucao        = x_atu,
        iteracoes      = iter,
        maxiter        = maxiter,
        erro_absoluto  = erro_abs,
        erro_relativo  = erro_rel,
        erro_residuo   = erro_res,
        fator_linhas   = crit_linhas.fator_maximo,
        criterio_usado = criterio,
        motivo_parada  = motivo_parada,
        convergiu      = convergiu
    )
end

"""
    gauss_jacobi_historico(A::AbstractMatrix, b::AbstractVector, n_passos::Int; x0=nothing)

Executa `n_passos` iterações de Gauss-Jacobi e retorna o histórico de aproximações e os 3 erros.
"""
function gauss_jacobi_historico(
    A::AbstractMatrix,
    b::AbstractVector,
    n_passos::Int;
    x0::Union{AbstractVector, Nothing} = nothing
)
    n = size(A, 1)
    T = float(promote_type(eltype(A), eltype(b)))
    x_atu = isnothing(x0) ? zeros(T, n) : Vector{T}(x0)
    x_novo = similar(x_atu)

    RowType = @NamedTuple{
        k::Int, x::Vector{T}, erro_absoluto::T, erro_relativo::T, erro_residuo::T
    }
    historico = Vector{RowType}(undef, n_passos)

    for k in 1:n_passos
        for i in 1:n
            soma = T(b[i])
            for j in 1:n
                j != i && (soma = muladd(-T(A[i, j]), x_atu[j], soma))
            end
            x_novo[i] = soma / T(A[i, i])
        end

        erro_abs = norm(x_novo - x_atu, Inf)
        norma_x  = norm(x_novo, Inf)
        erro_rel = iszero(norma_x) ? T(Inf) : erro_abs / norma_x
        erro_res = norm(b - A * x_novo, Inf)

        historico[k] = (
            k             = k,
            x             = copy(x_novo),
            erro_absoluto = erro_abs,
            erro_relativo = erro_rel,
            erro_residuo  = erro_res
        )
        copyto!(x_atu, x_novo)
    end

    return historico
end


A = [10.0  2.0   1.0;
      1.0  5.0   1.0;
      2.0  3.0  10.0]
b = [7.0, -8.0, 6.0]
x0 = [0.7, -1.6, 0.6] # Chute inicial clássico b_i / a_ii

println("=== Primeiras 3 Iterações do Método de Gauss-Jacobi ===")
hist = gauss_jacobi_historico(A, b, 3; x0=x0)
for p in hist
    @printf("k = %d | x = [%.6f, %.6f, %.6f] | E_abs = %.6f | E_rel = %.6f | Resíduo = %.6f\n",
            p.k, p.x[1], p.x[2], p.x[3], p.erro_absoluto, p.erro_relativo, p.erro_residuo)
end

println("\n=== Gauss-Jacobi até Convergência (tol = 1e-6) ===")
res = gauss_jacobi(A, b; x0=x0, criterio=:relativo, rtol=1e-6)
@printf("Fator das Linhas (α) : %.4f (< 1, converge!)\n", res.fator_linhas)
println("Solução x            : ", res.solucao)
println("Iterações            : ", res.iteracoes)
@printf("Erro Absoluto        : %.3e\n", res.erro_absoluto)
@printf("Erro Relativo        : %.3e\n", res.erro_relativo)
@printf("Erro Residual        : %.3e\n", res.erro_residuo)