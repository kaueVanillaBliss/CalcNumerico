# ==============================================================================
# Arquivo: gauss_seidel.jl
# Método : Método Iterativo de Gauss-Seidel (com Critério de Sassenfeld)
# ==============================================================================

using LinearAlgebra
using Printf

"""
    criterio_sassenfeld(A::AbstractMatrix)

Calcula os fatores `β_i` do Critério de Sassenfeld e retorna `(beta_maximo, betas, satisfeito)`.
Se `beta_maximo < 1`, o Método de Gauss-Seidel converge para qualquer aproximação inicial `x0`.
"""
function criterio_sassenfeld(A::AbstractMatrix)
    n, m = size(A)
    n == m || throw(DimensionMismatch("A matriz A deve ser quadrada."))
    T = float(eltype(A))
    betas = zeros(T, n)

    @inbounds for i in 1:n
        aii = abs(T(A[i, i]))
        iszero(aii) && return (beta_maximo = T(Inf), betas = betas, satisfeito = false)
        soma = zero(T)
        # Termos já atualizados multiplicados por β_j
        for j in 1:(i - 1)
            soma = muladd(abs(T(A[i, j])), betas[j], soma)
        end
        # Termos ainda não atualizados
        for j in (i + 1):n
            soma += abs(T(A[i, j]))
        end
        betas[i] = soma / aii
    end

    beta_max = maximum(betas)
    return (beta_maximo = beta_max, betas = betas, satisfeito = beta_max < one(T))
end

"""
    gauss_seidel(A::AbstractMatrix, b::AbstractVector; x0=nothing, omega::Real=1.0, criterio::Symbol=:combinado, atol=1e-8, rtol=1e-8, ftol=1e-8, maxiter::Int=1000)

Resolve o sistema linear `Ax = b` pelo Método Iterativo de Gauss-Seidel (ou SOR se `omega != 1.0`).

# Argumentos Opcionais (Keywords)
- `x0`: Vetor de aproximação inicial (padrão: `zeros(n)`).
- `omega::Real=1.0`: Fator de relaxação (`omega = 1.0` corresponde ao Gauss-Seidel puro).
- `criterio::Symbol=:combinado`: Critério de parada na norma infinito `||.||_∞`:
    - `:absoluto`  -> Para quando `||x^{(k)} - x^{(k-1)}||_∞ <= atol`.
    - `:relativo`  -> Para quando `||x^{(k)} - x^{(k-1)}||_∞ / ||x^{(k)}||_∞ <= rtol`.
    - `:residuo`   -> Para quando `||b - A*x^{(k)}||_∞ <= ftol`.
    - `:maxiter`   -> Executa exatamente `maxiter` iterações.
    - `:combinado` -> Para quando o erro no domínio ou no resíduo for atingido.
- `atol::Real=1e-8`: Tolerância para o erro absoluto `||x^{(k)} - x^{(k-1)}||_∞`.
- `rtol::Real=1e-8`: Tolerância para o erro relativo `||x^{(k)} - x^{(k-1)}||_∞ / ||x^{(k)}||_∞`.
- `ftol::Real=1e-8`: Tolerância para o resíduo `||b - A*x^{(k)}||_∞`.
- `maxiter::Int=1000`: Número máximo de iterações.

# Retorno
Uma `NamedTuple` contendo:
`(solucao, iteracoes, maxiter, erro_absoluto, erro_relativo, erro_residuo, beta_sassenfeld, criterio_usado, motivo_parada, convergiu)`
"""
function gauss_seidel(
    A::AbstractMatrix,
    b::AbstractVector;
    x0::Union{AbstractVector, Nothing} = nothing,
    omega::Real = 1.0,
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
    (0 < omega < 2) || throw(ArgumentError("O parâmetro de relaxação omega deve estar em (0, 2)."))

    T = float(promote_type(eltype(A), eltype(b)))
    tol_a, tol_r, tol_f, w = T(atol), T(rtol), T(ftol), T(omega)

    @inbounds for i in 1:n
        if iszero(A[i, i])
            throw(DomainError(i, "Elemento diagonal A[$i,$i] é nulo. Permute as linhas antes de aplicar Gauss-Seidel."))
        end
    end

    # Diagnóstico de Sassenfeld
    sassenfeld = criterio_sassenfeld(A)
    if !sassenfeld.satisfeito && !issymmetric(A)
        @warn "Critério de Sassenfeld não satisfeito (β_max = $(sassenfeld.beta_maximo) >= 1)."
    end

    # Apenas UM vetor de estado x é necessário (atualização in-place imediata!)
    x = isnothing(x0) ? zeros(T, n) : Vector{T}(x0)

    erro_abs = T(Inf)
    erro_rel = T(Inf)
    erro_res = norm(b - A * x, Inf)

    if iszero(erro_res)
        return (solucao = x, iteracoes = 0, maxiter = maxiter,
                erro_absoluto = zero(T), erro_relativo = zero(T), erro_residuo = zero(T),
                beta_sassenfeld = sassenfeld.beta_maximo, criterio_usado = criterio,
                motivo_parada = :raiz_exata, convergiu = true)
    end

    iter = 0
    convergiu = false
    motivo_parada = :maxiter_atingido

    while iter < maxiter
        iter += 1
        erro_abs = zero(T)
        norma_x  = zero(T)

        # Atualização sequencial in-place de Gauss-Seidel
        @inbounds for i in 1:n
            soma = T(b[i])
            for j in 1:n
                if j != i
                    # Para j < i, x[j] já contém o valor da iteração atual (k+1)!
                    # Para j > i, x[j] ainda contém o valor da iteração anterior (k)!
                    soma = muladd(-T(A[i, j]), x[j], soma)
                end
            end
            xi_gs = soma / T(A[i, i])
            xi_ant = x[i]
            xi_novo = ifelse(isone(w), xi_gs, muladd(one(T) - w, xi_ant, w * xi_gs))

            x[i] = xi_novo # Escrita imediata no próprio vetor x

            diff_i = abs(xi_novo - xi_ant)
            if diff_i > erro_abs
                erro_abs = diff_i
            end
            abs_xi = abs(xi_novo)
            if abs_xi > norma_x
                norma_x = abs_xi
            end
        end

        if isnan(erro_abs) || isinf(erro_abs)
            motivo_parada = :divergencia_numerica
            @warn "Divergência numérica detectada na iteração $iter."
            break
        end

        erro_rel = iszero(norma_x) ? T(Inf) : erro_abs / norma_x

        # Cálculo do resíduo ||b - A*x||_∞ sem alocar vetores temporários
        erro_res = zero(T)
        @inbounds for i in 1:n
            soma_ax = zero(T)
            for j in 1:n
                soma_ax = muladd(T(A[i, j]), x[j], soma_ax)
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
            convergiu = (erro_abs <= tol_a + tol_r * norma_x) || (erro_res <= tol_f)
        elseif criterio === :maxiter
            convergiu = (iter == maxiter)
        end

        if convergiu
            motivo_parada = ifelse(criterio === :maxiter, :maxiter_atingido, :tolerancia_atingida)
            break
        end
    end

    if !convergiu && motivo_parada === :maxiter_atingido
        @warn "Gauss-Seidel atingiu maxiter = $maxiter sem satisfazer o critério :$criterio."
    end

    return (
        solucao         = x,
        iteracoes       = iter,
        maxiter         = maxiter,
        erro_absoluto   = erro_abs,
        erro_relativo   = erro_rel,
        erro_residuo    = erro_res,
        beta_sassenfeld = sassenfeld.beta_maximo,
        criterio_usado  = criterio,
        motivo_parada   = motivo_parada,
        convergiu       = convergiu
    )
end

"""
    gauss_seidel_historico(A::AbstractMatrix, b::AbstractVector, n_passos::Int; x0=nothing)

Executa `n_passos` iterações de Gauss-Seidel e retorna o histórico de aproximações e os 3 erros.
"""
function gauss_seidel_historico(
    A::AbstractMatrix,
    b::AbstractVector,
    n_passos::Int;
    x0::Union{AbstractVector, Nothing} = nothing
)
    n = size(A, 1)
    T = float(promote_type(eltype(A), eltype(b)))
    x = isnothing(x0) ? zeros(T, n) : Vector{T}(x0)
    x_ant = similar(x)

    RowType = @NamedTuple{
        k::Int, x::Vector{T}, erro_absoluto::T, erro_relativo::T, erro_residuo::T
    }
    historico = Vector{RowType}(undef, n_passos)

    for k in 1:n_passos
        copyto!(x_ant, x)
        for i in 1:n
            soma = T(b[i])
            for j in 1:n
                j != i && (soma = muladd(-T(A[i, j]), x[j], soma))
            end
            x[i] = soma / T(A[i, i])
        end

        erro_abs = norm(x - x_ant, Inf)
        norma_x  = norm(x, Inf)
        erro_rel = iszero(norma_x) ? T(Inf) : erro_abs / norma_x
        erro_res = norm(b - A * x, Inf)

        historico[k] = (
            k             = k,
            x             = copy(x),
            erro_absoluto = erro_abs,
            erro_relativo = erro_rel,
            erro_residuo  = erro_res
        )
    end

    return historico
end

A = [10.0  2.0   1.0;
      1.0  5.0   1.0;
      2.0  3.0  10.0]
b = [7.0, -8.0, 6.0]
x0 = [0.7, -1.6, 0.6]

println("=== Primeiras 3 Iterações do Método de Gauss-Seidel ===")
hist = gauss_seidel_historico(A, b, 3; x0=x0)
for p in hist
    @printf("k = %d | x = [%.6f, %.6f, %.6f] | E_abs = %.6f | E_rel = %.6f | Resíduo = %.6f\n",
            p.k, p.x[1], p.x[2], p.x[3], p.erro_absoluto, p.erro_relativo, p.erro_residuo)
end

println("\n=== Gauss-Seidel até Convergência (tol = 1e-6) ===")
res = gauss_seidel(A, b; x0=x0, criterio=:relativo, rtol=1e-6)
@printf("Fator de Sassenfeld (β_max) : %.4f (< 1, converge mais rápido que Jacobi!)\n", res.beta_sassenfeld)
println("Solução x                   : ", res.solucao)
println("Iterações                   : ", res.iteracoes)
@printf("Erro Absoluto               : %.3e\n", res.erro_absoluto)
@printf("Erro Relativo               : %.3e\n", res.erro_relativo)
@printf("Erro Residual               : %.3e\n", res.erro_residuo)