using Printf

"""
    falsa_posicao(f, a::Real, b::Real; criterio::Symbol=:combinado, atol=1e-8, rtol=1e-8, ftol=1e-8, maxiter::Int=1000)

Encontra uma raiz de `f(x) = 0` no intervalo `[a, b]` utilizando o Método da Falsa Posição (Regula Falsi).

# Argumentos
- `f`: Função contínua univariada `f(x)`.
- `a::Real`: Limite inferior do intervalo inicial.
- `b::Real`: Limite superior do intervalo inicial.

# Argumentos Opcionais (Keywords)
- `criterio::Symbol=:combinado`: Regra de parada ativa:
    - `:absoluto`  -> Para quando `|x_k - x_{k-1}| <= atol` (a partir de `k >= 2`).
    - `:relativo`  -> Para quando `|x_k - x_{k-1}| / |x_k| <= rtol` (a partir de `k >= 2`).
    - `:residuo`   -> Para quando `|f(x_k)| <= ftol`.
    - `:maxiter`   -> Executa exatamente `maxiter` iterações.
    - `:combinado` -> Para quando o erro no domínio (`atol + rtol*|x_k|`) ou no resíduo (`ftol`) for atingido.
- `atol::Real=1e-8`: Tolerância para o erro absoluto `|x_k - x_{k-1}|`.
- `rtol::Real=1e-8`: Tolerância para o erro relativo `|x_k - x_{k-1}| / |x_k|`.
- `ftol::Real=1e-8`: Tolerância para o resíduo `|f(x_k)|`.
- `maxiter::Int=1000`: Número máximo de iterações permitidas.

# Retorno
Uma `NamedTuple` contendo:
`(solucao, iteracoes, maxiter, erro_absoluto, erro_relativo, erro_residuo, criterio_usado, motivo_parada, convergiu)`
"""
function falsa_posicao(
    f::F,
    a::Real,
    b::Real;
    criterio::Symbol = :combinado,
    atol::Real = 1e-8,
    rtol::Real = 1e-8,
    ftol::Real = 1e-8,
    maxiter::Int = 1000
) where {F}
    if !(criterio === :absoluto || criterio === :relativo ||
         criterio === :residuo  || criterio === :maxiter  || criterio === :combinado)
        throw(ArgumentError(
            "Critério inválido: :$criterio. Use :absoluto, :relativo, :residuo, :maxiter ou :combinado."
        ))
    end

    maxiter < 1 && throw(ArgumentError("O número máximo de iterações (maxiter) deve ser >= 1."))

    T = float(promote_type(typeof(a), typeof(b)))
    x_a, x_b = T(a), T(b)

    if x_a > x_b
        x_a, x_b = x_b, x_a
    elseif x_a == x_b
        throw(ArgumentError("Os extremos do intervalo devem ser distintos (a != b)."))
    end

    tol_a, tol_r, tol_f = T(atol), T(rtol), T(ftol)

    f_a, f_b = T(f(x_a)), T(f(x_b))
    if isnan(f_a) || isnan(f_b)
        throw(DomainError((x_a, x_b), "A função retornou NaN nos extremos do intervalo."))
    end

    # Verificação de raiz exata nos extremos
    if iszero(f_a)
        return (solucao = x_a, iteracoes = 0, maxiter = maxiter,
                erro_absoluto = zero(T), erro_relativo = zero(T), erro_residuo = zero(T),
                criterio_usado = criterio, motivo_parada = :raiz_exata, convergiu = true)
    elseif iszero(f_b)
        return (solucao = x_b, iteracoes = 0, maxiter = maxiter,
                erro_absoluto = zero(T), erro_relativo = zero(T), erro_residuo = zero(T),
                criterio_usado = criterio, motivo_parada = :raiz_exata, convergiu = true)
    end

    if signbit(f_a) == signbit(f_b)
        throw(ArgumentError("Condição de Bolzano violada: f(a) e f(b) possuem o mesmo sinal."))
    end

    x_ant = T(NaN)
    x_k = x_a
    f_k = f_a
    erro_abs = (x_b - x_a) / 2
    erro_rel = T(Inf)
    erro_res = abs(f_k)

    iter = 0
    convergiu = false
    motivo_parada = :maxiter_atingido

    while iter < maxiter
        iter += 1
        denom = f_b - f_a

        # Prevenção contra divisão por zero caso f_b - f_a sofra underflow
        if iszero(denom)
            motivo_parada = :divisao_por_zero
            @warn "Denominador (f(b) - f(a)) nulo na iteração $iter."
            break
        end

        # Fórmula numericamente estável da interpolação linear (mantém x_k dentro de [x_a, x_b])
        x_k = x_a - f_a * ((x_b - x_a) / denom)
        f_k = T(f(x_k))

        # Cálculo dos 3 erros
        erro_res = abs(f_k)
        if iter == 1
            # Na 1ª iteração ainda não há x_ant; usa-se a semilargura como referência inicial
            erro_abs = (x_b - x_a) / 2
        else
            erro_abs = abs(x_k - x_ant)
        end
        erro_rel = ifelse(iszero(x_k), T(Inf), erro_abs / abs(x_k))

        # 1. Raiz exata
        if iszero(erro_res)
            convergiu = true
            motivo_parada = :raiz_exata
            break
        end

        # 2. Avaliação do critério de parada
        # Nota: critérios baseados em |x_k - x_{k-1}| exigem pelo menos 2 iterações (iter >= 2)
        if criterio === :absoluto
            convergiu = (iter >= 2) && (erro_abs <= tol_a)
        elseif criterio === :relativo
            convergiu = (iter >= 2) && (erro_rel <= tol_r)
        elseif criterio === :residuo
            convergiu = erro_res <= tol_f
        elseif criterio === :combinado
            convergiu = ((iter >= 2) && (erro_abs <= tol_a + tol_r * abs(x_k))) || (erro_res <= tol_f)
        elseif criterio === :maxiter
            convergiu = (iter == maxiter)
        end

        if convergiu
            motivo_parada = ifelse(criterio === :maxiter, :maxiter_atingido, :tolerancia_atingida)
            break
        end

        # 3. Proteção contra estagnação de ponto flutuante
        if x_k == x_a || x_k == x_b
            motivo_parada = :precisao_maquina
            @warn "Limite da precisão de máquina atingido na iteração $iter."
            break
        end

        # Guarda aproximação atual para o cálculo de |x_k - x_{k-1}| no próximo passo
        x_ant = x_k

        # Atualização do intervalo de Bolzano
        if signbit(f_a) != signbit(f_k)
            x_b = x_k
            f_b = f_k
        else
            x_a = x_k
            f_a = f_k
        end
    end

    if !convergiu && motivo_parada === :maxiter_atingido
        @warn "Limite máximo de iterações (maxiter = $maxiter) atingido sem satisfazer o critério :$criterio."
    end

    return (
        solucao        = x_k,
        iteracoes      = iter,
        maxiter        = maxiter,
        erro_absoluto  = erro_abs,
        erro_relativo  = erro_rel,
        erro_residuo   = erro_res,
        criterio_usado = criterio,
        motivo_parada  = motivo_parada,
        convergiu      = convergiu
    )
end

"""
    falsa_posicao_historico(f, a::Real, b::Real, n_passos::Int)

Executa `n_passos` iterações do Método da Falsa Posição e retorna um vetor tipado
com os intervalos, aproximações e os 3 erros em cada passo `n`.
"""
function falsa_posicao_historico(f::F, a::Real, b::Real, n_passos::Int) where {F}
    n_passos < 1 && throw(ArgumentError("O número de passos deve ser >= 1."))

    T = float(promote_type(typeof(a), typeof(b)))
    x_a, x_b = T(a), T(b)
    x_a > x_b && ((x_a, x_b) = (x_b, x_a))

    f_a, f_b = T(f(x_a)), T(f(x_b))
    if signbit(f_a) == signbit(f_b)
        throw(ArgumentError("Condição de Bolzano violada: f(a) e f(b) possuem o mesmo sinal."))
    end

    RowType = @NamedTuple{
        n::Int, a::T, b::T, x::T, fx::T,
        erro_absoluto::T, erro_relativo::T, erro_residuo::T
    }
    historico = Vector{RowType}(undef, n_passos)
    x_ant = T(NaN)

    for n in 1:n_passos
        x_k = x_a - f_a * ((x_b - x_a) / (f_b - f_a))
        f_k = T(f(x_k))

        erro_res = abs(f_k)
        erro_abs = (n == 1) ? T(NaN) : abs(x_k - x_ant)
        erro_rel = (n == 1 || iszero(x_k)) ? T(NaN) : erro_abs / abs(x_k)

        historico[n] = (
            n             = n,
            a             = x_a,
            b             = x_b,
            x             = x_k,
            fx            = f_k,
            erro_absoluto = erro_abs,
            erro_relativo = erro_rel,
            erro_residuo  = erro_res
        )

        x_ant = x_k
        if signbit(f_a) != signbit(f_k)
            x_b, f_b = x_k, f_k
        else
            x_a, f_a = x_k, f_k
        end
    end

    return historico
end

# 1. Verificando as 2 primeiras iterações de f(x) = x^2 - 3 em [1, 2]
f(x) = x^2 - 3
passos = falsa_posicao_historico(f, 1.0, 2.0, 2)

println("=== Passo a Passo (n = 1 até n = 2) para f(x) = x^2 - 3 ===")
println(@sprintf("%-3s | %-18s | %-10s | %-11s | %-10s | %-10s | %-10s",
                 "n", "[a_n, b_n]", "x_n", "f(x_n)", "Erro Abs.", "Erro Rel.", "Resíduo"))
println("-"^86)

for p in passos
    str_abs = isnan(p.erro_absoluto) ? "   ---    " : @sprintf("%-10.6f", p.erro_absoluto)
    str_rel = isnan(p.erro_relativo) ? "   ---    " : @sprintf("%-10.6f", p.erro_relativo)
    @printf("%-3d | [%.6f, %.6f] | %-10.6f | %+11.6f | %s | %s | %-10.6f\n",
            p.n, p.a, p.b, p.x, p.fx, str_abs, str_rel, p.erro_residuo)
end

# 2. Testando os 4 critérios de parada até a convergência (tol = 1e-6)
println("\n=== Comparação dos 4 Critérios de Parada (tol = 1e-6) ===")
println(@sprintf("%-10s | %-8s | %-14s | %-10s | %-10s | %-10s | %-20s",
                 "Critério", "Iter/Max", "Solução", "Erro Abs.", "Erro Rel.", "Resíduo", "Motivo Parada"))
println("-"^96)

for (crit, max_it) in [(:absoluto, 100), (:relativo, 100), (:residuo, 100), (:maxiter, 2)]
    res = falsa_posicao(f, 1.0, 2.0; criterio=crit, atol=1e-6, rtol=1e-6, ftol=1e-6, maxiter=max_it)
    @printf("%-10s | %3d/%-4d | %-14.10f | %-10.2e | %-10.2e | %-10.2e | %-20s\n",
            res.criterio_usado, res.iteracoes, res.maxiter, res.solucao,
            res.erro_absoluto, res.erro_relativo, res.erro_residuo, res.motivo_parada)
end