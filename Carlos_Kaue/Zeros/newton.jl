using Printf

"""
    newton_raphson(f, df, x0::Real; criterio::Symbol=:combinado, atol=1e-8, rtol=1e-8, ftol=1e-8, df_tol=1e-14, maxiter::Int=100)

Encontra uma raiz de `f(x) = 0` a partir da aproximação inicial `x0` utilizando o Método de Newton-Raphson.

# Argumentos
- `f`: Função diferenciável univariada `f(x)`.
- `df`: Derivada primeira `f'(x)` (analítica ou via Diferenciação Automática).
- `x0::Real`: Aproximação inicial (chute inicial).

# Argumentos Opcionais (Keywords)
- `criterio::Symbol=:combinado`: Regra de parada ativa:
    - `:absoluto`  -> Para quando `|x_k - x_{k-1}| <= atol`.
    - `:relativo`  -> Para quando `|x_k - x_{k-1}| / |x_k| <= rtol`.
    - `:residuo`   -> Para quando `|f(x_k)| <= ftol`.
    - `:maxiter`   -> Executa exatamente `maxiter` iterações.
    - `:combinado` -> Para quando o erro no domínio (`atol + rtol*|x_k|`) ou no resíduo (`ftol`) for atingido.
- `atol::Real=1e-8`: Tolerância para o erro absoluto `|x_k - x_{k-1}|`.
- `rtol::Real=1e-8`: Tolerância para o erro relativo `|x_k - x_{k-1}| / |x_k|`.
- `ftol::Real=1e-8`: Tolerância para o resíduo `|f(x_k)|`.
- `df_tol::Real=1e-14`: Limite mínimo para `|f'(x_k)|` para evitar divisão por zero / explosão numérica.
- `maxiter::Int=100`: Número máximo de iterações permitidas.

# Retorno
Uma `NamedTuple` contendo:
`(solucao, iteracoes, maxiter, erro_absoluto, erro_relativo, erro_residuo, criterio_usado, motivo_parada, convergiu)`
"""
function newton_raphson(
    f::F,
    df::DF,
    x0::Real;
    criterio::Symbol = :combinado,
    atol::Real = 1e-8,
    rtol::Real = 1e-8,
    ftol::Real = 1e-8,
    df_tol::Real = 1e-14,
    maxiter::Int = 100
) where {F, DF}
    if !(criterio === :absoluto || criterio === :relativo ||
         criterio === :residuo  || criterio === :maxiter  || criterio === :combinado)
        throw(ArgumentError(
            "Critério inválido: :$criterio. Use :absoluto, :relativo, :residuo, :maxiter ou :combinado."
        ))
    end

    maxiter < 1 && throw(ArgumentError("O número máximo de iterações (maxiter) deve ser >= 1."))

    T = float(typeof(x0))
    x_k = T(x0)
    tol_a, tol_r, tol_f, tol_df = T(atol), T(rtol), T(ftol), T(df_tol)

    f_k = T(f(x_k))
    if isnan(f_k) || isinf(f_k)
        throw(DomainError(x_k, "A função retornou NaN ou Inf no ponto inicial x0."))
    end

    # Caso x0 já seja uma raiz exata
    if iszero(f_k)
        return (solucao = x_k, iteracoes = 0, maxiter = maxiter,
                erro_absoluto = zero(T), erro_relativo = zero(T), erro_residuo = zero(T),
                criterio_usado = criterio, motivo_parada = :raiz_exata, convergiu = true)
    end

    erro_abs = T(Inf)
    erro_rel = T(Inf)
    erro_res = abs(f_k)

    iter = 0
    convergiu = false
    motivo_parada = :maxiter_atingido

    while iter < maxiter
        df_k = T(df(x_k))

        # Proteção contra derivada nula ou quase nula (ponto estacionário)
        if abs(df_k) <= tol_df || isnan(df_k)
            motivo_parada = :derivada_nula
            @warn "Derivada nula ou próxima de zero (|f'(x)| = $(abs(df_k))) na iteração $(iter + 1)."
            break
        end

        iter += 1
        passo = f_k / df_k
        x_novo = x_k - passo
        f_novo = T(f(x_novo))

        if isnan(f_novo) || isinf(f_novo)
            motivo_parada = :divergencia_numerica
            @warn "Divergência numérica (NaN/Inf) detectada na iteração $iter."
            break
        end

        # Atualização dos 3 erros
        erro_abs = abs(x_novo - x_k)
        erro_rel = ifelse(iszero(x_novo), T(Inf), erro_abs / abs(x_novo))
        erro_res = abs(f_novo)

        x_k = x_novo
        f_k = f_novo

        # 1. Raiz exata
        if iszero(erro_res)
            convergiu = true
            motivo_parada = :raiz_exata
            break
        end

        # 2. Avaliação do critério selecionado
        if criterio === :absoluto
            convergiu = erro_abs <= tol_a
        elseif criterio === :relativo
            convergiu = erro_rel <= tol_r
        elseif criterio === :residuo
            convergiu = erro_res <= tol_f
        elseif criterio === :combinado
            convergiu = (erro_abs <= tol_a + tol_r * abs(x_k)) || (erro_res <= tol_f)
        elseif criterio === :maxiter
            convergiu = (iter == maxiter)
        end

        if convergiu
            motivo_parada = ifelse(criterio === :maxiter, :maxiter_atingido, :tolerancia_atingida)
            break
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
    newton_historico(f, df, x0::Real, n_passos::Int)

Executa `n_passos` iterações do Método de Newton-Raphson a partir de `x0` e retorna
um vetor estaticamente tipado com os valores e os 3 erros de cada iteração.
"""
function newton_historico(f::F, df::DF, x0::Real, n_passos::Int) where {F, DF}
    n_passos < 1 && throw(ArgumentError("O número de passos deve ser >= 1."))

    T = float(typeof(x0))
    x_k = T(x0)

    RowType = @NamedTuple{
        n::Int, x_ant::T, fx_ant::T, dfx_ant::T, x::T, fx::T,
        erro_absoluto::T, erro_relativo::T, erro_residuo::T
    }
    historico = Vector{RowType}(undef, n_passos)

    for n in 1:n_passos
        fx_ant = T(f(x_k))
        dfx_ant = T(df(x_k))
        iszero(dfx_ant) && throw(DomainError(x_k, "Derivada nula encontrada na iteração $n."))

        x_novo = x_k - fx_ant / dfx_ant
        fx_novo = T(f(x_novo))

        erro_abs = abs(x_novo - x_k)
        erro_rel = ifelse(iszero(x_novo), T(Inf), erro_abs / abs(x_novo))
        erro_res = abs(fx_novo)

        historico[n] = (
            n             = n,
            x_ant         = x_k,
            fx_ant        = fx_ant,
            dfx_ant       = dfx_ant,
            x             = x_novo,
            fx            = fx_novo,
            erro_absoluto = erro_abs,
            erro_relativo = erro_rel,
            erro_residuo  = erro_res
        )

        x_k = x_novo
    end

    return historico
end

# Definição da função f(x) = x^2 - 3 e sua derivada f'(x) = 2x
f(x)  = muladd(x, x, -3.0)
df(x) = 2.0 * x
x0    = 1.5

# 1. Tabela passo a passo até n = 2
passos = newton_historico(f, df, x0, 2)

println("=== Passo a Passo de Newton-Raphson (n = 1 até n = 2) com x0 = 1.5 ===")
println(@sprintf("%-3s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s",
                 "n", "x_{n-1}", "f(x_{n-1})", "f'(x_{n-1})", "x_n", "Erro Abs.", "Erro Rel.", "Resíduo"))
println("-"^92)

for p in passos
    @printf("%-3d | %-10.6f | %+10.6f | %-10.6f | %-10.6f | %-10.6f | %-10.6f | %-10.6f\n",
            p.n, p.x_ant, p.fx_ant, p.dfx_ant, p.x, p.erro_absoluto, p.erro_relativo, p.erro_residuo)
end

# 2. Comparando os 4 critérios de parada (tol = 1e-6)
println("\n=== Comparação dos 4 Critérios de Parada em Newton-Raphson (tol = 1e-6) ===")
println(@sprintf("%-10s | %-8s | %-16s | %-10s | %-10s | %-10s | %-20s",
                 "Critério", "Iter/Max", "Solução", "Erro Abs.", "Erro Rel.", "Resíduo", "Motivo Parada"))
println("-"^98)

for (crit, max_it) in [(:absoluto, 100), (:relativo, 100), (:residuo, 100), (:maxiter, 2)]
    res = newton_raphson(f, df, x0; criterio=crit, atol=1e-6, rtol=1e-6, ftol=1e-6, maxiter=max_it)
    @printf("%-10s | %3d/%-4d | %-16.12f | %-10.2e | %-10.2e | %-10.2e | %-20s\n",
            res.criterio_usado, res.iteracoes, res.maxiter, res.solucao,
            res.erro_absoluto, res.erro_relativo, res.erro_residuo, res.motivo_parada)
end