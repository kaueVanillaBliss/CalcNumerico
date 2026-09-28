using Printf

"""
    secante(f, x0::Real, x1::Real; criterio::Symbol=:combinado, atol=1e-8, rtol=1e-8, ftol=1e-8, df_tol=1e-14, maxiter::Int=100)

Encontra uma raiz de `f(x) = 0` a partir de duas aproximações iniciais `x0` e `x1` utilizando o Método da Secante.

# Argumentos
- `f`: Função contínua univariada `f(x)`.
- `x0::Real`: Primeira aproximação inicial.
- `x1::Real`: Segunda aproximação inicial (`x1 != x0`).

# Argumentos Opcionais (Keywords)
- `criterio::Symbol=:combinado`: Regra de parada ativa:
    - `:absoluto`  -> Para quando `|x_{k+1} - x_k| <= atol`.
    - `:relativo`  -> Para quando `|x_{k+1} - x_k| / |x_{k+1}| <= rtol`.
    - `:residuo`   -> Para quando `|f(x_{k+1})| <= ftol`.
    - `:maxiter`   -> Executa exatamente `maxiter` iterações.
    - `:combinado` -> Para quando o erro no domínio (`atol + rtol*|x_{k+1}|`) ou no resíduo (`ftol`) for atingido.
- `atol::Real=1e-8`: Tolerância para o erro absoluto `|x_{k+1} - x_k|`.
- `rtol::Real=1e-8`: Tolerância para o erro relativo `|x_{k+1} - x_k| / |x_{k+1}|`.
- `ftol::Real=1e-8`: Tolerância para o resíduo `|f(x_{k+1})|`.
- `df_tol::Real=1e-14`: Tolerância mínima para `|f(x_k) - f(x_{k-1})|` para prevenir divisão por zero.
- `maxiter::Int=100`: Número máximo de iterações permitidas.

# Retorno
Uma `NamedTuple` contendo:
`(solucao, iteracoes, maxiter, erro_absoluto, erro_relativo, erro_residuo, criterio_usado, motivo_parada, convergiu)`
"""
function secante(
    f::F,
    x0::Real,
    x1::Real;
    criterio::Symbol = :combinado,
    atol::Real = 1e-8,
    rtol::Real = 1e-8,
    ftol::Real = 1e-8,
    df_tol::Real = 1e-14,
    maxiter::Int = 100
) where {F}
    if !(criterio === :absoluto || criterio === :relativo ||
         criterio === :residuo  || criterio === :maxiter  || criterio === :combinado)
        throw(ArgumentError(
            "Critério inválido: :$criterio. Use :absoluto, :relativo, :residuo, :maxiter ou :combinado."
        ))
    end

    maxiter < 1 && throw(ArgumentError("O número máximo de iterações (maxiter) deve ser >= 1."))

    T = float(promote_type(typeof(x0), typeof(x1)))
    x_ant = T(x0)
    x_atu = T(x1)

    if x_ant == x_atu
        throw(ArgumentError("As aproximações iniciais devem ser distintas (x0 != x1)."))
    end

    tol_a, tol_r, tol_f, tol_df = T(atol), T(rtol), T(ftol), T(df_tol)

    f_ant = T(f(x_ant))
    f_atu = T(f(x_atu))

    if isnan(f_ant) || isnan(f_atu) || isinf(f_ant) || isinf(f_atu)
        throw(DomainError((x_ant, x_atu), "A função retornou NaN ou Inf nas aproximações iniciais."))
    end

    # Verificação de raiz exata nos pontos iniciais
    if iszero(f_ant)
        return (solucao = x_ant, iteracoes = 0, maxiter = maxiter,
                erro_absoluto = zero(T), erro_relativo = zero(T), erro_residuo = zero(T),
                criterio_usado = criterio, motivo_parada = :raiz_exata, convergiu = true)
    elseif iszero(f_atu)
        return (solucao = x_atu, iteracoes = 0, maxiter = maxiter,
                erro_absoluto = zero(T), erro_relativo = zero(T), erro_residuo = zero(T),
                criterio_usado = criterio, motivo_parada = :raiz_exata, convergiu = true)
    end

    erro_abs = abs(x_atu - x_ant)
    erro_rel = ifelse(iszero(x_atu), T(Inf), erro_abs / abs(x_atu))
    erro_res = abs(f_atu)

    iter = 0
    convergiu = false
    motivo_parada = :maxiter_atingido

    while iter < maxiter
        denom = f_atu - f_ant

        # Proteção contra secante horizontal (f(x_k) == f(x_{k-1}))
        if abs(denom) <= tol_df
            motivo_parada = :divisao_por_zero
            @warn "Diferença |f(x_k) - f(x_{k-1})| muito próxima de zero ($(abs(denom))) na iteração $(iter + 1)."
            break
        end

        iter += 1
        # Atualização incremental numericamente estável
        passo = f_atu * ((x_atu - x_ant) / denom)
        x_novo = x_atu - passo
        f_novo = T(f(x_novo))

        if isnan(f_novo) || isinf(f_novo)
            motivo_parada = :divergencia_numerica
            @warn "Divergência numérica (NaN/Inf) detectada na iteração $iter."
            break
        end

        # Cálculo das 3 métricas de erro
        erro_abs = abs(x_novo - x_atu)
        erro_rel = ifelse(iszero(x_novo), T(Inf), erro_abs / abs(x_novo))
        erro_res = abs(f_novo)

        # Avança a fila de pontos (sem teste de Bolzano)
        x_ant, f_ant = x_atu, f_atu
        x_atu, f_atu = x_novo, f_novo

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
            convergiu = (erro_abs <= tol_a + tol_r * abs(x_atu)) || (erro_res <= tol_f)
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
        solucao        = x_atu,
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
    secante_historico(f, x0::Real, x1::Real, n_passos::Int)

Executa `n_passos` iterações do Método da Secante a partir de `x0` e `x1` e retorna
um vetor estaticamente tipado com os pontos e os 3 erros em cada passo `n`.
"""
function secante_historico(f::F, x0::Real, x1::Real, n_passos::Int) where {F}
    n_passos < 1 && throw(ArgumentError("O número de passos deve ser >= 1."))

    T = float(promote_type(typeof(x0), typeof(x1)))
    x_ant, x_atu = T(x0), T(x1)
    f_ant, f_atu = T(f(x_ant)), T(f(x_atu))

    RowType = @NamedTuple{
        n::Int, x_ant::T, x_atu::T, x_novo::T, fx_novo::T,
        erro_absoluto::T, erro_relativo::T, erro_residuo::T
    }
    historico = Vector{RowType}(undef, n_passos)

    for n in 1:n_passos
        denom = f_atu - f_ant
        iszero(denom) && throw(DomainError((x_ant, x_atu), "Divisão por zero: f(x_k) == f(x_{k-1}) na iteração $n."))

        x_novo = x_atu - f_atu * ((x_atu - x_ant) / denom)
        f_novo = T(f(x_novo))

        erro_abs = abs(x_novo - x_atu)
        erro_rel = ifelse(iszero(x_novo), T(Inf), erro_abs / abs(x_novo))
        erro_res = abs(f_novo)

        historico[n] = (
            n             = n,
            x_ant         = x_ant,
            x_atu         = x_atu,
            x_novo        = x_novo,
            fx_novo       = f_novo,
            erro_absoluto = erro_abs,
            erro_relativo = erro_rel,
            erro_residuo  = erro_res
        )

        x_ant, f_ant = x_atu, f_atu
        x_atu, f_atu = x_novo, f_novo
    end

    return historico
end

# Função f(x) = x^2 - 3 com chutes iniciais x0 = 1.0 e x1 = 2.0
f(x) = muladd(x, x, -3.0)
x0, x1 = 1.0, 2.0

# 1. Tabela passo a passo (mostrando n = 1, n = 2 e também n = 3 para ver a diferença para a Falsa Posição)
passos = secante_historico(f, x0, x1, 3)

println("=== Passo a Passo da Secante para f(x) = x^2 - 3 (x0 = 1.0, x1 = 2.0) ===")
println(@sprintf("%-3s | %-10s | %-10s | %-10s | %-11s | %-10s | %-10s | %-10s",
                 "n", "x_{n-1}", "x_n", "x_{n+1}", "f(x_{n+1})", "Erro Abs.", "Erro Rel.", "Resíduo"))
println("-"^93)

for p in passos
    @printf("%-3d | %-10.6f | %-10.6f | %-10.6f | %+11.6f | %-10.6f | %-10.6f | %-10.6f\n",
            p.n, p.x_ant, p.x_atu, p.x_novo, p.fx_novo, p.erro_absoluto, p.erro_relativo, p.erro_residuo)
end

# 2. Comparação dos 4 critérios de parada (tol = 1e-6)
println("\n=== Comparação dos 4 Critérios de Parada na Secante (tol = 1e-6) ===")
println(@sprintf("%-10s | %-8s | %-16s | %-10s | %-10s | %-10s | %-20s",
                 "Critério", "Iter/Max", "Solução", "Erro Abs.", "Erro Rel.", "Resíduo", "Motivo Parada"))
println("-"^98)

for (crit, max_it) in [(:absoluto, 100), (:relativo, 100), (:residuo, 100), (:maxiter, 2)]
    res = secante(f, x0, x1; criterio=crit, atol=1e-6, rtol=1e-6, ftol=1e-6, maxiter=max_it)
    @printf("%-10s | %3d/%-4d | %-16.12f | %-10.2e | %-10.2e | %-10.2e | %-20s\n",
            res.criterio_usado, res.iteracoes, res.maxiter, res.solucao,
            res.erro_absoluto, res.erro_relativo, res.erro_residuo, res.motivo_parada)
end