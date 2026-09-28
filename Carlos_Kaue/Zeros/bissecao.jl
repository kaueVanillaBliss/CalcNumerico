"""
    iteracoes_teoricas(a::Real, b::Real, atol::Real) -> Int

Calcula o número mínimo teórico de iterações necessárias para que o Método da Bisseção
atinja o erro absoluto `atol` no intervalo `[a, b]`, segundo a fórmula:
`ceil(Int, log2((b - a) / (2 * atol)))`.
"""
function iteracoes_teoricas(a::Real, b::Real, atol::Real)::Int
    atol <= 0 && throw(ArgumentError("A tolerância absoluta (atol) deve ser estritamente positiva."))
    largura = abs(float(b) - float(a))
    largura == 0 && return 0
    k = ceil(Int, log2(largura / (2 * float(atol))))
    return max(0, k)
end

"""
    bissecao(f, a::Real, b::Real; criterio::Symbol=:combinado, atol=1e-8, rtol=1e-8, ftol=1e-8, maxiter::Int=1000)

Encontra uma raiz de `f(x) = 0` em `[a, b]` pelo Método da Bisseção, permitindo selecionar
entre os 3 critérios de erro (`:absoluto`, `:relativo`, `:residuo`), parada por número fixo
de iterações (`:maxiter`) ou critério `:combinado`.

# Argumentos
- `f`: Função contínua univariada `f(x)`.
- `a::Real`: Limite inferior do intervalo inicial.
- `b::Real`: Limite superior do intervalo inicial.

# Argumentos Opcionais (Keywords)
- `criterio::Symbol=:combinado`: Define a regra de parada principal:
    - `:absoluto`  -> Para quando `erro_absoluto <= atol` (ou ao atingir `maxiter`).
    - `:relativo`  -> Para quando `erro_relativo <= rtol` (ou ao atingir `maxiter`).
    - `:residuo`   -> Para quando `erro_residuo <= ftol` (ou ao atingir `maxiter`).
    - `:maxiter`   -> Ignora tolerâncias de erro e executa exatamente `maxiter` iterações (salvo raiz exata).
    - `:combinado` -> Para quando qualquer uma das 3 tolerâncias for satisfeita (ou ao atingir `maxiter`).
- `atol::Real=1e-8`: Tolerância para o erro absoluto `(b - a) / 2`.
- `rtol::Real=1e-8`: Tolerância para o erro relativo `(b - a) / (2 * |x|)`.
- `ftol::Real=1e-8`: Tolerância para o resíduo `|f(x)|`.
- `maxiter::Int=1000`: Número máximo de iterações permitidas.

# Retorno
Uma `NamedTuple` contendo:
- `solucao`: Aproximação final da raiz (`T`).
- `iteracoes`: Número de iterações executadas (`Int`).
- `maxiter`: Limite máximo de iterações configurado (`Int`).
- `erro_absoluto`: Limitante do erro absoluto `(b - a) / 2` (`T`).
- `erro_relativo`: Estimativa do erro relativo `(b - a) / (2 * |x|)` (`T`).
- `erro_residuo`: Módulo do resíduo `|f(solucao)|` (`T`).
- `criterio_usado`: Critério configurado na chamada (`Symbol`).
- `motivo_parada`: Causa exata do encerramento (`:tolerancia_atingida`, `:maxiter_atingido`, `:raiz_exata` ou `:precisao_maquina`).
- `convergiu`: `true` se o critério solicitado foi plenamente atendido (`Bool`).
"""
function bissecao(
    f::F,
    a::Real,
    b::Real;
    criterio::Symbol = :combinado,
    atol::Real = 1e-8,
    rtol::Real = 1e-8,
    ftol::Real = 1e-8,
    maxiter::Int = 1000
) where {F}
    # Validação do critério escolhido
    if !(criterio === :absoluto || criterio === :relativo ||
         criterio === :residuo  || criterio === :maxiter  || criterio === :combinado)
        throw(ArgumentError(
            "Critério inválido: :$criterio. Use :absoluto, :relativo, :residuo, :maxiter ou :combinado."
        ))
    end

    if maxiter < 1
        throw(ArgumentError("O número máximo de iterações (maxiter) deve ser >= 1."))
    end

    # Promoção de tipos para garantir estabilidade (Float32, Float64, BigFloat)
    T = float(promote_type(typeof(a), typeof(b)))
    x_a = T(a)
    x_b = T(b)

    if x_a > x_b
        x_a, x_b = x_b, x_a
    elseif x_a == x_b
        throw(ArgumentError("Os extremos do intervalo devem ser distintos (a != b)."))
    end

    tol_a = T(atol)
    tol_r = T(rtol)
    tol_f = T(ftol)

    f_a = T(f(x_a))
    f_b = T(f(x_b))

    if isnan(f_a) || isnan(f_b)
        throw(DomainError((x_a, x_b), "A função retornou NaN nos extremos do intervalo."))
    end

    # Verificação se um dos extremos já é raiz exata
    if iszero(f_a)
        return (solucao = x_a, iteracoes = 0, maxiter = maxiter,
                erro_absoluto = zero(T), erro_relativo = zero(T), erro_residuo = zero(T),
                criterio_usado = criterio, motivo_parada = :raiz_exata, convergiu = true)
    elseif iszero(f_b)
        return (solucao = x_b, iteracoes = 0, maxiter = maxiter,
                erro_absoluto = zero(T), erro_relativo = zero(T), erro_residuo = zero(T),
                criterio_usado = criterio, motivo_parada = :raiz_exata, convergiu = true)
    end

    # Verificação do Teorema de Bolzano
    if signbit(f_a) == signbit(f_b)
        throw(ArgumentError("Condição de Bolzano violada: f(a) e f(b) possuem o mesmo sinal."))
    end

    erro_abs = (x_b - x_a) / 2
    x_m = x_a + erro_abs
    f_m = T(f(x_m))
    erro_res = abs(f_m)
    erro_rel = ifelse(iszero(x_m), T(Inf), erro_abs / abs(x_m))

    iter = 0
    convergiu = false
    motivo_parada = :maxiter_atingido

    while iter < maxiter
        iter += 1
        erro_abs = (x_b - x_a) / 2
        x_m = x_a + erro_abs
        f_m = T(f(x_m))

        # Atualização das 3 métricas de erro
        erro_res = abs(f_m)
        erro_rel = ifelse(iszero(x_m), T(Inf), erro_abs / abs(x_m))

        # 1. Raiz exata em ponto flutuante
        if iszero(erro_res)
            convergiu = true
            motivo_parada = :raiz_exata
            break
        end

        # 2. Verificação da tolerância segundo o critério ativo
        if criterio === :absoluto
            convergiu = erro_abs <= tol_a
        elseif criterio === :relativo
            convergiu = erro_rel <= tol_r
        elseif criterio === :residuo
            convergiu = erro_res <= tol_f
        elseif criterio === :combinado
            convergiu = (erro_abs <= tol_a + tol_r * abs(x_m)) || (erro_res <= tol_f)
        elseif criterio === :maxiter
            convergiu = (iter == maxiter)
        end

        if convergiu
            motivo_parada = ifelse(criterio === :maxiter, :maxiter_atingido, :tolerancia_atingida)
            break
        end

        # 3. Proteção contra exaustão da precisão de ponto flutuante
        if x_m == x_a || x_m == x_b
            motivo_parada = :precisao_maquina
            @warn "Limite da precisão de máquina atingido na iteração $iter antes de satisfazer :$criterio."
            break
        end

        # Atualização do intervalo de confinamento
        if signbit(f_a) != signbit(f_m)
            x_b = x_m
            f_b = f_m
        else
            x_a = x_m
            f_a = f_m
        end
    end

    # Alerta caso os critérios de tolerância tenham sido interrompidos pelo teto maxiter
    if !convergiu && motivo_parada === :maxiter_atingido
        @warn "Limite máximo de iterações (maxiter = $maxiter) atingido sem satisfazer o critério :$criterio."
    end

    return (
        solucao        = x_m,
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

using Printf

f(x) = x^2 - 3
a, b = 1, 2
tol = 1e-3

k_teo = iteracoes_teoricas(a, b, tol)
println("Iterações teóricas previstas para atol = $tol: $k_teo iterações\n")

println(@sprintf("%-10s | %-8s | %-16s | %-10s | %-10s | %-10s | %-20s",
                 "Critério", "Iter/Max", "Solução", "Erro Abs.", "Erro Rel.", "Resíduo", "Motivo Parada"))
println("-"^98)

# 1. Comparando os 3 erros + modo :maxiter fixo (10 iterações) + combinado
configuracoes = [
    (:absoluto,  100),
    (:relativo,  100),
    (:residuo,   100),
    (:maxiter,   2),   # Executa exatamente 10 iterações
    (:absoluto,  15)    # Teto insuficiente (15 < 24 necessárias) -> dispara @warn
]

for (crit, max_it) in configuracoes
    res = bissecao(f, a, b; criterio=crit, atol=tol, rtol=tol, ftol=tol, maxiter=max_it)
    @printf("%-10s | %3d/%-4d | %-16.12f | %-10.2e | %-10.2e | %-10.2e | %-20s\n",
            res.criterio_usado,
            res.iteracoes,
            res.maxiter,
            res.solucao,
            res.erro_absoluto,
            res.erro_relativo,
            res.erro_residuo,
            res.motivo_parada)
end