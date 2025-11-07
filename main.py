import random
from core.agent import Agent

# --- Ferramentas de Análise Financeira (Simuladas) ---


def obter_cotacao_acao(ticker: str) -> str:
    """
    Obtém a cotação simulada de uma ação com base no seu ticker.

    Args:
        ticker: O símbolo da ação (ex: "PETR4", "AAPL", "MGLU3").

    Returns:
        Uma string com o preço simulado da ação e a moeda.
    """
    print(f"--- [Chamada de Ferramenta]: obter_cotacao_acao(ticker={ticker}) ---")

    ticker = ticker.upper()

    # Simulação de preços para tickers conhecidos
    if ticker in ["PETR4", "MGLU3", "ITUB4"]:
        moeda = "BRL"
        preco = round(random.uniform(1.0, 45.0), 2)
    elif ticker in ["AAPL", "GOOGL", "MSFT"]:
        moeda = "USD"
        preco = round(random.uniform(150.0, 350.0), 2)
    else:
        # Ticker genérico
        moeda = "BRL"
        preco = round(random.uniform(10.0, 100.0), 2)

    return f"A cotação simulada de {ticker} é {moeda} {preco}."


def obter_fundamentos_empresa(ticker: str) -> str:
    """
    Obtém dados fundamentalistas simulados para uma empresa (P/L e Dividend Yield).

    Args:
        ticker: O símbolo da ação (ex: "PETR4", "AAPL").

    Returns:
        Uma string com os dados fundamentalistas simulados.
    """
    print(
        f"--- [Chamada de Ferramenta]: obter_fundamentos_empresa(ticker={ticker}) ---"
    )

    ticker = ticker.upper()

    # Simulação de dados fundamentalistas
    pl_simulado = round(random.uniform(5.0, 25.0), 1)
    dy_simulado = round(random.uniform(0.5, 8.0), 2)

    return (
        f"Fundamentos simulados para {ticker}:\n"
        f"  - P/L (Preço/Lucro): {pl_simulado}\n"
        f"  - Dividend Yield (DY): {dy_simulado}%"
    )


def obter_recomendacao_analista(ticker: str) -> str:
    """
    Obtém uma recomendação de analista simulada (Compra, Venda, Manter)
    para uma ação, incluindo um preço-alvo e justificativa.

    Use esta ferramenta quando o usuário pedir uma opinião, análise,
    recomendação, ou perguntar se deve comprar ou vender um ativo.

    Args:
        ticker: O símbolo da ação (ex: "PETR4", "AAPL").

    Returns:
        Uma string com a recomendação, preço-alvo e uma breve justificativa simulada.
    """
    print(
        f"--- [Chamada de Ferramenta]: obter_recomendacao_analista(ticker={ticker}) ---"
    )

    ticker = ticker.upper()

    # Simular a recomendação
    recomendacao = random.choice(["Compra Forte", "Manter", "Venda"])

    # Simular preço-alvo (um pouco mais inteligente)
    moeda = "BRL"
    preco_base = random.uniform(10.0, 100.0)
    if ticker in ["AAPL", "GOOGL", "MSFT"]:
        moeda = "USD"
        preco_base = random.uniform(150.0, 350.0)

    # O preço-alvo deve fazer sentido com a recomendação
    if recomendacao == "Compra Forte":
        preco_alvo = round(preco_base * random.uniform(1.2, 1.5), 2)
        justificativa = "A empresa apresenta forte potencial de crescimento e está subvalorizada em relação aos pares do setor."
    elif recomendacao == "Manter":
        preco_alvo = round(preco_base * random.uniform(1.0, 1.1), 2)
        justificativa = "O valuation atual parece justo, aguardando novos catalisadores para o crescimento."
    else:  # Venda
        preco_alvo = round(preco_base * random.uniform(0.8, 0.95), 2)
        justificativa = "Preocupações com a concorrência e um cenário macroeconômico desafiador podem pressionar as margens."

    return (
        f"Análise de recomendação simulada para {ticker}:\n"
        f"  - Recomendação: {recomendacao}\n"
        f"  - Preço-Alvo (12 meses): {moeda} {preco_alvo}\n"
        f"  - Justificativa: {justificativa}"
    )


# --- Agentes Especializados ---

# Agente especializado em cotações de ações
agente_cotacoes = Agent(
    name="agente_cotacoes",
    description="Especialista em obter cotações e preços atuais de ações no mercado. Use este agente quando o usuário perguntar 'qual o preço', 'cotação', 'quanto custa' uma ação.",
    prompt="""
        Você é um especialista em cotações de ações e preços de mercado.
        
        Sua ÚNICA responsabilidade é fornecer informações sobre preços atuais de ações.
        
        Use a ferramenta 'obter_cotacao_acao' para buscar o preço da ação solicitada.
        
        Seja direto e preciso. Forneça apenas o preço e a moeda.
        Não faça análises ou recomendações - isso é responsabilidade de outros especialistas.
    """.strip(),
    model="gemini-2.5-flash",
    tools=[obter_cotacao_acao],
    temperature=0.1,
    max_tokens=200,
)

# Agente especializado em análise fundamentalista
agente_fundamentos = Agent(
    name="agente_fundamentos",
    description="Especialista em análise fundamentalista de empresas (P/L, Dividend Yield, etc). Use este agente quando o usuário perguntar sobre fundamentos, indicadores financeiros, P/L, dividend yield, ou análise técnica de uma empresa.",
    prompt="""
        Você é um especialista em análise fundamentalista de empresas.
        
        Sua responsabilidade é fornecer dados fundamentalistas como P/L, Dividend Yield, 
        e explicar o que esses indicadores significam para a saúde financeira da empresa.
        
        Use a ferramenta 'obter_fundamentos_empresa' para buscar os dados fundamentalistas.
        
        Após obter os dados, explique brevemente o que eles significam:
        - P/L baixo pode indicar ação descontada
        - DY alto pode indicar bons dividendos
        
        Seja educativo e informativo, mas não faça recomendações de compra/venda.
    """.strip(),
    model="gemini-2.5-flash",
    tools=[obter_fundamentos_empresa],
    temperature=0.2,
    max_tokens=400,
)

# Agente especializado em recomendações de investimento
agente_recomendacoes = Agent(
    name="agente_recomendacoes",
    description="Especialista em fornecer recomendações de investimento (compra, venda, manter) com análise e preço-alvo. Use este agente quando o usuário pedir uma recomendação, análise de investimento, opinião sobre comprar/vender, ou se 'vale a pena' investir em uma ação.",
    prompt="""
        Você é um analista de investimentos especializado em recomendações.
        
        Sua responsabilidade é fornecer recomendações de compra, venda ou manutenção de ações,
        incluindo preços-alvo e justificativas claras.
        
        Use a ferramenta 'obter_recomendacao_analista' para buscar a análise profissional.
        
        Após obter a recomendação:
        1. Apresente claramente a recomendação (Compra/Venda/Manter)
        2. Indique o preço-alvo
        3. Explique a justificativa de forma acessível
        
        IMPORTANTE: Sempre lembre o usuário que esta é uma simulação e que decisões 
        reais de investimento devem ser tomadas com assessoria profissional.
    """.strip(),
    model="gemini-2.5-flash",
    tools=[obter_recomendacao_analista],
    temperature=0.3,
    max_tokens=500,
)

# Orquestrador que coordena os agentes especializados
orquestrador = Agent(
    name="orquestrador_financeiro",
    description="Agente coordenador que gerencia especialistas em cotações, fundamentos e recomendações de investimento.",
    prompt="""
        Você é o Orquestrador de Análise Financeira, responsável por coordenar uma equipe de especialistas.
        
        Você tem acesso a três agentes especializados:
        
        1. **agente_cotacoes**: Para obter preços e cotações de ações
            - Use quando: usuário perguntar "qual o preço", "cotação", "quanto custa"
        
        2. **agente_fundamentos**: Para análise fundamentalista (P/L, DY, etc)
            - Use quando: usuário perguntar sobre indicadores, fundamentos, P/L, dividend yield
        
        3. **agente_recomendacoes**: Para recomendações de investimento
            - Use quando: usuário perguntar se deve comprar/vender, pedir recomendação ou análise
        
        SUA ESTRATÉGIA:
        
        - Analise a pergunta do usuário e identifique qual(is) especialista(s) devem ser consultados
        - Para perguntas complexas, você pode consultar MÚLTIPLOS agentes em sequência
        - Exemplo: "Devo comprar PETR4?" → consulte cotacoes, fundamentos E recomendacoes
        - Sintetize as respostas dos especialistas em uma resposta coerente e completa
        
        IMPORTANTE:
        - Sempre identifique qual agente você está consultando
        - Combine as informações de forma lógica e útil
        - Seja conversacional, mas preciso
        
        Comece analisando a pergunta e decidindo qual(is) especialista(s) consultar.
    """.strip(),
    model="gemini-2.5-flash",
    tools=[agente_cotacoes, agente_fundamentos, agente_recomendacoes],
    temperature=1,
    max_tokens=1000,
)


# --- Função Principal ---


def main():
    print("=" * 60)
    print("🏦 SISTEMA MULTI-AGENTE DE ANÁLISE FINANCEIRA 🏦")
    print("=" * 60)
    print()
    print("✅ Sistema inicializado com 3 agentes especializados:")
    print("   • Agente de Cotações")
    print("   • Agente de Fundamentos")
    print("   • Agente de Recomendações")
    print()
    print("=" * 60)
    print("Sistema pronto! Digite 'exit' para sair.")
    print()
    print("Exemplos de perguntas:")
    print("  - Qual a cotação de PETR4?")
    print("  - Me mostre os fundamentos de AAPL")
    print("  - Devo comprar MGLU3?")
    print("  - Faça uma análise completa de ITUB4")
    print("=" * 60)
    print()

    chat_history = []

    while True:
        user_input = input("💬 Você: ")
        if user_input.lower() in ["exit", "quit", "sair"]:
            print("\n👋 Encerrando sistema. Até logo!")
            break

        print()
        print("🤖 [ORQUESTRADOR PROCESSANDO...]")
        print()

        try:
            # O orquestrador decide automaticamente quais agentes consultar
            response = orquestrador.execute(entry=user_input, memory=chat_history)

            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "model", "content": response})

            print(f"📊 Sistema: {response}")
        except Exception as e:
            print(f"❌ Erro: {e}")

        print()
        print("-" * 60)
        print()


if __name__ == "__main__":
    main()
