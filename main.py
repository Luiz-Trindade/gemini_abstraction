from core.agent import Agent
from datetime import datetime


def obter_data_e_hora_atual(fuso_horario: str) -> str:
    """
    Retorna a data e hora atual em um fuso horário específico.

    Args:
        fuso_horario: O fuso horário desejado, como 'America/Sao_Paulo' ou 'Europe/London'.

    Returns:
        A data e hora atual formatada como string.
    """
    # Exemplo: Simulação de obtenção da hora real
    if fuso_horario == "America/Sao_Paulo":
        return f"A data e hora atual em São Paulo é {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    elif fuso_horario == "Europe/London":
        return f"A data e hora atual em Londres é {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    else:
        return f"Fuso horário '{fuso_horario}' não suportado pela ferramenta."


def main():
    agent_prompt = """
        You are an assistant that provides the current date and time for requested cities using the provided tool.
    """.strip()

    agent = Agent(
        prompt=agent_prompt,
        model="gemini-2.5-flash",
        tools=[obter_data_e_hora_atual],
        temperature=1.0,
        max_tokens=250,
    )

    chat_history = []

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = agent.execute(entry=user_input, memory=chat_history)
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "model", "content": response})

        print(f"Response: {response}")
        print("-" * 40)
        print("")


if __name__ == "__main__":
    main()
