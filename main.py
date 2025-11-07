import sys

from langchain_gigachat import GigaChat

from config import Config
from coolprompt.assistant import PromptTuner

GIGA_CHAT: GigaChat = GigaChat(
    credentials=Config.GIGA_CHAT_CREDENTIALS,
    verify_ssl_certs=False,
    model=Config.GIGA_CHAT_MODEL,
    timeout=600
)


def main() -> int:
    prompt_tuner: PromptTuner = PromptTuner(target_model=GIGA_CHAT)

    final_prompt: str = prompt_tuner.run(start_prompt="Write an essay about autumn", generate_num_samples=4, verbose=1)
    print(final_prompt)

    print(prompt_tuner.init_metric)
    print(prompt_tuner.final_metric)

    return 0


if __name__ == "__main__":
    sys.exit(main())
