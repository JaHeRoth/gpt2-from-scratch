# %%
import torch
from transformers import AutoTokenizer

from utilities.model_handler import print_stream
from utilities.models import ModelConfig, ParametersGPT2

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load("outputs/model.pt", map_location=device)
model_config = ModelConfig(**(checkpoint["model.config"] | {"device": device}))
model = ParametersGPT2(config=model_config)
model.load_state_dict(checkpoint["model.state_dict"])

tokenizer = AutoTokenizer.from_pretrained("outputs/tokenizer")

def complete(
    prompt: str, n_tries: int = 5, max_length: int | None = None, prob_threshold: float = 0.95, temperature: float = 1.0
) -> None:
    refined_prompt = f"{tokenizer.eos_token}{prompt}"
    for _ in range(n_tries):
        print_stream(
            model=model,
            tokenizer=tokenizer,
            prompt=refined_prompt,
            device=device,
            max_length=max_length,
            prob_threshold=prob_threshold,
            temperature=temperature,
        )
        print("\n" + "=" * 100 + "\n", flush=True)

# %%
complete(prompt="This was")

# %%
