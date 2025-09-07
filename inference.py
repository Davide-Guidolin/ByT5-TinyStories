from t5 import T5
from config import get_config, DataConfig
import torch
from typing import Generator

def generate_story(
    model: T5, 
    prompt: str, 
    data_config: DataConfig, 
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
    stream: bool = False
) -> str | Generator[str, None, None]:
    prompt = list(prompt.encode("utf-8"))
    device = next(model.parameters()).device
    
    # create input tensor of bytes
    prompt = torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0)
    
    # generate out tensor of bytes
    out = model.generate(
        prompt, 
        max_new_tokens=max_new_tokens, 
        eos_token_id=data_config.eos_token_id, 
        pad_token_id=data_config.pad_token_id,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        stream=stream
    )
    
    if not stream:
        out = out.squeeze().cpu().tolist()
    
        if isinstance(out, int):
            out = [out]
            
        return bytes(out).decode("utf-8", errors="replace")
    
    def streamer():
        # stream
        for tok_id in out:
            token = bytes([tok_id]).decode("utf-8", errors="replace")
            yield token
    
    return streamer()

def load_model(checkpoint_path: str) -> T5:
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    state_dict = checkpoint["model"]
    for k in list(state_dict.keys()):
        new_k = k.replace("_orig_mod.", "")
        state_dict[new_k] = state_dict.pop(k)
        
    model_size = checkpoint.get("model_size", "small")
    
    t5_config, _, _, _ = get_config(model_size)
    model = T5(t5_config)
    model.load_state_dict(state_dict)
    
    print("Model loaded")
    model.print_info()
    
    return model, t5_config
    

if __name__ == "__main__":
    from config import DataConfig
    model, t5_config = load_model("./runs/1/checkpoints/checkpoint_63999.pt")
    model.to('cuda')
    
    # print(generate_story(
    #     model=model, 
    #     prompt="Once upon a time ", 
    #     data_config=DataConfig(), 
    #     max_new_tokens=10,
    #     temperature=0,
    #     top_k=None,
    #     top_p=None,
    #     stream=False
    # ))
    
    for tok in generate_story(
        model=model, 
        prompt="Once upon a time ", 
        data_config=DataConfig(), 
        max_new_tokens=1024,
        temperature=1,
        top_k=None,
        top_p=None,
        stream=True
    ):
        print(tok, flush=True, end='')
    