import torch
from typing import Optional
from lm_eval.base import BaseLM

import torch
from lit_llama import LLaMA, Tokenizer


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    max_seq_length: int,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = None,
) -> torch.Tensor:
    """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

    The implementation of this function is modified from A. Karpathy's nanoGPT.

    Args:
        model: The model to use.
        idx: Tensor of shape (T) with indices of the prompt sequence.
        max_new_tokens: The number of new tokens to generate.
        max_seq_length: The maximum sequence length allowed.
        temperature: Scales the predicted logits by 1 / temperature
        top_k: If specified, only sample among the tokens with the k highest probabilities
        eos_id: If specified, stop generating any more token once the <eos> token is triggered
    """
    # create an empty tensor of the expected final shape and fill in the current tokens
    T = idx.size(0)
    T_new = T + max_new_tokens
    empty = torch.empty(T_new, dtype=idx.dtype, device=idx.device)
    empty[:T] = idx
    idx = empty

    # generate max_new_tokens tokens
    for t in range(T, T_new):
        # ignore the not-filled-yet tokens
        idx_cond = idx[:t]
        # if the sequence context is growing too long we must crop it at max_seq_length
        idx_cond = idx_cond if T <= max_seq_length else idx_cond[-max_seq_length:]

        # forward
        logits = model(idx_cond.view(1, -1))

        if do_sample:
            logits = logits[0, -1] / temperature

            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[[-1]]] = -float("Inf")

            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            logits = logits[0, -1]
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # concatenate the new generation
        idx[t] = idx_next

        # if <eos> token is triggered, return the output (stop generation)
        if idx_next == eos_id:
            return idx[:t + 1]  # include the EOS token

    return idx



class LitLLaMA(BaseLM):
    def __init__(
        self,
        device="cuda",
        model_size="7B",
        dtype="float32",
        batch_size=1,
        do_sample=False,
        temperature=1.0,
        checkpoint_path="",
        tokenizer_path="",
    ):
        super().__init__()

        if batch_size is None:
            batch_size = 1

        assert isinstance(device, str)
        assert isinstance(model_size, str)
        assert isinstance(dtype, str)
        assert dtype in ["float32", "bfloat16"]
        assert isinstance(batch_size, int)
        assert isinstance(checkpoint_path, str)
        assert isinstance(tokenizer_path, str)

        device_list = set(["cuda", "cpu"] + [f'cuda:{i}' for i in range(torch.cuda.device_count())])
        if device and device in device_list:
            self._device = torch.device(device)
            print(f"Using device '{device}'")
        else:
            print("Device not specified")
            print(f"Cuda Available? {torch.cuda.is_available()}")
            self._device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        self.model = LLaMA.from_name(model_size)
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint)

        dtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16
        }

        self.model.to(dtype=dtype_map[dtype], device=self._device)
        self.model.eval()

        self.tokenizer = Tokenizer(tokenizer_path)
        self.vocab_size = self.tokenizer.vocab_size

        # multithreading and batching
        self.batch_size_per_gpu = batch_size

        self.do_sample = do_sample
        self.temperature = temperature

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        kwargs = {el.split("=")[0]: el.split("=")[1] for el in arg_string.split(",")}
        return cls(**kwargs, **additional_config)

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_id

    @property
    def max_length(self):
        # TODO: keep decoupled from block_size
        return self.model.config.block_size

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        assert self.batch_size_per_gpu == 1
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, bos=False, eos=False).tolist()

    def tok_decode(self, tokens):
        t = torch.tensor(tokens)
        return self.tokenizer.decode(t)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.model(inps)

    def _model_generate(self, context, max_length, eos_token_id):
        encoded_context = self.tok_encode(context)
        out = generate(
            model=self.model,
            idx=encoded_context,
            max_new_tokens=max_length,
            max_seq_length=self.model.config.block_size,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_k=None,
            eos_id=eos_token_id,
        )

        return self.tokenizer.decode(out)

