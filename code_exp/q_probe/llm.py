import torch
import time
import transformers
from dataclasses import dataclass
from transformers import StoppingCriteriaList
from bigcode_eval import generation
from openai import OpenAI

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter


def ppprint(code):
    print(highlight(code, PythonLexer(), TerminalFormatter()))


@dataclass
class LLMConfig:
    model_name: str = "codellama/CodeLlama-7b-hf"
    num_gens: int = 10
    batch_size: int = 10
    temperature: float = 0.2
    top_p: float = 0.95
    device: str = "cuda"
    layer: int = 26

    return_hiddens: bool = True
    debug: bool = False


class LLM_base:
    def __init__(self, config: LLMConfig, task=None):
        self.config = config
        self.task = task

    def get_actions(self, prompt: str):
        raise NotImplementedError


class LLM_API(LLM_base):
    def __init__(self, config: LLMConfig, task=None):
        super().__init__(config, task)
        self.client = OpenAI()
        self.stop_words = []
        self.stop_words.extend(self.task.stop_words)

    def get_actions(self, prompt: str):
        messages = [{"role": "user", "content": prompt}]
        completion = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            n=self.config.num_gens,
            logprobs=False,
            stop=self.stop_words[:4],  # API allows only 4 stop words
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        top_k_strings = [c.message.content for c in completion.choices]
        top_k_strings = [
            self.task._stop_at_stop_token(_, self.task.stop_words)
            for _ in top_k_strings
        ]  # without prompt

        # Add embeddings if needed
        if self.config.return_hiddens:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=["\n".join([prompt, _]) for _ in top_k_strings],
                encoding_format="float",
            )
            hiddens = [c.embedding for c in response.data]  # list of list of floats
        else:
            hiddens = None

        return top_k_strings, hiddens


def get_stopping_criteria(stop_words, tokenizer, stop_on_newline=False):
    stopping_criteria = []
    if stop_words and tokenizer.eos_token:
        stop_words.append(tokenizer.eos_token)
    if stop_words and stop_on_newline:
        stop_words.append("\n")
    print("stop_words:", stop_words)
    if stop_words:
        stopping_criteria.append(
            generation.EndOfFunctionCriteria(0, stop_words, tokenizer)
        )
    return StoppingCriteriaList(stopping_criteria)


class LLM(LLM_base):
    def __init__(self, config: LLMConfig, task=None):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model_name,
            pad_token_id=self.tokenizer.eos_token_id,
            torch_dtype=torch.float16,
        )
        self.config = config

        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=True)

        self.eos_token_id = self.tokenizer.eos_token_id
        self.unk_token_id = self.tokenizer.unk_token_id
        self.task = task
        self.stopping_criteria = get_stopping_criteria(
            self.task.stop_words, self.tokenizer
        )

    def get_actions(self, prompt: str):
        assert self.config.num_gens % self.config.batch_size == 0
        num_batches = self.config.num_gens // self.config.batch_size
        top_k_strings, hiddens = [], []
        for _ in range(num_batches):
            s, h = self.get_actions_single_batch(prompt)
            top_k_strings.extend(s)
            if h is not None:
                hiddens.extend(h)
        return top_k_strings, hiddens

    def get_actions_single_batch(self, prompt: str):
        with torch.no_grad():
            prompt_ids = (
                self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
                .input_ids[0]
                .to(self.device)
            )

            start_time = time.time()
            self.stopping_criteria[0].start_length = len(prompt_ids) + 1

            model_output = self.model.generate(
                prompt_ids.unsqueeze(0),
                num_return_sequences=self.config.batch_size,
                # EOS parameters
                max_new_tokens=300,
                stopping_criteria=self.stopping_criteria,
                pad_token_id=self.unk_token_id,
                # Sampling
                do_sample=True if self.config.temperature > 0 else False,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                # Output parameters
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=self.config.return_hiddens,  # gen_toks x layers x (k x input_len x D)
                # Cache
                use_cache=True,
            )

            # delicate code: due to the annoying unexchangability between encode() and string concatenation()
            max_gen_length = len(model_output.scores)
            pre_decoded_sequences = self.tokenizer.batch_decode(
                model_output.sequences
            )  # with prompt and suffixes
            sequences_wp = [
                _[len(prompt) :] for _ in pre_decoded_sequences
            ]  # without prompt, with suffixes
            top_k_strings = [
                self.task._stop_at_stop_token(_, self.task.stop_words)
                for _ in sequences_wp
            ]  # without prompt, without suffixes

            useful_lengths = [
                self.tokenizer.encode_plus(
                    prompt + _, add_special_tokens=False, return_tensors="pt"
                )["input_ids"].size(1)
                for _ in top_k_strings
            ]  # token length with prompt, without suffixes
            useful_lengths = [
                min(_ - len(prompt_ids), max_gen_length) for _ in useful_lengths
            ]  # token length without prompt, without suffixes

            if self.config.return_hiddens:
                hiddens = []
                for i, l in enumerate(useful_lengths):  # index by action
                    hiddens.append(
                        model_output.hidden_states[l - 1][self.config.layer][i, 0, :]
                        .detach()
                        .cpu()
                        .numpy()
                    )
            else:
                hiddens = None

            if self.config.debug:
                print("generate top-k time: " + str(time.time() - start_time))

            return top_k_strings, hiddens
