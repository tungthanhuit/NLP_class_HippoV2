from __future__ import annotations

from typing import List, Optional

import os

import torch

from .base import LLMConfig
from ..utils.llm_utils import TextChatMessage
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

from transformers import PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer


def convert_text_chat_messages_to_strings(
    messages: List[TextChatMessage],
    tokenizer: PreTrainedTokenizer,
    add_assistant_header=True,
) -> List[str]:
    return tokenizer.apply_chat_template(conversation=messages, tokenize=False)


def convert_text_chat_messages_to_input_string(
    messages: List[TextChatMessage],
    tokenizer: PreTrainedTokenizer,
    add_assistant_header=True,
) -> str:
    return tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=False,
        add_generation_prompt=True,
    )


class TransformersOffline:

    def _init_llm_config(self) -> None:
        self.llm_config = LLMConfig()

    def __init__(
        self,
        global_config,
        cache_dir=None,
        cache_filename=None,
        max_model_len=4096,
        **kwargs,
    ):
        model_name = kwargs.get("model_name", global_config.llm_name)

        # Allow using the same naming convention as other parts of the codebase.
        if isinstance(model_name, str) and model_name.startswith("Transformers/"):
            model_name = model_name[len("Transformers/") :]

        # If a non-HF/OpenAI-style name is provided (e.g., "gpt-4o-mini"), fall back to a local HF model.
        if model_name is None or (
            isinstance(model_name, str)
            and ("/" not in model_name)
            and model_name.startswith(("gpt", "o"))
        ):
            model_name = "meta-llama/Llama-3.1-8B-Instruct"
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if cache_filename is None:
            cache_filename = f'{model_name.replace("/", "_")}_cache.sqlite'
        if cache_dir is None:
            cache_dir = os.path.join(global_config.save_dir, "llm_cache")
        self.cache_file_name = os.path.join(cache_dir, cache_filename)

    def infer(self, messages: List[TextChatMessage], max_tokens: int = 2048):
        logger.info(f"Calling Transformers offline, # of messages {len(messages)}")

        prompt_text = convert_text_chat_messages_to_input_string(
            messages, self.tokenizer
        )
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=0.0,
            )

        input_len = inputs["input_ids"].shape[-1]
        completion_ids = output_ids[0][input_len:]
        response = self.tokenizer.decode(completion_ids, skip_special_tokens=True)

        prompt_tokens = input_len
        completion_tokens = int(completion_ids.shape[-1])
        metadata = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
        return response, metadata

    def batch_infer(
        self,
        messages_list: List[List[TextChatMessage]],
        max_tokens: int = 2048,
        json_template: Optional[str] = None,
        batch_size: int = 4,
    ):
        # NOTE: `json_template` is accepted for API-compatibility with the OpenIE callers.
        # We do not enforce schemas here; the prompt templates already instruct JSON outputs.

        if len(messages_list) > 1:
            logger.info(
                f"Calling Transformers offline, # of messages {len(messages_list)}, using batchsize = {batch_size}"
            )

        all_prompt_texts = [
            convert_text_chat_messages_to_input_string(messages, self.tokenizer)
            for messages in messages_list
        ]

        device = next(self.model.parameters()).device
        all_responses: List[str] = []
        all_prompt_tokens: List[int] = []
        all_completion_tokens: List[int] = []

        with torch.inference_mode():
            for i in range(0, len(all_prompt_texts), batch_size):
                prompts = all_prompt_texts[i : i + batch_size]
                inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    temperature=0.0,
                )

                input_lens = inputs["attention_mask"].sum(dim=1).tolist()
                for row_idx, in_len in enumerate(input_lens):
                    completion_ids = output_ids[row_idx][int(in_len) :]
                    text = self.tokenizer.decode(
                        completion_ids, skip_special_tokens=True
                    )
                    all_responses.append(text)
                    all_prompt_tokens.append(int(in_len))
                    all_completion_tokens.append(int(completion_ids.shape[-1]))

        metadata = {
            "prompt_tokens": sum(all_prompt_tokens),
            "completion_tokens": sum(all_completion_tokens),
            "num_request": len(messages_list),
        }
        return all_responses, metadata
