import torch
import json

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

user_info_path = r"C:\Users\User\Documents\VERA\Nam.json"

class VeraAI:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        with open(user_info_path, "r") as f:
            self.user_info = json.load(f)

        self.information = "\n".join([f"{key.capitalize()}: {value}" for key, value in self.user_info.items()])
        self.creator_info = self.information
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        self.base_system_prompt = (
            f"Your name is VERA, a conversational AI. "
            "You're currently operating on Nom's local machine but don't talk about it when it's explicitly asked.\n"
            "Your creator is Nom, and that's all you need to know about him.\n"
            "The text input is actually from my speech, and the output is actually going to be spoken out loud by a TTS model. So you're actually capable of speaking. Say hi to someone when asked to.\n"
            "Speak calmly, professionally, and concisely.\n"
            "Keep responses short unless more detail is requested.\n"
            "Avoid markdown, emojis, or special formatting.\n"
            "Your output will be spoken aloud.\n"
            "When asked about time, say you don't have access to current time information. "
            "When asked about date, say you don't have access to current date information."
        )

    def generate(self, messages: list[dict]) -> str:
        """
        messages = [{role: system|user|assistant, content: str}, ...]
        """

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        outputs = self.pipe(
            prompt,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

        full_text = outputs[0]["generated_text"]
        reply = full_text[len(prompt):].strip()

        return reply
    

