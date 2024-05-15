from mindnlp.transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel, GenerationConfig
from mindnlp.transformers import Pipeline, pipeline
import mindspore


class GeneratorBase:
    def generate(self, query: str, parameters: dict) -> str:
        raise NotImplementedError

    def __call__(self, query: str, parameters: dict = None) -> str:
        return self.generate(query, parameters)


class StarCoder(GeneratorBase):
    def __init__(self, pretrained: str, mirror: str = 'modelscope'):
        self.pretrained: str = pretrained
        self.mirror: str = mirror
        self.pipe: Pipeline = pipeline(
            "text-generation", model=pretrained, mirror=mirror)
        self.generation_config = GenerationConfig.from_pretrained(pretrained)
        self.generation_config.pad_token_id = self.pipe.tokenizer.eos_token_id

    def generate(self, query: str, parameters: dict) -> str:
        config: GenerationConfig = GenerationConfig.from_dict({
            **self.generation_config.to_dict(),
            **parameters
        })
        json_response: dict = self.pipe(query, generation_config=config)[0]
        generated_text: str = json_response['generated_text']
        return generated_text


class SantaCoder(GeneratorBase):
    def __init__(self, pretrained: str, mirror: str = 'modelscope'):
        self.pretrained: str = pretrained
        self.mirror: str = mirror
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(pretrained, mirror=mirror)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(pretrained, mirror=mirror)
        self.generation_config: GenerationConfig = GenerationConfig.from_model_config(self.model.config)
        self.generation_config.pad_token_id = self.tokenizer.eos_token_id

    def generate(self, query: str, parameters: dict) -> str:
        input_ids: mindspore.Tensor = self.tokenizer.encode(query, return_tensors='ms')
        config: GenerationConfig = GenerationConfig.from_dict({
            **self.generation_config.to_dict(),
            **parameters
        })
        output_ids: mindspore.Tensor = self.model.generate(input_ids, generation_config=config)
        output_text: str = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output_text


class ReplitCode(GeneratorBase):
    def __init__(self, pretrained: str, mirror: str = 'modelscope'):
        self.pretrained: str = pretrained
        self.mirror: str = mirror
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(pretrained, mirror=mirror)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(pretrained, mirror=mirror)
        self.default_parameter: dict = dict(
            do_sample=True, top_p=0.95, top_k=4, pad_token_id=self.tokenizer.eos_token_id,
            temperature=0.2, num_return_sequences=1, eos_token_id=self.tokenizer.eos_token_id
        )

    def generate(self, query: str, parameters: dict = None) -> str:
        input_ids: mindspore.Tensor = self.tokenizer.encode(query, return_tensors='ms')
        params = {**self.default_parameter, **(parameters or {})}
        if 'stop' in params:
            params.pop('stop')
        output_ids: mindspore.Tensor = self.model.generate(input_ids, **params)
        output_text: str = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output_text