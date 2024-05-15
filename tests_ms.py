import unittest

# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import transformers

import mindspore
from mindnlp.transformers import AutoModelForCausalLM, AutoTokenizer
import mindnlp.transformers as transformers


class TestGenerator(unittest.TestCase):
    def test_replit(self):
        from generators_ms import ReplitCode
        pretrained = 'bigcode/starcoder2-7b'
        g = ReplitCode(pretrained)
        print(g('def fibonacci(n):'))

    def test_starcoder(self):
        from generators_ms import StarCoder
        pretrained = 'bigcode/starcoder2-7b'
        g = StarCoder(pretrained)
        print(g('def fibonacci(n):', {'max_new_tokens': 10}))


if __name__ == '__main__':
    unittest.main()
