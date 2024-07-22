import json
import os
import re
import textwrap
import time
import random
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Dict, List, Optional, Union
import jieba
import requests
from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList
from opencompass.utils.jwt_token import generate_token
from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]


def dedent(text: str) -> str:
    # Remove leading newline
    if text.startswith("\n"):
        text = text[1:]
    text = textwrap.dedent(text)
    # Remove trailing new line
    if text.endswith("\n"):
        text = text[:-1]
    return text


@MODELS.register_module()
class AilabCommon(BaseAPIModel):
    """Model wrapper around OpenAI's models.

    Args:
        path (str): The name of OpenAI's model.
        max_seq_len (int): The maximum allowed sequence length of a model.
            Note that the length of prompt + generated tokens shall not exceed
            this value. Defaults to 2048.
        query_per_second (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 1.
        retry (int): Number of retires if the API call fails. Defaults to 2.
        org (str or List[str], optional): OpenAI organization(s). If not
            specified, OpenAI uses the default organization bound to each API
            key. If specified, the orgs will be posted with each request in
            round-robin manner. Defaults to None.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        url (str): The base url
        mode (str, optional): The method of input truncation when input length
            exceeds max_seq_len. 'front','mid' and 'rear' represents the part
            of input to truncate. Defaults to 'none'.
        temperature (float, optional): What sampling temperature to use.
            If not None, will override the temperature in the `generate()`
            call. Defaults to None.
    """

    is_api: bool = True

    def __init__(self,
                 path: str = 'AilabCommon',
                 max_seq_len: int = 4096,
                 query_per_second: int = 1,
                 retry: int = 3,
                 org: Optional[Union[str, List[str]]] = None,
                 meta_template: Optional[Dict] = None,
                 url: str = None,
                 mode: str = 'none',
                 model_id: str = 'QW72Chat',
                 stream: bool = True,
                 kunlun: bool = False,
                 temperature: Optional[float] = None):

        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         meta_template=meta_template,
                         query_per_second=query_per_second,
                         retry=retry)
        import tiktoken
        self.tiktoken = tiktoken
        self.temperature = temperature
        assert mode in ['none', 'front', 'mid', 'rear']
        self.mode = mode

        self.key_ctr = 0
        if isinstance(org, str):
            self.orgs = [org]
        else:
            self.orgs = org
        self.org_ctr = 0
        self.url = url
        self.path = path
        self.model_id = model_id
        self.stream = stream
        self.kunlun = kunlun

    def generate(
            self,
            inputs: List[str or PromptList],
            max_out_len: int = 512,
            temperature: float = 0.7,
    ) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str or PromptList]): A list of strings or PromptDicts.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.
            temperature (float): What sampling temperature to use,
                between 0 and 2. Higher values like 0.8 will make the output
                more random, while lower values like 0.2 will make it more
                focused and deterministic. Defaults to 0.7.

        Returns:
            List[str]: A list of generated strings.
        """
        if self.temperature is not None:
            temperature = self.temperature

        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(self._generate, inputs,
                             [max_out_len] * len(inputs),
                             [temperature] * len(inputs)))
        return results

    def _generate(self, input: str or PromptList, max_out_len: int,
                  temperature: float) -> str:
        """Generate results given a list of inputs.

        Args:
            inputs (str or PromptList): A string or PromptDict.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.
            temperature (float): What sampling temperature to use,
                between 0 and 2. Higher values like 0.8 will make the output
                more random, while lower values like 0.2 will make it more
                focused and deterministic.

        Returns:
            str: The generated string.
        """
        assert isinstance(input, (str, PromptList))

        context_window = 32768

        # will leave 100 tokens as prompt buffer, triggered if input is str
        if isinstance(input, str) and self.mode != 'none':
            context_window = self.max_seq_len
            input = self.bin_trim(input, context_window - max_out_len - 8)

        if isinstance(input, str):
            messages = input
        else:
            messages = []
            for item in input:
                messages.append(str(item['prompt']).strip())
            messages = dedent("\n".join(messages))

        # Hold out 100 tokens due to potential errors in tiktoken calculation
        max_out_len = context_window - self.get_token_len(str(input)) - 8
        if max_out_len <= 0:
            return ''

        max_num_retries = 0

        while max_num_retries < self.retry:
            self.wait()
            header = {
                'content-type': 'application/json',
            }
            data = {
                "model": self.model_id,
                "messages": [{"role": "user", "content": messages}],
                "stream": self.stream,  # 接口请求不用stream
                "max_tokens": max_out_len,
                "stop_token_ids": [151645, 198, 151644],
            }
            try:
                self.logger.debug(f'start request data {data}, {self.url}, {self.stream}')
                time_begin = time.time()
                raw_response = requests.post(self.url, headers=header, data=json.dumps(data), stream=self.stream,
                                             timeout=2000)
            except requests.ConnectionError:
                self.logger.error('Got connection error, retrying...')
                continue
            self.logger.debug(f'response is {raw_response.status_code}')
            if not self.stream:
                try:
                    response = raw_response.json()
                    if self.kunlun:
                        self.logger.debug(f"response detail data is {response['choices'][0]['message']['content']}")
                        response = {"response": response['choices'][0]['message']['content']}
                except requests.JSONDecodeError:
                    self.logger.error(f'JsonDecode error, got {str(raw_response.content)}')
                    # raise
                    # continue
                    return ''
                except Exception as  e:
                    self.logger.error(f'{e} {response}')
                    # raise e
                    return ''
            else:
                response = self.parse_event_data(raw_response)
            try:
                time_end = time.time()
                self.logger.info(f'--------------******request http cost time is :{time_end-time_begin}******--------------')
                self.logger.debug(f"response is {response}")
                return response["response"]
            except Exception as e:
                self.logger.error("response error, not exists response key or None")
                time.sleep(60)
                return ''
            max_num_retries += 1

        raise RuntimeError(
            'Calling cmri playground failed after retrying for 'f'{max_num_retries} times. Check the logs for ''details.')

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized string. Only English and Chinese
        characters are counted for now. Users are encouraged to override this
        method if more accurate length is needed.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """
        # enc = self.tiktoken.encoding_for_model("gpt-3.5-turbo")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("/data/models/Qwen-14B-Chat", trust_remote_code=True)
        return len(tokenizer.encode(prompt))

    def bin_trim(self, prompt: str, num_token: int) -> str:
        """Get a suffix of prompt which is no longer than num_token tokens.

        Args:
            prompt (str): Input string.
            num_token (int): The upper bound of token numbers.

        Returns:
            str: The trimmed prompt.
        """
        token_len = self.get_token_len(prompt)
        if token_len <= num_token:
            return prompt
        pattern = re.compile(r'[\u4e00-\u9fa5]')
        if pattern.search(prompt):
            words = list(jieba.cut(prompt, cut_all=False))
            sep = ''
        else:
            words = prompt.split(' ')
            sep = ' '

        l, r = 1, len(words)
        while l + 2 < r:
            mid = (l + r) // 2
            if self.mode == 'front':
                cur_prompt = sep.join(words[-mid:])
            elif self.mode == 'mid':
                cur_prompt = sep.join(words[:mid]) + sep.join(words[-mid:])
            elif self.mode == 'rear':
                cur_prompt = sep.join(words[:mid])

            if self.get_token_len(cur_prompt) <= num_token:
                l = mid  # noqa: E741
            else:
                r = mid

        if self.mode == 'front':
            prompt = sep.join(words[-l:])
        elif self.mode == 'mid':
            prompt = sep.join(words[:l]) + sep.join(words[-l:])
        elif self.mode == 'rear':
            prompt = sep.join(words[:l])
        return prompt

    def parse_event_data(self, resp) -> Dict:
        """
        解析事件数据
        :return:
        """
        if resp.encoding is None:
            resp.encoding = 'utf-8'
        try:
            generated_text = ""
            for chunk in resp.iter_lines(decode_unicode=True):
                if not chunk:
                    continue

                stem = "data:"
                if not chunk.startswith(stem):
                    self.logger.warning(f"not start with data: {chunk}")
                    continue

                chunk = chunk.split(stem)[-1]
                if chunk.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(chunk)
                except Exception as e:
                    self.logger.warning("data paste failure " + str(e))
                    continue

                content = data["choices"][0]

                # if manufacturer in ["海光"]:
                #     delta = content.get("delta").get("content")
                # elif manufacturer in ["天数"]:
                #     delta = content.get("text")
                delta = None

                if "text" in content:  # 海光
                    delta = content.get("text")
                if "delta" in content:  # 天数
                    delta = content.get("delta").get("content")

                if delta is None:
                    continue

                generated_text += delta

            return {"response": generated_text}

        except:
            return None
