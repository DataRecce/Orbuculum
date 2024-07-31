import json
import os
from typing import List, Optional, Dict, Any, Mapping, Iterator, AsyncIterator

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLLM, BaseLanguageModel
from langchain_core.messages import BaseMessage, AIMessageChunk
from langchain_core.outputs import LLMResult, Generation, ChatGenerationChunk, GenerationChunk
from langchain_core.runnables import run_in_executor
from pydantic import BaseModel


class FFMEmbeddings(BaseModel, Embeddings):
    base_url: str = 'https://api-ams.twcc.ai/api'
    api_key: str = os.environ.get('AFS_API_KEY')
    model: str = ''

    def get_embeddings(self, payload):
        endpoint_url = f"{self.base_url}/models/embeddings"
        embeddings = []
        headers = {
            "Content-type": "application/json",
            "accept": "application/json",
            "X-API-KEY": self.api_key,
            "X-API-HOST": "afs-inference"
        }
        response = requests.post(endpoint_url, headers=headers, data=payload)
        body = response.json()
        datas = body.get('data', [])
        for data in datas:
            embeddings.append(data["embedding"])

        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        payload = json.dumps({"model": self.model, "inputs": texts})
        return self.get_embeddings(payload)

    def embed_query(self, text: str) -> List[List[float]]:
        payload = json.dumps({"model": self.model, "inputs": [text]})
        emb = self.get_embeddings(payload)
        return emb[0]


class _FormosaFoundationCommon(BaseLanguageModel):
    base_url: str = "https://api-ams.twcc.ai"
    """Base url the model is hosted under."""

    model: str = "meta-llama3-70b"
    """Model name to use."""

    temperature: Optional[float]
    """
    The temperature of the model. Increasing the temperature will
    make the model answer more creatively.
    """

    stop: Optional[List[str]]
    """Sets the stop tokens to use."""

    top_k: int = 50
    """
    Reduces the probability of generating nonsense.
    A higher value (e.g. 100) will give more diverse answers, while
    a lower value (e.g. 10) will be more conservative. (Default: 50)
    """

    top_p: float = 1
    """
    Works together with top-k. A higher value (e.g., 0.95) will lead
    to more diverse text, while a lower value (e.g., 0.5) will
    generate more focused and conservative text. (Default: 1)
    """

    max_new_tokens: int = 350
    """
    The maximum number of tokens to generate in the completion.
    -1 returns as many tokens as possible given the prompt and
    the models maximal context size.
    """

    frequence_penalty: float = 1
    """Penalizes repeated tokens according to frequency."""

    model_kwargs: Dict[str, Any] = {}
    """
    Holds any model parameters valid for `create` call not explicitly
    specified.
    """

    ffm_api_key: Optional[str] = os.environ.get('AFS_API_KEY')

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling FFM API."""
        normal_params = {
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "top_p": self.top_p,
            "frequence_penalty": self.frequence_penalty,
            "top_k": self.top_k,
        }
        return {**normal_params, **self.model_kwargs}

    def _call(
        self,
        prompt,
        service_path="/api/models/conversation",
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        if self.stop is not None and stop is not None:
            raise ValueError(
                "`stop` found in both the input and default params.")
        elif self.stop is not None:
            stop = self.stop
        elif stop is None:
            stop = []
        params = {**self._default_params, "stop": stop, **kwargs}
        parameter_payload = {"parameters": params, "model": self.model}
        if isinstance(prompt, str):
            parameter_payload = {"inputs": prompt, **parameter_payload}
            service_path = "/api/models/conversation"
        else:
            parameter_payload = {"messages": prompt, **parameter_payload}
            service_path = "/api/models/conversation"

        # HTTP headers for authorization
        headers = {
            'X-API-KEY': self.ffm_api_key,
            'Content-Type': 'application/json',
            'X-API-HOST': 'afs-inference',
        }
        endpoint_url = f"{self.base_url}{service_path}"
        # send request
        try:
            response = requests.post(
                url=endpoint_url,
                headers=headers,
                data=json.dumps(parameter_payload,
                                ensure_ascii=False).encode('utf8'),
                stream=False,
            )
            response.encoding = "utf-8"
            generated_text = response.json()
            if response.status_code != 200:
                detail = generated_text.get("detail")
                raise ValueError(
                    f"FormosaFoundationModel endpoint_url: {endpoint_url}\n"
                    f"error raised with status code {response.status_code}\n"
                    f"Details: {detail}\n"
                )

        except requests.exceptions.RequestException as e:
            # This is the correct syntax
            raise ValueError(f"FormosaFoundationModel error raised by \
                              inference endpoint: {e}\n")

        if generated_text.get('detail') is not None:
            detail = generated_text['detail']
            raise ValueError(
                f"FormosaFoundationModel endpoint_url: {endpoint_url}\n"
                f'error raised by inference API: {detail}\n'
            )

        if generated_text.get('generated_text') is None:
            raise ValueError(
                f"FormosaFoundationModel endpoint_url: {endpoint_url}\n"
                f'Response format error: {generated_text}\n'
            )

        return generated_text


class FormosaFoundationModel(BaseLLM, _FormosaFoundationCommon):
    """Formosa Foundation Model

    Example:
        .. code-block:: python
            ffm = FormosaFoundationModel(model_name="llama2-7b-chat-meta")
    """

    @property
    def _llm_type(self) -> str:
        return 'FormosaFoundationModel'

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        '''Get the identifying parameters.'''

        return {
            **{
                "model": self.model,
                "base_url": self.base_url
            },
            **self._default_params
        }

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Call out to FormosaFoundationModel's generate endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                response = FormosaFoundationModel("Tell me a joke.")
        """

        generations = []
        token_usage = 0
        for prompt in prompts:
            final_chunk = super()._call(
                prompt,
                stop=stop,
                **kwargs,
            )
            generations.append(
                [
                    Generation(
                        text=final_chunk["generated_text"],
                        generation_info=dict(
                            finish_reason=final_chunk["finish_reason"]
                        )
                    )
                ]
            )
            token_usage += final_chunk["generated_tokens"]

        llm_output = {"token_usage": token_usage, "model": self.model}
        return LLMResult(generations=generations, llm_output=llm_output)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:

        service_path = "/api/models/conversation"
        endpoint_url = f"{self.base_url}{service_path}"

        headers = {
            "Content-type": "application/json",
            "accept": "application/json",
            "X-API-KEY": self.ffm_api_key,
            'X-API-HOST': 'afs-inference',
        }

        payload = {
            "model": self.model,
            "messages": kwargs['kwargs']['messages'],
            "parameters": {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_k": self.top_k,
                "top_p": self.top_p,
                "frequence_penalty": self.frequence_penalty
            },
        }

        response = requests.post(endpoint_url, headers=headers,
                                 json=payload, stream=True)
        for chunktxt in response.iter_lines(decode_unicode=True):
            content = chunktxt.lstrip('data: ')
            if len(content) == 0:
                yield ChatGenerationChunk(message=AIMessageChunk(content=''))
            else:
                contentresult = json.loads(content)
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=contentresult['generated_text']))

                yield chunk

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        result = await run_in_executor(
            None,
            self._stream,
            messages,
            stop=stop,
            run_manager=run_manager.get_sync() if run_manager else None,
            **kwargs,
        )
        for chunk in result:
            yield chunk
