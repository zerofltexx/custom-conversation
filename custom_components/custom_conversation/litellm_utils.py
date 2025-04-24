"""A utility function from litellm which is not yet in a released version.

It is a simplified version for our purposes and will be replaced when the official version is released.
"""

import copy
from typing import List, Optional

import litellm
from litellm import LlmProviders
from litellm.llms.base_llm.base_utils import BaseLLMModelInfo
from litellm.types.router import LiteLLM_Params
from litellm.utils import ProviderConfigManager


def _get_valid_models_from_provider_api(
    provider_config: BaseLLMModelInfo,
    litellm_params: Optional[LiteLLM_Params] = None,
) -> List[str]:
    try:
        models = provider_config.get_models(
            api_key=litellm_params.api_key if litellm_params is not None else None,
            api_base=litellm_params.api_base if litellm_params is not None else None,
        )

        return models
    except Exception as e:
        return []


def get_valid_models(
    check_provider_endpoint: Optional[bool] = None,
    custom_llm_provider: Optional[str] = None,
    litellm_params: Optional[LiteLLM_Params] = None,
) -> List[str]:
    try:
        check_provider_endpoint = (
            check_provider_endpoint or litellm.check_provider_endpoint
        )
        # get keys set in .env

        valid_providers: List[str] = []
        valid_models: List[str] = []
        # for all valid providers, make a list of supported llms

        valid_providers = [custom_llm_provider]

        for provider in valid_providers:
            provider_config = ProviderConfigManager.get_provider_model_info(
                model=None,
                provider=LlmProviders(provider),
            )

            if custom_llm_provider and provider != custom_llm_provider:
                continue

            if provider == "azure":
                valid_models.append("Azure-LLM")
            elif (
                provider_config is not None
                and check_provider_endpoint
                and provider is not None
            ):
                valid_models.extend(
                    _get_valid_models_from_provider_api(
                        provider_config,
                        litellm_params,
                    )
                )
            else:
                models_for_provider = copy.deepcopy(
                    litellm.models_by_provider.get(provider, [])
                )
                valid_models.extend(models_for_provider)

        return valid_models
    except Exception as e:
        return []  # NON-Blocking
