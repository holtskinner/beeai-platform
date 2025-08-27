# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import logging
from asyncio import TaskGroup
from datetime import timedelta
from http.client import HTTPException
from uuid import UUID

from cachetools import TTLCache
from kink import inject
from openai.types import Model
from pydantic import HttpUrl

from beeai_server.domain.constants import MODEL_API_KEY_SECRET_NAME
from beeai_server.domain.models.model_provider import ModelProvider, ModelProviderType
from beeai_server.domain.repositories.env import EnvStoreEntity
from beeai_server.exceptions import EntityNotFoundError
from beeai_server.service_layer.unit_of_work import IUnitOfWorkFactory

logger = logging.getLogger(__name__)


@inject
class ModelProviderService:
    _provider_models: TTLCache[UUID, list[Model]] = TTLCache(maxsize=100, ttl=timedelta(days=1).total_seconds())

    def __init__(self, uow: IUnitOfWorkFactory):
        self._uow = uow

    async def create_provider(
        self,
        *,
        name: str | None,
        description: str | None,
        type: ModelProviderType,
        base_url: HttpUrl,
        watsonx_project_id: str | None = None,
        watsonx_space_id: str | None = None,
        api_key: str,
    ) -> ModelProvider:
        model_provider = ModelProvider(
            name=name,
            description=description,
            type=type,
            base_url=base_url,
            watsonx_project_id=watsonx_project_id,
            watsonx_space_id=watsonx_space_id,
        )
        async with self._uow() as uow:
            await uow.model_providers.create(model_provider=model_provider)
            await uow.env.update(
                parent_entity=EnvStoreEntity.model_provider,
                parent_entity_id=model_provider.id,
                variables={MODEL_API_KEY_SECRET_NAME: api_key},
            )
            await uow.commit()
        return model_provider

    async def get_provider(self, *, model_provider_id: UUID) -> ModelProvider:
        """Get a model provider by ID."""
        async with self._uow() as uow:
            return await uow.model_providers.get(model_provider_id=model_provider_id)

    async def get_provider_by_model_id(self, *, model_id: str) -> ModelProvider:
        all_models = await self.get_all_models()
        if model_id not in all_models:
            raise EntityNotFoundError("llm_provider", id=model_id)
        return all_models[model_id][0]

    async def list_providers(self) -> list[ModelProvider]:
        """List model providers, optionally filtered by capability."""
        async with self._uow() as uow:
            return [provider async for provider in uow.model_providers.list()]

    async def delete_provider(self, *, model_provider_id: UUID) -> None:
        """Delete a model provider and its environment variables."""
        async with self._uow() as uow:
            await uow.model_providers.delete(model_provider_id=model_provider_id)
            await uow.commit()

    async def get_provider_api_key(self, *, model_provider_id: UUID) -> str:
        async with self._uow() as uow:
            # Check permissions
            await uow.model_providers.get(model_provider_id=model_provider_id)
            result = await uow.env.get(
                parent_entity=EnvStoreEntity.model_provider,
                parent_entity_id=model_provider_id,
                key=MODEL_API_KEY_SECRET_NAME,
            )
            if not result:
                raise EntityNotFoundError("provider_variable", id=MODEL_API_KEY_SECRET_NAME)
            return result

    async def _get_provider_models(self, provider: ModelProvider, api_key: str):
        try:
            if not self._provider_models.get(provider.id):
                self._provider_models[provider.id] = await provider.load_models(api_key=api_key)
            return self._provider_models[provider.id]
        except HTTPException as ex:
            logger.warning(f"Failed to load models for provider {provider.id}: {ex}")

    async def get_all_models(self) -> dict[str, tuple[ModelProvider, Model]]:
        async with self._uow() as uow, TaskGroup() as tg:
            providers = [provider async for provider in uow.model_providers.list()]
            all_env = await uow.env.get_all(
                parent_entity=EnvStoreEntity.model_provider, parent_entity_ids=[p.id for p in providers]
            )
            for provider in providers:
                tg.create_task(self._get_provider_models(provider=provider, api_key=all_env[provider.id]["API_KEY"]))

        result = {}
        for provider in providers:
            for model in self._provider_models[provider.id]:
                result[model.id] = (provider, model)
        return result
