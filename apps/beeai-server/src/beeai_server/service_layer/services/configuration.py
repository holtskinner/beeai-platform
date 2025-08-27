# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import logging

from kink import inject

from beeai_server.domain.models.configuration import SystemConfiguration
from beeai_server.domain.models.user import User
from beeai_server.exceptions import EntityNotFoundError
from beeai_server.service_layer.unit_of_work import IUnitOfWorkFactory

logger = logging.getLogger(__name__)


@inject
class ConfigurationService:
    def __init__(self, uow: IUnitOfWorkFactory):
        self._uow = uow

    async def get_system_configuration(self, *, user: User) -> SystemConfiguration:
        """Get the current system configuration."""
        async with self._uow() as uow:
            try:
                return await uow.configuration.get_system_configuration()
            except EntityNotFoundError:
                # Return a default configuration if none exists
                return SystemConfiguration(
                    default_llm_model=None,
                    default_embedding_model=None,
                    created_by=user.id,
                )

    async def update_system_configuration(
        self,
        *,
        default_llm_model: str | None = None,
        default_embedding_model: str | None = None,
        user: User,
    ) -> SystemConfiguration:
        configuration = SystemConfiguration(
            default_llm_model=default_llm_model, default_embedding_model=default_embedding_model, created_by=user.id
        )

        async with self._uow() as uow:
            await uow.configuration.create_or_update_system_configuration(configuration=configuration)
            await uow.commit()

        return configuration
