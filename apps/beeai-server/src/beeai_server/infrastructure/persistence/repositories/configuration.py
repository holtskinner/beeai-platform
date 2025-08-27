# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0


from sqlalchemy import UUID as SQL_UUID
from sqlalchemy import Column, DateTime, ForeignKey, Index, Row, String, Table
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncConnection
from sqlalchemy.sql import select

from beeai_server.domain.models.configuration import SystemConfiguration
from beeai_server.domain.repositories.configuration import IConfigurationRepository
from beeai_server.exceptions import EntityNotFoundError
from beeai_server.infrastructure.persistence.repositories.db_metadata import metadata


class ConfigurationType(str):
    SYSTEM = "system"
    USER = "user"


configurations_table = Table(
    "configurations",
    metadata,
    Column("id", SQL_UUID, primary_key=True),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("created_by", ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    Column("configuration_type", String(50), nullable=False),
    Column("default_llm_model", String(256), nullable=True),
    Column("default_embedding_model", String(256), nullable=True),
    Index(
        "ix_unique_system_configuration",
        "configuration_type",
        unique=True,
        postgresql_where="configuration_type = 'system'",
    ),
)


class SqlAlchemyConfigurationRepository(IConfigurationRepository):
    def __init__(self, connection: AsyncConnection):
        self.connection = connection

    async def get_system_configuration(self) -> SystemConfiguration:
        query = select(configurations_table).where(
            configurations_table.c.configuration_type == ConfigurationType.SYSTEM
        )
        result = await self.connection.execute(query)

        if not (row := result.fetchone()):
            raise EntityNotFoundError(entity="configuration", id="system")

        return self._row_to_system_configuration(row)

    async def create_or_update_system_configuration(self, *, configuration: SystemConfiguration) -> None:
        # Use PostgreSQL's UPSERT (INSERT ... ON CONFLICT ... DO UPDATE)
        stmt = insert(configurations_table).values(
            id=configuration.id,
            configuration_type=ConfigurationType.SYSTEM,
            created_by=configuration.created_by,
            updated_at=configuration.updated_at,
            default_llm_model=configuration.default_llm_model,
            default_embedding_model=configuration.default_embedding_model,
        )

        # On conflict, update all fields except id, configuration_type, and created_by
        stmt = stmt.on_conflict_do_update(
            index_elements=["configuration_type"],
            index_where=configurations_table.c.configuration_type == ConfigurationType.SYSTEM,
            set_={
                "default_llm_model": stmt.excluded.default_llm_model,
                "default_embedding_model": stmt.excluded.default_embedding_model,
                "updated_at": stmt.excluded.updated_at,
                "created_by": stmt.excluded.created_by,
            },
        )

        await self.connection.execute(stmt)

    def _row_to_system_configuration(self, row: Row) -> SystemConfiguration:
        return SystemConfiguration(
            default_llm_model=row.default_llm_model,
            default_embedding_model=row.default_embedding_model,
            updated_at=row.updated_at,
            created_by=row.created_by,
        )
