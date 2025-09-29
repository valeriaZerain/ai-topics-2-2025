from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from enum import Enum
from datetime import datetime
import random
import csv
import io


class RaceEnum(str, Enum):
    orc = "ORC"
    elf = "ELF"
    human = "HUMAN"
    goblin = "GOBLIN"

class Guild(BaseModel):
    id: int
    name: str
    realm: str
    created: datetime

class Character(BaseModel):
    id: int
    name: str
    level: int
    race: RaceEnum
    hp: int
    damage: int | None = None
    guild: Guild

class CharacterCreate(BaseModel):
    name: str
    level: int
    race: RaceEnum
    hp: int
    damage: int
    guild_id: int

app = FastAPI(title="validacion")

guilds: list[Guild] = []
characters: list[Character] = []


@app.post("/guilds", status_code=201)
def create_guild(guild: Guild) -> list[Guild]:
    guilds.append(guild)
    return guilds


@app.post("/characters", status_code=201)
def create_character(character: CharacterCreate):
    id = random.randint(0, 9999)
    guilds_found = [g for g in guilds if g.id == character.guild_id]
    if not guilds_found:
        raise HTTPException(status_code=404, detail="guild not found")
    guild = guilds_found[0]
    new_character = Character(
        id=id,
        guild=guild,
        **character.model_dump(exclude={"guild_id"})
    )
    characters.append(new_character)
    return characters

@app.get("/characters/report", responses={200: {"content": {"text/csv": {}}}})
def download_characters_csv() -> Response:
    if not characters:
        raise HTTPException(status_code=404, detail="No characters found")

    csv_stream = io.StringIO()
    writer = csv.DictWriter(
        csv_stream,
        fieldnames=[
            "id",
            "name",
            "level",
            "race",
            "hp",
            "damage",
            "guild_id",
            "guild_name",
            "guild_realm",
            "guild_created",
        ],
        quoting=csv.QUOTE_ALL
    )
    writer.writeheader()

    for c in characters:
        writer.writerow({
            "id": c.id,
            "name": c.name,
            "level": c.level,
            "race": c.race,
            "hp": c.hp,
            "damage": c.damage if c.damage is not None else "",
            "guild_id": c.guild.id,
            "guild_name": c.guild.name,
            "guild_realm": c.guild.realm,
            "guild_created": c.guild.created.isoformat(),
        })

    return Response(content=csv_stream.getvalue(), media_type="text/csv")
