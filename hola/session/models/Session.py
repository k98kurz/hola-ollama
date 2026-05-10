from sqloquent import SqlModel, RelatedCollection
import packify


_empty = packify.pack({})


class Session(SqlModel):
    """Database model for storing session metadata and FSM state."""
    connection_info: str = 'hola.db'
    table: str = 'sessions'
    id_column: str = 'id'
    columns: tuple[str,] = ('id', 'cwd', 'metadata', 'state')
    id: str
    cwd: str | None
    metadata: bytes | None
    state: bytes | None
    memories: RelatedCollection
    messages: RelatedCollection

    @property
    def metadata(self) -> dict[str, object]:
        """Return the deserialized session metadata dictionary. Can be set
            using the setter to serialize and store a new dictionary.
        """
        return packify.unpack(self.data.get('metadata', None) or _empty)

    @metadata.setter
    def metadata(self, val: dict[str, object]) -> None:
        self.data['metadata'] = packify.pack(val)

