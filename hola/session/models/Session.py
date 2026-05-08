from sqloquent import SqlModel, RelatedCollection
import packify


_empty = packify.pack({})


class Session(SqlModel):
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
        return packify.unpack(self.data.get('metadata', None) or _empty)

    @metadata.setter
    def metadata(self, val: dict[str, object]) -> None:
        self.data['metadata'] = packify.pack(val)

