from sqloquent import SqlModel, RelatedModel
import packify


_None = packify.pack(None)


class Memory(SqlModel):
    """Database model for storing key-value memory entries for a session."""
    connection_info: str = 'hola.db'
    table: str = 'memories'
    id_column: str = 'id'
    columns: tuple[str, ...] = ('id', 'session_id', 'name', 'content')
    id: str
    session_id: str
    name: str
    content: bytes
    session: RelatedModel

    @property
    def content(self) -> packify.SerializableType:
        """Return the deserialized memory content value. Can be set
            using the setter to serialize and store a new value.
        """
        return packify.unpack(self.data.get('content', None) or _None)

    @content.setter
    def content(self, val: packify.SerializableType) -> None:
        self.data['content'] = packify.pack(val)

