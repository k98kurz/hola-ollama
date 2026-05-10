from enum import Enum
from sqloquent import SqlModel, Default, RelatedModel
import packify


_empty = packify.pack({})


class MType(Enum):
    """Message type enumeration for conversation messages."""
    USER = 'USER'
    CLANKER = 'CLANKER'
    TOOL = 'TOOL'


class Message(SqlModel):
    """Database model for storing conversation messages with optional
        tool call details.
    """
    connection_info: str = 'hola.db'
    table: str = 'messages'
    id_column: str = 'id'
    columns: tuple[str,] = (
        'id', 'session_id', 'sequence', 'type', 'model', 'call_id', 'content',
        'details'
    )
    id: str
    session_id: str
    sequence: int
    type: str | Default['USER']
    model: str | None
    call_id: str | None
    content: str
    details: bytes | None
    session: RelatedModel

    @property
    def type(self) -> MType:
        """Return the message type as an `MType` enum. Can be set using the
            setter to change the message type.
        """
        return MType(self.data.get('type', 'USER'))

    @type.setter
    def type(self, val: MType) -> None:
        self.data['type'] = val.value

    @property
    def details(self) -> packify.SerializableType:
        """Return the deserialized message details dictionary. Can be set
            using the setter to serialize and store a new dictionary.
        """
        return packify.unpack(self.data.get('details', None) or _empty)

    @details.setter
    def details(self, val: packify.SerializableType) -> None:
        self.data['details'] = packify.pack(val)

