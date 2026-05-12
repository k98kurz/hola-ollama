from enum import Enum
from sqloquent import SqlModel, Default, RelatedModel
from hola.ollama.types import Role
import packify


_empty = packify.pack({})


class Message(SqlModel):
    """Database model for storing conversation messages with optional
        tool call details.
    """
    connection_info: str = 'hola.db'
    table: str = 'messages'
    id_column: str = 'id'
    columns: tuple[str,] = (
        'id', 'session_id', 'sequence', 'role', 'model', 'call_id', 'content',
        'details'
    )
    id: str
    session_id: str
    sequence: int
    role: str | Default['user']
    model: str | None
    call_id: str | None
    content: str
    details: bytes | None
    session: RelatedModel

    @property
    def role(self) -> Role:
        """Return the message role as a `Role` enum. Can be set using the
            setter to change the message role.
        """
        return Role(self.data.get('role', 'user'))

    @role.setter
    def role(self, val: Role) -> None:
        self.data['role'] = val.value

    @property
    def details(self) -> packify.SerializableType:
        """Return the deserialized message details dictionary. Can be set
            using the setter to serialize and store a new dictionary.
        """
        return packify.unpack(self.data.get('details', None) or _empty)

    @details.setter
    def details(self, val: packify.SerializableType) -> None:
        self.data['details'] = packify.pack(val)

