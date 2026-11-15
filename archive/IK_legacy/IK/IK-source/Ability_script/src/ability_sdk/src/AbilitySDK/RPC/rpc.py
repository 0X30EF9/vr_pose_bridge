import enum


class ProtocolType(enum.Enum):
    HTTP = "http"


class RpcCall:
    """
    规定RPC调用的接口
    """

    def __init__(self, uri: str, porotocol: ProtocolType):
        self.uri = uri
        self.protocol = porotocol

    def call(self, payload: dict):
        raise NotImplementedError()
