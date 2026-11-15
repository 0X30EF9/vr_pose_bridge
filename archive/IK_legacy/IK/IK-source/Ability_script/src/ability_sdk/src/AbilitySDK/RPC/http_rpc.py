import dataclasses
import enum

import requests

from . import rpc


class HttpMethod(enum.Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"


@dataclasses.dataclass
class HttpHost:
    """
    HTTP主机信息
    """
    hostname: str
    port: int

    def __post_init__(self):
        if not self.hostname:
            raise ValueError("Hostname cannot be empty")
        if not (0 <= self.port <= 65535):
            raise ValueError(f"Invalid port: {self.port}")


class HttpRpcCall(rpc.RpcCall):
    """
    HTTP RPC实现。
    """

    def __init__(self, uri: str, host: HttpHost, method: HttpMethod):
        super().__init__(uri, rpc.ProtocolType.HTTP)
        self.host = host
        self.method = method

    def call(self, payload=None, params=None):
        """
        执行HTTP RPC调用。
        """
        url = f"http://{self.host.hostname}:{self.host.port}{self.uri}"
        headers = {"Content-Type": "application/json"}

        try:
            if self.method == HttpMethod.GET:
                response = requests.get(url, params=params, headers=headers)
            elif self.method == HttpMethod.POST:
                response = requests.post(
                    url, params=params, json=payload, headers=headers)
            elif self.method == HttpMethod.PUT:
                response = requests.put(
                    url, params=params, json=payload, headers=headers)
            elif self.method == HttpMethod.DELETE:
                response = requests.delete(
                    url, params=params, json=payload, headers=headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {self.method}")

            response.raise_for_status()

            if response.headers.get("Content-Type", "").startswith("application/json"):
                return response.json()
            return response.text

        except requests.RequestException as e:
            raise RuntimeError(f"HTTP RPC call failed: {e}")
