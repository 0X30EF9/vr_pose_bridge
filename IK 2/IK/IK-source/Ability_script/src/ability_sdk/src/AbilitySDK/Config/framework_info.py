import uuid

from ..RPC import http_rpc


class FrameworkInfo:
    """
    框架信息"""

    def __init__(self, port: int, redis_url: str = None):
        self.host = http_rpc.HttpHost(hostname="localhost", port=port)
        self.redis_url = redis_url


class FrameworkInterfaceUri:
    """
    框架HTTP服务器接口URI
    """

    @staticmethod
    def cr_by_id(ability_instance_id: uuid.UUID) -> str:
        """
        从框架获取CR
        """
        return f"/api/cr/{str(ability_instance_id)}"

    @staticmethod
    def crd_by_id(ability_instance_id: uuid.UUID) -> str:
        """
        从框架获取CRD
        """
        return f"/api/crd/{str(ability_instance_id)}"

    @staticmethod
    def ability_heartbeat() -> str:
        """
        能力心跳接口
        """
        return "/api/ability-heartbeat"

    @staticmethod
    def lifecycle_request() -> str:
        """
        生命周期请求接口
        """
        return "/api/lifecycle-request"

    @staticmethod
    def task_status() -> str:
        """
        任务状态接口
        """
        return "/api/task-status"
