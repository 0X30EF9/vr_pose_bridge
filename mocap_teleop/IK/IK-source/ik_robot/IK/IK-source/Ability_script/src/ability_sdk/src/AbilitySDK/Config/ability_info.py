import dataclasses
import enum
import tempfile
import time
import uuid

from . import framework_info


class LifecycleState(enum.Enum):
    """
    生命周期状态枚举
    """
    Unknown = 10
    Inactive = 0
    Init = 1
    Standby = 2
    Running = 3
    Suspend = 4
    Terminated = 5
    Error = 6


@dataclasses.dataclass
class AbilityBaseInfo:
    """
    存储能力的标识信息
    """
    ability_instance_id: uuid.UUID
    ability_name: str = None
    instance_name: str = None
    version: str = None
    config: dict = None # CR中的spec/config部分，JSON格式


class AbilityInfo:
    """
    能力信息
    """

    def __init__(self, base_info: AbilityBaseInfo, framework_info: framework_info.FrameworkInfo):
        # 基本信息
        self.base_info = base_info
        # 框架信息
        self.fwk_info = framework_info
        # 运行状态信息
        self.status = LifecycleState.Init
        self.ipc_port = 0
        self.ability_port = 0
        # 目录信息
        self.instance_directory = None
        self.assets_directory = None
        self.tmp_directory = tempfile.gettempdir() + "/" + str(base_info.ability_instance_id) + \
            "/" + str(int(time.time_ns())) + "/"

    def heartbeat_info(self) -> dict:
        """
        返回能力信息的JSON表示，只包含心跳包所需信息
        """
        return {
            "id": str(self.base_info.ability_instance_id),
            "abilityName": self.base_info.ability_name,
            "instanceName": self.base_info.instance_name,
            "state": self.status.name,
            "IPCPort": self.ipc_port,
            "abilityPort": self.ability_port,
            "version": self.base_info.version,
            "IPCProtocol": "http"
        }
