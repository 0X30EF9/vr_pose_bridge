import dataclasses
import enum
import inspect
import logging
import threading
import time
import uuid

from . import ipc_server_handle
from .Config import framework_info
from .RPC import http_rpc


class TaskState(enum.Enum):
    unstarted = 0
    running = 1
    finished = 2
    error = 3
    cancelled = 4


class ExecutorType(enum.Enum):
    unknown = 0
    ability = 1
    controller = 2
    framework = 3


def get_time_str():
    """格式：%Y-%m-%d %H:%M:%S.%3N"""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + f".{int(time.time() * 1000) % 1000:03d}"


@dataclasses.dataclass
class TaskStatus:
    executor_id: uuid.UUID
    id: uuid.UUID = None
    executor_type: ExecutorType = ExecutorType.ability
    state: TaskState = TaskState.unstarted
    start_time: str = None
    end_time: str = None
    timeout: int = 0
    payload: dict = dataclasses.field(default_factory=dict)
    message: str = None

    def __post_init__(self):
        # 生成任务id
        if self.id is None:
            if self.executor_id is not None:
                self.id = uuid.uuid5(uuid.NAMESPACE_DNS,
                                     f"{self.executor_id}-{uuid.uuid1().time}")
            else:
                raise ValueError("executor_id must be provided")

    def __dict__(self):
        return {
            "id": str(self.id),
            "executor_id": str(self.executor_id),
            "executor_type": self.executor_type.name,
            "state": self.state.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "timeout": self.timeout,
            "payload": self.payload,
            "message": self.message
        }


def update_task_status(task_status: TaskStatus) -> bool:
    """更新任务状态"""
    host = ipc_server_handle.ABILITY_INFO.fwk_info.host
    uri = framework_info.FrameworkInterfaceUri.task_status()
    caller = http_rpc.HttpRpcCall(uri, host, http_rpc.HttpMethod.POST)
    try:
        caller.call(payload=task_status.__dict__())
    except Exception as e:
        logging.error(f"Failed to update task status: {e}")
        return False
    return True


def execute_task_simple(task_func, executor_id: uuid.UUID, input: dict = {}, executor_type: ExecutorType = ExecutorType.ability, timeout: int = 0):
    """
    任务执行器的封装。
    @param task_func: 任务函数，要求输入必须包含一个dict，可选一个TaskStatus用于实时更新状态，
        使用两个参数时，dict参数名为input，TaskStatus参数名为task_status；
        输出为一个dict，（如果有）.output键放入payload（dict），
            .message键放入message（str），.state键放入state（TaskState）。
        示例task_func：myfunc1(input: dict) -> dict；myfunc2(input: dict, task_status: TaskStatus = None) -> dict。
    @return: dict，task_id键为str类型的任务ID。
    """
    # 注册任务
    stat = TaskStatus(executor_id=executor_id, executor_type=ExecutorType.ability,
                      timeout=timeout, state=TaskState.unstarted)
    update_task_status(stat)

    # 执行任务
    threading.Thread(target=__execute_task_core,
                     args=(task_func, input, stat)).start()

    # 返回任务id
    return {"task_id": str(stat.id)}


def __execute_task_core(task_func, input: dict, stat: TaskStatus):
    # 任务开始
    stat.start_time = get_time_str()
    stat.state = TaskState.running
    update_task_status(stat)

    # 执行任务
    try:
        if not callable(task_func):
            raise TypeError("task_func must be callable")
        params = inspect.signature(task_func).parameters
        if len(params) == 1:
            ret = task_func(input)
        elif len(params) == 2:
            param_names = [p.name for p in params.values()]
            if set(param_names) == {"input", "task_status"}:
                ret = task_func(input=input, task_status=stat)
            else:
                raise ValueError(f"invalid parameters name {params}")
        else:
            raise ValueError(f"invalid parameter length {params}")
    except Exception as e:
        logging.error(f"Task {stat.id} execution failed: {e}")
        stat.state = TaskState.error
        stat.message = str(e)
        update_task_status(stat)
        return

    # 检查输出
    state = ret.get("state", TaskState.finished)
    payload = ret.get("output", {})
    message = ret.get("message", "")
    if state != TaskState.finished and state != TaskState.error and state != TaskState.cancelled:
        logging.error(
            f"Invalid state: {state}. Using TaskState.finished.")
        stat.state = TaskState.finished
    else:
        stat.state = state
    if not isinstance(payload, dict):
        logging.error(
            f"Invalid type for payload: {type(payload)}. Using empty dict.")
        stat.payload = {}
    else:
        stat.payload = payload
    if not isinstance(message, str):
        logging.error(
            f"Invalid type for message: {type(message)}. Using empty string.")
        stat.message = ""
    else:
        stat.message = message
    # 任务结束
    stat.end_time = get_time_str()
    update_task_status(stat)
