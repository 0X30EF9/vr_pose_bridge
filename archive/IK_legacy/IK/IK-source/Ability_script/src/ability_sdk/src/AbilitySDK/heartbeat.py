import logging
import threading
import time

from .Config import ability_info
from .RPC import rpc


class Heartbeater:
    def __init__(self, ab_info: ability_info.AbilityInfo, interval: int = 5, caller: rpc.RpcCall = None):
        self.ability_info = ab_info
        self.interval = interval
        self.caller = caller
        self.is_running = False
        self.thread = None

    def start(self):
        if self.is_running:
            return
        if self.thread and self.thread.is_alive():
            # 重新创建线程
            self.thread.join()
        self.is_running = True
        self.thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.is_running = False

    def refresh(self):
        if not self.caller:
            raise ValueError("RPC caller is not set.")
        try:
            py = self.ability_info.heartbeat_info()
            # logging.warning(f"Sending heartbeat with payload: {py}")
            self.caller.call(payload=self.ability_info.heartbeat_info())
        except Exception as e:
            logging.error(f"Failed to send heartbeat: {e}")

    def _refresh_loop(self):
        while self.is_running and self.ability_info.status != ability_info.LifecycleState.Terminated:
            self.refresh()
            time.sleep(self.interval)
