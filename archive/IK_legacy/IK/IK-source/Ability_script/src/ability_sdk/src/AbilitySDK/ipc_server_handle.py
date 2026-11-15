import json
import logging
import socket
import uuid

import werkzeug.serving
from flask import Flask, jsonify

from . import ability_stub, heartbeat
from .Config import ability_info, framework_info
from .log import configure_logging
from .RPC import http_rpc

# 能力信息，在IpcServerHandle实例化时设置，此引用只读
ABILITY_INFO = None


class IpcServerHandle:
    def __init__(self, argv, ability: ability_stub.AbilityInterface):
        configure_logging(argv[0])
        self._parse_info_from_argv(argv)
        self._parse_info_from_cr_warp()
        if self.protocol == "http":
            hb_caller = http_rpc.HttpRpcCall(
                framework_info.FrameworkInterfaceUri.ability_heartbeat(),
                self.ability_info.fwk_info.host,
                http_rpc.HttpMethod.POST)
        self.heartbeater = heartbeat.Heartbeater(
            self.ability_info,
            caller=hb_caller
        )
        self.ability = ability
        global ABILITY_INFO
        ABILITY_INFO = self.ability_info

    def run(self):
        raise NotImplementedError()

    def _parse_info_from_argv(self, argv):
        if len(argv) < 3:
            raise ValueError("cmd args must be > 3!")
        ab_inst_id = uuid.UUID(argv[1])
        if not ab_inst_id:
            raise ValueError("Invalid instance ID!")
        j = json.loads(argv[2])
        # 框架信息
        protocol = j.get("protocol", "http")
        self.protocol = protocol
        if protocol == "http":
            port = j["port"]
            redis_url = j.get("redisUrl", "redis://localhost:9090")
            fwk_info = framework_info.FrameworkInfo(
                port=port, redis_url=redis_url)
        # 能力信息
        inst_dir = j["instanceDirectory"]
        assets_dir = j["assetsDirectory"]
        ab_base_info = ability_info.AbilityBaseInfo(
            ability_instance_id=ab_inst_id)

        self.ability_info = ability_info.AbilityInfo(
            base_info=ab_base_info,
            framework_info=fwk_info
        )
        self.ability_info.instance_directory = inst_dir
        self.ability_info.assets_directory = assets_dir

    def _parse_info_from_cr_warp(self):
        if self.protocol == "http":
            caller = http_rpc.HttpRpcCall(framework_info.FrameworkInterfaceUri.cr_by_id(
                self.ability_info.base_info.ability_instance_id), self.ability_info.fwk_info.host, http_rpc.HttpMethod.GET)
            res = caller.call()
            if res is None:
                raise ValueError("Failed to get CR info from framework!")
            self._parse_info_from_cr(res)
        else:
            raise NotImplementedError()

    def _parse_info_from_cr(self, cr):
        j = cr
        self.ability_info.base_info.instance_name = j["metadata"]["name"]
        self.ability_info.base_info.ability_name = j["spec"]["abilityName"]
        self.ability_info.base_info.version = j["spec"]["version"]
        self.ability_info.base_info.config = j["spec"]["config"]
        self.ability_info.lifecycle_state = ability_info.LifecycleState[
            j["runInfo"]["lifecycleState"]]

    def _handle_on_start(self):
        logging.warning("receive START command")
        if self.ability_info.status not in (ability_info.LifecycleState.Inactive, ability_info.LifecycleState.Init):
            raise RuntimeError("wrong lifecycle, now is %s)" %
                               self.ability_info.status.name)
        self.ability.load_info(self.ability_info)
        self.ability.on_start()
        self.ability_info.status = ability_info.LifecycleState.Standby
        self.heartbeater.refresh()

    def _handle_on_connect(self):
        logging.warning("receive CONNECT command")
        if self.ability_info.status not in (ability_info.LifecycleState.Standby, ability_info.LifecycleState.Suspend):
            raise RuntimeError("wrong lifecycle, now is %s)" %
                               self.ability_info.status.name)
        self.ability.on_connect()
        if not 0 <= self.ability_info.ability_port <= 65535:
            raise ValueError("invalid ability port: %s" %
                             self.ability_info.ability_port)
        self.ability_info.status = ability_info.LifecycleState.Running
        self.heartbeater.refresh()

    def _handle_on_disconnect(self):
        logging.warning("receive DISCONNECT command")
        # fix: argument of type "LifecycleState" is not iterable
        if self.ability_info.status != ability_info.LifecycleState.Running:
            raise RuntimeError("wrong lifecycle, now is %s)" %
                               self.ability_info.status.name)
        self.ability.on_disconnect()
        self.ability_info.status = ability_info.LifecycleState.Suspend
        self.heartbeater.refresh()

    def _handle_on_terminate(self):
        logging.warning("receive TERMINATE command")
        self.ability.on_terminate()
        self.ability_info.status = ability_info.LifecycleState.Terminated
        self.heartbeater.refresh()


class HttpIpcServerHandle(IpcServerHandle):
    def __init__(self, argv, ability: ability_stub.AbilityInterface):
        super().__init__(argv, ability)

        # 初始化服务器

        self.server = None

        # 初始化app

        self.app = Flask(__name__)

        def handle_start():
            try:
                self._handle_on_start()
                return jsonify({"status": "success"}), 200
            except Exception as e:
                logging.error(f"Error in handle_start: {e}")
                return jsonify({"status": "error", "message": str(e)}), 500

        def handle_connect():
            try:
                self._handle_on_connect()
                return jsonify({"status": "success"}), 200
            except Exception as e:
                logging.error(f"Error in handle_connect: {e}")
                return jsonify({"status": "error", "message": str(e)}), 500

        def handle_disconnect():
            try:
                self._handle_on_disconnect()
                return jsonify({"status": "success"}), 200
            except Exception as e:
                logging.error(f"Error in handle_disconnect: {e}")
                return jsonify({"status": "error", "message": str(e)}), 500

        def handle_terminate():
            try:
                self._handle_on_terminate()
                if self.server:
                    self.server.shutdown()
                    self.server.server_close()
                return jsonify({"status": "success"}), 200
            except Exception as e:
                logging.error(f"Error in handle_terminate: {e}")
                return jsonify({"status": "error", "message": str(e)}), 500

        self.app.add_url_rule("/api/lifecycle/start", "handle_start",
                              handle_start, methods=["POST"])
        self.app.add_url_rule("/api/lifecycle/connect",
                              "handle_connect", handle_connect, methods=["POST"])
        self.app.add_url_rule("/api/lifecycle/disconnect",
                              "handle_disconnect", handle_disconnect, methods=["POST"])
        self.app.add_url_rule("/api/lifecycle/terminate",
                              "handle_terminate", handle_terminate, methods=["POST"])
        

    def run(self):
        # 框架只从第一个心跳包读取ipc_port，所以必须确定端口再启动心跳器
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            random_port = s.getsockname()[1]
        self.ability_info.ipc_port = random_port
        self.heartbeater.start()
        self._handle_on_start()

        self.__run_flask_server()

        self.heartbeater.stop()

    def __run_flask_server(self):
        self.server = werkzeug.serving.make_server("127.0.0.1", self.ability_info.ipc_port, self.app, threaded=True)
        self.server.serve_forever()
