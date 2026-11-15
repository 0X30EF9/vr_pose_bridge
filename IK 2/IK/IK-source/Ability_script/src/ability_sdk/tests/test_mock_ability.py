import logging
import sys
import threading
import uuid

import flask
import preload

from AbilitySDK import ability_stub, ipc_server_handle, log, task


class MyAbility(ability_stub.AbilityInterface):
    def __init__(self):
        logging.warning("MyAbility initialized")
        self.app = flask.Flask(__name__)

        @self.app.route('/test', methods=['GET'])
        def test_route():
            logging.warning("Test route called")
            return "Test successful", 200

        @self.app.route('/task', methods=['GET'])
        def task_route():
            logging.warning("Task route called")
            ret = task.execute_task_simple(
                self.my_task_func,
                executor_id=uuid.uuid4(),
            )
            return ret, 200

    def my_task_func(self, input: dict = None) -> dict:
        ret = {"output": "task executed successfully",
               "message": "Task completed"}
        logging.warning(
            f"my_task_func called with input: {input}, returning: {ret}")
        return ret

    def on_start(self):
        logging.warning("on_start called")

    def on_connect(self):
        threading.Thread(target=self.app.run, kwargs={
                         "host": '0.0.0.0', "port": 5000}).start()
        self.ability_info.ability_port = 5000
        logging.warning("on_connect called")

    def on_disconnect(self):
        logging.warning("on_disconnect called")

    def on_terminate(self):
        logging.warning("on_terminate called")


if __name__ == "__main__":
    log.configure_logging(sys.argv[0])
    ability = MyAbility()
    handle = ipc_server_handle.HttpIpcServerHandle(sys.argv, ability)
    handle.run()
    logging.warning("process ended")
