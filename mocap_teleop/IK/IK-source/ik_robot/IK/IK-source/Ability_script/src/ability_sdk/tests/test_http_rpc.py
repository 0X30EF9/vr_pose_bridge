import os
import threading
import time

import preload
from flask import Flask, request

from AbilitySDK.log import configure_logging
from AbilitySDK.RPC import http_rpc

host_info = http_rpc.HttpHost(hostname="127.0.0.1", port=8088)
caller = http_rpc.HttpRpcCall(
    uri="/test", host=host_info, method=http_rpc.HttpMethod.GET)
try:
    response = caller.call()
    print("Response:", response)
except RuntimeError as e:
    print("Error during RPC call:", e)


configure_logging()

app = Flask(__name__)
app_name = "test_http_rpc:app"


@app.route('/test', methods=['GET'])
def test_endpoint():
    return {"message": "Hello, World!"}, 200


@app.route('/shutdown', methods=['GET'])
def shutdown():
    # 关闭app
    def shutdown_server():
        time.sleep(1)
        os._exit(0)
    threading.Thread(target=shutdown_server).start()
    return {"message": "Server shutting down..."}, 200


app.run(host=host_info.hostname, port=host_info.port)
