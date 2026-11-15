import logging

import preload

from AbilitySDK.log import configure_logging

if __name__ == "__main__":
    configure_logging("test")
    logging.info("日志配置测试 - 信息级别")
    logging.warning("这是一个警告消息")
    logging.error("这是一个错误消息")
    try:
        1 / 0
    except Exception as e:
        logging.error("发生了一个错误", exc_info=True)