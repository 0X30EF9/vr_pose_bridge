import logging
import logging.handlers
import os
import time
from datetime import datetime
from pathlib import Path

IS_LOGGING_CONFIGURED = False


def get_default_log_path():
    """获取默认日志路径，如果不存在则返回None"""
    home_path = os.getenv("ABILITY_FRAMEWORK_HOME")
    if not home_path:
        return "./log"

    log_path = Path(home_path) / "log"
    if log_path.exists() and log_path.is_dir():
        return str(log_path)
    return "./log"


def parse_file_name(argv0: str):
    """解析文件名，仅去除路径"""
    if not argv0:
        return "app"
    return Path(argv0).name


class CustomFileFormatter(logging.Formatter):
    """文件日志格式化器，不带颜色"""

    def formatTime(self, record, datefmt=None):
        """重写时间格式化方法"""
        ct = self.converter(record.created)
        if datefmt:
            return time.strftime(datefmt, ct)
        t = time.strftime("%Y%m%d %H:%M:%S", ct)
        s = "%s.%06d" % (t, record.msecs * 1000)
        return s

    def format(self, record):
        """重写日志记录格式化方法"""
        message = super().format(record)
        # 格式处理
        level_abbr = record.levelname[0]
        timestamp = self.formatTime(record)
        thread_id = record.thread
        filename = record.filename
        lineno = record.lineno
        # 目标格式
        return f"{level_abbr}{timestamp} {thread_id} {filename}:{lineno}] {message}"


class CustomConsoleFormatter(logging.Formatter):
    """控制台日志格式化器，使用ANSI字符控制颜色"""

    COLORS = {
        "INFO": "\033[0m",  # 默认白色
        "WARNING": "\033[33m",  # 黄色
        "ERROR": "\033[31m",  # 红色
        "CRITICAL": "\033[1;31m",  # 粗体红色
        "DEBUG": "\033[32m",  # 绿色
        "FATAL": "\033[1;31m",  # 粗体红色
    }
    RESET = "\033[0m"

    def formatTime(self, record, datefmt=None):
        """重写时间格式化方法"""
        ct = self.converter(record.created)
        if datefmt:
            return time.strftime(datefmt, ct)
        t = time.strftime("%Y%m%d %H:%M:%S", ct)
        s = "%s.%06d" % (t, record.msecs * 1000)
        return s

    def format(self, record):
        """重写日志记录格式化方法"""
        # 获取原始消息
        message = record.getMessage()
        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)
        # 获取颜色
        color = self.COLORS.get(record.levelname, self.RESET)
        # 格式处理
        level_abbr = record.levelname[0]
        timestamp = self.formatTime(record)
        thread_id = record.thread
        filename = record.filename
        lineno = record.lineno
        # 目标格式
        return f"{color}{level_abbr}{timestamp} {thread_id} {filename}:{lineno}] {message}{self.RESET}"


def configure_logging(argv0=None):
    """配置logging日志系统"""
    global IS_LOGGING_CONFIGURED
    if IS_LOGGING_CONFIGURED:
        return

    # 创建logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # 清除现有handler
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 文件路径
    log_path = get_default_log_path()
    Path(log_path).mkdir(parents=True, exist_ok=True)
    file_name = parse_file_name(argv0)
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S.%f')
    file_formatter = CustomFileFormatter(
        fmt="%(message)s",
        style="%"
    )
    # 文件handler
    for level_name, level in [("INFO", logging.INFO), ("WARNING", logging.WARNING), ("ERROR", logging.ERROR)]:
        level_log_file = Path(log_path) / \
            f"{file_name}.log.{level_name}.{timestamp}"
        file_handler = logging.handlers.RotatingFileHandler(
            filename=level_log_file,
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=10,
            encoding="utf-8"
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
        # 创建符号链接
        symlink_path = Path(log_path) / f"{file_name}.{level_name}"
        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()
        symlink_path.symlink_to(level_log_file)

    # 控制台handler
    console_formatter = CustomConsoleFormatter(
        fmt="%(message)s",
        style="%"
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    IS_LOGGING_CONFIGURED = True
