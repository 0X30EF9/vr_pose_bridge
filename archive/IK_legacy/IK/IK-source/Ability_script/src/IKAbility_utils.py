import socket, os, sys, re, logging, math, copy
import numpy as np

from contextlib import closing
from pathlib import Path
from typing import Tuple
from urllib.parse import urlparse

essential_env_keys = ('ABILITY_ROOT', 'ABILITY_FRAMEWORK_HOME', 'ABILITY_HOME')

left_init_position = np.asarray([0.0250, 0.2670, -0.3909])
right_init_position = np.asarray([0.0250, -0.2670, -0.3909])
left_init_rotation = np.asarray([[1.0, 0.0, -0.0], [-0.0, 0.99876, -0.1566], [0.0, 0.1566, 0.9876]])
right_init_rotation = np.asarray([[1.0, 0.0, -0.0], [-0.0, 0.99876, 0.1566], [0.0, -0.1566, 0.9876]])

joint_control_dict = {"name": 0.0, "pos": 0.0, "spd": 0.3, "cur": 4.0}

motors = [copy.deepcopy(joint_control_dict) for _ in range(14)]

for i, motor in enumerate(motors, 11):
    motor["name"] = (i if i < 18 else i + 3)

def find_free_port():
    """获取当前机器的随机端口，将其设置为当前能力的端口号"""

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def detect_ability_name(default: str = "HeadAbility"):
    """根据当前输入信息再次获取ability_name"""

    env_name = os.environ.get('ABILITY_NAME')
    if env_name:
        return env_name.strip()
    try:
        base = Path(sys.argv[0]).name
        if base.endswith('.ability'):
            base = base[:-len('.ability')]
        if '.' in base:
            base = base.split('.', 1)[0]

        base = base or default

        if default.endswith('Ability') and not base.endswith('Ability'):
            return default
        return base

    except Exception:
        return default


def iter_candidate_bases():
    """根据当前运行文件遍历当前目录并且输出目录信息"""
    bases = []
    # __file__ directory
    try:
        bases.append(Path(__file__).resolve())
    except Exception:
        pass
    # CWD
    try:
        bases.append(Path.cwd().resolve())
    except Exception:
        pass
    # PyInstaller temp dir
    meipass = getattr(sys, '_MEIPASS', None)
    if meipass:
        try:
            bases.append(Path(meipass).resolve())
        except Exception:
            pass
    return bases


def extract_field(text: str, key: str):
    """关键词提取"""

    m = re.search(rf'"{re.escape(key)}"\s*:\s*"([^"]+)"', text)
    if m:
        return m.group(1)

    m = re.search(rf'^[ \t]*{re.escape(key)}\s*:\s*([\"\']?)([^\n\r#\"\']+)\1\s*$', text, re.MULTILINE)
    if m:
        return m.group(2).strip()
    return None


def find_crs_dir():
    """获取当前的crs目录"""

    # Environment overrides first
    for env_key in essential_env_keys:
        env_home = os.environ.get(env_key)
        if env_home:
            try:
                p = Path(env_home).resolve()
                crs = p / 'crs'
                if crs.exists() and crs.is_dir():
                    return crs
            except Exception:
                pass

    candidates = iter_candidate_bases()
    best_match = None
    for base in candidates:
        for parent in [base] + list(base.parents):
            crs = parent / 'crs'
            if crs.exists() and crs.is_dir():

                packages = parent / 'packages'
                if packages.exists() and packages.is_dir():
                    return crs
                if best_match is None:
                    best_match = crs
    return best_match


def find_config_for_ability(ability_name: str):
    """根据当前传入的ability名称匹配最优的config文件"""

    crs_dir = find_crs_dir()
    if not crs_dir:
        return None
    yaml_files = list(crs_dir.glob('*.yaml')) + list(crs_dir.glob('*.yml'))
    matches = []
    for p in yaml_files:
        try:
            text = p.read_text(encoding='utf-8')
        except Exception:
            continue
        name = extract_field(text, 'abilityName')
        if name and name.strip() == ability_name:
            matches.append((p, text))
    if not matches:
        return None
    # Prefer file name containing the ability name (loose heuristic)
    lower = ability_name.lower()
    for p, _ in matches:
        if lower in p.name.lower():
            return p
    return matches[0][0]


def find_crs_config_file(ABILITY_NAME: str):
    crs_dir = find_crs_dir()
    if not crs_dir:
        return None, 'crs dir not found'
    # First try ability-specific yaml selection
    p = find_config_for_ability(ABILITY_NAME or '')
    if p:
        return p, 'matched by abilityName'
    # Fallback to generic names
    candidates = [
        crs_dir / 'config',
        crs_dir / 'config.yaml',
        crs_dir / 'config.yml',
        crs_dir / 'config.json',
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p, 'fallback generic config name'
    return None, 'no known config file'


def load_robot_host_port(default_host: str = '192.168.26.1', default_port: int = 65521, LAST_CFG_PATH=None, ABILITY_NAME=''):

    """根据当前获取的ABILITY_NAME获取当前config文件中的host和address"""

    try:
        primary_name = (ABILITY_NAME or '').strip()
        fallback_name = detect_ability_name('HeadAbility')
        logging.warning(f"[CFG] lookup start: primary='{primary_name}', fallback='{fallback_name}'")

    except Exception:
        primary_name = (ABILITY_NAME or '').strip()
        fallback_name = 'HeadAbility'

    cfg_path = None

    if primary_name:
        cfg_path = find_config_for_ability(primary_name)
        if cfg_path:
            try:
                logging.warning(f"[CFG] matched by primary abilityName: {primary_name} -> {cfg_path}")
            except Exception:
                pass

    if not cfg_path and fallback_name and fallback_name != primary_name:
        retry = find_config_for_ability(fallback_name)
        if retry:
            cfg_path = retry
            try:

                ABILITY_NAME = fallback_name
                logging.warning(f"[CFG] matched by fallback abilityName: {fallback_name} -> {cfg_path}")
            except Exception:
                pass

    # Step 3: generic fallback files in crs
    if not cfg_path:
        cfg_path_generic, reason = find_crs_config_file(ABILITY_NAME=ABILITY_NAME)
        cfg_path = cfg_path_generic
        try:
            crs_dir = find_crs_dir()
            logging.warning(f"[CFG] generic fallback: reason='{reason}', crs_dir={crs_dir}, chosen={cfg_path}")
            if not cfg_path and crs_dir:
                yfiles = list(crs_dir.glob('*.yaml')) + list(crs_dir.glob('*.yml'))
                logging.warning(f"[CFG] yaml candidates: {[p.name for p in yfiles]}")
        except Exception:
            pass

    if not cfg_path:
        LAST_CFG_PATH = None
        logging.warning(f"[CFG] no config matched; using defaults {default_host}:{default_port}")
        return default_host, default_port, ABILITY_NAME, LAST_CFG_PATH

    try:

        text = cfg_path.read_text(encoding='utf-8')

        uri = extract_field(text, 'robotUri')
        if not uri:
            # Fallback to permissive regex (legacy)
            m = re.search(r'robotUri\s*:\s*[\"\']?([^\n\r#\"\']+)', text)
            uri = m.group(1).strip() if m else None
        if not uri:
            LAST_CFG_PATH = str(cfg_path)
            logging.warning(f"[CFG] robotUri not found in {cfg_path}; using defaults {default_host}:{default_port}")

            return default_host, default_port, ABILITY_NAME, LAST_CFG_PATH

        parsed = urlparse(uri)
        host = parsed.hostname or default_host
        port = parsed.port or default_port
        LAST_CFG_PATH = str(cfg_path)
        logging.warning(f"[CFG] using {host}:{port} from {cfg_path}")
        return host, int(port), ABILITY_NAME, LAST_CFG_PATH

    except Exception as e:

        LAST_CFG_PATH = str(cfg_path)

        return default_host, default_port, ABILITY_NAME, LAST_CFG_PATH


# 以下为暂时未用到的函数
def _parse_robot_uri_from_text(text: str):
    """Extract robotUri from either top-level or nested under 'config:'.
    Tries JSON style and YAML style. Quotes are optional in YAML.
    """
    # Try top-level first
    uri = extract_field(text, 'robotUri')
    if uri:
        return uri
    # If nested, try to roughly capture a 'config: ... robotUri: ...' block
    # This is already handled by allowing leading spaces in _extract_field, so return None
    return None


def rotation_matrix_to_euler_zyx(R):
    """
    将3x3旋转矩阵转换为ZYX欧拉角 (yaw, pitch, roll)
    """

    if not np.allclose(np.dot(R, R.T), np.eye(3), atol=1e-6):
        print("警告: 输入矩阵可能不是有效的旋转矩阵")

    # ZYX顺序 (yaw-pitch-roll)
    # 当pitch接近±90度时会有万向节锁问题
    if abs(R[2, 0]) != 1:
        pitch = -math.asin(R[2, 0])
        roll = math.atan2(R[2, 1] / math.cos(pitch), R[2, 2] / math.cos(pitch))
        yaw = math.atan2(R[1, 0] / math.cos(pitch), R[0, 0] / math.cos(pitch))
    else:
        # 万向节锁情况
        yaw = 0  # 可以任意设置
        if R[2, 0] == -1:
            pitch = math.pi / 2
            roll = yaw + math.atan2(R[0, 1], R[0, 2])
        else:
            pitch = -math.pi / 2
            roll = -yaw + math.atan2(-R[0, 1], -R[0, 2])

    return roll, pitch, yaw

