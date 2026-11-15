from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from revo2_hand_ik.udp_bridge import DIRECT_CONTROL_FORMAT, load_dual_bridge_config  # noqa: E402


class DualBridgeConfigTest(unittest.TestCase):
    def test_config_resolves_two_sdk_configs_and_direct_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "left.json").write_text("{}", encoding="utf-8")
            (base / "right.json").write_text("{}", encoding="utf-8")
            config = {
                "bridge_mode": "dual_hand",
                "input_udp": {"listen_ip": "0.0.0.0", "listen_port": 15000},
                "output_udp": {"target_ip": "127.0.0.1", "target_port": 5555, "packet_format": DIRECT_CONTROL_FORMAT},
                "left_hand": {"sdk_config": "left.json"},
                "right_hand": {"sdk_config": "right.json"},
            }
            path = base / "direct.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            parsed = load_dual_bridge_config(path)
            self.assertEqual(parsed.left_sdk_config, (base / "left.json").resolve())
            self.assertEqual(parsed.right_sdk_config, (base / "right.json").resolve())
            self.assertEqual(parsed.udp.listen_port, 15000)
            self.assertEqual(parsed.udp.target_port, 5555)


if __name__ == "__main__":
    unittest.main(verbosity=2)
