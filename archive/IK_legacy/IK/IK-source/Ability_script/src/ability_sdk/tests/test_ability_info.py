import json
import uuid

import preload

from AbilitySDK.Config import ability_info, framework_info

if __name__ == "__main__":
    base = ability_info.AbilityBaseInfo(
        ability_instance_id=uuid.uuid4(),
        ability_name="TestAbility",
        instance_name="TestInstance",
        version="1.0.0",
        config=None
    )
    fi = framework_info.FrameworkInfo(port=8080)
    a = ability_info.AbilityInfo(base_info=base, framework_info=fi)
    a.assets_directory = "/path/to/assets"
    a.instance_directory = "/path/to/instance"
    print(a.heartbeat_info())
    print(a.tmp_directory)

    stat = ability_info.LifecycleState.Suspend
    if stat not in (ability_info.LifecycleState.Standby): # fix: argument of type "LifecycleState" is not iterable
            raise RuntimeError("wrong lifecycle, now is %s)" %
                               stat.name)
