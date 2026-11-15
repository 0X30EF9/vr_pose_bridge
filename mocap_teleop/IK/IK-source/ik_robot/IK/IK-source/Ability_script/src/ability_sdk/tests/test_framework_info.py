import uuid

import preload

from AbilitySDK.Config import framework_info

print(framework_info.FrameworkInterfaceUri.cr_by_id(uuid.uuid4()))
print(framework_info.FrameworkInterfaceUri.crd_by_id(uuid.uuid4()))
print(framework_info.FrameworkInterfaceUri.ability_heartbeat())
print(framework_info.FrameworkInterfaceUri.lifecycle_request())
print(framework_info.FrameworkInterfaceUri.task_status())