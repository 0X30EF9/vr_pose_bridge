from .Config import ability_info


class AbilityInterface:
    def load_info(self, ability_info: ability_info.AbilityInfo):
        self.ability_info = ability_info

    def on_start(self):
        raise NotImplementedError()

    def on_connect(self):
        """
        如果启动了能力服务器，需要设置ability_info.ability_port
        """
        raise NotImplementedError()

    def on_disconnect(self):
        raise NotImplementedError()

    def on_terminate(self):
        raise NotImplementedError()
