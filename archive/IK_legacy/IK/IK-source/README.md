# 基于Python的能力封装

## 文件树

- XXX能力目录
  - XXX-src (c++源码)
  - XXX-script(python源码)
    - packages(借鉴了底层能力框架中的文件目录结构，crds中保存能力相关的配置文件，bin中保存相关的可执行文件，暂时没看出来两个有啥特别大的区别，python中使用到了packages中的crs文件 )
    - src(具体代码，后续尝试按照script和src区分相关的python和c++)
      - ability_sdk(理解为底层的类)
        - src (底层能力父类和相关的函数)
          - AbilitySDK(相关父类SDK文件)
          - Config
            - ability_info.py 规定了底层能力的状态和其他的类型等
            - framework_info.py 规定了框架的端口和ip，注意框架请求的名称来自与这里
          - RPC
            - http_rpc.py HTTP请求函数
            - rpc 用处可忽略
          - ability_stub.py 底层基类，当前的能力类继承自本函数
          - heartbeat.py 开启心跳包的相关能力
          - ipc_server_handle.py  （*****主要能力，能力的运行等都是通过此函数调用）
          - log.py logger函数
          - task.py 相关的任务执行函数
        - tests(基于python的测试函数)
      - build(没用上，不清楚有什么用处)
      - dist(通过pyinstaller -F your_script.py 打包的过程中会出现这样的包，里边包含打包的可执行文件)
      - log(log文件夹)
      - XXXAbility.py 能力封装主函数
      - requirments.txt(因为是通过pyinstaller封装，默认会把py文件所依赖的环境也封装，所以猜测是必须的)


- Step1、尝试将现有的代码进行解析
  - 代码依赖项：借鉴于大型深度学习网络，现有的依赖项安装通常采取两种方式：1、requirements.txt（常用方式）2、pyproject.toml和uv.lock形式
  - 工作空间文件树: src：主要核心代码保存位置，packages：相关依赖项，一般不保存底层代码，只保存代码封装后的启动脚本
  - 具体能力代码解析 ybxbase-python\src\HeadAbility.py.
    - 程序入口：main.py函数 - log.configure_logging(sys.argv[0]) logger函数（暂时没看，后续如果有兴趣倒回来再看）- Head()【能力主函数】：继承自src/ability_sdk/src/AbilitySDK/ability_stub.py函数，其中定义了ability_info类和AbilityInfo类
    - main.py函数
      - base = Path(sys.argv[0]).name 获取当前的ability名称[非必须，因为会在底下进行二次和三次监测，都是为了获取当前的Ability_name，第一次监测其实没有起作用]
      - Step1、根据上述内容获取相关的ability_name，然后将ABILITY_NAME变成全局变量
      - Step2、调用_load_robot_host_port()获取当前ip地址和端口号
        - A 规定 primary_name 继承自ability_name，获取fallback_name的名称_detect_ability_name，函数内部其实做了跟获取ability_name相同的操作，只是多了一个最后如果输出格式不对，需要指定成XXXAbility(可以尽量保证当前的文件名称和所封装的能力还有就是当前的default中是一致的)
        - B 分为三个情况，第一primary_name是具有名称的，调用_find_config_for_ability在当前函数中在调用_find_crs_dir() 首先查找当前的环境变量中是否包含essential_env_keys = ('ABILITY_ROOT', 'ABILITY_FRAMEWORK_HOME', 'ABILITY_HOME')相关内容，一般是没有，然后通过_iter_candidate_bases()获取当前工作文件的相关路径
        遍历当前文件夹和父文件夹，获取crs路径, 然后根据传入的primary_name参数和当前yaml中的abilityName进行匹配，匹配成功最后返回当前的yaml文件,
        - C 第二种情况匹配不成功使用刚才获取的fallback_name进行匹配，匹配成功最后返回当前的yaml文件,
        - D 还是不成功再此调用_find_crs_config_file进行匹配
        - E 依旧不存在返回固定ip和端口
        - F 存在获取当前config中的ip地址和端口号
      - Step3、获取当前的ip和端口以后，初始化能力相关内容
        - 自己的代码 后续代码为具体封装的能力代码，不在进行详细解析
    - find_free_port()【能力代码必须】: 获取临时端口号（调试时可以将其置为固定端口号），代码位置具体能力XXX.on_connect, 参数相关属性socket.AF_INET，socket.SOCK_STREAM前者代表使用IPv4地址族，后者代表使用TCP协议

# 必须代码和非必须代码
  - 上述Step1和Step2可以省略，代码是通用代码，使用直接使用即可
  - Step3为特有代码，需要根据自己的内容进行编写
  - 后续能力中代码如on_start，on_connect，on_disconnect，on_terminate均为通用代码，可以直接复用

- Step2、构建基于IK的Python能力
  - pass
