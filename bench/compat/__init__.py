"""兼容垫片：把 legacy 时代的模块级客户端接口原样提供出来。

用途：让 baselines/ 下的基线脚本能【逐行原样】使用 legacy 的实验脚本
（只改 import 三行），从而保证行为与历史成绩可比。

⚠️ 这两个模块是 legacy/qwen_client.py 与 legacy/ds_client.py 的**逐字拷贝**，
   刻意不重构、不统一签名——重构会改变行为（曾经因此让 b3 的调用数从 2.68 变成 5.10）。
   新代码请用 bench.clients 的统一接口。
"""
