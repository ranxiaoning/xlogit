import jax
import jax.numpy as jnp
import numpy as np

class Device:
    """
    一个用于 JAX 设备管理的包装器，以检查 GPU 的可用性。
    """
    def __init__(self):
        try:
            self._backend = jax.default_backend()
            self._using_gpu = (self._backend == 'gpu')
        except RuntimeError:
            # 在没有 GPU 驱动等情况下 JAX 可能会失败
            self._backend = 'cpu'
            self._using_gpu = False
            
        self.np = jnp # 对所有计算使用 jax.numpy

    @property
    def using_gpu(self):
        """如果 JAX 正在使用 GPU 后端，则返回 True。"""
        return self._using_gpu

    def to_cpu(self, arr):
        """将 JAX 数组从设备移动到主机 (CPU)。"""
        if arr is not None:
            return np.asarray(arr)
        return None

    def to_device(self, arr):
        """将 numpy 数组从主机移动到 JAX 设备。"""
        if arr is not None:
            # jax.device_put 会自动处理 JAX 数组和 numpy 数组
            return jax.device_put(arr)
        return None

    def get_device_count(self):
        """返回可用的 JAX 设备数量。"""
        try:
            return jax.device_count()
        except RuntimeError:
            return 0

# 全局设备实例
device = Device()
