import functools

import jax
import numpy as np


class _ShardingAPI:
    @functools.cached_property
    def REPLICATED(self):
        devices = np.asarray(jax.devices())
        mesh = jax.sharding.Mesh(devices, axis_names=("devices",))
        return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    def with_sharding_constraint(self, arrays, shardings):
        if shardings is None:
            return arrays
        return jax.tree.map(
            lambda array: jax.lax.with_sharding_constraint(array, shardings),
            arrays,
        )


sharding = _ShardingAPI()
