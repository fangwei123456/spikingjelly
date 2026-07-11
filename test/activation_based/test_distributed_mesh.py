# ruff: noqa: F401,F403,F405
from test.activation_based._distributed_dtensor_test_support import *


def test_topology_from_mapping_orders_named_dims():
    dims = {"tp": 2, "dp": 2}
    topology = SNNDistributedTopology.from_mapping(dims)
    dims["pp"] = 4  # mutate original dict — topology must stay unaffected
    assert topology.world_size == 4
    assert topology.ordered_dim_names == ("dp", "tp")
    assert topology.mesh_shape == (2, 2)
    assert "pp" not in topology.dims

def test_topology_rejects_non_integer_dim_sizes():
    with pytest.raises(TypeError, match="must be an integer"):
        SNNDistributedTopology.from_mapping({"dp": 1.5, "tp": 2}, world_size=3)

def test_topology_rejects_non_integer_world_size():
    with pytest.raises(TypeError, match="world_size must be an integer"):
        SNNDistributedTopology.from_mapping({"dp": 2}, world_size=1.5)

def test_topology_rejects_non_integral_numeric_types():
    with pytest.raises(TypeError, match="must be an integer"):
        SNNDistributedTopology.from_mapping({"dp": Decimal("1.5")})
    with pytest.raises(TypeError, match="world_size must be an integer"):
        SNNDistributedTopology.from_mapping({"dp": 1}, world_size=Fraction(3, 2))

def test_topology_rejects_non_string_dim_names():
    with pytest.raises(TypeError, match="Topology dimension names must be strings"):
        SNNDistributedTopology.from_mapping({1: 2})

def test_partition_helpers_respect_2d_mesh_coordinates():
    with single_rank_process_group():
        model = ToyDistributedSNN()
        config = SNNDistributedConfig(
            device_type="cpu",
            mesh_shape=(1, 1),
            tensor_parallel_roots=["features"],
            auto_tensor_parallel=True,
            enable_data_parallel=False,
            tp_mesh_dim=1,
            dp_mesh_dim=0,
        )
        _, mesh, _ = configure_snn_distributed(model, config)
        assert resolve_data_parallel_partition(
            mesh, dp_mesh_dim=0, sharded_by_data_parallel=False
        ) == (1, 0)
        assert resolve_data_parallel_partition(
            mesh, dp_mesh_dim=0, sharded_by_data_parallel=True
        ) == (1, 0)
        assert (
            resolve_tensor_parallel_group_size(
                mesh, tp_mesh_dim=1, tensor_parallel_enabled=True
            )
            == 1
        )

def test_resolve_data_parallel_partition_rejects_mesh_without_rank_coordinate():
    class FakeMesh:
        mesh = torch.zeros(2)

        def get_coordinate(self):
            return None

    with pytest.raises(ValueError, match="Current rank does not belong"):
        resolve_data_parallel_partition(
            FakeMesh(), dp_mesh_dim=None, sharded_by_data_parallel=True
        )
    with pytest.raises(ValueError, match="Current rank does not belong"):
        resolve_data_parallel_partition(
            FakeMesh(), dp_mesh_dim=0, sharded_by_data_parallel=True
        )
