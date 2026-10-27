# Exports resolve lazily (PEP 562) so that light submodules such as
# ``utils.eval_report`` import without pulling in sapien / mani_skill.
from importlib import import_module

_EXPORTS = {
    # Grasp computation
    'compute_grasp_info_by_point': '.grasp_compute',
    'compute_grasp_info_by_obb': '.grasp_compute',
    'get_actor_obb': '.grasp_compute',
    'compute_grasp_pose_from_json_matrix': '.grasp_compute',
    'get_grasp_pose_from_config': '.grasp_compute',
    'build_grasp_pose_from_info': '.grasp_compute',
    'visualize_grasp_pose': '.grasp_compute',
    # Grasp helpers
    'compute_smart_pregrasp_pose': '.util',
    # Misc utilities
    'plan_arc_path': '.util',
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
