"""Use a node-local package mirror without modifying the shared runtime."""

import os
import sys
import typing


def _activate_local_site_packages():
    local_site = os.environ.get("VIDEOX_LOCAL_SITE_PACKAGES", "")
    if not local_site:
        return

    local_site = os.path.realpath(local_site)
    if not os.path.isdir(local_site):
        raise RuntimeError(f"VIDEOX_LOCAL_SITE_PACKAGES does not exist: {local_site}")

    filtered_paths = []
    for path in sys.path:
        if not path:
            filtered_paths.append(path)
            continue
        real_path = os.path.realpath(path)
        if real_path != local_site and real_path.endswith("/site-packages"):
            continue
        filtered_paths.append(path)

    sys.path[:] = [local_site] + [
        path
        for path in filtered_paths
        if os.path.realpath(path or ".") != local_site
    ]


def _patch_torch_custom_op_schema_annotations():
    try:
        import torch._custom_op.impl as custom_op_impl
        import torch._library.infer_schema as infer_schema_mod
    except Exception:
        return

    original_infer_schema = infer_schema_mod.infer_schema

    def infer_schema_with_resolved_hints(fn, mutates_args=()):
        annotations = getattr(fn, "__annotations__", None)
        if not annotations or not any(isinstance(value, str) for value in annotations.values()):
            return original_infer_schema(fn, mutates_args)

        try:
            resolved = typing.get_type_hints(fn)
        except Exception:
            return original_infer_schema(fn, mutates_args)

        old_annotations = dict(annotations)
        try:
            fn.__annotations__.update(resolved)
            return original_infer_schema(fn, mutates_args)
        finally:
            fn.__annotations__.clear()
            fn.__annotations__.update(old_annotations)

    infer_schema_mod.infer_schema = infer_schema_with_resolved_hints
    custom_op_impl.infer_schema = infer_schema_with_resolved_hints


_activate_local_site_packages()
_patch_torch_custom_op_schema_annotations()
