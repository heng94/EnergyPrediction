import logging
from iopath.common.file_io import PathManager as PathManagerBase
from fvcore.common.config import CfgNode as _CfgNode


PathManager = PathManagerBase()


class CfgNode(_CfgNode):
    """
    The same as `fvcore.common.config.CfgNode`, but different in:

    1. Use unsafe yaml loading by default.
       Note that this may lead to arbitrary code execution: you must not
       load a config file from untrusted sources before manually inspecting
       the content of the file.
    2. Support config versioning.
       When attempting to merge an old config, it will convert the old config automatically.

    .. automethod:: clone
    .. automethod:: freeze
    .. automethod:: defrost
    .. automethod:: is_frozen
    .. automethod:: load_yaml_with_base
    .. automethod:: merge_from_list
    .. automethod:: merge_from_other_cfg
    """

    @classmethod
    def _open_cfg(cls, filename):
        return PathManager.open(filename, "r")

    # Note that the default value of allow_unsafe is changed to True
    def merge_from_file(self, cfg_filename: str, allow_unsafe: bool = True) -> None:
        """
        Load content from the given config file and merge it into self.

        Args:
            cfg_filename: config filename
            allow_unsafe: allow unsafe yaml syntax
        """
        assert PathManager.isfile(cfg_filename), f"Config file '{cfg_filename}' does not exist!"
        loaded_cfg = self.load_yaml_with_base(cfg_filename, allow_unsafe=allow_unsafe)
        loaded_cfg = type(self)(loaded_cfg)

        # defaults.py needs to import CfgNode
        from .defaults import _C

        latest_ver = _C.VERSION
        assert (
            latest_ver == self.VERSION
        ), "CfgNode.merge_from_file is only allowed on a config object of latest version!"

        logger = logging.getLogger(__name__)

        loaded_ver = loaded_cfg.get("VERSION", None)
        if loaded_ver is None:
            from .compat import guess_version

            loaded_ver = guess_version(loaded_cfg, cfg_filename)
        assert loaded_ver <= self.VERSION, "Cannot merge a v{} config into a v{} config.".format(
            loaded_ver, self.VERSION
        )

        if loaded_ver == self.VERSION:
            self.merge_from_other_cfg(loaded_cfg)
        else:
            # compat.py needs to import CfgNode
            from .compat import upgrade_config, downgrade_config

            logger.warning(
                "Loading an old v{} config file '{}' by automatically upgrading to v{}. "
                "See docs/CHANGELOG.md for instructions to update your files.".format(
                    loaded_ver, cfg_filename, self.VERSION
                )
            )
            # To convert, first obtain a full config at an old version
            old_self = downgrade_config(self, to_version=loaded_ver)
            old_self.merge_from_other_cfg(loaded_cfg)
            new_config = upgrade_config(old_self)
            self.clear()
            self.update(new_config)

    def dump(self, *args, **kwargs):
        """
        Returns:
            str: a yaml string representation of the config
        """
        # to make it show up in docs
        return super().dump(*args, **kwargs)
    

def get_cfg() -> CfgNode:
    """
    Get a copy of the default config.

    Returns:
        a detectron2 CfgNode instance.
    """
    from .defaults import _C

    return _C.clone()