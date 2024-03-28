import logging as log
import os.path
import configparser


class RunFile(configparser.ConfigParser):
    TWEAK_FILE = "tweak.ini"

    logger = log.getLogger("RunFile")

    """
    Configuration for the run with way to reload options.
    """
    def __init__(self, file_name):
        super(RunFile, self).__init__()
        if not self.read(file_name):
            raise FileNotFoundError(file_name)
        self.file_name = file_name
        self.mtime = os.path.getmtime(file_name)

    def check_and_reload(self):
        mtime = os.path.getmtime(self.file_name)
        if self.mtime != mtime:
            self.clear()
            self.read(self.file_name)
        if self.tweak_file_enabled:
            if os.path.exists(self.TWEAK_FILE):
                res = self._merge_tweak_file(self.TWEAK_FILE)
                os.unlink(self.TWEAK_FILE)
                return res
        return []

    def _merge_tweak_file(self, file_name):
        """
        Reads and merges config file, returning list of (section, name) tuples which were updated
        :param file_name: 
        :return: None if nothing was updated, or list of tuples (section, name) of changed options
        """
        updated = []
        c = configparser.ConfigParser()
        if not c.read(file_name):
            return updated
        for s in c.sections():
            for k in c[s].keys():
                if k not in self[s].keys():
                    self.logger.warning("Tweak file contains unknown option %s in section %s", k, s)
                else:
                    if c[s][k] != self[s][k]:
                        updated.append((s, k))
                        self[s][k] = c[s][k]
        return updated

    @property
    def tweak_file_enabled(self):
        return self.getboolean("defaults", "tweak_file", fallback=False)

    @property
    def cuda_enabled(self):
        return self.getboolean("defaults", "cuda", fallback=False)


