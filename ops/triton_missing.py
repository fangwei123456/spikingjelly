class DummyImport:
    """
    Dummy class as an import placeholder.
    """

    def __getattr__(self, name):
        return DummyImport()

    def __call__(self, *args, **kwargs):
        return DummyImport()

    def __getitem__(self, item):
        return DummyImport()

    def __bool__(self):
        return False

    def __repr__(self):
        return "<DummyImport>"
