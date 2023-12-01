from lightning_factory import LightningFactory


def ffnn(**kwargs):
    # Create an instance of LightningFactory with the given keyword arguments
    factory = LightningFactory()
    return factory.ffnn(**kwargs)
