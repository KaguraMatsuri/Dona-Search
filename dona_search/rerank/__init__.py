__all__ = ['CrossEncoderReranker']


def __getattr__(name):
    if name == 'CrossEncoderReranker':
        from .cross_encoder import CrossEncoderReranker
        return CrossEncoderReranker
    raise AttributeError(name)
