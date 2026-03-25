from .v1 import v1_router

from litestar import Router

router = Router(
    path="/api",
    route_handlers=[v1_router]
)
__all__ = ['router']