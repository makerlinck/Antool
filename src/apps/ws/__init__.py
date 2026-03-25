from litestar import Router
from .image_evaluation import evaluate

ws_router = Router(
    path="/ws",
    route_handlers=[evaluate],
)
