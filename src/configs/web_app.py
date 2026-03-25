class WebAppConfig:
    """Web 应用配置"""

    app_name: str = "Antool API"
    app_version: str = "0.1.1"
    app_description: str = "图片分类归档服务程序"
    host: str = "127.0.0.1"
    listen_port: int = 6500

    def __init__(
        self,
        app_name: str = "Antool API",
        app_version: str = "0.1.1",
        app_description: str = "图片分类归档服务程序",
    ) -> None:
        self.app_name = app_name
        self.app_version = app_version
        self.app_description = app_description
