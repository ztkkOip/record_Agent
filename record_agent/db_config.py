"""
SQLAlchemy 数据库配置和连接管理

支持 MySQL 数据库的连接管理和会话管理。
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from contextlib import contextmanager
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建引擎的配置类
class DatabaseConfig:
    """数据库配置"""

    # MySQL 配置
    MYSQL_HOST = 'localhost'
    MYSQL_PORT = 3306
    MYSQL_DATABASE = 'record'
    MYSQL_USER = 'root'
    MYSQL_PASSWORD = '235703101'
    MYSQL_CHARSET = 'utf8mb4'  # 使用 utf8mb4 支持中文

    # 连接池配置
    POOL_SIZE = 10
    MAX_OVERFLOW = 20
    POOL_RECYCLE = 3600  # 连接回收时间（秒）
    POOL_PRE_PING = True  # 连接前测试连接

    @classmethod
    def get_database_url(cls):
        """
        获取数据库连接URL

        Returns:
            str: 数据库连接URL
        """
        return (
            f"mysql+pymysql://{cls.MYSQL_USER}:{cls.MYSQL_PASSWORD}@"
            f"{cls.MYSQL_HOST}:{cls.MYSQL_PORT}/{cls.MYSQL_DATABASE}"
            f"?charset={cls.MYSQL_CHARSET}"
        )

    @classmethod
    def set_config(cls, host=None, port=None, database=None,
                  user=None, password=None):
        """
        设置数据库配置

        Args:
            host: 数据库主机
            port: 数据库端口
            database: 数据库名称
            user: 用户名
            password: 密码
        """
        if host:
            cls.MYSQL_HOST = host
        if port:
            cls.MYSQL_PORT = port
        if database:
            cls.MYSQL_DATABASE = database
        if user:
            cls.MYSQL_USER = user
        if password:
            cls.MYSQL_PASSWORD = password


# 创建引擎
def create_db_engine():
    """
    创建数据库引擎

    Returns:
        Engine: SQLAlchemy 引擎对象
    """
    engine = create_engine(
        DatabaseConfig.get_database_url(),
        pool_size=DatabaseConfig.POOL_SIZE,
        max_overflow=DatabaseConfig.MAX_OVERFLOW,
        pool_recycle=DatabaseConfig.POOL_RECYCLE,
        pool_pre_ping=DatabaseConfig.POOL_PRE_PING,
        echo=False,  # 设置为 True 可以查看 SQL 语句
        echo_pool=False
    )
    logger.info(f"数据库引擎创建成功: {DatabaseConfig.MYSQL_DATABASE}")
    return engine


# 创建会话工厂
def create_session_factory(engine=None):
    """
    创建会话工厂

    Args:
        engine: 数据库引擎，如果为 None 则创建新引擎

    Returns:
        sessionmaker: 会话工厂
    """
    if engine is None:
        engine = create_db_engine()
    return sessionmaker(bind=engine)


# 全局引擎和会话工厂
_engine = create_db_engine()
_session_factory = create_session_factory(_engine)

# 创建线程安全的会话
Session = scoped_session(_session_factory)

# 声明基类
Base = declarative_base()


def get_session():
    """
    获取数据库会话

    Returns:
        Session: 数据库会话对象

    Example:
        session = get_session()
        users = session.query(User).all()
        session.close()
    """
    return Session()


def close_session():
    """关闭所有会话"""
    Session.remove()
    logger.info("数据库会话已关闭")


def init_db():
    """
    初始化数据库（创建所有表）

    Example:
        from record_agent.models import Base
        init_db()
    """
    Base.metadata.create_all(bind=_engine)
    logger.info("数据库表创建完成")


def drop_db():
    """
    删除所有表（慎用！）

    Example:
        from record_agent.models import Base
        drop_db()
    """
    Base.metadata.drop_all(bind=_engine)
    logger.warning("数据库表已删除")


def get_engine():
    """
    获取数据库引擎

    Returns:
        Engine: 数据库引擎对象
    """
    return _engine


# 上下文管理器，自动管理会话
class SessionContext:
    """会话上下文管理器"""

    def __enter__(self):
        self.session = get_session()
        return self.session

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type:
                # 发生异常时回滚
                self.session.rollback()
                logger.error(f"数据库操作失败，已回滚: {exc_val}")
            else:
                # 正常提交
                self.session.commit()
        finally:
            self.session.close()


@contextmanager
def session_scope():
    """
    会话作用域上下文管理器

    Yields:
        Session: 数据库会话对象

    Example:
        with session_scope() as session:
            user = User(account='test')
            session.add(user)
    """
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"数据库操作失败，已回滚: {e}")
        raise
    finally:
        session.close()


if __name__ == '__main__':
    # 测试数据库连接
    try:
        with SessionContext() as session:
            result = session.execute("SELECT 1")
            print("数据库连接测试成功！")
            print(f"数据库: {DatabaseConfig.MYSQL_DATABASE}")
    except Exception as e:
        print(f"数据库连接测试失败: {e}")
