"""
数据库连接工具类

支持多种数据库类型的连接管理，提供连接池、连接复用和自动关闭功能。
"""

import logging
from contextlib import contextmanager
from typing import Optional, Union
import sqlite3
from abc import ABC, abstractmethod

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseConnection(ABC):
    """数据库连接抽象基类"""

    @abstractmethod
    def get_connection(self):
        """获取数据库连接"""
        pass

    @abstractmethod
    def close(self):
        """关闭数据库连接"""
        pass

    @contextmanager
    def cursor(self):
        """
        上下文管理器，自动管理游标和事务

        Yields:
            数据库游标对象

        Example:
            with mysql_conn.cursor() as cursor:
                cursor.execute("SELECT * FROM users")
                results = cursor.fetchall()
        """
        conn = self.get_connection()
        cursor_obj = conn.cursor()

        try:
            yield cursor_obj
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"数据库操作失败，已回滚: {e}")
            raise
        finally:
            cursor_obj.close()
            # 如果有连接池，归还连接
            if hasattr(self, 'return_connection'):
                self.return_connection(conn)


class SQLiteConnection(DatabaseConnection):
    """SQLite数据库连接实现"""

    def __init__(self, database_path: str):
        """
        初始化SQLite连接

        Args:
            database_path: SQLite数据库文件路径
        """
        self.database_path = database_path
        self._connection: Optional[sqlite3.Connection] = None

    def get_connection(self) -> sqlite3.Connection:
        """
        获取SQLite连接

        Returns:
            sqlite3.Connection: 数据库连接对象
        """
        if self._connection is None:
            try:
                self._connection = sqlite3.connect(
                    self.database_path,
                    check_same_thread=False
                )
                # 启用外键约束
                self._connection.execute("PRAGMA foreign_keys=ON")
                # 设置行工厂，返回字典格式
                self._connection.row_factory = sqlite3.Row
                logger.info(f"SQLite数据库连接成功: {self.database_path}")
            except sqlite3.Error as e:
                logger.error(f"SQLite数据库连接失败: {e}")
                raise
        return self._connection

    def close(self):
        """关闭数据库连接"""
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.info("SQLite数据库连接已关闭")


class MySQLConnection(DatabaseConnection):
    """MySQL数据库连接实现"""

    def __init__(self, host: str, port: int, database: str,
                 user: str, password: str, pool_size: int = 5):
        """
        初始化MySQL连接

        Args:
            host: 数据库主机地址
            port: 数据库端口
            database: 数据库名称
            user: 用户名
            password: 密码
            pool_size: 连接池大小
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.pool_size = pool_size
        self._connection_pool = []
        self._connection = None

    def get_connection(self):
        """
        获取MySQL连接

        Returns:
            数据库连接对象
        """
        try:
            import pymysql
        except ImportError:
            raise ImportError(
                "pymysql is required for MySQL connection. "
                "Install it with: pip install pymysql"
            )

        if not self._connection_pool:
            # 创建新连接
            try:
                connection = pymysql.connect(
                    host=self.host,
                    port=self.port,
                    database=self.database,
                    user=self.user,
                    password=self.password,
                    charset='utf8mb4',
                    cursorclass=pymysql.cursors.DictCursor
                )
                logger.info(f"MySQL数据库连接成功: {self.host}:{self.port}/{self.database}")
                return connection
            except Exception as e:
                logger.error(f"MySQL数据库连接失败: {e}")
                raise
        else:
            # 从连接池获取连接
            return self._connection_pool.pop()

    def return_connection(self, connection):
        """
        将连接归还到连接池

        Args:
            connection: 数据库连接对象
        """
        if len(self._connection_pool) < self.pool_size:
            self._connection_pool.append(connection)
        else:
            connection.close()

    def close(self):
        """关闭所有数据库连接"""
        for connection in self._connection_pool:
            connection.close()
        self._connection_pool = []
        logger.info("MySQL数据库连接池已关闭")


class PostgreSQLConnection(DatabaseConnection):
    """PostgreSQL数据库连接实现"""

    def __init__(self, host: str, port: int, database: str,
                 user: str, password: str, pool_size: int = 5):
        """
        初始化PostgreSQL连接

        Args:
            host: 数据库主机地址
            port: 数据库端口
            database: 数据库名称
            user: 用户名
            password: 密码
            pool_size: 连接池大小
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.pool_size = pool_size
        self._connection_pool = []
        self._connection = None

    def get_connection(self):
        """
        获取PostgreSQL连接

        Returns:
            数据库连接对象
        """
        try:
            import psycopg2
        except ImportError:
            raise ImportError(
                "psycopg2 is required for PostgreSQL connection. "
                "Install it with: pip install psycopg2-binary"
            )

        if not self._connection_pool:
            # 创建新连接
            try:
                connection = psycopg2.connect(
                    host=self.host,
                    port=self.port,
                    database=self.database,
                    user=self.user,
                    password=self.password
                )
                logger.info(
                    f"PostgreSQL数据库连接成功: {self.host}:{self.port}/{self.database}"
                )
                return connection
            except Exception as e:
                logger.error(f"PostgreSQL数据库连接失败: {e}")
                raise
        else:
            # 从连接池获取连接
            return self._connection_pool.pop()

    def return_connection(self, connection):
        """
        将连接归还到连接池

        Args:
            connection: 数据库连接对象
        """
        if len(self._connection_pool) < self.pool_size:
            self._connection_pool.append(connection)
        else:
            connection.close()

    def close(self):
        """关闭所有数据库连接"""
        for connection in self._connection_pool:
            connection.close()
        self._connection_pool = []
        logger.info("PostgreSQL数据库连接池已关闭")


class DatabaseConnectionManager:
    """数据库连接管理器"""

    _instances = {}

    @classmethod
    def get_connection(cls, database_type: str, **kwargs) -> DatabaseConnection:
        """
        获取数据库连接实例（单例模式）

        Args:
            database_type: 数据库类型 ('sqlite', 'mysql', 'postgresql')
            **kwargs: 数据库连接参数

        Returns:
            DatabaseConnection: 数据库连接对象

        Raises:
            ValueError: 不支持的数据库类型
        """
        # 创建唯一标识符
        connection_key = f"{database_type}_{hash(str(sorted(kwargs.items())))}"

        if connection_key not in cls._instances:
            if database_type == 'sqlite':
                cls._instances[connection_key] = SQLiteConnection(
                    kwargs.get('database_path')
                )
            elif database_type == 'mysql':
                cls._instances[connection_key] = MySQLConnection(
                    host=kwargs.get('host', 'localhost'),
                    port=kwargs.get('port', 3306),
                    database=kwargs.get('database'),
                    user=kwargs.get('user'),
                    password=kwargs.get('password'),
                    pool_size=kwargs.get('pool_size', 5)
                )
            elif database_type == 'postgresql':
                cls._instances[connection_key] = PostgreSQLConnection(
                    host=kwargs.get('host', 'localhost'),
                    port=kwargs.get('port', 5432),
                    database=kwargs.get('database'),
                    user=kwargs.get('user'),
                    password=kwargs.get('password'),
                    pool_size=kwargs.get('pool_size', 5)
                )
            else:
                raise ValueError(f"不支持的数据库类型: {database_type}")

        return cls._instances[connection_key]

    @classmethod
    def close_all(cls):
        """关闭所有数据库连接"""
        for connection in cls._instances.values():
            connection.close()
        cls._instances.clear()
        logger.info("所有数据库连接已关闭")

    @classmethod
    @contextmanager
    def get_cursor(cls, database_type: str, **kwargs):
        """
        上下文管理器，自动获取和关闭游标

        Args:
            database_type: 数据库类型
            **kwargs: 数据库连接参数

        Yields:
            数据库游标对象

        Example:
            with DatabaseConnectionManager.get_cursor('sqlite', database_path='test.db') as cursor:
                cursor.execute("SELECT * FROM users")
                results = cursor.fetchall()
        """
        connection = cls.get_connection(database_type, **kwargs)
        conn = connection.get_connection()
        cursor = conn.cursor()

        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"数据库操作失败，已回滚: {e}")
            raise
        finally:
            cursor.close()
            # 如果不是SQLite且有连接池，归还连接
            if isinstance(connection, (MySQLConnection, PostgreSQLConnection)):
                connection.return_connection(conn)


# 便捷函数
def get_sqlite_connection(database_path: str) -> SQLiteConnection:
    """
    获取SQLite数据库连接

    Args:
        database_path: 数据库文件路径

    Returns:
        SQLiteConnection: SQLite连接对象
    """
    return DatabaseConnectionManager.get_connection('sqlite', database_path=database_path)


def get_mysql_connection(host: str = 'localhost', port: int = 3306,
                        database: str = None, user: str = None,
                        password: str = None, pool_size: int = 5) -> MySQLConnection:
    """
    获取MySQL数据库连接

    Args:
        host: 数据库主机地址
        port: 数据库端口
        database: 数据库名称
        user: 用户名
        password: 密码
        pool_size: 连接池大小

    Returns:
        MySQLConnection: MySQL连接对象
    """
    return DatabaseConnectionManager.get_connection(
        'mysql',
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
        pool_size=pool_size
    )


def get_postgresql_connection(host: str = 'localhost', port: int = 5432,
                             database: str = None, user: str = None,
                             password: str = None, pool_size: int = 5) -> PostgreSQLConnection:
    """
    获取PostgreSQL数据库连接

    Args:
        host: 数据库主机地址
        port: 数据库端口
        database: 数据库名称
        user: 用户名
        password: 密码
        pool_size: 连接池大小

    Returns:
        PostgreSQLConnection: PostgreSQL连接对象
    """
    return DatabaseConnectionManager.get_connection(
        'postgresql',
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
        pool_size=pool_size
    )


if __name__ == '__main__':
    # # SQLite 示例
    # print("=== SQLite 示例 ===")
    # sqlite_conn = get_sqlite_connection("test.db")
    #
    # # 使用上下文管理器
    # with DatabaseConnectionManager.get_cursor('sqlite', database_path='test.db') as cursor:
    #     # 创建测试表
    #     cursor.execute("""
    #         CREATE TABLE IF NOT EXISTS test_users (
    #             id INTEGER PRIMARY KEY AUTOINCREMENT,
    #             name TEXT NOT NULL,
    #             email TEXT
    #         )
    #     """)
    #     # 插入数据
    #     cursor.execute("INSERT INTO test_users (name, email) VALUES (?, ?)", ('张三', 'zhangsan@example.com'))
    #     cursor.execute("INSERT INTO test_users (name, email) VALUES (?, ?)", ('李四', 'lisi@example.com'))
    #
    #     # 查询数据
    #     cursor.execute("SELECT * FROM test_users")
    #     for row in cursor.fetchall():
    #         print(dict(row))

    # MySQL 示例（需要安装pymysql）
    print("=== MySQL 示例 ===")
    mysql_conn = get_mysql_connection(
        host='localhost',
        port=3306,
        database='record',
        user='root',
        password='235703101'
    )
    # 使用连接对象的 cursor 方法
    with mysql_conn.cursor() as cursor:
        cursor.execute("SELECT * FROM user")
        results = cursor.fetchall()
        print(results)
        print(results[0]['id'])
        print(type(results[0]))

    # PostgreSQL 示例（需要安装psycopg2-binary）
    # print("=== PostgreSQL 示例 ===")
    # postgres_conn = get_postgresql_connection(
    #     host='localhost',
    #     port=5432,
    #     database='test_db',
    #     user='postgres',
    #     password='password'
    # )
    # with DatabaseConnectionManager.get_cursor('postgresql', **postgres_config) as cursor:
    #     cursor.execute("SELECT * FROM users")
    #     results = cursor.fetchall()
    #     print(results)
    # 关闭所有连接
    DatabaseConnectionManager.close_all()
    print("所有连接已关闭")
