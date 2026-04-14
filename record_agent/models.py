"""
SQLAlchemy 实体模型

定义所有数据库表的 ORM 模型。
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Date, JSON, BigInteger, SmallInteger
from sqlalchemy.types import DECIMAL
from sqlalchemy.sql import func
from record_agent.db_config import Base


class User(Base):
    """用户表"""
    __tablename__ = 'user'

    id = Column(Integer, primary_key=True, autoincrement=True, comment='用户ID')
    account = Column(String(10), nullable=False, comment='账号')
    password = Column(String(50), nullable=False, comment='密码')
    profile_picture = Column(String(100), comment='头像')
    phone = Column(String(20), comment='手机号')
    email = Column(String(20), comment='邮箱')
    create_time = Column(DateTime, comment='创建时间')
    update_time = Column(DateTime, comment='更新时间')


class Account(Base):
    """账户表"""
    __tablename__ = 'account'

    id = Column(Integer, primary_key=True, autoincrement=True, comment='账户ID')
    user_id = Column(Integer, comment='用户ID')
    name = Column(String(20), comment='账户名称')
    type_id = Column(Integer, comment='类型ID')
    balance = Column(Float, comment='余额')
    create_time = Column(DateTime, comment='创建时间')
    update_time = Column(DateTime, comment='更新时间')


class Type(Base):
    """类型表"""
    __tablename__ = 'type'

    id = Column(Integer, primary_key=True, autoincrement=True, comment='类型ID')
    name = Column(String(20), comment='类型名称')
    image = Column(String(50), comment='图片')
    create_time = Column(DateTime, comment='创建时间')
    update_time = Column(DateTime, comment='更新时间')


class Bill(Base):
    """账单表"""
    __tablename__ = 'bill'

    id = Column(Integer, primary_key=True, autoincrement=True, comment='账单ID')
    account_id = Column(Integer, comment='账户ID')
    number = Column(Float, comment='金额')
    comsuption = Column(String(100), comment='消费描述')
    consumption_time = Column(DateTime, comment='消费时间')


class Diary(Base):
    """日记表"""
    __tablename__ = 'diary'

    id = Column(Integer, primary_key=True, autoincrement=True, comment='日记ID')
    userId = Column(Integer, comment='用户ID')
    recordDate = Column(Date, comment='记录日期')
    detail = Column(JSON, comment='详情（JSON格式）')


class Order(Base):
    """订单表"""
    __tablename__ = 'order'

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment='订单ID')
    user_id = Column(BigInteger, nullable=False, comment='用户ID')
    product_id = Column(BigInteger, nullable=False, comment='商品ID')
    amount = Column(DECIMAL(18, 2), nullable=False, comment='订单金额')
    status = Column(String(20), default='pending', comment='订单状态')
    create_time = Column(DateTime, comment='创建时间')
    order_time = Column(DateTime, comment='下单时间')


class Product(Base):
    """商品表"""
    __tablename__ = 'product'

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment='商品ID')
    name = Column(String(255), nullable=False, comment='商品名称')
    price = Column(DECIMAL(18, 2), nullable=False, comment='商品价格')
    stock = Column(Integer, default=0, comment='库存数量')
    status = Column(Integer, default=0, comment='商品状态：0-创建，1-在售，2-下架，3-删除')
    description = Column(Text, comment='商品详情描述')
    image_url = Column(String(500), comment='商品图片地址')
    create_time = Column(DateTime, comment='创建时间')
    update_time = Column(DateTime, comment='更新时间')


class UserProduct(Base):
    """用户商品关联表"""
    __tablename__ = 'user_product'

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment='主键ID')
    user_id = Column(BigInteger, nullable=False, comment='用户ID')
    product_id = Column(BigInteger, nullable=False, comment='商品ID')


class ChatSession(Base):
    """聊天会话表"""
    __tablename__ = 'chat_session'

    id = Column(Integer, primary_key=True, autoincrement=True, comment='会话ID')
    user_id = Column(Integer, comment='用户ID')
    title = Column(String(200), default='新对话', comment='会话标题')
    is_deleted = Column(Integer, default=0, comment='是否已删除：0-否 1-是')
    meta_data = Column('metadata', Text, default='{}', comment='扩展元数据（JSON格式）')
    create_time = Column(DateTime, comment='创建时间')
    update_time = Column(DateTime, comment='更新时间')


class ChatHistory(Base):
    """聊天记录表"""
    __tablename__ = 'chat_history'

    id = Column(Integer, primary_key=True, autoincrement=True, comment='记录ID')
    session_id = Column(Integer, comment='会话ID')
    user_id = Column(Integer, comment='用户ID')
    role = Column(Integer, default=0, comment='角色类型：0-user, 1-assistant, 2-system')
    content = Column(Text, comment='对话内容')
    message_type = Column(String(20), default='text', comment='消息类型：text/image/audio/file/summary')
    tokens_used = Column(Integer, default=0, comment='使用的token数量')
    model = Column(String(50), comment='使用的大模型名称')
    meta_data = Column('metadata', Text, default='{}', comment='扩展元数据（JSON格式）')
    create_time = Column(DateTime, comment='消息时间')


class UserMemory(Base):
    """用户记忆表"""
    __tablename__ = 'user_memory'

    id = Column(Integer, primary_key=True, autoincrement=True, comment='记忆ID')
    user_id = Column(Integer, comment='用户ID')
    memory_type = Column(String(50), comment='记忆类型：preference/context/knowledge')
    memory_key = Column(String(100), comment='记忆键')
    memory_value = Column(Text, comment='记忆值')
    importance = Column(Integer, default=5, comment='重要性等级 1-10')
    access_count = Column(Integer, default=0, comment='访问次数')
    last_access_time = Column(DateTime, comment='最后访问时间')
    create_time = Column(DateTime, comment='创建时间')
    update_time = Column(DateTime, comment='更新时间')


class ChatSessionTags(Base):
    """聊天会话标签表"""
    __tablename__ = 'chat_session_tags'

    id = Column(Integer, primary_key=True, autoincrement=True, comment='标签ID')
    session_id = Column(Integer, comment='会话ID')
    tag_name = Column(String(50), comment='标签名称')
    tag_type = Column(String(20), default='custom', comment='标签类型：category/priority/custom')
    create_time = Column(DateTime, comment='创建时间')


# 导出所有模型类
__all__ = [
    'User',
    'Account',
    'Type',
    'Bill',
    'Diary',
    'Order',
    'Product',
    'UserProduct',
    'ChatSession',
    'ChatHistory',
    'UserMemory',
    'ChatSessionTags',
]
