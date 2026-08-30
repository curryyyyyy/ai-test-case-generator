from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import logging
import os
from pathlib import Path
import sqlite3
import uuid


from langgraph.checkpoint.sqlite import SqliteSaver


logger = logging.getLogger(__name__)

# Checkpoint 落盘位置（data/ 已被 .gitignore 忽略）。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DB_PATH = PROJECT_ROOT / "data" / "checkpoints.sqlite"

# 过期会话保留天数。每次上传都会生成新的 thread_id，旧会话不会自动消失，
# 若不清理数据库会持续膨胀。可用环境变量覆盖。
DEFAULT_RETENTION_DAYS = 7

THREAD_ID_PREFIX = "web"
# thread_id 形如 web_20260831003000_ab12cd34
THREAD_TS_FORMAT = "%Y%m%d%H%M%S"
THREAD_TS_LENGTH = 14


def new_thread_id(prefix: str = THREAD_ID_PREFIX) -> str:
    """生成带创建时间的会话 ID。

    把创建时间编码进 thread_id，是为了让保留策略有可靠的时间依据：
    LangGraph 的 checkpoint metadata 只有 source/step/parents，**不含时间戳**，
    而 checkpoint_id 是 UUID、不可按时间排序，因此只能由我们自行记录。
    """
    stamp = datetime.now(timezone.utc).strftime(THREAD_TS_FORMAT)
    return f"{prefix}_{stamp}_{uuid.uuid4().hex[:8]}"


def _thread_created_at(thread_id: str) -> datetime | None:
    """从 thread_id 解析创建时间；无法识别时返回 None（表示不删）。"""
    parts = str(thread_id).split("_")
    if len(parts) < 3:
        return None
    stamp = parts[1]
    if len(stamp) != THREAD_TS_LENGTH or not stamp.isdigit():
        return None
    try:
        parsed = datetime.strptime(stamp, THREAD_TS_FORMAT)
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc)


def _retention_days() -> int:
    raw = os.getenv("CHECKPOINT_RETENTION_DAYS", "").strip()
    if not raw:
        return DEFAULT_RETENTION_DAYS
    try:
        days = int(raw)
    except ValueError:
        logger.warning("CHECKPOINT_RETENTION_DAYS 不是整数，回退为 %s 天", DEFAULT_RETENTION_DAYS)
        return DEFAULT_RETENTION_DAYS
    return max(days, 0)


def _collect_stale_threads(
    conn: sqlite3.Connection,
    saver: SqliteSaver,
    cutoff: datetime,
) -> list[str]:
    """筛选出过期会话。查询走 SqliteSaver 内部锁，避免与其后台写入并发。"""
    # SqliteSaver 内部用 self.lock 串行化访问（LangGraph 会在自己的线程池里
    # 写 checkpoint），因此这里的直接查询也要持同一把锁。
    # 注意：不能在持锁期间调用 saver.delete_thread()，那会再次获取同一把
    # 非可重入锁而死锁，所以拆成「先收集、后删除」两步。
    lock = getattr(saver, "lock", None)
    with lock if lock is not None else nullcontext():
        try:
            rows = conn.execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall()
        except sqlite3.Error as exc:
            logger.warning("读取 checkpoint 会话列表失败，跳过清理: %s", exc)
            return []

    stale: list[str] = []
    for (thread_id,) in rows:
        created_at = _thread_created_at(thread_id)
        # 无法判定时间的会话一律保留，宁可留着也不误删。
        if created_at is not None and created_at < cutoff:
            stale.append(thread_id)
    return stale


def prune_old_checkpoints(
    conn: sqlite3.Connection,
    saver: SqliteSaver,
    retention_days: int | None = None,
) -> int:
    """删除超过保留期的会话，返回被清理的会话数。

    只走 SqliteSaver.delete_thread() 这个公开 API 做删除，不手写 DELETE，
    以免遗漏 writes 表留下孤儿数据。
    """
    days = _retention_days() if retention_days is None else retention_days
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    stale = _collect_stale_threads(conn, saver, cutoff)

    removed = 0
    for thread_id in stale:
        try:
            saver.delete_thread(thread_id)
            removed += 1
        except Exception as exc:  # noqa: BLE001 - 单个失败不影响其余清理
            logger.warning("删除过期 checkpoint 会话失败 thread_id=%s: %s", thread_id, exc)

    if removed:
        logger.info("已清理过期 checkpoint 会话 %s 个（保留期 %s 天）", removed, days)
    return removed


def build_checkpointer() -> SqliteSaver:
    """构建基于 SQLite 的 checkpointer，并在启动时执行保留策略。

    MemorySaver 会把所有会话的检查点堆在进程内存里且永不释放，
    进程重启即全部丢失；改用 SQLite 可持久化并解除内存压力。

    这里自行持有连接而非用 SqliteSaver.from_conn_string()：后者是上下文管理器，
    退出时会关闭连接，不适合长期存活的应用。连接方式与 LangGraph 文档一致
    （check_same_thread=False —— SqliteSaver 内部自带锁，可安全跨线程使用）。
    """
    CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(CHECKPOINT_DB_PATH),
        check_same_thread=False,
        timeout=30.0,
    )
    saver = SqliteSaver(conn)
    saver.setup()

    # 清理失败不应阻断应用启动，因此整体兜住异常。
    try:
        prune_old_checkpoints(conn, saver)
    except Exception as exc:  # noqa: BLE001
        logger.warning("checkpoint 保留策略执行失败（不影响启动）: %s", exc)

    return saver
