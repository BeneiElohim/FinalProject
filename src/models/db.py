import os
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
import math

from sqlalchemy import (
    create_engine, Column, String, Float, Integer, JSON,
    DateTime, Text, ForeignKey, event
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# --- Database Setup ---
DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "strategies.db"
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH}")

if DATABASE_URL.startswith("sqlite"):
    db_path = Path(DATABASE_URL.replace("sqlite:///", ""))
    db_path.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    pool_pre_ping=True,
)

if "sqlite" in DATABASE_URL:
    def _fk_pragma_on_connect(dbapi_con, con_record):
        dbapi_con.execute('pragma foreign_keys=ON')
    event.listen(engine, 'connect', _fk_pragma_on_connect)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(elem) for elem in obj]
    elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj

# --- Models ---

class Strategy(Base):
    __tablename__ = 'strategies'

    id = Column(String, primary_key=True)
    symbol = Column(String, nullable=False, index=True)
    model_type = Column(String, nullable=False, index=True)
    rules = Column(JSON, nullable=False)
    hyperparameters = Column(JSON)
    strategic_params = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    backtests = relationship("Backtest", back_populates="strategy", cascade="all, delete-orphan")

    def to_dict(self):
        best_backtest = max(self.backtests, key=lambda b: b.sharpe_ratio or -999, default=None)
        
        d = {
            'id': self.id,
            'symbol': self.symbol,
            'model_type': self.model_type,
            'rules': self.rules,
            'hyperparameters': self.hyperparameters,
            'strategic_params': self.strategic_params,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'best_backtest_metrics': best_backtest.to_dict().get('metrics') if best_backtest else {}
        }
        return sanitize_for_json(d)


class Backtest(Base):
    __tablename__ = 'backtests'

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_id = Column(String, ForeignKey('strategies.id', ondelete='CASCADE'), nullable=False, index=True)
    sharpe_ratio = Column(Float, index=True)
    max_drawdown = Column(Float)
    annual_return = Column(Float)
    win_rate = Column(Float)
    num_trades = Column(Integer)
    metrics = Column(JSON)
    equity_curve = Column(JSON)
    trade_log = Column(JSON)
    train_start = Column(DateTime)
    train_end = Column(DateTime)
    test_start = Column(DateTime)
    test_end = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    strategy = relationship("Strategy", back_populates="backtests")
    
    def to_dict(self):
        d = {
            'id': self.id,
            'strategy_id': self.strategy_id,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'annual_return': self.annual_return,
            'win_rate': self.win_rate,
            'num_trades': self.num_trades,
            'metrics': self.metrics,
            'equity_curve': self.equity_curve,
            'trade_log': self.trade_log,
            'test_period': f"{self.test_start.date() if self.test_start else 'N/A'} to {self.test_end.date() if self.test_end else 'N/A'}",
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
        return sanitize_for_json(d)


# --- Database Utilities ---
def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def generate_strategy_id(
    symbol: str, 
    model_type: str, 
    rules: dict, 
    hyperparameters: Optional[dict] = None,
    strategic_params: Optional[dict] = None
) -> str:
    rules_str = json.dumps(rules, sort_keys=True)
    params_str = json.dumps(hyperparameters, sort_keys=True) if hyperparameters else ""
    strat_params_str = json.dumps(strategic_params, sort_keys=True) if strategic_params else ""
    key_string = f"{symbol}-{model_type}-{rules_str}-{params_str}-{strat_params_str}"
    return hashlib.sha256(key_string.encode()).hexdigest()[:16]


def init_db():
    print(f"Initializing database at {DATABASE_URL}...")
    Base.metadata.create_all(bind=engine)
    print("Database initialized.")

def migrate_json_to_db():
    print("This function is for legacy migration and should not be needed for new setups.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "init":
        init_db()
    elif len(sys.argv) > 1 and sys.argv[1] == "migrate":
        migrate_json_to_db()
    else:
        print("Commands: init, migrate")