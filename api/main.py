import os
import sys
from datetime import datetime
from typing import Optional
import json
import httpx
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import func, desc
from sqlalchemy.orm import Session, joinedload

# Add project root to path to allow imports from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signal_engine.paths import FEAT_DIR
from src.models.db import get_session, Strategy, Backtest

# --- App Setup ---
app = FastAPI(
    title="Signal Engine API",
    version="1.0.0",
    description="API for accessing and analyzing AI-generated trading strategies."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"], # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoints ---
@app.get("/", tags=["General"])
async def root():
    return {
        "status": "running",
        "service": "Signal Engine API",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health", tags=["General"])
async def health_check(db: Session = Depends(get_session)):
    checks = { "api": "ok", "database": "ok", "features_dir": "ok" }
    try:
        db.execute("SELECT 1")
    except Exception:
        checks["database"] = "failed"
    if not FEAT_DIR.exists() or not any(FEAT_DIR.glob("*.csv")):
        checks["features_dir"] = "empty"
    status_code = 503 if "failed" in checks.values() or "empty" in checks.values() else 200
    return JSONResponse(checks, status_code=status_code)

@app.get("/symbols", tags=["Symbols"])
async def list_symbols(db: Session = Depends(get_session)):
    symbols_from_db = db.query(Strategy.symbol).distinct().all()
    symbols = sorted([s[0] for s in symbols_from_db])
    if not symbols:
        return {"symbols": [], "total": 0}
    query = db.query(Strategy.symbol, func.count(Strategy.id)).group_by(Strategy.symbol).all()
    count_by_symbol = {sym: count for sym, count in query}
    symbols_data = [
        {
            "symbol": sym,
            "strategy_count": count_by_symbol.get(sym, 0),
            "has_features": True,
        }
        for sym in symbols
    ]
    return {"symbols": symbols_data, "total": len(symbols_data)}

@app.get("/strategies", tags=["Strategies"])
async def list_strategies(
    symbol: Optional[str] = Query(None),
    model_type: Optional[str] = Query(None),
    min_sharpe: Optional[float] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_session)
):
    query = db.query(Strategy)
    if symbol:
        query = query.filter(Strategy.symbol == symbol)
    if model_type:
        query = query.filter(Strategy.model_type == model_type)
    best_backtest_subq = (
        db.query(
            Backtest.strategy_id,
            func.max(Backtest.sharpe_ratio).label("max_sharpe")
        )
        .group_by(Backtest.strategy_id)
        .subquery()
    )
    query = query.join(best_backtest_subq, Strategy.id == best_backtest_subq.c.strategy_id)
    if min_sharpe is not None:
        query = query.filter(best_backtest_subq.c.max_sharpe >= min_sharpe)
    total_count = query.count()
    strategies = (
        query.order_by(desc(best_backtest_subq.c.max_sharpe))
        .options(joinedload(Strategy.backtests))
        .limit(limit)
        .offset(offset)
        .all()
    )
    return {
        "strategies": [s.to_dict() for s in strategies],
        "total": total_count,
        "limit": limit,
        "offset": offset,
    }

@app.get("/strategies/{strategy_id}", tags=["Strategies"])
async def get_strategy_details(strategy_id: str, db: Session = Depends(get_session)):
    strategy = db.query(Strategy).options(joinedload(Strategy.backtests)).filter(Strategy.id == strategy_id).first()
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")
    result = strategy.to_dict()
    result['backtests'] = sorted(
        [b.to_dict() for b in strategy.backtests],
        key=lambda x: x.get('sharpe_ratio', -999),
        reverse=True
    )
    return result

@app.post("/strategies/{strategy_id}/explain", tags=["Strategies"])
async def explain_strategy(strategy_id: str, db: Session = Depends(get_session)):
    strategy_obj = db.query(Strategy).options(joinedload(Strategy.backtests)).filter(Strategy.id == strategy_id).first()
    if not strategy_obj:
        raise HTTPException(status_code=404, detail="Strategy not found")

    strategy = strategy_obj.to_dict()
    metrics = strategy.get('best_backtest_metrics', {})
    rules_text = strategy.get('rules', {}).get('text', json.dumps(strategy.get('rules'), indent=2))

    prompt = f"""You are a financial analyst. Explain this trading strategy in simple terms.

Strategy Details:
- Symbol: {strategy['symbol']}
- Model Type: {strategy['model_type']}
- Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A'):.2f}
- Max Drawdown: {metrics.get('max_drawdown', 'N/A'):.1f}%
- Annual Return: {metrics.get('annual_return', 'N/A'):.1f}%
- Trades: {metrics.get('num_trades', 'N/A')}

Trading Rules:
{rules_text}
Provide a concise explanation covering:
1. What triggers a buy/sell signal based on the rules?
2. What kind of market conditions might this strategy perform well in?
3. What are the main risks associated with this strategy?
"""
    ollama_url = os.environ.get("OLLAMA_URL", "http://ollama:11434")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": "gemma:2b",
                    "prompt": prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            explanation = response.json().get("response", "Failed to get explanation.")
            return {"explanation": explanation.strip()}
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        fallback = f"Could not get AI explanation. This {strategy['model_type']} strategy for {strategy['symbol']} has a Sharpe ratio of {metrics.get('sharpe_ratio', 0):.2f}. Error: {e}"
        return {"explanation": fallback, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)