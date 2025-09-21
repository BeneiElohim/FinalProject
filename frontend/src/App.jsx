import React, { useEffect, useState } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import "./App.css";


const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";
async function apiCall(endpoint, options = {}) {
    const res = await fetch(`${API_BASE}${endpoint}`, {
        ...options,
        headers: {
            "Content-Type": "application/json",
            ...(options.headers || {}),
        },
    });
    if (!res.ok) {
        const errorBody = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(`API Error: ${res.status} - ${errorBody.detail}`);
    }
    return res.json();
}

const formatMetric = (value, unit = "") => {
    if (value === null || typeof value === "undefined") return "N/A";
    if (typeof value === "number") {
        if (Math.abs(value) > 1000) {
            return `${Math.round(value).toLocaleString()}${unit}`;
        }
        return `${value.toFixed(2)}${unit}`;
    }
    return value;
};


function StrategyTable({ strategies, onRowClick }) {
    if (strategies.length === 0) {
        return <div className="no-results">No strategies found matching your criteria.</div>;
    }

    const renderMetricCell = (value, unit = '') => {
        const isNum = typeof value === 'number';
        const className = isNum ? (value >= 0 ? 'metric-positive' : 'metric-negative') : '';
        return <td className={className}>{formatMetric(value, unit)}</td>;
    };

    return (
        <table className="strategy-table">
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Model</th>
                    <th>Sharpe</th>
                    <th>Ann. Return</th>
                    <th>Max Drawdown</th>
                    <th>Trades</th>
                    <th>Win Rate</th>
                </tr>
            </thead>
            <tbody>
                {strategies.map(s => {
                    const metrics = s.best_backtest_metrics || {};
                    return (
                        <tr key={s.id} onClick={() => onRowClick(s)}>
                            <td><strong>{s.symbol}</strong></td>
                            <td><span className="model-type-badge">{s.model_type}</span></td>
                            {renderMetricCell(metrics.sharpe_ratio)}
                            {renderMetricCell(metrics.annual_return, '%')}
                            {renderMetricCell(metrics.max_drawdown, '%')}
                            <td>{metrics.num_trades ?? 0}</td>
                            {renderMetricCell(metrics.win_rate, '%')}
                        </tr>
                    );
                })}
            </tbody>
        </table>
    );
}

function EquityChart({ data }) {
    if (!data || data.length === 0) return <div>No equity data available.</div>;
    const chartData = data.map((value, index) => ({ day: index + 1, equity: value }));
    return (
        <div style={{ height: '300px', width: '100%' }}>
            <ResponsiveContainer>
                <LineChart data={chartData} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                    <XAxis dataKey="day" label={{ value: 'Time (Days)', position: 'insideBottom', offset: -5 }} />
                    <YAxis domain={['auto', 'auto']} tickFormatter={(tick) => tick.toLocaleString()} />
                    <Tooltip formatter={(value) => [`$${value.toLocaleString()}`, "Equity"]} />
                    <Line type="monotone" dataKey="equity" stroke="#007bff" strokeWidth={2} dot={false} />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
}

function StrategyDetailsModal({ strategyId, onClose }) {
    const [details, setDetails] = useState(null);
    const [explanation, setExplanation] = useState('');
    const [explaining, setExplaining] = useState(false);

    useEffect(() => {
        if (strategyId) {
            apiCall(`/strategies/${strategyId}`).then(setDetails);
        }
    }, [strategyId]);

    const handleExplain = async (e) => {
        e.stopPropagation();
        setExplaining(true);
        try {
            const data = await apiCall(`/strategies/${strategyId}/explain`, { method: 'POST' });
            setExplanation(data.explanation);
        } catch (err) {
            setExplanation(`Failed to get explanation: ${err.message}`);
        } finally {
            setExplaining(false);
        }
    };

    if (!details) return null;

    const metrics = details.best_backtest_metrics || {};

    return (
        <div className="strategy-details-overlay" onClick={onClose}>
            <div className="strategy-details" onClick={(e) => e.stopPropagation()}>
                <button className="close-btn" onClick={onClose}>&times;</button>
                <div className="details-header">
                    <h2>{details.symbol} - {details.model_type.toUpperCase()}</h2>
                    <p>Strategy ID: {details.id}</p>
                </div>
                
                <div className="details-section">
                    <h3>Equity Curve</h3>
                    <EquityChart data={details.backtests?.[0]?.equity_curve} />
                </div>
                
                <div className="details-grid">
                    <div className="details-section">
                        <h3>Performance Metrics</h3>
                        {Object.entries(metrics).map(([key, value]) => (
                            <div className="metric-row" key={key}>
                                <span>{key.replace(/_/g, ' ')}</span>
                                <strong>{formatMetric(value, key.includes('rate') || key.includes('return') ? '%' : '')}</strong>
                            </div>
                        ))}
                    </div>

                    <div className="details-section">
                        <h3>Trading Rules</h3>
                        <pre className="rules-display">{details.rules?.text || JSON.stringify(details.rules, null, 2)}</pre>
                    </div>
                </div>
                
                <div className="details-section" style={{marginTop: '1.5rem'}}>
                    <h3>AI Explanation</h3>
                    <button className="explain-btn" onClick={handleExplain} disabled={explaining}>
                        {explaining ? "Thinking..." : "Explain This Strategy"}
                    </button>
                    {explanation && <div className="explanation-text"><pre>{explanation}</pre></div>}
                </div>
            </div>
        </div>
    );
}
export default function App() {
    const [symbols, setSymbols] = useState([]);
    const [strategies, setStrategies] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [selectedStrategyId, setSelectedStrategyId] = useState(null);
    const [filters, setFilters] = useState({ symbol: '', min_sharpe: '' });

    useEffect(() => {
        apiCall('/symbols')
            .then(data => setSymbols(data.symbols || []))
            .catch(err => {
                console.error("Failed to fetch symbols:", err);
                setError("Could not connect to the API. Is the backend running?");
            });
    }, []);
    
    useEffect(() => {
        const params = new URLSearchParams({ limit: 500 });
        if (filters.symbol) params.set('symbol', filters.symbol);
        if (filters.min_sharpe) params.set('min_sharpe', filters.min_sharpe);
        
        setLoading(true);
        setError(null);
        apiCall(`/strategies?${params.toString()}`)
            .then(data => setStrategies(data.strategies || []))
            .catch(err => {
                console.error("Failed to fetch strategies:", err);
                setError("Could not fetch strategies. The API may have returned an error.");
            })
            .finally(() => setLoading(false));
    }, [filters]);

    const handleFilterChange = (e) => {
        setFilters(prev => ({ ...prev, [e.target.name]: e.target.value }));
    };

    const clearFilters = () => {
        setFilters({ symbol: '', min_sharpe: '' });
    };

    return (
        <div className="App">
            <header className="app-header">
                <h1>Signal Engine Dashboard</h1>
            </header>

            {error && <div className="error-banner">{error}<button onClick={() => setError(null)}>&times;</button></div>}

            <main className="main-content">
                <div className="filters-section">
                    <div className="filters">
                        <select name="symbol" value={filters.symbol} onChange={handleFilterChange} className="filter-select">
                            <option value="">All Symbols</option>
                            {symbols.map(s => <option key={s.symbol} value={s.symbol}>{s.symbol} ({s.strategy_count})</option>)}
                        </select>
                        <input
                            type="number"
                            name="min_sharpe"
                            placeholder="Min Sharpe Ratio (e.g., 0.5)"
                            value={filters.min_sharpe}
                            onChange={handleFilterChange}
                            className="filter-input"
                        />
                        <button onClick={clearFilters} className="clear-filters-btn">Clear Filters</button>
                    </div>
                </div>

                <div className="strategies-section">
                    {loading ? (
                        <div className="loading">Loading strategies...</div>
                    ) : (
                        <StrategyTable strategies={strategies} onRowClick={(s) => setSelectedStrategyId(s.id)} />
                    )}
                </div>
            </main>

            {selectedStrategyId && (
                <StrategyDetailsModal
                    strategyId={selectedStrategyId}
                    onClose={() => setSelectedStrategyId(null)}
                />
            )}
        </div>
    );
}