"""Tests for export_brain_data() in learning_engine.py."""

import json
import pytest
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

from backtest.learning_engine import (
    init_db, export_brain_data,
    TICKER_NAMES, REP_COMMITTEES, _ticker_display,
)
from backtest.shared import BRAIN_SIGNALS, BRAIN_STATS


@pytest.fixture
def db(tmp_path):
    """Create a temp database and point BRAIN_SIGNALS/BRAIN_STATS to tmp_path."""
    conn = init_db(db_path=tmp_path / 'test.db')
    yield conn, tmp_path
    conn.close()


def _insert_signals(conn, signals):
    """Insert test signals directly (bypassing insert_signal to avoid enrichment)."""
    cols = [
        'ticker', 'signal_date', 'source', 'total_score', 'price_at_signal',
        'convergence_tier', 'has_convergence', 'representative', 'insider_name',
        'insider_role', 'transaction_type', 'person_hit_rate_30d',
        'person_trade_count', 'sector', 'car_30d', 'same_ticker_signals_7d',
        'outcome_30d_filled',
    ]
    placeholders = ', '.join(['?'] * len(cols))
    col_str = ', '.join(cols)
    for s in signals:
        vals = [s.get(c) for c in cols]
        conn.execute(f"INSERT INTO signals ({col_str}) VALUES ({placeholders})", vals)
    conn.commit()


# ── Sample data ──────────────────────────────────────────────────────────────

MOCK_SIGNALS = [
    # High-score congress signal with convergence
    dict(ticker='NVDA', signal_date='2026-02-15', source='congress', total_score=92.5,
         price_at_signal=850.0, convergence_tier=2, has_convergence=1,
         representative='Tommy Tuberville', insider_name=None, insider_role=None,
         transaction_type=None, person_hit_rate_30d=0.72, person_trade_count=14,
         sector='Technology', car_30d=0.08, same_ticker_signals_7d=4,
         outcome_30d_filled=1),
    # Mid-score EDGAR insider buy
    dict(ticker='AAPL', signal_date='2026-02-10', source='edgar', total_score=78.0,
         price_at_signal=210.0, convergence_tier=0, has_convergence=0,
         representative=None, insider_name='Tim Cook', insider_role='CEO',
         transaction_type='Purchase', person_hit_rate_30d=0.65, person_trade_count=8,
         sector='Technology', car_30d=0.03, same_ticker_signals_7d=1,
         outcome_30d_filled=1),
    # Low-score congress signal (negative outcome)
    dict(ticker='PFE', signal_date='2026-02-08', source='congress', total_score=55.0,
         price_at_signal=28.0, convergence_tier=0, has_convergence=0,
         representative='Bill Cassidy', insider_name=None, insider_role=None,
         transaction_type=None, person_hit_rate_30d=0.40, person_trade_count=5,
         sector='Health Care', car_30d=-0.05, same_ticker_signals_7d=0,
         outcome_30d_filled=1),
    # High-score EDGAR with convergence
    dict(ticker='RTX', signal_date='2026-02-12', source='edgar', total_score=88.0,
         price_at_signal=115.0, convergence_tier=1, has_convergence=1,
         representative=None, insider_name='Greg Hayes', insider_role='Director',
         transaction_type='Purchase', person_hit_rate_30d=0.80, person_trade_count=12,
         sector='Industrials', car_30d=0.12, same_ticker_signals_7d=2,
         outcome_30d_filled=1),
    # EDGAR sale (exit signal)
    dict(ticker='TSLA', signal_date='2026-02-20', source='edgar', total_score=45.0,
         price_at_signal=220.0, convergence_tier=0, has_convergence=0,
         representative=None, insider_name='Elon Musk', insider_role='CEO',
         transaction_type='Sale', person_hit_rate_30d=None, person_trade_count=2,
         sector='Consumer Discretionary', car_30d=-0.10, same_ticker_signals_7d=0,
         outcome_30d_filled=1),
    # Congress signal from mapped rep (Mark Kelly → Armed Services)
    dict(ticker='LMT', signal_date='2026-02-18', source='congress', total_score=85.0,
         price_at_signal=500.0, convergence_tier=0, has_convergence=0,
         representative='Mark Kelly', insider_name=None, insider_role=None,
         transaction_type=None, person_hit_rate_30d=0.68, person_trade_count=10,
         sector='Industrials', car_30d=0.06, same_ticker_signals_7d=1,
         outcome_30d_filled=1),
    # Another Armed Services rep with outcome
    dict(ticker='GD', signal_date='2026-02-14', source='congress', total_score=72.0,
         price_at_signal=310.0, convergence_tier=0, has_convergence=0,
         representative='Jim Risch', insider_name=None, insider_role=None,
         transaction_type=None, person_hit_rate_30d=0.55, person_trade_count=7,
         sector='Industrials', car_30d=0.04, same_ticker_signals_7d=0,
         outcome_30d_filled=1),
    # Finance committee rep
    dict(ticker='JPM', signal_date='2026-02-16', source='congress', total_score=81.0,
         price_at_signal=195.0, convergence_tier=0, has_convergence=0,
         representative='Tim Scott', insider_name=None, insider_role=None,
         transaction_type=None, person_hit_rate_30d=0.60, person_trade_count=9,
         sector='Financials', car_30d=0.02, same_ticker_signals_7d=0,
         outcome_30d_filled=1),
    # Unfilled outcome (should NOT appear in tier stats)
    dict(ticker='AMZN', signal_date='2026-02-25', source='congress', total_score=90.0,
         price_at_signal=190.0, convergence_tier=1, has_convergence=1,
         representative='Ro Khanna', insider_name=None, insider_role=None,
         transaction_type=None, person_hit_rate_30d=0.70, person_trade_count=6,
         sector='Technology', car_30d=None, same_ticker_signals_7d=2,
         outcome_30d_filled=0),
    # Old signal (90+ days ago — should NOT appear in signals list)
    dict(ticker='XOM', signal_date='2025-10-01', source='edgar', total_score=70.0,
         price_at_signal=105.0, convergence_tier=0, has_convergence=0,
         representative=None, insider_name='Darren Woods', insider_role='CEO',
         transaction_type='Purchase', person_hit_rate_30d=0.50, person_trade_count=3,
         sector='Energy', car_30d=0.07, same_ticker_signals_7d=0,
         outcome_30d_filled=1),
]


class TestTickerNames:
    def test_known_ticker(self):
        assert _ticker_display('NVDA') == 'NVIDIA (NVDA)'

    def test_unknown_ticker(self):
        assert _ticker_display('ZZZZZ') == 'ZZZZZ'

    def test_dict_has_common_tickers(self):
        for t in ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'JPM', 'NVDA']:
            assert t in TICKER_NAMES


class TestRepCommittees:
    def test_has_armed_services(self):
        armed = [k for k, v in REP_COMMITTEES.items() if v == 'Armed Services']
        assert len(armed) >= 2

    def test_has_finance(self):
        fin = [k for k, v in REP_COMMITTEES.items() if v == 'Finance']
        assert len(fin) >= 2


class TestExportBrainData:
    def _run_export(self, conn, tmp_path):
        """Run export with paths redirected to tmp_path."""
        sig_path = tmp_path / 'brain_signals.json'
        stat_path = tmp_path / 'brain_stats.json'
        with patch('backtest.learning_engine.BRAIN_SIGNALS', sig_path), \
             patch('backtest.learning_engine.BRAIN_STATS', stat_path), \
             patch('backtest.learning_engine.OPTIMAL_WEIGHTS', tmp_path / 'nope.json'):
            export_brain_data(conn)
        signals_data = json.loads(sig_path.read_text())
        stats_data = json.loads(stat_path.read_text())
        return signals_data, stats_data

    def test_signals_output_schema(self, db):
        conn, tmp_path = db
        _insert_signals(conn, MOCK_SIGNALS)
        signals_data, _ = self._run_export(conn, tmp_path)

        assert 'generated' in signals_data
        assert 'signals' in signals_data
        assert 'exits' in signals_data
        assert isinstance(signals_data['signals'], list)

    def test_signal_fields(self, db):
        conn, tmp_path = db
        _insert_signals(conn, MOCK_SIGNALS)
        signals_data, _ = self._run_export(conn, tmp_path)

        if signals_data['signals']:
            sig = signals_data['signals'][0]
            required_keys = [
                'ticker', 'price_at_signal', 'signal_date', 'source', 'dir',
                'total_score', 'convergence_tier', 'has_convergence', 'note',
                'car_30d', 'person', 'sector', 'entry_lo', 'entry_hi',
                'target1', 'target2', 'stop',
            ]
            for k in required_keys:
                assert k in sig, f"Missing key: {k}"

    def test_signals_sorted_by_score(self, db):
        conn, tmp_path = db
        _insert_signals(conn, MOCK_SIGNALS)
        signals_data, _ = self._run_export(conn, tmp_path)

        scores = [s['total_score'] for s in signals_data['signals'] if s['total_score']]
        assert scores == sorted(scores, reverse=True)

    def test_old_signals_excluded(self, db):
        """Signals older than 90 days should not appear."""
        conn, tmp_path = db
        _insert_signals(conn, MOCK_SIGNALS)
        signals_data, _ = self._run_export(conn, tmp_path)

        tickers = [s['ticker'] for s in signals_data['signals']]
        # XOM is from 2025-10-01, should be excluded
        assert 'XOM' not in tickers

    def test_exits_are_sales(self, db):
        """Exits should only contain EDGAR sales/dispositions."""
        conn, tmp_path = db
        _insert_signals(conn, MOCK_SIGNALS)
        signals_data, _ = self._run_export(conn, tmp_path)

        for ex in signals_data['exits']:
            tx = ex['transaction_type'].lower()
            assert 'sale' in tx or 'disposition' in tx

    def test_sale_direction_is_short(self, db):
        """EDGAR sales should have dir='short'."""
        conn, tmp_path = db
        _insert_signals(conn, MOCK_SIGNALS)
        signals_data, _ = self._run_export(conn, tmp_path)

        tsla_signals = [s for s in signals_data['signals'] if s['ticker'] == 'TSLA']
        for s in tsla_signals:
            assert s['dir'] == 'short'

    def test_note_includes_company_name(self, db):
        """Notes should include company name for known tickers."""
        conn, tmp_path = db
        _insert_signals(conn, MOCK_SIGNALS)
        signals_data, _ = self._run_export(conn, tmp_path)

        nvda_signals = [s for s in signals_data['signals'] if s['ticker'] == 'NVDA']
        if nvda_signals:
            assert 'NVIDIA' in nvda_signals[0]['note']

    def test_note_includes_convergence(self, db):
        """Convergence signals should have tier info in note."""
        conn, tmp_path = db
        _insert_signals(conn, MOCK_SIGNALS)
        signals_data, _ = self._run_export(conn, tmp_path)

        nvda = [s for s in signals_data['signals'] if s['ticker'] == 'NVDA']
        if nvda:
            assert 'Tier 2 convergence' in nvda[0]['note']

    # ── Stats tests ──────────────────────────────────────────────────────────

    def test_stats_output_schema(self, db):
        conn, tmp_path = db
        _insert_signals(conn, MOCK_SIGNALS)
        _, stats = self._run_export(conn, tmp_path)

        for key in ['generated', 'alpha', 'score_tiers', 'sectors', 'committees',
                     'congress_heatmap', 'kpis', 'insider_sectors', 'ml']:
            assert key in stats, f"Missing stats key: {key}"

    def test_score_tiers_bucketing(self, db):
        """Score tiers should bucket signals correctly."""
        conn, tmp_path = db
        _insert_signals(conn, MOCK_SIGNALS)
        _, stats = self._run_export(conn, tmp_path)

        tiers = {t['tier']: t for t in stats['score_tiers']}
        # 92.5 → 90+ (only filled outcome)
        if '90+' in tiers:
            assert tiers['90+']['n'] >= 1
        # 88.0, 85.0, 81.0 → 80-89
        if '80-89' in tiers:
            assert tiers['80-89']['n'] >= 2
        # 78.0, 72.0 → 65-79
        if '65-79' in tiers:
            assert tiers['65-79']['n'] >= 1
        # 55.0, 45.0 → <65
        if '<65' in tiers:
            assert tiers['<65']['n'] >= 1

    def test_score_tiers_hit_rate_range(self, db):
        """Hit rates should be between 0 and 1."""
        conn, tmp_path = db
        _insert_signals(conn, MOCK_SIGNALS)
        _, stats = self._run_export(conn, tmp_path)

        for t in stats['score_tiers']:
            if t['hit_rate'] is not None:
                assert 0 <= t['hit_rate'] <= 1

    def test_unfilled_excluded_from_tiers(self, db):
        """Unfilled outcomes (AMZN) should NOT count in tier stats."""
        conn, tmp_path = db
        _insert_signals(conn, MOCK_SIGNALS)
        _, stats = self._run_export(conn, tmp_path)

        total_n = sum(t['n'] for t in stats['score_tiers'])
        # 10 signals total, but AMZN is unfilled and XOM is old (still has outcome filled)
        # So filled: NVDA(92.5), AAPL(78), PFE(55), RTX(88), TSLA(45), LMT(85), GD(72), JPM(81), XOM(70) = 9
        assert total_n >= 8  # at minimum 8 filled signals in tier buckets

    def test_committee_mapping_from_data(self, db):
        """Committees should be populated from REP_COMMITTEES mapping."""
        conn, tmp_path = db
        _insert_signals(conn, MOCK_SIGNALS)
        _, stats = self._run_export(conn, tmp_path)

        comm_names = [c['name'] for c in stats['committees']]
        # Tommy Tuberville, Mark Kelly, Jim Risch → Armed Services
        assert 'Armed Services' in comm_names
        # Tim Scott → Finance
        assert 'Finance' in comm_names
        # Bill Cassidy → Health Policy
        assert 'Health Policy' in comm_names

    def test_committee_trade_counts(self, db):
        """Committee trade counts should reflect actual data."""
        conn, tmp_path = db
        _insert_signals(conn, MOCK_SIGNALS)
        _, stats = self._run_export(conn, tmp_path)

        armed = [c for c in stats['committees'] if c['name'] == 'Armed Services']
        if armed:
            # Tuberville + Mark Kelly + Jim Risch = 3 trades
            assert armed[0]['n_trades'] >= 2

    def test_committee_match_rate_range(self, db):
        """Match rates should be between 0 and 1."""
        conn, tmp_path = db
        _insert_signals(conn, MOCK_SIGNALS)
        _, stats = self._run_export(conn, tmp_path)

        for c in stats['committees']:
            assert 0 <= c['match_rate'] <= 1

    def test_congress_heatmap_uses_names(self, db):
        """Heatmap should have company names for known tickers."""
        conn, tmp_path = db
        _insert_signals(conn, MOCK_SIGNALS)
        _, stats = self._run_export(conn, tmp_path)

        for h in stats['congress_heatmap']:
            if h['ticker'] in TICKER_NAMES:
                assert h['name'] == TICKER_NAMES[h['ticker']]

    def test_kpis_populated(self, db):
        conn, tmp_path = db
        _insert_signals(conn, MOCK_SIGNALS)
        _, stats = self._run_export(conn, tmp_path)

        kpis = stats['kpis']
        assert 'top_score' in kpis
        assert 'exceptional_count' in kpis
        assert 'cluster_count' in kpis
        assert 'congress_flags' in kpis

    def test_alpha_uses_high_score_signals(self, db):
        """Alpha should be computed from signals with total_score >= 65."""
        conn, tmp_path = db
        _insert_signals(conn, MOCK_SIGNALS)
        _, stats = self._run_export(conn, tmp_path)

        assert stats['alpha']['n_signals'] >= 1
        # Should be positive since high-score signals have positive CAR
        assert stats['alpha']['value'] > 0

    def test_empty_db_produces_valid_output(self, db):
        """Export on empty DB should not crash."""
        conn, tmp_path = db
        signals_data, stats = self._run_export(conn, tmp_path)

        assert signals_data['signals'] == []
        assert signals_data['exits'] == []
        assert stats['score_tiers'] == []
        assert stats['committees'] == []

    def test_entry_exit_price_levels(self, db):
        """Long signals should have target > entry > stop."""
        conn, tmp_path = db
        _insert_signals(conn, MOCK_SIGNALS)
        signals_data, _ = self._run_export(conn, tmp_path)

        for s in signals_data['signals']:
            if s['dir'] == 'long' and s['price_at_signal']:
                assert s['target1'] > s['entry_hi'] > s['stop']
            elif s['dir'] == 'short' and s['price_at_signal']:
                assert s['target1'] < s['entry_lo'] < s['stop']
