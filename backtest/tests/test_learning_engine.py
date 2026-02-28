"""Tests for backtest/learning_engine.py — Adaptive Learning Engine."""

import pytest
import sqlite3
import tempfile
from pathlib import Path
from backtest.learning_engine import (
    init_db, insert_signal, backfill_outcomes,
    compute_feature_stats, generate_dashboard,
    ingest_congress_feed, ingest_edgar_feed,
    update_aggregate_features,
    _buy_value_to_points, _compute_cluster_velocity, _normalize_role,
)
from backtest.shared import save_json


@pytest.fixture
def db():
    """Create a temporary in-memory database."""
    conn = init_db(db_path=Path(tempfile.mktemp(suffix='.db')))
    yield conn
    conn.close()


@pytest.fixture
def sample_signals():
    """Sample signal dicts for testing."""
    return [
        {
            'ticker': 'NVDA', 'signal_date': '2025-06-15', 'source': 'congress',
            'representative': 'Rep A', 'trade_size_range': '$100,001 - $250,000',
            'trade_size_points': 8.0, 'insider_name': None,
        },
        {
            'ticker': 'NVDA', 'signal_date': '2025-06-16', 'source': 'edgar',
            'insider_name': 'John Smith', 'representative': None,
            'ticker': 'NVDA', 'signal_date': '2025-06-16', 'source': 'edgar',
        },
        {
            'ticker': 'AAPL', 'signal_date': '2025-06-10', 'source': 'congress',
            'representative': 'Rep B', 'trade_size_range': '$1,001 - $15,000',
            'trade_size_points': 3.0, 'insider_name': None,
        },
    ]


class TestInitDb:
    def test_creates_tables(self, db):
        """init_db should create signals, feature_stats, and weight_history tables."""
        tables = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = [t['name'] for t in tables]
        assert 'signals' in table_names
        assert 'feature_stats' in table_names
        assert 'weight_history' in table_names


class TestInsertSignal:
    def test_insert_basic(self, db):
        """Should insert a signal and return True."""
        result = insert_signal(db, {
            'ticker': 'NVDA', 'signal_date': '2025-06-15', 'source': 'congress',
            'representative': 'Rep A', 'insider_name': None,
        })
        assert result is True
        count = db.execute("SELECT COUNT(*) as cnt FROM signals").fetchone()['cnt']
        assert count == 1

    def test_duplicate_ignored(self, db):
        """Duplicate signals should be ignored (UNIQUE constraint)."""
        sig = {
            'ticker': 'NVDA', 'signal_date': '2025-06-15', 'source': 'congress',
            'representative': 'Rep A', 'insider_name': None,
        }
        insert_signal(db, sig)
        insert_signal(db, sig)  # duplicate
        count = db.execute("SELECT COUNT(*) as cnt FROM signals").fetchone()['cnt']
        assert count == 1

    def test_different_sources_not_duplicate(self, db):
        """Same ticker+date but different source should be separate entries."""
        insert_signal(db, {
            'ticker': 'NVDA', 'signal_date': '2025-06-15', 'source': 'congress',
            'representative': 'Rep A', 'insider_name': None,
        })
        insert_signal(db, {
            'ticker': 'NVDA', 'signal_date': '2025-06-15', 'source': 'edgar',
            'representative': None, 'insider_name': 'CEO',
        })
        count = db.execute("SELECT COUNT(*) as cnt FROM signals").fetchone()['cnt']
        assert count == 2


class TestBackfillOutcomes:
    def test_backfill_with_no_price_data(self, db):
        """Should gracefully handle missing price data."""
        insert_signal(db, {
            'ticker': 'FAKE', 'signal_date': '2025-01-15', 'source': 'congress',
            'representative': 'Rep A', 'insider_name': None,
        })
        filled = backfill_outcomes(db, spy_index={})
        assert filled == 0


class TestComputeFeatureStats:
    def test_no_data(self, db):
        """Should return empty dict when no outcome data exists."""
        stats = compute_feature_stats(db)
        assert stats == {}

    def test_with_outcome_data(self, db):
        """Should compute stats for signals with backfilled outcomes."""
        # Insert signals with pre-filled outcomes
        for i in range(10):
            db.execute(
                """INSERT INTO signals (ticker, signal_date, source, representative, insider_name,
                   trade_size_range, has_convergence, outcome_30d_filled, car_30d)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)""",
                (f'T{i}', f'2025-06-{10+i:02d}', 'congress', f'Rep {i}', None,
                 '$100,001 - $250,000', 0, 0.02 if i < 6 else -0.01)
            )
        db.commit()

        stats = compute_feature_stats(db)
        assert len(stats) > 0
        # Check that source=congress has stats
        assert ('source', 'congress') in stats
        s = stats[('source', 'congress')]
        assert s['n'] == 10
        assert s['positive_rate_30d'] == 0.6  # 6 out of 10 positive


class TestBuyValueToPoints:
    def test_million_plus(self):
        assert _buy_value_to_points(2_500_000) == 15

    def test_mid_range(self):
        assert _buy_value_to_points(120_000) == 8

    def test_small(self):
        assert _buy_value_to_points(5_000) == 3

    def test_zero(self):
        assert _buy_value_to_points(0) == 0

    def test_exact_boundaries(self):
        assert _buy_value_to_points(1_000_000) == 15
        assert _buy_value_to_points(250_000) == 10
        assert _buy_value_to_points(50_000) == 6
        assert _buy_value_to_points(15_000) == 5


class TestIngestEdgarFeed:
    def test_filters_non_buys(self, db, tmp_path):
        """Should skip filings that are not purchases."""
        feed = {
            'filings': [
                {'company': 'Test Corp', 'insider': 'A', 'date': '2025-06-15',
                 'ticker': 'TST', 'direction': 'sell', 'roles': ['CEO'],
                 'title': 'CEO', 'buy_value': 0, 'is_10b5_1': False},
                {'company': 'Test Corp', 'insider': 'B', 'date': '2025-06-16',
                 'ticker': 'TST', 'direction': 'buy', 'roles': ['CFO'],
                 'title': 'CFO', 'buy_value': 500000, 'is_10b5_1': False},
            ]
        }
        feed_path = tmp_path / 'edgar_feed.json'
        save_json(feed_path, feed)
        inserted = ingest_edgar_feed(db, feed_path=feed_path)
        assert inserted == 1
        row = db.execute("SELECT * FROM signals WHERE ticker='TST'").fetchone()
        assert row['insider_role'] == 'CFO'
        assert row['transaction_type'] == 'Purchase'

    def test_uses_xml_ticker(self, db, tmp_path):
        """Should prefer the XML-extracted ticker over company name matching."""
        feed = {
            'filings': [
                {'company': 'Unknown Random Corp', 'insider': 'X', 'date': '2025-06-15',
                 'ticker': 'AAPL', 'direction': 'buy', 'roles': ['Director'],
                 'title': '', 'buy_value': 100000, 'is_10b5_1': False},
            ]
        }
        feed_path = tmp_path / 'edgar_feed.json'
        save_json(feed_path, feed)
        inserted = ingest_edgar_feed(db, feed_path=feed_path)
        assert inserted == 1
        row = db.execute("SELECT ticker FROM signals").fetchone()
        assert row['ticker'] == 'AAPL'


class TestGenerateDashboard:
    def test_empty_db(self, db):
        """Dashboard should work with empty database."""
        dashboard = generate_dashboard(db)
        assert dashboard['database_stats']['total_signals'] == 0
        assert 'scoring_performance' in dashboard
        assert 'top_features' in dashboard

    def test_with_data(self, db):
        """Dashboard should include signal counts."""
        insert_signal(db, {
            'ticker': 'NVDA', 'signal_date': '2025-06-15', 'source': 'congress',
            'representative': 'Rep A', 'insider_name': None,
        })
        insert_signal(db, {
            'ticker': 'AAPL', 'signal_date': '2025-06-16', 'source': 'edgar',
            'representative': None, 'insider_name': 'CEO',
        })
        dashboard = generate_dashboard(db)
        assert dashboard['database_stats']['total_signals'] == 2
        assert dashboard['database_stats']['congress_signals'] == 1
        assert dashboard['database_stats']['edgar_signals'] == 1


class TestMultiTierConvergence:
    def test_tier1_ticker_convergence(self, db):
        """Tier 1: same ticker in congress + edgar within 60d window."""
        insert_signal(db, {
            'ticker': 'NVDA', 'signal_date': '2025-06-15', 'source': 'congress',
            'representative': 'Rep A', 'insider_name': None,
        })
        insert_signal(db, {
            'ticker': 'NVDA', 'signal_date': '2025-06-20', 'source': 'edgar',
            'representative': None, 'insider_name': 'CEO',
        })
        update_aggregate_features(db)
        row = db.execute(
            "SELECT convergence_tier, has_convergence FROM signals WHERE source='edgar' AND ticker='NVDA'"
        ).fetchone()
        assert row['convergence_tier'] == 1
        assert row['has_convergence'] == 1

    def test_no_convergence_single_source(self, db):
        """No convergence if all signals from same source."""
        insert_signal(db, {
            'ticker': 'AAPL', 'signal_date': '2025-06-15', 'source': 'edgar',
            'representative': None, 'insider_name': 'A',
        })
        insert_signal(db, {
            'ticker': 'AAPL', 'signal_date': '2025-06-16', 'source': 'edgar',
            'representative': None, 'insider_name': 'B',
        })
        update_aggregate_features(db)
        row = db.execute(
            "SELECT convergence_tier, has_convergence FROM signals WHERE ticker='AAPL' LIMIT 1"
        ).fetchone()
        assert row['convergence_tier'] == 0
        assert row['has_convergence'] == 0

    def test_tier2_sector_convergence(self, db):
        """Tier 2: 3+ signals from 2+ sources in same sector within 30d."""
        # Congress signal in Industrials
        db.execute(
            """INSERT INTO signals (ticker, signal_date, source, representative, insider_name, sector)
               VALUES ('LMT', '2025-06-15', 'congress', 'Rep A', '', 'Industrials')"""
        )
        # Two edgar signals in same sector, different tickers
        db.execute(
            """INSERT INTO signals (ticker, signal_date, source, representative, insider_name, sector)
               VALUES ('RTX', '2025-06-16', 'edgar', '', 'CEO X', 'Industrials')"""
        )
        db.execute(
            """INSERT INTO signals (ticker, signal_date, source, representative, insider_name, sector)
               VALUES ('NOC', '2025-06-17', 'edgar', '', 'CFO Y', 'Industrials')"""
        )
        db.commit()
        update_aggregate_features(db)
        rows = db.execute(
            "SELECT convergence_tier FROM signals WHERE sector='Industrials'"
        ).fetchall()
        assert any(r['convergence_tier'] == 2 for r in rows)

    def test_tier1_60d_window(self, db):
        """Congress signal 50 days before edgar should still be Tier 1 (within 60d)."""
        insert_signal(db, {
            'ticker': 'MSFT', 'signal_date': '2025-04-01', 'source': 'congress',
            'representative': 'Rep B', 'insider_name': None,
        })
        insert_signal(db, {
            'ticker': 'MSFT', 'signal_date': '2025-05-20', 'source': 'edgar',
            'representative': None, 'insider_name': 'CEO',
        })
        update_aggregate_features(db)
        row = db.execute(
            "SELECT convergence_tier FROM signals WHERE source='edgar' AND ticker='MSFT'"
        ).fetchone()
        assert row['convergence_tier'] == 1

    def test_cluster_counts_still_work(self, db):
        """Cluster 7d/30d counts should still be computed."""
        insert_signal(db, {
            'ticker': 'GOOG', 'signal_date': '2025-06-15', 'source': 'edgar',
            'representative': None, 'insider_name': 'A',
        })
        insert_signal(db, {
            'ticker': 'GOOG', 'signal_date': '2025-06-16', 'source': 'edgar',
            'representative': None, 'insider_name': 'B',
        })
        update_aggregate_features(db)
        rows = db.execute(
            "SELECT same_ticker_signals_7d, same_ticker_signals_30d FROM signals WHERE ticker='GOOG' ORDER BY signal_date DESC LIMIT 1"
        ).fetchall()
        # The later signal (June 16) should see the earlier one (excludes self)
        assert rows[0]['same_ticker_signals_7d'] >= 1
        assert rows[0]['same_ticker_signals_30d'] >= 1


class TestClusterVelocity:
    def test_burst(self):
        assert _compute_cluster_velocity(['2025-06-15', '2025-06-16', '2025-06-17']) == 'burst'

    def test_fast(self):
        assert _compute_cluster_velocity(['2025-06-15', '2025-06-20']) == 'fast'

    def test_moderate(self):
        assert _compute_cluster_velocity(['2025-06-01', '2025-06-10']) == 'moderate'

    def test_slow(self):
        assert _compute_cluster_velocity(['2025-06-01', '2025-06-30']) == 'slow'

    def test_single_signal(self):
        assert _compute_cluster_velocity(['2025-06-15']) == 'n/a'

    def test_empty(self):
        assert _compute_cluster_velocity([]) == 'n/a'


class TestNormalizeRole:
    def test_ceo(self):
        assert _normalize_role('Chief Executive Officer') == 'CEO'

    def test_cfo(self):
        assert _normalize_role('CFO') == 'CFO'

    def test_director(self):
        assert _normalize_role('Director') == 'Director'

    def test_vp(self):
        assert _normalize_role('Senior Vice President') == 'VP'

    def test_president(self):
        assert _normalize_role('President') == 'President'

    def test_empty(self):
        assert _normalize_role('') == 'n/a'

    def test_none(self):
        assert _normalize_role(None) == 'n/a'

    def test_other(self):
        assert _normalize_role('Secretary') == 'Other'
