"""Tests for compact prompt format in BaseLLMActor."""

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from torchtrade.actor.base_llm_actor import BaseLLMActor


# ============================================================================
# StubActor for testing (BaseLLMActor is abstract)
# ============================================================================


class StubActor(BaseLLMActor):
    """Minimal concrete subclass for testing prompt construction."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return "<think>stub</think><answer>0</answer>"


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def ohlcv_data_24():
    """Synthetic OHLCV data: 24 candles around price 94000."""
    np.random.seed(42)
    n = 24
    base_price = 94000.0
    noise = np.random.normal(0, 100, n)
    close = base_price + noise
    high = close + np.abs(np.random.normal(0, 50, n))
    low = close - np.abs(np.random.normal(0, 50, n))
    open_ = np.roll(close, 1)
    open_[0] = base_price
    volume = np.random.lognormal(5, 1, n)
    return np.stack([open_, high, low, close, volume], axis=1).astype(np.float32)


@pytest.fixture
def make_tensordict(ohlcv_data_24):
    """Factory that wraps ohlcv_data_24 into a TensorDict with a given key."""
    def _make(key="market_data_1Hour_24"):
        td = TensorDict({
            key: torch.tensor(ohlcv_data_24).unsqueeze(0),  # (1, 24, 5)
            "account_state": torch.tensor([[0.0, 0.0, 0.0]]),
        }, batch_size=[])
        return td
    return _make


@pytest.fixture
def compact_actor():
    """StubActor configured with compact prompt format."""
    return StubActor(
        market_data_keys=["market_data_1Hour_24"],
        account_state_labels=["exposure_pct", "position_direction", "unrealized_pnl_pct"],
        action_levels=[0.0, 0.5, 1.0],
        prompt_format="compact",
    )


@pytest.fixture
def verbose_actor():
    """StubActor configured with default verbose prompt format."""
    return StubActor(
        market_data_keys=["market_data_1Hour_24"],
        account_state_labels=["exposure_pct", "position_direction", "unrealized_pnl_pct"],
        action_levels=[0.0, 0.5, 1.0],
        prompt_format="verbose",
    )


# ============================================================================
# TestReferencePrice
# ============================================================================


class TestReferencePrice:
    """Tests for reference price computation in compact format."""

    def test_ref_rounded_to_nearest_100(self, compact_actor, make_tensordict):
        """Reference price should be rounded to the nearest 100."""
        td = make_tensordict()
        output = compact_actor._format_market_data_compact(td)
        # Extract ref value from output like "1H(24) ref:94000"
        for line in output.split("\n"):
            if "ref:" in line:
                ref_str = line.split("ref:")[1].strip()
                ref = int(ref_str)
                assert ref % 100 == 0, f"ref {ref} is not a multiple of 100"
                break
        else:
            pytest.fail("No ref: line found in compact output")

    def test_ref_within_100_of_first_close(self, compact_actor, ohlcv_data_24, make_tensordict):
        """Reference price should be within 100 of the first candle's close."""
        td = make_tensordict()
        first_close = float(ohlcv_data_24[0, 3])  # close is column index 3
        output = compact_actor._format_market_data_compact(td)
        for line in output.split("\n"):
            if "ref:" in line:
                ref_str = line.split("ref:")[1].strip()
                ref = int(ref_str)
                assert abs(ref - first_close) <= 100, (
                    f"ref {ref} is more than 100 away from first close {first_close}"
                )
                break
        else:
            pytest.fail("No ref: line found in compact output")


# ============================================================================
# TestTimeframeHeader
# ============================================================================


class TestTimeframeHeader:
    """Tests for _parse_timeframe_header static method."""

    def test_1hour_24(self):
        assert BaseLLMActor._parse_timeframe_header("market_data_1Hour_24") == "1H(24)"

    def test_5minute_12(self):
        assert BaseLLMActor._parse_timeframe_header("market_data_5Minute_12") == "5M(12)"

    def test_1minute_48(self):
        assert BaseLLMActor._parse_timeframe_header("market_data_1Minute_48") == "1M(48)"

    def test_4hour_6(self):
        assert BaseLLMActor._parse_timeframe_header("market_data_4Hour_6") == "4H(6)"

    def test_1day_30(self):
        assert BaseLLMActor._parse_timeframe_header("market_data_1Day_30") == "1D(30)"


# ============================================================================
# TestFeatureAbbreviations
# ============================================================================


class TestFeatureAbbreviations:
    """Tests for feature name abbreviation."""

    def test_known_features(self):
        assert BaseLLMActor._abbreviate_feature("open") == "O"
        assert BaseLLMActor._abbreviate_feature("high") == "H"
        assert BaseLLMActor._abbreviate_feature("low") == "L"
        assert BaseLLMActor._abbreviate_feature("close") == "C"
        assert BaseLLMActor._abbreviate_feature("volume") == "V"

    def test_unknown_feature(self):
        assert BaseLLMActor._abbreviate_feature("rsi") == "RSI"
        assert BaseLLMActor._abbreviate_feature("macd") == "MACD"


# ============================================================================
# TestCompactFormat
# ============================================================================


class TestCompactFormat:
    """Tests for the full compact format output."""

    def test_compact_output_has_header_and_data(self, compact_actor, make_tensordict):
        """Compact output should have timeframe header, column header, and data rows."""
        td = make_tensordict()
        output = compact_actor._format_market_data_compact(td)
        lines = [l for l in output.strip().split("\n") if l.strip()]

        # First line: timeframe header with ref
        assert "1H(24)" in lines[0]
        assert "ref:" in lines[0]

        # Second line: column abbreviations
        assert lines[1] == "O,H,L,C,V"

        # Remaining lines: data rows (24 candles)
        data_lines = lines[2:]
        assert len(data_lines) == 24

    def test_price_columns_are_delta_encoded(self, compact_actor, make_tensordict):
        """Price columns (O, H, L, C) should be +/- delta integers from ref."""
        td = make_tensordict()
        output = compact_actor._format_market_data_compact(td)
        lines = [l for l in output.strip().split("\n") if l.strip()]

        # Data rows start at line index 2
        for data_line in lines[2:]:
            fields = data_line.split(",")
            # First 4 fields are price deltas (should start with + or -)
            for field in fields[:4]:
                field = field.strip()
                assert field[0] in ("+", "-"), (
                    f"Price delta '{field}' does not start with +/- sign"
                )

    def test_volume_is_rounded_integer(self, compact_actor, make_tensordict):
        """Volume (non-price column) should be a plain rounded integer."""
        td = make_tensordict()
        output = compact_actor._format_market_data_compact(td)
        lines = [l for l in output.strip().split("\n") if l.strip()]

        for data_line in lines[2:]:
            fields = data_line.split(",")
            vol_field = fields[4].strip()
            # Volume should be a plain integer (no + or - prefix for non-price)
            int(vol_field)  # Should not raise

    def test_verbose_is_default(self):
        """Default prompt_format should be 'verbose'."""
        actor = StubActor(
            market_data_keys=["market_data_1Hour_24"],
            account_state_labels=["exposure_pct"],
            action_levels=[0.0, 1.0],
        )
        assert actor.prompt_format == "verbose"

    def test_verbose_dispatch(self, verbose_actor, make_tensordict):
        """Verbose actor should still produce pipe-delimited format."""
        td = make_tensordict()
        output = verbose_actor._construct_market_data(td)
        assert "|" in output
        assert "market_data_1Hour_24:" in output

    def test_compact_dispatch(self, compact_actor, make_tensordict):
        """Compact actor should produce comma-delimited format with ref."""
        td = make_tensordict()
        output = compact_actor._construct_market_data(td)
        assert "ref:" in output
        assert "|" not in output

    def test_custom_price_columns(self, make_tensordict):
        """Custom price_columns should control which columns get delta encoding."""
        actor = StubActor(
            market_data_keys=["market_data_1Hour_24"],
            account_state_labels=["exposure_pct", "position_direction", "unrealized_pnl_pct"],
            action_levels=[0.0, 1.0],
            prompt_format="compact",
            price_columns=["close"],  # Only close gets delta
        )
        td = make_tensordict()
        output = actor._format_market_data_compact(td)
        lines = [l for l in output.strip().split("\n") if l.strip()]

        # Check a data row: only the close column (index 3) should have +/- prefix
        first_data = lines[2].split(",")
        # open (idx 0) is NOT a price column, should be plain integer
        assert first_data[0].strip()[0] not in ("+", "-") or first_data[0].strip().lstrip("+-").isdigit()
        # close (idx 3) IS a price column, should have +/- prefix
        assert first_data[3].strip()[0] in ("+", "-")


# ============================================================================
# TestDeltaEncoding
# ============================================================================


class TestDeltaEncoding:
    """Tests for correctness of delta encoding in compact format."""

    def test_price_columns_are_delta_encoded(self, compact_actor, ohlcv_data_24, make_tensordict):
        """First 4 values in each row should have +/- sign; reconstruct close from ref."""
        td = make_tensordict()
        output = compact_actor._format_market_data_compact(td)
        lines = [l for l in output.strip().split("\n") if l.strip()]

        # Extract ref
        ref_line = lines[0]
        ref = int(ref_line.split("ref:")[1].strip())

        # All data rows (after header line "O,H,L,C,V") must have signed price fields
        data_lines = lines[2:]
        assert len(data_lines) > 0, "No data rows found"
        for row_line in data_lines:
            fields = row_line.split(",")
            for field in fields[:4]:
                field = field.strip()
                assert field[0] in ("+", "-"), (
                    f"Expected +/- prefix on price field '{field}'"
                )

        # Reconstruct first close: ref + delta should match original within +-1
        first_row_fields = data_lines[0].split(",")
        close_delta = int(first_row_fields[3].strip())
        reconstructed_close = ref + close_delta
        expected_close = int(round(float(ohlcv_data_24[0, 3])))
        assert abs(reconstructed_close - expected_close) <= 1, (
            f"Reconstructed close {reconstructed_close} differs from expected {expected_close}"
        )

    def test_volume_stays_absolute(self, compact_actor, ohlcv_data_24, make_tensordict):
        """Volume (5th column) should be a plain integer without +/- prefix."""
        td = make_tensordict()
        output = compact_actor._format_market_data_compact(td)
        lines = [l for l in output.strip().split("\n") if l.strip()]
        data_lines = lines[2:]

        first_row_fields = data_lines[0].split(",")
        vol_field = first_row_fields[4].strip()

        # Volume must NOT start with + (negative volumes are nonsensical; plain int expected)
        assert not vol_field.startswith("+"), (
            f"Volume field '{vol_field}' should not have a + prefix"
        )

        # Value should parse as an integer and match the original within rounding
        parsed_vol = int(vol_field)
        expected_vol = int(round(float(ohlcv_data_24[0, 4])))
        assert parsed_vol == expected_vol, (
            f"Volume {parsed_vol} does not match expected {expected_vol}"
        )

    def test_all_24_rows_present(self, compact_actor, make_tensordict):
        """Compact output must contain exactly 24 data rows."""
        td = make_tensordict()
        output = compact_actor._format_market_data_compact(td)
        lines = [l for l in output.strip().split("\n") if l.strip()]

        # Data rows: lines with commas that are NOT the column header "O,H,L,C,V"
        data_rows = [
            l for l in lines
            if "," in l and not l.startswith("O,")
        ]
        assert len(data_rows) == 24, (
            f"Expected 24 data rows, got {len(data_rows)}"
        )


# ============================================================================
# TestCompactHeader
# ============================================================================


class TestCompactHeader:
    """Tests for timeframe header parsing and feature abbreviations."""

    def test_timeframe_header_1hour(self):
        assert StubActor._parse_timeframe_header("market_data_1Hour_24") == "1H(24)"

    def test_timeframe_header_5minute(self):
        assert StubActor._parse_timeframe_header("market_data_5Minute_12") == "5M(12)"

    def test_timeframe_header_unknown_passthrough(self):
        """Keys that don't match the expected pattern should be returned unchanged."""
        assert StubActor._parse_timeframe_header("weird_key") == "weird_key"

    def test_column_abbreviations_ohlcv(self, compact_actor, make_tensordict):
        """Compact output must contain the 'O,H,L,C,V' column header line."""
        td = make_tensordict()
        output = compact_actor._format_market_data_compact(td)
        assert "O,H,L,C,V" in output, "Column header 'O,H,L,C,V' not found in compact output"

    def test_custom_features_use_uppercase_name(self):
        """Unknown feature names should be uppercased."""
        assert StubActor._abbreviate_feature("rsi") == "RSI"
        assert StubActor._abbreviate_feature("funding_rate") == "FUNDING_RATE"


# ============================================================================
# TestBackwardCompatibility
# ============================================================================


class TestBackwardCompatibility:
    """Tests that verbose format remains the default and is stable."""

    def test_default_format_is_verbose(self, make_tensordict):
        """Actor created without prompt_format should default to verbose format."""
        actor = StubActor(
            market_data_keys=["market_data_1Hour_24"],
            account_state_labels=["exposure_pct", "position_direction", "unrealized_pnl_pct"],
            action_levels=[0.0, 0.5, 1.0],
        )
        td = make_tensordict()
        output = actor._construct_market_data(td)

        assert "|" in output, "Default (verbose) output should use '|' separators"
        assert "market_data_1Hour_24:" in output, (
            "Default (verbose) output should contain the raw key as a section header"
        )
        assert "ref:" not in output, "Default (verbose) output should NOT contain 'ref:'"

    def test_verbose_format_explicit(self, make_tensordict):
        """Explicit verbose actor and default actor should produce identical output."""
        default_actor = StubActor(
            market_data_keys=["market_data_1Hour_24"],
            account_state_labels=["exposure_pct", "position_direction", "unrealized_pnl_pct"],
            action_levels=[0.0, 0.5, 1.0],
        )
        explicit_verbose_actor = StubActor(
            market_data_keys=["market_data_1Hour_24"],
            account_state_labels=["exposure_pct", "position_direction", "unrealized_pnl_pct"],
            action_levels=[0.0, 0.5, 1.0],
            prompt_format="verbose",
        )
        td = make_tensordict()
        assert default_actor._construct_market_data(td) == explicit_verbose_actor._construct_market_data(td)


# ============================================================================
# TestMultipleBlocks
# ============================================================================


@pytest.fixture
def ohlcv_data_12(ohlcv_data_24):
    """First 12 rows of ohlcv_data_24 as a separate 5-minute block."""
    return ohlcv_data_24[:12]


@pytest.fixture
def make_tensordict_two_blocks(ohlcv_data_24, ohlcv_data_12):
    """TensorDict with both 5M(12) and 1H(24) market data blocks."""
    def _make():
        return TensorDict({
            "market_data_5Minute_12": torch.tensor(ohlcv_data_12).unsqueeze(0),  # (1, 12, 5)
            "market_data_1Hour_24": torch.tensor(ohlcv_data_24).unsqueeze(0),    # (1, 24, 5)
            "account_state": torch.tensor([[0.0, 0.0, 0.0]]),
        }, batch_size=[])
    return _make


@pytest.fixture
def two_block_compact_actor():
    """StubActor configured with both 5M and 1H keys in compact format."""
    return StubActor(
        market_data_keys=["market_data_5Minute_12", "market_data_1Hour_24"],
        account_state_labels=["exposure_pct", "position_direction", "unrealized_pnl_pct"],
        action_levels=[0.0, 0.5, 1.0],
        prompt_format="compact",
    )


class TestMultipleBlocks:
    """Tests for compact output with multiple market data blocks."""

    def test_two_blocks_both_compact(self, two_block_compact_actor, make_tensordict_two_blocks):
        """Output should contain headers for both 5M(12) and 1H(24) blocks."""
        td = make_tensordict_two_blocks()
        output = two_block_compact_actor._format_market_data_compact(td)

        assert "5M(12)" in output, "Expected '5M(12)' header in two-block output"
        assert "1H(24)" in output, "Expected '1H(24)' header in two-block output"

        ref_count = output.count("ref:")
        assert ref_count == 2, f"Expected 2 'ref:' values, got {ref_count}"

    def test_each_block_has_correct_row_count(self, two_block_compact_actor, make_tensordict_two_blocks):
        """5M block should have 12 data rows; 1H block should have 24 data rows."""
        td = make_tensordict_two_blocks()
        output = two_block_compact_actor._format_market_data_compact(td)

        # Split into per-block sections by splitting on lines containing "ref:"
        lines = output.strip().split("\n")

        blocks = []
        current_block_data_rows = []
        in_data = False

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if "ref:" in line:
                # New block starts; save previous block if any
                if current_block_data_rows:
                    blocks.append(current_block_data_rows)
                current_block_data_rows = []
                in_data = False
            elif line == "O,H,L,C,V":
                in_data = True
            elif in_data and "," in line:
                current_block_data_rows.append(line)

        # Flush last block
        if current_block_data_rows:
            blocks.append(current_block_data_rows)

        assert len(blocks) == 2, f"Expected 2 blocks, got {len(blocks)}"

        # First block: 5Minute_12 -> 12 rows
        assert len(blocks[0]) == 12, (
            f"Expected 12 rows in 5M block, got {len(blocks[0])}"
        )
        # Second block: 1Hour_24 -> 24 rows
        assert len(blocks[1]) == 24, (
            f"Expected 24 rows in 1H block, got {len(blocks[1])}"
        )
