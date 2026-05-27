# tests/test_ml_pipeline.py

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Make sure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml_pipeline.disruption_predictor import engineer_features, generate_predictions
from sklearn.ensemble import GradientBoostingClassifier


# ──────────────────────────────────────────
# FIXTURES — reusable test data
# ──────────────────────────────────────────
@pytest.fixture
def sample_supplier_df():
    """Creates a minimal realistic supplier dataframe for testing."""
    return pd.DataFrame({
        "supplier_id"          : [f"SUP-{i:03d}" for i in range(1, 21)],
        "supplier_name"        : [f"Supplier {i}" for i in range(1, 21)],
        "country"              : ["Kenya", "China", "India", "Italy", "Vietnam"] * 4,
        "product_category"     : ["Frames", "Prescription Lenses", "Sunglasses", "Cases", "Lens Coatings"] * 4,
        "risk_level"           : ["LOW", "MEDIUM", "HIGH", "CRITICAL", "LOW"] * 4,
        "reliability_score"    : np.round(np.random.uniform(0.5, 1.0, 20), 2),
        "lead_time_days"       : np.random.randint(7, 90, 20),
        "is_active"            : [True] * 18 + [False, True],
        "annual_spend_usd"     : np.round(np.random.uniform(10000, 2000000, 20), 2),
        "total_shipments"      : np.random.randint(5, 50, 20),
        "disrupted_shipments"  : np.random.randint(0, 10, 20),
        "avg_delay_days"       : np.round(np.random.uniform(0, 30, 20), 2),
        "max_delay_days"       : np.random.randint(0, 60, 20),
        "disruption_rate_pct"  : np.round(np.random.uniform(0, 50, 20), 2),
    })


@pytest.fixture
def engineered_df(sample_supplier_df):
    """Returns feature-engineered dataframe and feature column list."""
    df, feature_cols = engineer_features(sample_supplier_df.copy())
    return df, feature_cols


@pytest.fixture
def trained_model(engineered_df):
    """Trains a quick model on the sample data."""
    df, feature_cols = engineered_df
    X = df[feature_cols]
    y = df["target"]
    model = GradientBoostingClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model, feature_cols


# ──────────────────────────────────────────
# TESTS
# ──────────────────────────────────────────
class TestFeatureEngineering:

    def test_required_encoded_columns_exist(self, engineered_df):
        """Feature engineering must produce encoded columns."""
        df, _ = engineered_df
        required = ["country_enc", "category_enc", "risk_enc", "is_active_int"]
        for col in required:
            assert col in df.columns, f"Missing encoded column: {col}"

    def test_target_column_exists(self, engineered_df):
        """Target column must be created."""
        df, _ = engineered_df
        assert "target" in df.columns

    def test_target_is_binary(self, engineered_df):
        """Target must only contain 0 or 1."""
        df, _ = engineered_df
        unique_values = set(df["target"].unique())
        assert unique_values.issubset({0, 1}), f"Target has non-binary values: {unique_values}"

    def test_feature_cols_count(self, engineered_df):
        """Must return exactly 12 feature columns."""
        _, feature_cols = engineered_df
        assert len(feature_cols) == 12, f"Expected 12 features, got {len(feature_cols)}"

    def test_no_nulls_in_features(self, engineered_df):
        """No null values should exist in feature columns."""
        df, feature_cols = engineered_df
        null_counts = df[feature_cols].isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        assert len(cols_with_nulls) == 0, f"Null values found in: {cols_with_nulls.to_dict()}"

    def test_is_active_int_is_binary(self, engineered_df):
        """is_active_int must only be 0 or 1."""
        df, _ = engineered_df
        assert set(df["is_active_int"].unique()).issubset({0, 1})


class TestModelPredictions:

    def test_prediction_count_matches_input(self, engineered_df, trained_model):
        """Number of predictions must equal number of input rows."""
        df, feature_cols = engineered_df
        model, _ = trained_model
        result = generate_predictions(df.copy(), model, feature_cols)
        assert len(result) == len(df), "Prediction count does not match input row count"

    def test_risk_probability_range(self, engineered_df, trained_model):
        """All risk probabilities must be between 0 and 1."""
        df, feature_cols = engineered_df
        model, _ = trained_model
        result = generate_predictions(df.copy(), model, feature_cols)
        assert result["risk_probability"].between(0, 1).all(), \
            "Some risk probabilities are outside [0, 1]"

    def test_valid_risk_tiers(self, engineered_df, trained_model):
        """Risk tiers must only contain valid labels."""
        df, feature_cols = engineered_df
        model, _ = trained_model
        result = generate_predictions(df.copy(), model, feature_cols)
        valid_tiers = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        actual_tiers = set(result["risk_tier"].astype(str).unique())
        invalid = actual_tiers - valid_tiers
        assert not invalid, f"Invalid risk tiers found: {invalid}"

    def test_predicted_risk_is_binary(self, engineered_df, trained_model):
        """Predicted risk must be 0 or 1."""
        df, feature_cols = engineered_df
        model, _ = trained_model
        result = generate_predictions(df.copy(), model, feature_cols)
        assert set(result["predicted_risk"].unique()).issubset({0, 1})

    def test_risk_probability_column_exists(self, engineered_df, trained_model):
        """Output must contain risk_probability column."""
        df, feature_cols = engineered_df
        model, _ = trained_model
        result = generate_predictions(df.copy(), model, feature_cols)
        assert "risk_probability" in result.columns

    def test_risk_tier_column_exists(self, engineered_df, trained_model):
        """Output must contain risk_tier column."""
        df, feature_cols = engineered_df
        model, _ = trained_model
        result = generate_predictions(df.copy(), model, feature_cols)
        assert "risk_tier" in result.columns


class TestDataIntegrity:

    def test_no_duplicate_supplier_ids(self, sample_supplier_df):
        """Supplier IDs must be unique."""
        assert sample_supplier_df["supplier_id"].nunique() == len(sample_supplier_df)

    def test_reliability_score_in_range(self, sample_supplier_df):
        """Reliability scores must be between 0.5 and 1.0."""
        assert sample_supplier_df["reliability_score"].between(0, 1).all()

    def test_lead_time_positive(self, sample_supplier_df):
        """Lead times must be positive."""
        assert (sample_supplier_df["lead_time_days"] > 0).all()

    def test_disruption_rate_in_range(self, sample_supplier_df):
        """Disruption rate must be between 0 and 100."""
        assert sample_supplier_df["disruption_rate_pct"].between(0, 100).all()