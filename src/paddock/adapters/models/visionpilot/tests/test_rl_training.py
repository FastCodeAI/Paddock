"""Tests for VisionPilot RL fine-tuning infrastructure."""

import json
import tempfile
from pathlib import Path

from paddock.adapters.models.visionpilot.rl_training import (
    FusionWeightPolicy,
    IDMParameterPolicy,
    RLAction,
    RLConfig,
    RLState,
    create_fusion_rl_experiment,
    create_idm_rl_experiment,
)


class TestRLConfig:
    """Test RL configuration."""

    def test_default_config(self):
        config = RLConfig()
        assert config.phase == "idm"
        assert config.learning_rate == 3e-4
        assert config.gamma == 0.99
        assert config.train_datasets == ["nuscenes", "comma2k19"]

    def test_custom_config(self):
        config = RLConfig(
            phase="fusion",
            learning_rate=5e-4,
            num_samples=5000,
        )
        assert config.phase == "fusion"
        assert config.learning_rate == 5e-4
        assert config.num_samples == 5000


class TestRLAction:
    """Test RL action space."""

    def test_action_creation(self):
        action = RLAction(headway_t=1.5, a_max=2.0, b_min=2.5)
        assert action.headway_t == 1.5
        assert action.a_max == 2.0
        assert action.b_min == 2.5

    def test_action_clamping(self):
        # Test that out-of-range values are clamped
        action = RLAction(headway_t=5.0, a_max=10.0, b_min=0.5)
        clamped = action.clamp()

        assert clamped.headway_t == 2.5  # max clamp
        assert clamped.a_max == 4.0  # max clamp
        assert clamped.b_min == 1.5  # min clamp

    def test_action_clamp_low_values(self):
        action = RLAction(headway_t=0.1, a_max=0.5, b_min=0.5)
        clamped = action.clamp()

        assert clamped.headway_t == 0.8  # min clamp
        assert clamped.a_max == 1.0  # min clamp
        assert clamped.b_min == 1.5  # min clamp


class TestIDMParameterPolicy:
    """Test IDM parameter learning policy."""

    def test_initial_parameters(self):
        config = RLConfig(phase="idm")
        policy = IDMParameterPolicy(config)

        assert policy.scenario_params["highway"]["T"] == 1.8
        assert policy.scenario_params["urban"]["T"] == 1.2
        assert policy.scenario_params["suburban"]["T"] == 1.5

    def test_compute_action_highway(self):
        config = RLConfig(phase="idm")
        policy = IDMParameterPolicy(config)

        state = RLState(
            speed_ms=25.0,
            lead_distance_m=100.0,
            lead_prob=0.1,
            curvature=0.0,
            time_step=0,
            scenario="highway",
        )

        action = policy.compute_action(state)
        assert action.headway_t == 1.8
        assert action.a_max == 2.0

    def test_compute_action_urban(self):
        config = RLConfig(phase="idm")
        policy = IDMParameterPolicy(config)

        state = RLState(
            speed_ms=10.0,
            lead_distance_m=30.0,
            lead_prob=0.9,
            curvature=0.05,
            time_step=0,
            scenario="urban",
        )

        action = policy.compute_action(state)
        assert action.headway_t == 1.2
        assert action.a_max == 1.5

    def test_save_and_load_parameters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RLConfig(phase="idm")
            policy = IDMParameterPolicy(config)

            # Modify parameters
            policy.scenario_params["highway"]["T"] = 2.0

            # Save
            save_path = Path(tmpdir) / "policy.json"
            policy.save(save_path)
            assert save_path.exists()

            # Load into new policy
            new_policy = IDMParameterPolicy(config)
            new_policy.load(save_path)
            assert new_policy.scenario_params["highway"]["T"] == 2.0


class TestFusionWeightPolicy:
    """Test sensor fusion weight learning policy."""

    def test_initial_weight(self):
        config = RLConfig(phase="fusion")
        policy = FusionWeightPolicy(config)
        assert policy.fusion_weight == 0.5

    def test_fusion_weight_high_uncertainty(self):
        config = RLConfig(phase="fusion")
        policy = FusionWeightPolicy(config)

        # High uncertainty (far field) + low speed → trust boxes
        weight = policy.compute_fusion_weight(
            cipo_distance=50.0,
            cipo_stddev=15.5,  # High uncertainty
            box_distance=48.0,
            speed_ms=8.0,  # Low speed
            lead_prob=0.7,
        )
        assert weight == 0.2  # Trust boxes 80%

    def test_fusion_weight_low_uncertainty(self):
        config = RLConfig(phase="fusion")
        policy = FusionWeightPolicy(config)

        # Low uncertainty + high confidence → trust camera
        weight = policy.compute_fusion_weight(
            cipo_distance=15.0,
            cipo_stddev=0.9,  # Low uncertainty
            box_distance=15.2,
            speed_ms=15.0,
            lead_prob=0.95,  # High confidence
        )
        assert weight == 0.9  # Trust camera 90%

    def test_fusion_weight_balanced(self):
        config = RLConfig(phase="fusion")
        policy = FusionWeightPolicy(config)

        # Medium uncertainty, medium conditions → balance
        weight = policy.compute_fusion_weight(
            cipo_distance=30.0,
            cipo_stddev=5.0,  # Medium
            box_distance=30.0,
            speed_ms=15.0,  # Medium speed
            lead_prob=0.5,  # Uncertain detection
        )
        assert weight == 0.5  # Balance

    def test_save_and_load_weight(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RLConfig(phase="fusion")
            policy = FusionWeightPolicy(config)

            # Modify weight (would be learned)
            policy.fusion_weight = 0.7

            # Save
            save_path = Path(tmpdir) / "fusion_policy.json"
            policy.save(save_path)
            assert save_path.exists()

            # Load
            new_policy = FusionWeightPolicy(config)
            new_policy.load(save_path)
            assert new_policy.fusion_weight == 0.7


class TestRLExperiment:
    """Test RL experiment management."""

    def test_create_idm_experiment(self):
        exp = create_idm_rl_experiment(name="test_idm")
        assert exp.config.phase == "idm"
        assert isinstance(exp.policy, IDMParameterPolicy)
        assert "test_idm" in str(exp.results_dir)

    def test_create_fusion_experiment(self):
        exp = create_fusion_rl_experiment(name="test_fusion")
        assert exp.config.phase == "fusion"
        assert isinstance(exp.policy, FusionWeightPolicy)
        assert "test_fusion" in str(exp.results_dir)

    def test_save_experiment_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir)
            exp = create_idm_rl_experiment(results_dir=results_dir)
            exp.save_config()

            config_file = results_dir / "config.json"
            assert config_file.exists()

            with open(config_file) as f:
                config_data = json.load(f)
            assert config_data["phase"] == "idm"
            assert "learning_rate" in config_data
            assert "reward_weights" in config_data

    def test_save_and_load_policy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir)
            exp = create_idm_rl_experiment(results_dir=results_dir)

            # Modify policy
            exp.policy.scenario_params["highway"]["T"] = 2.1
            exp.save_policy()

            # Load in new experiment
            new_exp = create_idm_rl_experiment(results_dir=results_dir)
            new_exp.load_policy()
            assert new_exp.policy.scenario_params["highway"]["T"] == 2.1


class TestRLState:
    """Test RL state representation."""

    def test_state_creation(self):
        state = RLState(
            speed_ms=20.0,
            lead_distance_m=50.0,
            lead_prob=0.8,
            curvature=0.01,
            time_step=10,
            scenario="highway",
        )
        assert state.speed_ms == 20.0
        assert state.scenario == "highway"
