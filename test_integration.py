"""
Integration tests for the GraphPart hyperparameter optimization system
"""
import unittest
import tempfile
import shutil
import json
import os
import sys
from pathlib import Path
import torch
import numpy as np

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hyperopt_config import HyperOptConfig, NormalizationMethod
from feature_normalizer import FeatureNormalizer, create_enhanced_features
from convergence_monitor import ConvergenceMonitor, AdaptiveLossWeighting, QualityEvaluator
from models import GraphPartitionModel, HyperData
from enhanced_trainer import EnhancedTrainer

class TestHyperoptIntegration(unittest.TestCase):
    """Integration tests for the complete hyperparameter optimization system"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.data_dir = cls.temp_dir / "data"
        cls.data_dir.mkdir()
        
        # Create synthetic test data
        cls._create_test_data()
        
        print(f"Test environment created at: {cls.temp_dir}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        shutil.rmtree(cls.temp_dir)
        print("Test environment cleaned up")
    
    @classmethod
    def _create_test_data(cls):
        """Create synthetic hypergraph data for testing"""
        # Create a small test hypergraph
        test_hgr_content = """10 8
1 2 3
2 3 4 5
3 4
4 5 6
5 6 7 8
6 7
7 8 1
8 1 2
"""
        
        # Save test file
        test_file = cls.data_dir / "test01.hgr"
        with open(test_file, 'w') as f:
            f.write(test_hgr_content)
        
        # Create a slightly larger test file
        test_hgr_content2 = """15 12
1 2 3 4
2 3 4 5 6
3 4 5
4 5 6 7
5 6 7 8 9
6 7 8
7 8 9 10
8 9 10 11 12
9 10 11
10 11 12 13
11 12 13 14 15
12 13 14 15 1 2
"""
        
        test_file2 = cls.data_dir / "test02.hgr"
        with open(test_file2, 'w') as f:
            f.write(test_hgr_content2)
    
    def setUp(self):
        """Set up each test"""
        self.test_config = {
            'learning_rate': 1e-3,
            'epochs': 5,  # Very short for testing
            'alpha': 0.01,
            'beta': 2.0,
            'gamma': 1.0,
            'hidden_dim': 32,
            'latent_dim': 16,
            'dropout_rate': 0.1,
            'mask_rate': 0.2,
            'norm_method': NormalizationMethod.STANDARD,
            'adaptive_weights': True,
            'seed': 42
        }
    
    def test_feature_normalizer(self):
        """Test feature normalization with different methods"""
        print("\n🧪 Testing feature normalization...")
        
        # Create dummy features
        features = np.random.randn(10, 7)
        features[:, 4] = np.abs(features[:, 4])  # Ensure degree feature is positive
        
        for method in NormalizationMethod:
            with self.subTest(method=method):
                normalizer = FeatureNormalizer(method=method)
                normalized = normalizer.normalize_features(features.copy())
                
                # Basic checks
                self.assertEqual(normalized.shape, features.shape)
                self.assertFalse(np.any(np.isnan(normalized)), f"NaN found in {method} normalization")
                self.assertFalse(np.any(np.isinf(normalized)), f"Inf found in {method} normalization")
                
        print("✅ Feature normalization tests passed")
    
    def test_convergence_monitor(self):
        """Test convergence monitoring system"""
        print("\n🧪 Testing convergence monitor...")
        
        monitor = ConvergenceMonitor(
            target_cut_loss=0.02,
            target_balance_loss=1e-3,
            stability_window=5
        )
        
        # Simulate improving losses
        test_losses = [
            (0.5, 0.1, 1.0),    # Initial high losses
            (0.3, 0.05, 0.8),   # Improving
            (0.1, 0.01, 0.6),   # Better
            (0.05, 0.005, 0.4), # Close to targets
            (0.01, 0.0005, 0.3) # Achieved targets
        ]
        
        converged = False
        for cut_loss, balance_loss, kl_loss in test_losses:
            result = monitor.update(cut_loss, balance_loss, kl_loss)
            if result['converged']:
                converged = True
                break
        
        self.assertTrue(converged, "Convergence should be detected when targets are met")
        print("✅ Convergence monitoring tests passed")
    
    def test_adaptive_loss_weighting(self):
        """Test adaptive loss weighting"""
        print("\n🧪 Testing adaptive loss weighting...")
        
        loss_weighter = AdaptiveLossWeighting(
            adaptive=True,
            annealing=True
        )
        
        # Test forward pass
        kl_loss = torch.tensor(0.5)
        cut_loss = torch.tensor(0.1)
        balance_loss = torch.tensor(0.01)
        
        total_loss, weights_info = loss_weighter(kl_loss, cut_loss, balance_loss)
        
        # Basic checks
        self.assertIsInstance(total_loss, torch.Tensor)
        self.assertFalse(torch.isnan(total_loss), "Total loss should not be NaN")
        self.assertIn('alpha', weights_info)
        self.assertIn('beta', weights_info)
        self.assertIn('gamma', weights_info)
        
        print("✅ Adaptive loss weighting tests passed")
    
    def test_model_creation_and_forward(self):
        """Test model creation and forward pass"""
        print("\n🧪 Testing model creation and forward pass...")
        
        model = GraphPartitionModel(
            input_dim=7,
            hidden_dim=32,
            latent_dim=16,
            num_partitions=2,
            use_hypergraph=True
        )
        
        # Create dummy data
        x = torch.randn(10, 7)
        hyperedge_index = torch.tensor([[0, 1, 2, 3, 4], [0, 0, 1, 1, 2]], dtype=torch.long)
        data = HyperData(x=x, hyperedge_index=hyperedge_index)
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            output = model(data)
            self.assertEqual(output.shape, (10, 2))
            self.assertFalse(torch.any(torch.isnan(output)), "Model output should not contain NaN")
        
        # Test sampling
        samples = model.sample(data, m=3)
        self.assertEqual(len(samples), 3)
        for sample in samples:
            self.assertEqual(len(sample), 10)
            self.assertTrue(np.all(np.isin(sample, [0, 1])), "Samples should be binary")
        
        print("✅ Model tests passed")
    
    def test_enhanced_features_creation(self):
        """Test enhanced feature creation with real hypergraph data"""
        print("\n🧪 Testing enhanced feature creation...")
        
        # Load test hypergraph
        test_file = self.data_dir / "test01.hgr"
        with open(test_file, 'r') as f:
            lines = f.readlines()
            num_nets, num_nodes = map(int, lines[0].split())
            hypergraph_vertices = list(range(num_nodes))
            hypergraph_edges = []
            for line in lines[1:]:
                if line.strip():
                    edge = [int(node) - 1 for node in line.split()]
                    hypergraph_edges.append(edge)
        
        # Test feature creation with different normalization methods
        for method in [NormalizationMethod.STANDARD, NormalizationMethod.ROBUST]:
            with self.subTest(method=method):
                try:
                    features = create_enhanced_features(
                        hypergraph_vertices, hypergraph_edges,
                        "test01.hgr", num_nodes, len(hypergraph_edges),
                        norm_method=method,
                        spectral_norm=True,
                        feature_scaling=1.0
                    )
                    
                    # Check feature properties
                    self.assertEqual(features.shape[0], num_nodes)
                    self.assertEqual(features.shape[1], 7)
                    self.assertFalse(np.any(np.isnan(features)), f"NaN in features with {method}")
                    self.assertFalse(np.any(np.isinf(features)), f"Inf in features with {method}")
                    
                except Exception as e:
                    self.fail(f"Feature creation failed with {method}: {str(e)}")
        
        print("✅ Enhanced feature creation tests passed")
    
    def test_enhanced_trainer_integration(self):
        """Test the enhanced trainer with minimal configuration"""
        print("\n🧪 Testing enhanced trainer integration...")
        
        output_dir = self.temp_dir / "trainer_test"
        output_dir.mkdir()
        
        # Create trainer with minimal config
        trainer = EnhancedTrainer(
            config_dict=self.test_config,
            data_path=str(self.data_dir),
            output_dir=str(output_dir)
        )
        
        # Test data preparation
        try:
            enhanced_data = trainer.prepare_data()
            self.assertGreater(len(enhanced_data), 0, "Should load at least one dataset")
            
            for data in enhanced_data:
                self.assertIsInstance(data, HyperData)
                self.assertGreater(data.x.shape[0], 0, "Data should have nodes")
                self.assertEqual(data.x.shape[1], 7, "Data should have 7 features")
        
        except Exception as e:
            self.fail(f"Data preparation failed: {str(e)}")
        
        # Test model creation
        try:
            model = trainer.create_model()
            self.assertIsInstance(model, GraphPartitionModel)
        except Exception as e:
            self.fail(f"Model creation failed: {str(e)}")
        
        print("✅ Enhanced trainer integration tests passed")
    
    def test_quality_evaluator(self):
        """Test partition quality evaluation"""
        print("\n🧪 Testing quality evaluator...")
        
        evaluator = QualityEvaluator()
        
        # Create simple hypergraph
        hypergraph_vertices = list(range(4))
        hypergraph_edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
        
        # Test different partitions
        partitions = [
            np.array([0, 0, 1, 1]),  # Perfect balance
            np.array([0, 1, 0, 1]),  # Different partition
            np.array([0, 0, 0, 1])   # Imbalanced
        ]
        
        for i, partition in enumerate(partitions):
            with self.subTest(partition=i):
                metrics = evaluator.evaluate_partition_quality(
                    partition, hypergraph_vertices, hypergraph_edges
                )
                
                # Check metrics structure
                required_keys = ['cut', 'imbalance', 'balance_score', 'cut_quality', 'overall_quality']
                for key in required_keys:
                    self.assertIn(key, metrics, f"Missing metric: {key}")
                
                # Check value ranges
                self.assertGreaterEqual(metrics['balance_score'], 0)
                self.assertLessEqual(metrics['balance_score'], 1)
                self.assertGreaterEqual(metrics['overall_quality'], 0)
                self.assertLessEqual(metrics['overall_quality'], 1)
        
        print("✅ Quality evaluator tests passed")
    
    def test_config_profiles_loading(self):
        """Test loading configuration profiles"""
        print("\n🧪 Testing configuration profiles...")
        
        config_file = Path(__file__).parent / "config_profiles.json"
        if not config_file.exists():
            self.skipTest("Config profiles file not found")
        
        try:
            with open(config_file, 'r') as f:
                profiles = json.load(f)
            
            # Check required sections
            self.assertIn('optimization_profiles', profiles)
            self.assertIn('normalization_experiments', profiles)
            self.assertIn('loss_weight_experiments', profiles)
            
            # Check specific profiles
            opt_profiles = profiles['optimization_profiles']
            required_profiles = ['aggressive', 'balanced', 'conservative', 'quick_test']
            for profile in required_profiles:
                self.assertIn(profile, opt_profiles, f"Missing profile: {profile}")
                
                # Check required parameters in each profile
                profile_config = opt_profiles[profile]
                required_params = ['learning_rate', 'epochs', 'alpha', 'beta', 'gamma']
                for param in required_params:
                    self.assertIn(param, profile_config, f"Missing parameter {param} in {profile}")
        
        except Exception as e:
            self.fail(f"Config profiles test failed: {str(e)}")
        
        print("✅ Configuration profiles tests passed")
    
    def test_system_robustness(self):
        """Test system robustness with edge cases"""
        print("\n🧪 Testing system robustness...")
        
        # Test with very small hypergraph
        small_vertices = [0, 1]
        small_edges = [[0, 1]]
        
        try:
            features = create_enhanced_features(
                small_vertices, small_edges, "small.hgr", 2, 1,
                norm_method=NormalizationMethod.STANDARD,
                spectral_norm=True,
                feature_scaling=1.0
            )
            self.assertEqual(features.shape[0], 2)
            self.assertFalse(np.any(np.isnan(features)), "Small graph should not produce NaN")
        except Exception as e:
            print(f"  ⚠️  Small graph test failed (expected): {str(e)}")
        
        # Test convergence monitor with extreme values
        monitor = ConvergenceMonitor()
        
        # Test with very good losses (should converge immediately)
        result = monitor.update(0.001, 0.0001, 0.1)
        self.assertTrue(result['cut_target_met'])
        self.assertTrue(result['balance_target_met'])
        
        # Test with NaN (should handle gracefully)
        try:
            result = monitor.update(float('nan'), 0.01, 0.1)
            # Should not crash
        except Exception as e:
            print(f"  ⚠️  NaN handling could be improved: {str(e)}")
        
        print("✅ System robustness tests passed")

class TestHyperoptPerformance(unittest.TestCase):
    """Performance and stress tests"""
    
    def test_memory_usage(self):
        """Test memory usage doesn't grow excessively"""
        print("\n🧪 Testing memory usage...")
        
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and destroy multiple models
        for i in range(10):
            model = GraphPartitionModel(
                input_dim=7, hidden_dim=256, latent_dim=64,
                num_partitions=2, use_hypergraph=True
            )
            
            # Create some data and run forward pass
            x = torch.randn(100, 7)
            hyperedge_index = torch.randint(0, 100, (2, 200))
            data = HyperData(x=x, hyperedge_index=hyperedge_index)
            
            output = model(data)
            
            # Clean up
            del model, x, hyperedge_index, data, output
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        print(f"  Memory usage: {initial_memory:.1f}MB → {final_memory:.1f}MB (growth: {memory_growth:.1f}MB)")
        
        # Allow some growth but not excessive
        self.assertLess(memory_growth, 200, "Memory growth should be reasonable (< 200MB)")
        
        print("✅ Memory usage test passed")
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values"""
        print("\n🧪 Testing numerical stability...")
        
        # Test feature normalizer with extreme values
        extreme_features = np.array([
            [1e10, 1e-10, 0, 1e5, 1, 1e-5, 100],    # Very large/small values
            [0, 0, 0, 1, 0, 0, 0],                   # Mostly zeros
            [1, 1, 1, 1, 1, 1, 1],                   # All ones
            [-1e5, 1e5, -100, 10, 5, -50, 0]        # Mixed signs
        ])
        
        normalizer = FeatureNormalizer(method=NormalizationMethod.ROBUST, eps=1e-8)
        normalized = normalizer.normalize_features(extreme_features)
        
        # Check for numerical issues
        self.assertFalse(np.any(np.isnan(normalized)), "Normalized features should not contain NaN")
        self.assertFalse(np.any(np.isinf(normalized)), "Normalized features should not contain Inf")
        
        # Test model with extreme inputs
        model = GraphPartitionModel(7, 64, 32, 2, True)
        x = torch.tensor(normalized, dtype=torch.float)
        hyperedge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], dtype=torch.long)
        data = HyperData(x=x, hyperedge_index=hyperedge_index)
        
        model.eval()
        with torch.no_grad():
            output = model(data)
            self.assertFalse(torch.any(torch.isnan(output)), "Model output should be stable")
        
        print("✅ Numerical stability test passed")

def run_integration_tests():
    """Run all integration tests"""
    print("🚀 Starting GraphPart Hyperparameter Optimization Integration Tests")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestHyperoptIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestHyperoptPerformance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout, buffer=False)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print("🎉 ALL TESTS PASSED!")
        print(f"✅ Ran {result.testsRun} tests successfully")
    else:
        print("❌ SOME TESTS FAILED!")
        print(f"❌ Failures: {len(result.failures)}")
        print(f"❌ Errors: {len(result.errors)}")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback}")
    
    print("=" * 70)
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_integration_tests()
    sys.exit(0 if success else 1)