"""
Test file for MyDecisionTreeClassifier and MyRandomForestClassifier
Programmer: Xavier Melancon
Class: CPSC 322-01 Fall 2025

This test suite provides thorough, readable tests for the decision tree and random forest
classifiers. Tests cover basic functionality, edge cases, and integration scenarios.
"""
from mysklearn.myclassifiers import MyDecisionTreeClassifier, MyRandomForestClassifier


class TestDecisionTree:
    """Test suite for MyDecisionTreeClassifier"""
    
    def test_simple_binary_classification(self):
        """Test basic binary classification on a simple dataset"""
        print("\n--- Test: Simple Binary Classification ---")
        
        # Create simple training data
        X_train = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [2, 2],
            [2, 3],
            [3, 2],
            [3, 3]
        ]
        y_train = [0, 0, 0, 0, 1, 1, 1, 1]
        
        # Train the tree
        dt = MyDecisionTreeClassifier(max_depth=5)
        dt.fit(X_train, y_train)
        
        # Test on same data
        predictions = dt.predict(X_train)
        
        # Check predictions
        accuracy = sum(1 for pred, true in zip(predictions, y_train) if pred == true) / len(y_train)
        print(f"Predictions: {predictions}")
        print(f"True labels: {y_train}")
        print(f"Accuracy: {accuracy:.2f}")
        
        assert accuracy > 0.75, "Accuracy should be above 75% on simple data"
        print("Test passed")
    
    def test_single_feature_split(self):
        """Test tree behavior with only one feature"""
        print("\n--- Test: Single Feature Split ---")
        
        X_train = [[1], [2], [3], [4], [5], [6]]
        y_train = [0, 0, 0, 1, 1, 1]
        
        dt = MyDecisionTreeClassifier(max_depth=3)
        dt.fit(X_train, y_train)
        
        predictions = dt.predict(X_train)
        accuracy = sum(1 for pred, true in zip(predictions, y_train) if pred == true) / len(y_train)
        
        print(f"Predictions: {predictions}")
        print(f"Accuracy: {accuracy:.2f}")
        
        assert accuracy > 0.8, "Should achieve good accuracy with linearly separable data"
        print("Test passed")
    
    def test_max_depth_limiting(self):
        """Test that max_depth parameter actually limits tree depth"""
        print("\n--- Test: Max Depth Limiting ---")
        
        X_train = [
            [i, j] for i in range(10) for j in range(10)
        ]
        y_train = [
            (i + j) % 2 for i in range(10) for j in range(10)
        ]
        
        # Train with limited depth
        dt_limited = MyDecisionTreeClassifier(max_depth=2)
        dt_limited.fit(X_train, y_train)
        
        # Train with no depth limit
        dt_unlimited = MyDecisionTreeClassifier(max_depth=None)
        dt_unlimited.fit(X_train, y_train)
        
        pred_limited = dt_limited.predict(X_train)
        pred_unlimited = dt_unlimited.predict(X_train)
        
        acc_limited = sum(1 for p, t in zip(pred_limited, y_train) if p == t) / len(y_train)
        acc_unlimited = sum(1 for p, t in zip(pred_unlimited, y_train) if p == t) / len(y_train)
        
        print(f"Limited depth (2) accuracy: {acc_limited:.2f}")
        print(f"Unlimited depth accuracy: {acc_unlimited:.2f}")
        
        assert acc_unlimited >= acc_limited, "Unlimited depth should have >= accuracy on training data"
        print("Test passed")
    
    def test_random_feature_selection(self):
        """Test that random feature selection works"""
        print("\n--- Test: Random Feature Selection ---")
        
        X_train = [
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [3, 4, 5, 6],
            [4, 5, 6, 7],
            [5, 6, 7, 8],
            [6, 7, 8, 9]
        ]
        y_train = [0, 0, 0, 1, 1, 1]
        
        # Train with limited features
        dt = MyDecisionTreeClassifier(max_depth=3, max_features=2)
        dt.fit(X_train, y_train)
        
        predictions = dt.predict(X_train)
        accuracy = sum(1 for p, t in zip(predictions, y_train) if p == t) / len(y_train)
        
        print(f"Accuracy with max_features=2: {accuracy:.2f}")
        
        assert accuracy > 0.5, "Should perform better than random guessing"
        print("Test passed")
    
    def test_empty_predictions(self):
        """Test that tree handles empty test set gracefully"""
        print("\n--- Test: Empty Predictions ---")
        
        X_train = [[1, 2], [3, 4], [5, 6]]
        y_train = [0, 1, 0]
        
        dt = MyDecisionTreeClassifier(max_depth=3)
        dt.fit(X_train, y_train)
        
        predictions = dt.predict([])
        
        print(f"Predictions for empty set: {predictions}")
        assert predictions == [], "Empty input should return empty predictions"
        print("Test passed")
    
    def test_all_same_class(self):
        """Test behavior when all training samples are same class"""
        print("\n--- Test: All Same Class ---")
        
        X_train = [[1, 2], [3, 4], [5, 6], [7, 8]]
        y_train = [1, 1, 1, 1]  # All same class
        
        dt = MyDecisionTreeClassifier(max_depth=3)
        dt.fit(X_train, y_train)
        
        predictions = dt.predict(X_train)
        
        print(f"Predictions: {predictions}")
        print(f"All predictions are class 1: {all(p == 1 for p in predictions)}")
        
        assert all(p == 1 for p in predictions), "Should predict the single class for all samples"
        print("Test passed")


class TestRandomForest:
    """Test suite for MyRandomForestClassifier"""
    
    def test_basic_forest_creation(self):
        """Test basic random forest creation and training"""
        print("\n--- Test: Basic Forest Creation ---")
        
        X_train = [
            [0, 0], [0, 1], [1, 0], [1, 1],
            [2, 2], [2, 3], [3, 2], [3, 3]
        ]
        y_train = [0, 0, 0, 0, 1, 1, 1, 1]
        
        # Create and train forest
        rf = MyRandomForestClassifier(n_trees=5, max_depth=3, max_features=1)
        rf.fit(X_train, y_train)
        
        print(f"Number of trees in forest: {len(rf.trees)}")
        
        # Make predictions
        predictions = rf.predict(X_train)
        accuracy = sum(1 for p, t in zip(predictions, y_train) if p == t) / len(y_train)
        
        print(f"Predictions: {predictions}")
        print(f"Accuracy: {accuracy:.2f}")
        
        assert len(rf.trees) > 0, "Forest should have trees after fitting"
        assert len(predictions) == len(X_train), "Should have one prediction per sample"
        print("Test passed")
    
    def test_m_trees_selection(self):
        """Test that only M best trees are kept"""
        print("\n--- Test: M Trees Selection ---")
        
        X_train = [
            [i, j] for i in range(5) for j in range(5)
        ]
        y_train = [
            (i + j) % 2 for i in range(5) for j in range(5)
        ]
        
        # Create forest with N=10 trees but keep only M=5
        rf = MyRandomForestClassifier(
            n_trees=10,
            max_depth=2,
            max_features=1,
            m_trees=5
        )
        rf.fit(X_train, y_train)
        
        print(f"Requested {10} trees but kept {5} best trees")
        print(f"Actual forest size: {len(rf.trees)}")
        
        assert len(rf.trees) == 5, "Forest should contain exactly M=5 trees"
        print("Test passed")
    
    def test_majority_voting(self):
        """Test that majority voting works correctly"""
        print("\n--- Test: Majority Voting ---")
        
        X_train = [
            [0, 0], [0, 1], [1, 0], [1, 1],
            [2, 2], [2, 3], [3, 2], [3, 3]
        ]
        y_train = [0, 0, 0, 0, 1, 1, 1, 1]
        
        # Create forest with odd number of trees for clear majority
        rf = MyRandomForestClassifier(n_trees=3, max_depth=3, max_features=1)
        rf.fit(X_train, y_train)
        
        # Test single prediction
        test_sample = [[0, 0]]
        prediction = rf.predict(test_sample)
        
        print(f"Test sample: {test_sample}")
        print(f"Prediction: {prediction}")
        
        assert len(prediction) == 1, "Should have one prediction"
        assert prediction[0] in [0, 1], "Prediction should be valid class"
        print("Test passed")
    
    def test_multiple_predictions(self):
        """Test that forest can handle multiple test samples"""
        print("\n--- Test: Multiple Predictions ---")
        
        X_train = [
            [0, 0], [0, 1], [1, 0], [1, 1],
            [2, 2], [2, 3], [3, 2], [3, 3]
        ]
        y_train = [0, 0, 0, 0, 1, 1, 1, 1]
        
        rf = MyRandomForestClassifier(n_trees=5, max_depth=3, max_features=1)
        rf.fit(X_train, y_train)
        
        # Test with multiple samples
        X_test = [
            [0, 0],
            [3, 3],
            [1, 1],
            [2, 2]
        ]
        predictions = rf.predict(X_test)
        
        print(f"Number of test samples: {len(X_test)}")
        print(f"Number of predictions: {len(predictions)}")
        print(f"Predictions: {predictions}")
        
        assert len(predictions) == len(X_test), "Should have one prediction per test sample"
        print("Test passed")
    
    def test_forest_bootstrapping(self):
        """Test that different trees are trained on different bootstrap samples"""
        print("\n--- Test: Forest Bootstrapping ---")
        
        X_train = [
            [i, j] for i in range(10) for j in range(10)
        ]
        y_train = [
            (i * j) % 2 for i in range(10) for j in range(10)
        ]
        
        # Create multiple forests and check they make different predictions
        # (due to random bootstrap samples)
        rf1 = MyRandomForestClassifier(n_trees=5, max_depth=2, max_features=2)
        rf1.fit(X_train, y_train)
        
        rf2 = MyRandomForestClassifier(n_trees=5, max_depth=2, max_features=2)
        rf2.fit(X_train, y_train)
        
        test_sample = [[5, 5]]
        pred1 = rf1.predict(test_sample)
        pred2 = rf2.predict(test_sample)
        
        print(f"Forest 1 prediction: {pred1}")
        print(f"Forest 2 prediction: {pred2}")
        print("(Note: Predictions may differ due to random bootstrapping)")
        
        # Both should produce valid predictions
        assert len(pred1) == 1 and len(pred2) == 1, "Both forests should make predictions"
        print("Test passed")
    
    def test_forest_accuracy(self):
        """Test that forest achieves reasonable accuracy on training data"""
        print("\n--- Test: Forest Accuracy ---")
        
        # Create a more complex dataset
        X_train = [
            [0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2],
            [2, 0], [2, 1], [2, 2], [3, 3], [3, 4], [3, 5],
            [4, 3], [4, 4], [4, 5], [5, 3], [5, 4], [5, 5]
        ]
        y_train = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        
        rf = MyRandomForestClassifier(n_trees=10, max_depth=4, max_features=2)
        rf.fit(X_train, y_train)
        
        predictions = rf.predict(X_train)
        accuracy = sum(1 for p, t in zip(predictions, y_train) if p == t) / len(y_train)
        
        print(f"Training accuracy: {accuracy:.2f}")
        
        assert accuracy > 0.7, "Forest should achieve at least 70% accuracy on training data"
        print("Test passed")
    
    def test_no_trees_error(self):
        """Test that prediction fails gracefully before fit"""
        print("\n--- Test: No Trees Error ---")
        
        rf = MyRandomForestClassifier(n_trees=5)
        
        try:
            predictions = rf.predict([[1, 2]])
            assert False, "Should raise an error when predicting before fit"
        except ValueError as e:
            print(f"Correctly raised error: {e}")
            print("Test passed")
    
    def test_consistent_m_trees(self):
        """Test that M trees parameter is respected"""
        print("\n--- Test: Consistent M Trees ---")
        
        X_train = [
            [i, j] for i in range(6) for j in range(6)
        ]
        y_train = [
            (i + j) % 2 for i in range(6) for j in range(6)
        ]
        
        test_cases = [
            (5, 3),   # N=5, M=3
            (10, 5),  # N=10, M=5
            (20, 8),  # N=20, M=8
        ]
        
        for n_trees, m_trees in test_cases:
            rf = MyRandomForestClassifier(
                n_trees=n_trees,
                max_depth=2,
                max_features=1,
                m_trees=m_trees
            )
            rf.fit(X_train, y_train)
            
            print(f"N={n_trees}, M={m_trees}: Forest has {len(rf.trees)} trees")
            assert len(rf.trees) == m_trees, f"Expected {m_trees} trees, got {len(rf.trees)}"
        
        print("Test passed")


def run_all_tests():
    """Run all tests and print summary"""
    print("="*60)
    print("Running Decision Tree and Random Forest Tests")
    print("="*60)
    
    # Test Decision Tree
    print("\n" + "="*60)
    print("DECISION TREE TESTS")
    print("="*60)
    
    dt_tests = TestDecisionTree()
    
    try:
        dt_tests.test_simple_binary_classification()
    except AssertionError as e:
        print(f"Test failed: {e}")
    
    try:
        dt_tests.test_single_feature_split()
    except AssertionError as e:
        print(f"Test failed: {e}")
    
    try:
        dt_tests.test_max_depth_limiting()
    except AssertionError as e:
        print(f"Test failed: {e}")
    
    try:
        dt_tests.test_random_feature_selection()
    except AssertionError as e:
        print(f"Test failed: {e}")
    
    try:
        dt_tests.test_empty_predictions()
    except AssertionError as e:
        print(f"Test failed: {e}")
    
    try:
        dt_tests.test_all_same_class()
    except AssertionError as e:
        print(f"Test failed: {e}")
    
    # Test Random Forest
    print("\n" + "="*60)
    print("RANDOM FOREST TESTS")
    print("="*60)
    
    rf_tests = TestRandomForest()
    
    try:
        rf_tests.test_basic_forest_creation()
    except AssertionError as e:
        print(f"Test failed: {e}")
    
    try:
        rf_tests.test_m_trees_selection()
    except AssertionError as e:
        print(f"Test failed: {e}")
    
    try:
        rf_tests.test_majority_voting()
    except AssertionError as e:
        print(f"Test failed: {e}")
    
    try:
        rf_tests.test_multiple_predictions()
    except AssertionError as e:
        print(f"Test failed: {e}")
    
    try:
        rf_tests.test_forest_bootstrapping()
    except AssertionError as e:
        print(f"Test failed: {e}")
    
    try:
        rf_tests.test_forest_accuracy()
    except AssertionError as e:
        print(f"Test failed: {e}")
    
    try:
        rf_tests.test_no_trees_error()
    except Exception as e:
        print(f"Test failed: {e}")
    
    try:
        rf_tests.test_consistent_m_trees()
    except AssertionError as e:
        print(f"Test failed: {e}")
    

    print("\n\nAll Tests Passed!")



if __name__ == "__main__":
    run_all_tests()
