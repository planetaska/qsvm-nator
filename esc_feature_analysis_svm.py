import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import seaborn as sns

# Set paths
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = CURRENT_DIR / "datasets"


def load_dataset(dataset_path):
    """Load a preprocessed audio feature dataset."""
    print(f"Loading dataset from {dataset_path}")

    # Load features and labels
    X = np.load(dataset_path / "X.npy")
    y = np.load(dataset_path / "y.npy")

    # Load metadata
    with open(dataset_path / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Features: {', '.join(metadata['feature_set'])}")

    # Get class information from metadata
    if metadata["dataset_type"] == "esc10":
        # For ESC-10 dataset, load class names from metadata file
        meta_df = pd.read_csv(CURRENT_DIR / "data" / "esc50" / "meta" / "esc50.csv")
        # Get unique categories for the 10 classes in ESC-10
        class_names = meta_df[meta_df["esc10"] == True]["category"].unique()
        print(f"Classes: {', '.join(class_names)}")

    return X, y, metadata


def prepare_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and testing sets and standardize features."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")

    # Standardize features (important for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_svm(X_train, y_train, kernel="rbf", C=1.0, gamma="scale"):
    """Train an SVM classifier."""
    print(f"\nTraining SVM classifier with kernel={kernel}, C={C}, gamma={gamma}")

    # Create and train the SVM classifier
    svm = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
    svm.fit(X_train, y_train)

    return svm


def evaluate_model(model, X_test, y_test, class_names=None):
    """Evaluate model performance on test data."""
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")

    # Classification report
    print("\nClassification Report:")
    if class_names is not None:
        # Map numeric class labels to class names
        target_names = [class_names[i] for i in sorted(np.unique(y_test))]
        print(classification_report(y_test, y_pred, target_names=target_names))
    else:
        print(classification_report(y_test, y_pred))

    return y_pred, accuracy


def plot_confusion_matrix(y_test, y_pred, class_names=None):
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot with seaborn
    if class_names is not None:
        # Map numeric class labels to class names
        labels = [class_names[i] for i in sorted(np.unique(y_test))]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
    else:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.show()


def plot_feature_importance(X, y, feature_names):
    """Plot feature importance using a simple SVM-based approach with empirical cutoff detection."""
    # Train a linear SVM
    svm = SVC(kernel="linear")
    svm.fit(X, y)

    # Get feature importance (absolute value of coefficients)
    # For multiclass, average the importance across all one-vs-rest classifiers
    if hasattr(svm, "coef_"):
        importance = np.abs(svm.coef_).mean(axis=0)
    else:
        print("Cannot extract feature importance from non-linear kernel.")
        return

    # Create Series for better indexing
    importance_series = pd.Series(importance, index=feature_names)

    # Sort all features by importance (descending)
    sorted_features = importance_series.sort_values(ascending=False)

    # Calculate cumulative importance
    cumulative_importance = sorted_features.cumsum() / sorted_features.sum()

    # Find cutoff points using different methods

    # Method 1: Elbow detection using second derivatives
    # Calculate approximate second derivative to find where the curve flattens
    importance_values = sorted_features.values
    # Pad to avoid edge issues
    padded = np.pad(importance_values, (1, 1), "edge")
    second_deriv = np.diff(np.diff(padded))
    # Find the first significant drop (where second derivative has a peak)
    # Add minimal threshold to avoid noise
    significant_points = np.where(second_deriv > 0.5 * np.max(second_deriv))[0]
    elbow_point = significant_points[0] if len(significant_points) > 0 else 10

    # Method 2: Cumulative importance threshold (e.g., 80%)
    threshold = 0.8
    cumulative_cutoff = np.where(cumulative_importance >= threshold)[0][0] + 1

    # Choose the cutoff to mark on the plot
    cutoff_point = max(elbow_point, 3)  # Make sure we have at least 3 features

    # Determine figure size based on number of features
    n_features = len(feature_names)
    fig_width = 16  # Wider to accommodate the text column
    fig_height = max(12, min(25, n_features * 0.15))  # Adjust for vertical layout

    # =============================================================
    # FIGURE 1: Feature importance by rank and by feature category
    # =============================================================

    # Create a figure with gridspec for more control
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 0.01, 1, 0.2], width_ratios=[1, 3])

    # Text explanation areas (one for each row)
    explanation_ax1 = fig.add_subplot(gs[0, 0])
    explanation_ax1.axis("off")  # Hide axes

    # Empty middle row for spacing
    empty_ax = fig.add_subplot(gs[1, :])
    empty_ax.axis("off")  # Hide axes

    explanation_ax2 = fig.add_subplot(gs[2, 0])
    explanation_ax2.axis("off")  # Hide axes

    # Empty bottom row for extra margin space
    bottom_empty_ax = fig.add_subplot(gs[3, :])
    bottom_empty_ax.axis("off")  # Hide axes

    # Feature importance bar chart
    ax1 = fig.add_subplot(gs[0, 1])

    # Categorical feature importance chart (replacing cumulative chart)
    ax2 = fig.add_subplot(gs[2, 1])

    # Set figure margins with more bottom space for vertical labels
    plt.subplots_adjust(
        top=0.96,
        bottom=0.16,
        left=0.01,
        right=0.98,
    )

    # Add explanatory text for first plot (feature importance)
    explanation_text1 = """
    FEATURE IMPORTANCE
    -----------------
    
    Magnitude of each feature's 
    contribution to the SVM model.
    
    Orange dotted line marks the 
    "elbow point" - where importance 
    drops significantly.
    
    INTERPRETATION
    -------------
    
    The elbow method identifies a 
    natural division between essential 
    and less useful features.
    """

    explanation_ax1.text(
        0.05,
        0.5,
        explanation_text1,
        va="center",
        ha="left",
        fontsize=10,
        family="monospace",
        transform=explanation_ax1.transAxes,
    )

    # Add explanatory text for second plot (feature categories)
    explanation_text2 = """
    FEATURE IMPORTANCE 
    BY CATEGORY
    -------------------
    
    Features grouped by category
    in original dataset order.
    
    Shows relative importance
    within feature groups.
    
    INTERPRETATION
    -------------
    
    Identifies which feature types
    contribute most to the model.
    """

    explanation_ax2.text(
        0.05,
        0.5,
        explanation_text2,
        va="center",
        ha="left",
        fontsize=10,
        family="monospace",
        transform=explanation_ax2.transAxes,
    )

    # Plot 1: Feature importance
    sorted_features.plot(kind="bar", color="teal", ax=ax1)

    # Add a vertical line to indicate the elbow cutoff point
    ax1.axvline(
        x=cutoff_point - 0.5, color="orange", linestyle=":", linewidth=2, alpha=0.7
    )

    # Add annotation at the top of the plot - moved further right and up to 98% height
    max_y = sorted_features.max()
    y_pos = max_y
    ax1.annotate(
        f"Elbow method cutoff: {cutoff_point} features",
        xy=(cutoff_point - 0.5, y_pos),
        xytext=(cutoff_point + 2, y_pos),  # Shifted further to the right
        arrowprops=dict(arrowstyle="->", color="orange", alpha=0.7),
        color="orange",
        fontweight="bold",
        va="top",
    )

    ax1.set_title(f"Feature Importance Distribution (All {n_features} Features)")
    ax1.set_ylabel("Feature Importance (abs. coefficient magnitude)")
    ax1.set_xlabel("")  # Remove x label from top plot

    # Adjust x-axis tick parameters to prevent overlapping labels
    ax1.tick_params(axis="x", rotation=90, labelsize=8)

    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    # Plot 2: Feature importance by original categories
    # Group features according to their names/prefixes
    # Create a Series with features in their original order (as in features.csv)
    original_order_importance = pd.Series(importance, index=feature_names)
    original_order_importance.plot(kind="bar", color="purple", ax=ax2)

    ax2.set_title("Feature Importance By Category (Original Order)")
    ax2.set_ylabel("Feature Importance")
    ax2.set_xlabel("")  # Remove x label from bottom plot

    # Adjust x-axis tick parameters to prevent overlapping labels
    ax2.tick_params(axis="x", rotation=90, labelsize=8)

    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()

    # =============================================================
    # FIGURE 2: Cumulative feature importance (separate figure)
    # =============================================================

    # Call the new function for cumulative importance chart
    cumulative_cutoff = plot_cumulative_importance(
        sorted_features=sorted_features,
        threshold=threshold,
        elbow_point=elbow_point,
        cutoff_point=cutoff_point,
    )

    print(f"\nFeature importance analysis:")
    print(f"- Elbow method detected {elbow_point} significant features")
    print(
        f"- Top {cumulative_cutoff} features capture {threshold*100:.0f}% of total importance"
    )
    print(f"- Recommended cutoff: {cutoff_point} features")

    print("\nNote: For optimal feature selection, consider incremental feature")
    print(
        "addition with cross-validation to evaluate model performance vs. complexity."
    )

    # Return the top features according to the cutoff point
    return sorted_features.index[:cutoff_point].tolist()


def plot_cumulative_importance(
    sorted_features, threshold=0.8, elbow_point=10, cutoff_point=10
):
    """
    Plot cumulative feature importance chart with tick marks for every feature.

    Parameters:
    -----------
    sorted_features : pd.Series
        Features sorted by importance (descending)
    threshold : float, default=0.8
        Importance threshold (e.g., 80%)
    elbow_point : int
        Point detected by elbow method
    cutoff_point : int
        Recommended cutoff point for feature selection
    """
    # Calculate cumulative importance
    cumulative_importance = sorted_features.cumsum() / sorted_features.sum()

    # Find the cumulative cutoff point for the threshold
    cumulative_cutoff = np.where(cumulative_importance >= threshold)[0][0] + 1

    # Create a new figure for cumulative importance
    fig, ax_cum = plt.subplots(figsize=(12, 6))

    # Plot cumulative importance
    cumulative_importance.plot(ax=ax_cum, color="blue")

    # Set tick marks for every feature using integer values
    feature_indices = np.arange(len(sorted_features))
    ax_cum.set_xticks(feature_indices)
    ax_cum.set_xticklabels(feature_indices + 1, rotation=90, fontsize=8)

    # Mark the threshold
    ax_cum.axhline(y=threshold, color="red", linestyle="--", alpha=0.7)
    ax_cum.annotate(
        f"{threshold*100:.0f}% threshold cutoff: {cumulative_cutoff} features",
        xy=(cumulative_cutoff, threshold),
        xytext=(cumulative_cutoff + 1, threshold - 0.15),
        arrowprops=dict(arrowstyle="->", color="red", alpha=0.7),
        color="red",
        fontweight="bold",
    )

    # Mark the elbow cutoff
    ax_cum.axvline(x=cutoff_point, color="orange", linestyle=":", alpha=0.7)

    # Add annotation for elbow point
    ax_cum.annotate(
        f"Elbow method cutoff: {cutoff_point} features",
        xy=(cutoff_point, cumulative_importance.iloc[cutoff_point]),
        xytext=(cutoff_point + 2, cumulative_importance.iloc[cutoff_point]),
        arrowprops=dict(arrowstyle="->", color="orange", alpha=0.7),
        color="orange",
        fontweight="bold",
    )

    ax_cum.set_title("Cumulative Feature Importance")
    ax_cum.set_ylabel("Cumulative Importance Ratio")
    ax_cum.set_xlabel("Feature Number")
    ax_cum.grid(True, linestyle="--", alpha=0.7)

    # Add explanatory text with mathematical explanation of elbow method
    ax_cum.text(
        0.40,
        0.05,
        """Cumulative importance shows how total importance accumulates as features are added.
Red dashed line: 80% threshold    |    Orange dotted line: elbow cutoff

Elbow method: Uses 2nd derivative (∇²f) of importance curve to detect
significant curvature change where f''(x) > 0.5*max(f''(x)).
This identifies where feature importance begins to drop.""",
        transform=ax_cum.transAxes,
        bbox=dict(
            facecolor="white", alpha=0.7, edgecolor="gray", boxstyle="round,pad=0.5"
        ),
        fontsize=9,
    )

    plt.tight_layout()
    plt.show()

    return cumulative_cutoff


def plot_pca(X, y, class_names=None):
    """Plot PCA visualization of the data."""
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Create a DataFrame for easier plotting
    df = pd.DataFrame(data=X_pca, columns=["PC1", "PC2"])
    df["target"] = y

    # Define a list of markers to cycle through
    markers = ["o", "s", "^", "v", "D", "P", "*", "X", "d", "p", "h", "8", "+", "x"]

    # Plot with different markers for each class
    plt.figure(figsize=(10, 8))

    # Use class names if available
    if class_names is not None:
        df["class"] = df["target"].apply(lambda x: class_names[x])

        # Get unique classes
        unique_classes = df["class"].unique()

        # Plot each class with a different marker
        for i, cls in enumerate(unique_classes):
            marker = markers[i % len(markers)]  # Cycle through markers
            subset = df[df["class"] == cls]
            plt.scatter(
                subset["PC1"],
                subset["PC2"],
                marker=marker,
                label=cls,
                s=80,  # Slightly larger points for better visibility
                alpha=0.7,
            )

        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        # If no class names, use numeric targets
        unique_targets = df["target"].unique()

        # Plot each target with a different marker
        for i, tgt in enumerate(unique_targets):
            marker = markers[i % len(markers)]  # Cycle through markers
            subset = df[df["target"] == tgt]
            plt.scatter(
                subset["PC1"],
                subset["PC2"],
                marker=marker,
                label=f"Class {tgt}",
                s=80,
                alpha=0.7,
            )

        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.title("PCA Visualization")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    return pca


def optimize_hyperparameters(X_train, y_train):
    """Find optimal hyperparameters using GridSearchCV."""
    print("\nOptimizing hyperparameters...")

    # Define parameter grid
    param_grid = {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.1, 0.01, 0.001],
        "kernel": ["rbf", "linear", "poly"],
    }

    # Create grid search
    grid_search = GridSearchCV(
        SVC(probability=True),
        param_grid,
        cv=5,
        scoring="accuracy",
        verbose=1,
        n_jobs=-1,
    )

    # Fit grid search
    grid_search.fit(X_train, y_train)

    # Print results
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_


def main():
    """Main function to run the SVM classifier on audio datasets."""
    print("=" * 60)
    print("🎵 AUDIO CLASSIFICATION WITH SVM 🎵")
    print("=" * 60)

    # List available datasets
    print("\nAvailable datasets:")
    datasets = [d for d in DATASETS_DIR.iterdir() if d.is_dir()]
    # Sort datasets alphabetically by name
    datasets = sorted(datasets, key=lambda x: x.name)

    if not datasets:
        print("No datasets found. Please run audio_feature_extractor.py first.")
        return

    for i, dataset in enumerate(datasets, 1):
        print(f"{i}. {dataset.name}")

    # Select dataset
    dataset_idx = int(input("\nSelect dataset (number): ")) - 1
    if dataset_idx < 0 or dataset_idx >= len(datasets):
        print("Invalid selection. Using the first dataset.")
        dataset_idx = 0

    dataset_path = datasets[dataset_idx]

    # Load dataset
    X, y, metadata = load_dataset(dataset_path)
    feature_names = metadata["feature_names"]
    dataset_type = metadata["dataset_type"]  # Get dataset type from metadata

    # Get class mapping
    meta_df = pd.read_csv(CURRENT_DIR / "data" / "esc50" / "meta" / "esc50.csv")

    # Create class mapping based on dataset type
    if dataset_type == "esc10":
        class_filter = meta_df["esc10"] == True
    else:  # esc50
        class_filter = meta_df["target"].isin(np.unique(y))

    class_indices = sorted(meta_df[class_filter]["target"].unique())
    class_names = {
        idx: meta_df[meta_df["target"] == idx]["category"].iloc[0]
        for idx in class_indices
    }

    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data(X, y)

    # Ask user if they want to optimize hyperparameters
    optimize = input("\nOptimize hyperparameters? (y/n, can be slow): ").lower() == "y"

    if optimize:
        # Find best hyperparameters
        best_model, best_params = optimize_hyperparameters(X_train, y_train)
        model = best_model
    else:
        # Train SVM with default parameters
        model = train_svm(X_train, y_train)

    # Evaluate model
    y_pred, accuracy = evaluate_model(model, X_test, y_test, class_names)

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, class_names)

    # Plot PCA visualization
    pca = plot_pca(X, y, class_names)

    # Plot feature importance if linear kernel was used
    if not optimize or best_params["kernel"] == "linear":
        # Need to train a linear SVM for feature importance
        top_features = plot_feature_importance(X_train, y_train, feature_names)
        print(f"\nTop features: {', '.join(top_features)}")

    print("\n✅ SVM classification complete!")


if __name__ == "__main__":
    main()
