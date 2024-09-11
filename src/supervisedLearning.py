import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import learning_curve, cross_validate, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif

def select_features(X, y, max_features=20):
    best_score = 0
    best_n_features = 0
    for n in range(1, max_features + 1):
        selector = SelectKBest(f_classif, k=n)
        X_selected = selector.fit_transform(X, y)
        
        # cross-validation per feature selection
        clf = RandomForestClassifier(random_state=42)
        cv_score = cross_validate(clf, X_selected, y, cv=5, scoring='accuracy')['test_score'].mean()
        
        if cv_score > best_score:
            best_score = cv_score
            best_n_features = n

    final_selector = SelectKBest(f_classif, k=best_n_features)
    X_selected = final_selector.fit_transform(X, y)
    selected_features = X.columns[final_selector.get_support()].tolist()
    
    return X_selected, selected_features

def create_metrics_table(results_df):
    plt.figure(figsize=(14, 8))
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    table_data = [
        results_df['Model'],
        results_df['CV Accuracy Mean'].round(4),
        results_df['CV Accuracy Std'].round(4),
        results_df['Test Accuracy'].round(4),
        results_df['CV Precision Mean'].round(4),
        results_df['CV Precision Std'].round(4),
        results_df['Test Precision'].round(4),
        results_df['CV Recall Mean'].round(4),
        results_df['CV Recall Std'].round(4),
        results_df['Test Recall'].round(4),
        results_df['CV F1-score Mean'].round(4),
        results_df['CV F1-score Std'].round(4),
        results_df['Test F1-score'].round(4),
        results_df['CV MSE Mean'].round(4),
        results_df['CV MSE Std'].round(4),
        results_df['Test MSE'].round(4),
        results_df['CV R2 Mean'].round(4),
        results_df['CV R2 Std'].round(4),
        results_df['Test R2'].round(4)
    ]

    table = plt.table(cellText=list(zip(*table_data)),
                      colLabels=['Model', 'CV Acc Mean', 'CV Acc Std', 'Test Acc',
                                 'CV Prec Mean', 'CV Prec Std', 'Test Prec',
                                 'CV Rec Mean', 'CV Rec Std', 'Test Rec',
                                 'CV F1 Mean', 'CV F1 Std', 'Test F1',
                                 'CV MSE Mean', 'CV MSE Std', 'Test MSE',
                                 'CV R2 Mean', 'CV R2 Std', 'Test R2'],
                      loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)  
    table.scale(1.2, 1.5)
    
    plt.title("Model Performance Metrics", fontsize=16)
    plt.tight_layout()
    plt.show()
    
def perform_supervised_learning(X, y):
    print("\nPerforming Supervised Learning with Feature Selection:")
    
    # Feature selection
    X_selected, selected_features = select_features(X, y)
    X_selected = pd.DataFrame(X_selected, columns=selected_features)
    
    print(f"Selected features: {selected_features}")
    print(f"Number of selected features: {len(selected_features)}")
    
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(X_selected.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix of Selected Features')
    plt.tight_layout()
    plt.show()
    
    
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    scaler = MinMaxScaler()
    
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    X_train_scaled.to_csv('train.csv', index=False)
    X_test_scaled.to_csv('test.csv', index=False)
    
    classifiers = {
        'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance'),
        'DecisionTree': DecisionTreeClassifier(max_depth=3, min_samples_split=15, min_samples_leaf=10),
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=15, min_samples_leaf=10, max_features='sqrt', random_state=42),
        'SVC': SVC(kernel='rbf', C=0.1, gamma='scale', probability=True)
    }

    metrics_list = ['accuracy', 'precision', 'recall', 'f1']
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='weighted'),
        'recall': make_scorer(recall_score, average='weighted'),
        'f1': make_scorer(f1_score, average='weighted'),
        'mse': make_scorer(mean_squared_error),
        'r2': make_scorer(r2_score)
    }


    results = []
    
    for name, clf in classifiers.items():
        print(f"\nTraining and evaluating {name}...")
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = cross_validate(clf, X_train_scaled, y_train, cv=cv, scoring=scoring, error_score='raise')
        
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        
        fig, axs = plt.subplots(2, 2, figsize=(10, 14))
        fig.suptitle(f'Model Evaluation - {name}', fontsize=15, y=1.0)
        
        cm = metrics.confusion_matrix(y_test, y_pred)
        disp = metrics.ConfusionMatrixDisplay(cm, display_labels=['Benign', 'Malignant'])
        disp.plot(ax=axs[0, 0], cmap='YlOrRd', values_format='d')
        disp.im_.set_clim(0, np.max(cm))  # Set color scale to max value
        axs[0, 0].set_title('Confusion Matrix', fontsize=16)
        axs[0, 0].tick_params(axis='both', which='major', labelsize=12)
        
        box = axs[0, 0].get_position()
        axs[0, 0].set_position([box.x0 - 0.05, box.y0, box.width * 0.6, box.height * 0.6])
        
        # ROC Curve
        y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
        roc_auc = metrics.auc(fpr, tpr)
        axs[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        axs[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axs[0, 1].set_xlim([0.0, 1.0])
        axs[0, 1].set_ylim([0.0, 1.05])
        axs[0, 1].set_xlabel('False Positive Rate', fontsize=12)
        axs[0, 1].set_ylabel('True Positive Rate', fontsize=12)
        axs[0, 1].set_title('ROC Curve', fontsize=16)
        axs[0, 1].legend(loc="lower right", fontsize=12)
        axs[0, 1].tick_params(axis='both', which='major', labelsize=10)
        
        box = axs[0, 1].get_position()
        axs[0, 1].set_position([box.x0 + 0.05, box.y0, box.width * 0.9, box.height * 0.9])
        
        # Learning Curve
        train_sizes, train_scores, test_scores = learning_curve(
            clf, X_train_scaled, y_train, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 10))
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        axs[1, 0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1, color="r")
        axs[1, 0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color="g")
        axs[1, 0].plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        axs[1, 0].plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        
        axs[1, 0].set_xlabel("Training examples", fontsize=10)
        axs[1, 0].set_ylabel("Score", fontsize=10)
        axs[1, 0].set_title("Learning Curve", fontsize=14)
        axs[1, 0].legend(loc="best", fontsize=10)
        axs[1, 0].tick_params(axis='both', which='major', labelsize=8)
        
        # Adjust learning curve position
        box = axs[1, 0].get_position()
        axs[1, 0].set_position([box.x0, box.y0 + 0.05, box.width * 0.9, box.height * 0.75])
        
        summary_text = (
            f"Model: {name}\n\n"
            f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}\n\n"
            f"Test Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}\n\n"
            f"Test Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}\n\n"
            f"Test F1-score: {f1_score(y_test, y_pred, average='weighted'):.4f}\n\n"
            f"ROC AUC: {roc_auc:.4f}"
        )
        axs[1, 1].text(0.5, 0.5, summary_text, fontsize=14, va='center', ha='center', 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1))
        axs[1, 1].axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.905, hspace=0.352, wspace=0.25, bottom=0.07)
        plt.show()

        # Calculate metrics
        results.append({
            'Model': name,
            'CV Accuracy Mean': cv_results['test_accuracy'].mean(),
            'CV Accuracy Std': cv_results['test_accuracy'].std(),
            'CV Precision Mean': cv_results['test_precision'].mean(),
            'CV Precision Std': cv_results['test_precision'].std(),
            'CV Recall Mean': cv_results['test_recall'].mean(),
            'CV Recall Std': cv_results['test_recall'].std(),
            'CV F1-score Mean': cv_results['test_f1'].mean(),
            'CV F1-score Std': cv_results['test_f1'].std(),
            'CV MSE Mean': cv_results['test_mse'].mean(),
            'CV MSE Std': cv_results['test_mse'].std(),
            'CV R2 Mean': cv_results['test_r2'].mean(),
            'CV R2 Std': cv_results['test_r2'].std(),
            'Test Accuracy': accuracy_score(y_test, y_pred),
            'Test Precision': precision_score(y_test, y_pred, average='weighted'),
            'Test Recall': recall_score(y_test, y_pred, average='weighted'),
            'Test F1-score': f1_score(y_test, y_pred, average='weighted'),
            'Test MSE': mean_squared_error(y_test, y_pred),
            'Test R2': r2_score(y_test, y_pred)
        })

    results_df = pd.DataFrame(results)
    print("\nResults summary:")
    print(results_df)

    create_metrics_table(results_df)


    return X_selected, y
