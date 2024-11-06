import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    RandomizedSearchCV, 
    GridSearchCV,
    learning_curve,
    cross_validate,
    StratifiedKFold,
    train_test_split
)
from sklearn import metrics
from sklearn.metrics import (
    make_scorer, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    log_loss,
    roc_auc_score,
    average_precision_score
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
#from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import (
    SelectKBest,
    f_classif
)
from scipy.stats import randint, uniform

def metrics_table(results_df):
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
        results_df['CV Log Loss Mean'].round(4),
        results_df['CV Log Loss Std'].round(4),
        results_df['Test Log Loss'].round(4),
        results_df['CV ROC AUC Mean'].round(4),
        results_df['CV ROC AUC Std'].round(4),
        results_df['Test ROC AUC'].round(4),
        results_df['CV Avg Precision Mean'].round(4),
        results_df['CV Avg Precision Std'].round(4),
        results_df['Test Avg Precision'].round(4)
    ]

    table = plt.table(cellText=list(zip(*table_data)),
                      colLabels=['Model', 'CV Acc Mean', 'CV Acc Std', 'Test Acc',
                                 'CV Prec Mean', 'CV Prec Std', 'Test Prec',
                                 'CV Rec Mean', 'CV Rec Std', 'Test Rec',
                                 'CV F1 Mean', 'CV F1 Std', 'Test F1',
                                 'CV Log Loss Mean', 'CV Log Loss Std', 'Test Log Loss',
                                 'CV ROC AUC Mean', 'CV ROC AUC Std', 'Test ROC AUC',
                                 'CV Avg Precision Mean', 'CV Avg Precision Std', 'Test Avg Precision'],
                      loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)  
    table.scale(1.2, 1.5)
    
    plt.title("Model Performance Metrics", fontsize=16)
    plt.tight_layout()
    plt.show()


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

    
def supervised_learning(X, y):
    print("\nPerforming Supervised Learning with Feature Selection and Hyperparameter Tuning:")
    
    # Feature selection
    X_selected, selected_features = select_features(X, y)
    X_selected = pd.DataFrame(X_selected, columns=selected_features)
    
    print(f"Features selezionate: {selected_features}")
    print(f"Numero di features selezionate: {len(selected_features)}")
    
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
    
    param_grids = {
        'KNN': {
            'n_neighbors': [5, 7, 9, 11, 13],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto'],
            'leaf_size': [20, 30, 40]
        },
        'DecisionTree': {
            'max_depth': [3, 4, 5, 6, 7],
            'min_samples_split': [5, 10, 15, 20],
            'min_samples_leaf': [5, 10, 15],
            'max_features': ['sqrt', 'log2']
        },
        'RandomForest': {
            'n_estimators': [100, 150, 200],
            'max_depth': [5, 7, 9],
            'min_samples_split': [5, 10, 15],
            'min_samples_leaf': [4, 6, 8],
            'max_features': ['sqrt', 'log2'],
            'bootstrap': [True],
            'oob_score': [True]
        },
        'SVC': {
            'C': [0.1, 0.5, 1.0],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf'],
            'class_weight': ['balanced']
        }
    }

    classifiers = {
        'KNN': KNeighborsClassifier(),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'SVC': SVC(probability=True, random_state=42)
    }

    #metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'log_loss', 'roc_auc', 'average_precision']
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='weighted'),
        'recall': make_scorer(recall_score, average='weighted'),
        'f1': make_scorer(f1_score, average='weighted'),
        'log_loss': 'neg_log_loss',  #scorer predefinito di sklearn
        'roc_auc': make_scorer(roc_auc_score, needs_proba=True),
        'average_precision': make_scorer(average_precision_score, needs_proba=True)
    }

    results = []
    
     # Definizione di quale search method usare per ogni classificatore
    search_methods = {
        'KNN': (GridSearchCV, {}),  
        'DecisionTree': (RandomizedSearchCV, {'n_iter': 20}),  
        'RandomForest': (RandomizedSearchCV, {'n_iter': 20}),
        'SVC': (GridSearchCV, {})
    }

    for name, clf in classifiers.items():
        print(f"\nAddestramento e valutazione del modello {name}...")
        
        # Stratified K-Fold con più split per una validazione più robusta
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        if name in ['KNN', 'SVC']:  # Pochi iperparametri -> GridSearchCV
            search = GridSearchCV(
                clf,
                param_grids[name],
                cv=cv,
                scoring='f1_weighted',
                n_jobs=-1,
                refit=True
            )
        else:  # Molti iperparametri -> RandomizedSearchCV
            search = RandomizedSearchCV(
                clf,
                param_grids[name],
                n_iter=20,
                cv=cv,
                scoring='f1_weighted',
                n_jobs=-1,
                random_state=42,
                refit=True
            )
        
        search.fit(X_train_scaled, y_train)
        best_clf = search.best_estimator_
        print(f"Parameters migliori evidenziati per {name}: {search.best_params_}")
        
        # Valutazione delle performance
        cv_results = cross_validate(
            best_clf, 
            X_train_scaled, 
            y_train, 
            cv=cv, 
            scoring=scoring, 
            return_train_score=True  # Aggiunto per monitorare l'overfitting
        )
        
        # Calcolo del gap tra training e validation score per monitorare l'overfitting
        train_scores = cv_results['train_f1'].mean()
        val_scores = cv_results['test_f1'].mean()
        overfitting_gap = train_scores - val_scores
        print(f"Gap di overfitting per {name}: {overfitting_gap:.4f}")

        # Predictions on test set
        y_pred_proba = best_clf.predict_proba(X_test_scaled)
        log_loss_score = log_loss(y_test, y_pred_proba[:, 1])
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        avg_precision = average_precision_score(y_test, y_pred_proba[:, 1])

        y_pred = best_clf.predict(X_test_scaled)
        
        # Plotting section (same as before)
        fig, axs = plt.subplots(2, 2, figsize=(10, 14))
        fig.suptitle(f'Model Evaluation - {name} (Hyperparameter Tuned)', fontsize=15, y=1.0)
        
        cm = metrics.confusion_matrix(y_test, y_pred)
        disp = metrics.ConfusionMatrixDisplay(cm, display_labels=['Benign', 'Malignant'])
        disp.plot(ax=axs[0, 0], cmap='YlOrRd', values_format='d')
        disp.im_.set_clim(0, np.max(cm))
        axs[0, 0].set_title('Confusion Matrix', fontsize=16)
        axs[0, 0].tick_params(axis='both', which='major', labelsize=12)
        
        box = axs[0, 0].get_position()
        axs[0, 0].set_position([box.x0 - 0.05, box.y0, box.width * 0.6, box.height * 0.6])
        
        # ROC Curve
        y_pred_proba = best_clf.predict_proba(X_test_scaled)[:, 1]
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
            best_clf, X_train_scaled, y_train, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
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

        # Raccolta dei risultati
        results.append({
            'Model': name,
            'Best Parameters': str(search.best_params_),
            'CV Accuracy Mean': cv_results['test_accuracy'].mean(),
            'CV Accuracy Std': cv_results['test_accuracy'].std(),
            'CV Precision Mean': cv_results['test_precision'].mean(),
            'CV Precision Std': cv_results['test_precision'].std(),
            'CV Recall Mean': cv_results['test_recall'].mean(),
            'CV Recall Std': cv_results['test_recall'].std(),
            'CV F1-score Mean': cv_results['test_f1'].mean(),
            'CV F1-score Std': cv_results['test_f1'].std(),
            'CV Log Loss Mean': -cv_results['test_log_loss'].mean(),  
            'CV Log Loss Std': cv_results['test_log_loss'].std(),   
            'CV ROC AUC Mean': cv_results['test_roc_auc'].mean(),   
            'CV ROC AUC Std': cv_results['test_roc_auc'].std(),      
            'CV Avg Precision Mean': cv_results['test_average_precision'].mean(),  
            'CV Avg Precision Std': cv_results['test_average_precision'].std(),    
            'Test Accuracy': accuracy_score(y_test, y_pred),
            'Test Precision': precision_score(y_test, y_pred, average='weighted'),
            'Test Recall': recall_score(y_test, y_pred, average='weighted'),
            'Test F1-score': f1_score(y_test, y_pred, average='weighted'),
            'Test Log Loss': log_loss_score,      
            'Test ROC AUC': roc_auc,              
            'Test Avg Precision': avg_precision    
        })

    results_df = pd.DataFrame(results)
    print("\nResults summary:")
    print(results_df)

    metrics_table(results_df)

    return X_selected, y
