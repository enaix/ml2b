import numpy as np
import pandas as pd
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
import traceback
import shutil

# Import from our architecture
from grade_functions import GRADERS
import common
from src.bench import Competition, DATA_LOADERS


class ModernGrader:
    """Modern grading system that works independently of BenchPipeline"""
    
    def __init__(self, base_path: str):
        """
        Initialize the modern grader.
        
        Args:
            base_path: Base path for competition data and results
        """
        self.base_path = base_path
        self.comp_cache = {}
    
    def load_competition(self, competition_id: str) -> Competition:
        """Load competition directly from competitions.json"""
        if competition_id in self.comp_cache:
            return self.comp_cache[competition_id]
        
        # Load competition metadata
        comp_json_path = os.path.join(self.base_path, "data", "competitions.json")
        if not os.path.exists(comp_json_path):
            raise FileNotFoundError(f"Competitions JSON not found at {comp_json_path}")
        
        with open(comp_json_path, "r") as f:
            comp_data = json.load(f)
        
        if competition_id not in comp_data:
            raise ValueError(f"Competition {competition_id} not found in competitions.json")
        
        # Create competition object for grading stage
        comp = Competition(
            comp_id=competition_id,
            bench=None,  # No bench reference needed for grading
            metadata=comp_data[competition_id],
            tasks={},  # No tasks needed for grading
            grading_stage=True  # Skip file validation
        )
        
        self.comp_cache[competition_id] = comp
        return comp
    
    def grade_code(self, train_code: Dict[str, Callable], competition_id: str, 
                  language: str, mono_predict: bool, folds: Optional[int] = None) -> Dict[str, Any]:
        """
        Grade LLM-generated code using hardcoded loader paths.
        
        Args:
            train_code: Dictionary containing train/predict functions
            competition_id: ID of the competition to grade
            language: Language of the submission
            mono_predict: Whether using monolithic predict mode
            folds: Optional override for number of folds
            
        Returns:
            Dictionary with grading results and metrics
        """
        try:
            # Validate train_code input
            if not isinstance(train_code, dict):
                raise ValueError("train_code must be a dictionary of functions")
            
            # Validate required functions exist
            if mono_predict:
                if "train_and_predict" not in train_code:
                    raise ValueError("train_and_predict function required for mono_predict mode")
            else:
                required_funcs = ["train", "prepare_val", "predict"]
                for func_name in required_funcs:
                    if func_name not in train_code:
                        raise ValueError(f"{func_name} function required for modular mode")
            
            # Load competition
            comp = self.load_competition(competition_id)
            
            # Get number of folds
            num_folds = folds if folds is not None else comp.metadata.get("cv_folds", 5)
            
            # Get grader function
            grader_name = comp.metadata.get("grader", "default")
            if grader_name not in GRADERS:
                raise ValueError(f"Grader '{grader_name}' not found in grade_functions.GRADERS")
            
            grader_func = GRADERS[grader_name]
            
            scores = []
            fold_details = []
            
            # Grade each fold
            for fold_idx in range(num_folds):
                try:
                    # Load separate datasets using hardcoded loader paths
                    loader_name = comp.metadata.get("data_loader", "default")
                    loader_class = DATA_LOADERS.get(loader_name)
                    if loader_class is None:
                        raise ValueError(f"Data loader '{loader_name}' not found")
                    
                    loader = loader_class()
                    
                    # Load training data
                    train_dataset = loader.load_train_data(comp, fold_idx, self.base_path)
                    
                    # Load validation features
                    val_features_dataset = loader.load_validation_features(comp, fold_idx, self.base_path)
                    
                    # Load validation labels
                    val_labels = loader.load_validation_labels(comp, fold_idx, self.base_path)
                    
                    # Execute submission code and grade
                    fold_score = self._grade_fold(
                        train_code, train_dataset, val_features_dataset, val_labels, 
                        comp, grader_func, mono_predict, fold_idx
                    )
                    
                    scores.append(fold_score)
                    fold_details.append({
                        "fold": fold_idx,
                        "score": fold_score,
                        "status": "success"
                    })
                    
                    print(f"Fold {fold_idx + 1}/{num_folds} completed with score: {fold_score:.4f}")
                    
                except Exception as e:
                    print(f"Fold {fold_idx + 1} failed: {str(e)}")
                    scores.append(np.nan)
                    fold_details.append({
                        "fold": fold_idx,
                        "score": None,
                        "status": "failed",
                        "error": str(e)
                    })
                    traceback.print_exc()
            
            # Clean up fold data
            self._cleanup_fold_data(comp)
            
            # Calculate final score
            valid_scores = [s for s in scores if not np.isnan(s)]
            if not valid_scores:
                raise ValueError("All folds failed during grading")
            
            final_score = np.mean(valid_scores)
            
            return {
                "score": final_score,
                "cv_scores": scores,
                "fold_details": fold_details,
                "competition_id": competition_id,
                "grader": grader_name,
                "folds_used": num_folds,
                "successful_folds": len(valid_scores),
                "failed_folds": len(scores) - len(valid_scores),
                "success": True
            }
            
        except Exception as e:
            error_msg = f"Grading failed for competition {competition_id}: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return {
                "score": None,
                "error": error_msg,
                "success": False
            }

    def _grade_fold(self, train_code: Dict[str, Callable], train_dataset: Dict[str, Any], 
                   val_features_dataset: Dict[str, Any], val_labels: pd.DataFrame, 
                   comp: Competition, grader_func: Callable, mono_predict: bool, fold_idx: int) -> float:
        """
        Grade a single fold using separate datasets.
        
        Args:
            train_code: Dictionary of LLM-generated functions
            train_dataset: Training data with target
            val_features_dataset: Validation features without target
            val_labels: True labels for validation
            comp: Competition object
            grader_func: Grader function to use
            mono_predict: Whether using monolithic predict mode
            fold_idx: Fold index for logging
            
        Returns:
            Score for this fold
        """
        # Validate that all required functions are callable
        if mono_predict:
            if not callable(train_code.get("train_and_predict")):
                raise ValueError("train_and_predict is not a callable function")
        else:
            for func_name in ["train", "prepare_val", "predict"]:
                if not callable(train_code.get(func_name)):
                    raise ValueError(f"{func_name} is not a callable function")
        
        # Execute the appropriate prediction function
        try:
            if mono_predict:
                predictions = train_code["train_and_predict"](train_dataset, val_features_dataset)
            else:
                # Train phase
                train_output = train_code["train"](train_dataset)
                
                # Prepare validation phase
                val_prepared = train_code["prepare_val"](val_features_dataset, train_output)
                
                # Predict phase
                predictions = train_code["predict"](train_output, val_prepared)
            
            # Grade the predictions against true labels
            score = grader_func(predictions, val_labels, comp.metadata)
            return score
            
        except Exception as e:
            print(f"Error during fold {fold_idx} execution: {str(e)}")
            traceback.print_exc()
            raise

    def _cleanup_fold_data(self, comp: Competition):
        """Clean up temporary fold data"""
        fold_dir = os.path.join(self.base_path, "competitions", "folds", comp.comp_id)
        private_dir = os.path.join(self.base_path, "competitions", "validation", comp.comp_id)
        
        if os.path.exists(fold_dir):
            shutil.rmtree(fold_dir)
        if os.path.exists(private_dir):
            shutil.rmtree(private_dir)


def grade_llm_code(train_code: dict, competition_id: str, language: str, 
                  mono_predict: bool, folds: int | None = None) -> dict:
    """
    Modern replacement for the old grade_llm_code function.
    
    Args:
        train_code: Dictionary containing train/predict functions
        competition_id: ID of the competition to grade
        language: Language of the submission
        mono_predict: Whether using monolithic predict mode
        folds: Optional override for number of folds
        
    Returns:
        Dictionary with grading results
    """
    # Use current directory as base path
    base_path = Path.cwd()
    grader = ModernGrader(base_path)
    
    # Grade the code
    return grader.grade_code(train_code, competition_id, language, mono_predict, folds)


# For direct execution compatibility
if __name__ == "__main__":
    # This allows the old script to still be called directly
    # You would need to adapt this based on how the old script was called
    pass
