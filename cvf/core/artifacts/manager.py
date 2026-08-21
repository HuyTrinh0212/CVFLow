"""Artifacts manager - handles saving/loading models, results, reports."""
from pathlib import Path
from typing import Any, Dict, Optional
import json
import pickle


class ArtifactsManager:
    """Manages CVF artifacts (models, results, reports)."""
    
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_model(self, model: Any, name: str, metadata: Optional[Dict] = None) -> Path:
        """Save model artifact."""
        path = self.output_dir / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        if metadata:
            meta_path = self.output_dir / f"{name}.meta.json"
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)
        return path
    
    def load_model(self, name: str) -> Any:
        """Load model artifact."""
        path = self.output_dir / f"{name}.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)
    
    def save_results(self, results: Dict, name: str) -> Path:
        """Save results as JSON."""
        path = self.output_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        return path
    
    def load_results(self, name: str) -> Dict:
        """Load results from JSON."""
        path = self.output_dir / f"{name}.json"
        with open(path, "r") as f:
            return json.load(f)
    
    def save_report(self, report: str, name: str) -> Path:
        """Save text report."""
        path = self.output_dir / f"{name}.txt"
        with open(path, "w") as f:
            f.write(report)
        return path
    
    def list_artifacts(self) -> list:
        """List all artifacts in output directory."""
        return list(self.output_dir.glob("*"))


# Global instance
artifacts_manager = ArtifactsManager()

__all__ = ["ArtifactsManager", "artifacts_manager"]