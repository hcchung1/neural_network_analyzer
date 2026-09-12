"""Training results endpoints."""

import os
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()


class FolderScanRequest(BaseModel):
    folder_path: str


class TrainingFile(BaseModel):
    name: str
    path: str
    size: int  # File size in bytes


class FolderScanResponse(BaseModel):
    success: bool
    folder_path: str
    files: List[TrainingFile]
    error: Optional[str] = None


@router.post("/scan", response_model=FolderScanResponse)
async def scan_training_folder(request: FolderScanRequest) -> FolderScanResponse:
    """Scan a folder for training history and phase analysis CSV files."""
    folder_path = os.path.expanduser(request.folder_path)
    
    if not os.path.exists(folder_path):
        return FolderScanResponse(
            success=False,
            folder_path=folder_path,
            files=[],
            error=f"Folder not found: {folder_path}"
        )
    
    if not os.path.isdir(folder_path):
        return FolderScanResponse(
            success=False,
            folder_path=folder_path,
            files=[],
            error=f"Path is not a directory: {folder_path}"
        )
    
    files = []
    try:
        # Phase analysis outputs are often saved in nested folders such as
        # output/analysis/phase_analysis_YYYYMMDD_HHMMSS/, so scan recursively.
        for root, _, filenames in os.walk(folder_path):
            for filename in sorted(filenames):
                if filename.endswith('_history.csv') or filename.endswith('_phase_summary.csv') or filename.endswith('_turn_metrics.csv') or filename == 'comparison_table.csv':
                    file_path = os.path.join(root, filename)
                    if os.path.isfile(file_path):
                        files.append(TrainingFile(
                            name=filename,
                            path=file_path,
                            size=os.path.getsize(file_path)
                        ))
        files.sort(key=lambda f: f.path)
    except Exception as e:
        return FolderScanResponse(
            success=False,
            folder_path=folder_path,
            files=files,
            error=f"Error scanning folder: {str(e)}"
        )
    
    return FolderScanResponse(
        success=True,
        folder_path=folder_path,
        files=files,
        error=None
    )


class ReadFileRequest(BaseModel):
    file_path: str


class ReadFileResponse(BaseModel):
    success: bool
    content: str = ""
    error: Optional[str] = None


@router.post("/read_file", response_model=ReadFileResponse)
async def read_training_file(request: ReadFileRequest) -> ReadFileResponse:
    """Read a single training history file."""
    file_path = os.path.expanduser(request.file_path)
    
    if not os.path.exists(file_path):
        return ReadFileResponse(
            success=False,
            error=f"File not found: {file_path}"
        )
    
    if not os.path.isfile(file_path):
        return ReadFileResponse(
            success=False,
            error=f"Path is not a file: {file_path}"
        )
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return ReadFileResponse(success=True, content=content)
    except Exception as e:
        return ReadFileResponse(
            success=False,
            error=f"Error reading file: {str(e)}"
        )
