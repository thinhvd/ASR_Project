import os
import torch
import numpy as np
from nemo.collections.asr.data.audio_to_text import AudioToCharDataset

def to_numpy(tensor):
    """Converts a PyTorch tensor to a NumPy array."""
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def setup_transcribe_dataloader(cfg, vocabulary, manifest_path=None):
    """
    Sets up the data loader for transcription.
    
    Args:
        cfg (dict): Configuration dictionary containing temp_dir, batch_size, and paths2audio_files.
        vocabulary (list): The vocabulary labels from the model decoder.
        manifest_path (str, optional): Custom path to the manifest file.
        
    Returns:
        torch.utils.data.DataLoader: The PyTorch dataloader ready for inference.
    """
    if manifest_path is None:
        manifest_path = os.path.join(cfg['temp_dir'], 'manifest.json')
        
    config = {
        'manifest_filepath': manifest_path,
        'sample_rate': 16000,
        'labels': vocabulary,
        'batch_size': min(cfg['batch_size'], len(cfg['paths2audio_files'])),
        'trim_silence': True,
        'shuffle': False,
    }
    
    dataset = AudioToCharDataset(
        manifest_filepath=config['manifest_filepath'],
        labels=config['labels'],
        sample_rate=config['sample_rate'],
        int_values=config.get('int_values', False),
        augmentor=None,
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        max_utts=config.get('max_utts', 0),
        blank_index=config.get('blank_index', -1),
        unk_index=config.get('unk_index', -1),
        normalize=config.get('normalize_transcripts', False),
        trim=config.get('trim_silence', True),
        parser=config.get('parser', 'en'),
    )
    
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config['batch_size'],
        collate_fn=dataset.collate_fn,
        drop_last=config.get('drop_last', False),
        shuffle=False,
        num_workers=config.get('num_workers', 0),
        pin_memory=config.get('pin_memory', False),
    )