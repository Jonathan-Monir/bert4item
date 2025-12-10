# solvers/bert4rec.py
import os
import pickle

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from models import BERT4Rec
from datasets import (
    MLMTrainDataset,
    MLMEvalDataset,
)

from .base import BaseSolver


__all__ = (
    'BERT4RecSolver',
)


class BERT4RecSolver(BaseSolver):

    def __init__(self, config: dict) -> None:
        C = config

        # before super
        with open(os.path.join(C['envs']['DATA_ROOT'], C['dataset'], 'iid2iindex.pkl'), 'rb') as fp:
            self.iid2iindex = pickle.load(fp)
        self.num_items = len(self.iid2iindex)

        super().__init__(config)

    # override
    def init_model(self) -> None:
        C = self.config
        cm = C['model']
        self.model = BERT4Rec(
            num_items=self.num_items,
            sequence_len=C['dataloader']['sequence_len'],
            max_num_segments=C['dataloader']['max_num_segments'],
            use_session_token=C['dataloader']['use_session_token'],
            num_layers=cm['num_layers'],
            hidden_dim=cm['hidden_dim'],
            temporal_dim=cm['temporal_dim'],
            num_heads=cm['num_heads'],
            dropout_prob=cm['dropout_prob'],
            random_seed=cm['random_seed']
        ).to(self.device)

    # override
    def init_criterion(self) -> None:
        self.ce_losser = CrossEntropyLoss(ignore_index=0)

    # override
    def init_dataloader(self) -> None:
        C = self.config
        name = C['dataset']
        print(f'jon {name}')
        sequence_len = C['dataloader']['sequence_len']
        max_num_segments = C['dataloader']['max_num_segments']
        use_session_token = C['dataloader']['use_session_token']
        self.train_dataloader = DataLoader(
            MLMTrainDataset(
                name=name,
                sequence_len=sequence_len,
                max_num_segments=max_num_segments,
                random_cut_prob=C['dataloader']['random_cut_prob'],
                mask_prob=C['dataloader']['mlm_mask_prob'],
                use_session_token=use_session_token,
                random_seed=C['dataloader']['random_seed']
            ),
            batch_size=C['train']['batch_size'],
            shuffle=True,
            num_workers=C['envs']['CPU_COUNT'],
            pin_memory=True,
            drop_last=False
        )
        self.valid_dataloader = DataLoader(
            MLMEvalDataset(
                name=name,
                target='valid',
                ns='random',
                sequence_len=sequence_len,
                max_num_segments=max_num_segments,
                use_session_token=use_session_token
            ),
            batch_size=C['train']['batch_size'],
            shuffle=False,
            num_workers=C['envs']['CPU_COUNT'],
            pin_memory=True,
            drop_last=False
        )
        self.test_ns_random_dataloader = DataLoader(
            MLMEvalDataset(
                name=name,
                target='test',
                ns='random',
                sequence_len=sequence_len,
                max_num_segments=max_num_segments,
                use_session_token=use_session_token
            ),
            batch_size=C['train']['batch_size'],
            shuffle=False,
            num_workers=C['envs']['CPU_COUNT'],
            pin_memory=True,
            drop_last=False,
        )
        self.test_ns_popular_dataloader = DataLoader(
            MLMEvalDataset(
                name=name,
                target='test',
                ns='popular',
                sequence_len=sequence_len,
                max_num_segments=max_num_segments,
                use_session_token=use_session_token
            ),
            batch_size=C['train']['batch_size'],
            shuffle=False,
            num_workers=C['envs']['CPU_COUNT'],
            pin_memory=True,
            drop_last=False,
        )
        self.test_ns_all_dataloader = DataLoader(
            MLMEvalDataset(
                name=name,
                target='test',
                ns='all',
                sequence_len=sequence_len,
                max_num_segments=max_num_segments,
                use_session_token=use_session_token
            ),
            batch_size=C['train']['batch_size'],
            shuffle=False,
            num_workers=C['envs']['CPU_COUNT'],
            pin_memory=True,
            drop_last=False
        )

    # override
    def calculate_loss(self, batch):

        # device
        tokens = batch['tokens'].to(self.device)  # b x L
        labels = batch['labels'].to(self.device)  # b x L

        # use segments
        if self.config['dataloader']['max_num_segments']:
            segments = batch['segments'].to(self.device)  # b x L
        else:
            segments = None

        # use stamps
        if self.config['model']['temporal_dim']:
            stamps = batch['stamps'].to(self.device)  # b x L
        else:
            stamps = None

        # forward
        logits = self.model(tokens, segments=segments, stamps=stamps)  # b x L x (V + 1)

        # loss
        logits = logits.view(-1, logits.size(-1))  # bL x (V + 1)
        labels = labels.view(-1)  # bL
        loss = self.ce_losser(logits, labels)

        return loss

    # override
    def calculate_rankers(self, batch):

        # device
        tokens = batch['tokens'].to(self.device)  # b x L
        cands = batch['cands'].to(self.device)  # b x C

        # use segments
        if self.config['dataloader']['max_num_segments']:
            segments = batch['segments'].to(self.device)  # b x L
        else:
            segments = None

        # use stamps
        if self.config['model']['temporal_dim']:
            stamps = batch['stamps'].to(self.device)  # b x L
        else:
            stamps = None

        # forward
        logits = self.model(tokens, segments=segments, stamps=stamps)  # b x L x (V + 1)

        # gather
        logits = logits[:, -1, :]  # b x (V + 1)
        scores = logits.gather(1, cands)  # b x C
        rankers = scores.argsort(dim=1, descending=True)

        return rankers

    def predict(self, user_history, current_cart_items=None, top_k=10):
        """
        Predict next items for recommendation at checkout.
        
        Args:
            user_history: List of (iid, sid, timestamp) tuples from user's history
            current_cart_items: List of (iid, timestamp) tuples currently in cart
            top_k: Number of recommendations to return
        
        Returns:
            List of (iid, score) tuples - recommended items with scores
        """
        import torch
        
        # Prepare the sequence
        self.set_model_mode('eval')
        
        # Combine history with current cart
        if current_cart_items:
            # Get the last session ID from history
            last_sid = user_history[-1][1] if user_history else 0
            # Add current cart items as a new session
            for iid, timestamp in current_cart_items:
                user_history.append((iid, last_sid + 1, timestamp))
        
        # Build tokens, segments, stamps (similar to MLMEvalDataset)
        tokens = []
        segments = []
        stamps = []
        current_segment = 0
        previous_sid = None
        start_stamp = user_history[0][2] if user_history else 0
        
        for iindex, current_sid, stamp in user_history:
            # New session detection
            if current_sid != previous_sid:
                current_segment += 1
                previous_sid = current_sid
                if self.config['dataloader']['use_session_token']:
                    tokens.append(self.model.module.session_token if hasattr(self.model, 'module') 
                                 else self.model.session_token)
                    segments.append(current_segment)
                    stamps.append(stamp)
            
            tokens.append(iindex)
            segments.append(current_segment)
            stamps.append(stamp)
        
        # Add mask token at the end for prediction
        mask_token = self.num_items + 1
        tokens.append(mask_token)
        segments.append(current_segment)
        stamps.append(stamps[-1] if stamps else 0)
        
        # Cut to sequence length
        sequence_len = self.config['dataloader']['sequence_len']
        tokens = tokens[-sequence_len:]
        segments = segments[-sequence_len:]
        stamps = stamps[-sequence_len:]
        
        # Rename segments
        num_sessions = max(segments)
        max_num_segments = self.config['dataloader']['max_num_segments']
        segments = [max(1, max_num_segments - num_sessions + seg) for seg in segments]
        
        # Pad
        padding_len = sequence_len - len(tokens)
        tokens = [0] * padding_len + tokens
        segments = [0] * padding_len + segments
        stamps = [start_stamp] * padding_len + stamps
        
        # Convert to tensors
        tokens_tensor = torch.LongTensor(tokens).unsqueeze(0).to(self.device)
        segments_tensor = torch.LongTensor(segments).unsqueeze(0).to(self.device) if max_num_segments else None
        stamps_tensor = torch.LongTensor(stamps).unsqueeze(0).to(self.device) if self.config['model']['temporal_dim'] else None
        
        # Get predictions
        with torch.no_grad():
            logits = self.model(tokens_tensor, segments=segments_tensor, stamps=stamps_tensor)
            scores = logits[0, -1, :].cpu()  # Get scores for the last position
        
        # Filter out already seen items
        seen_items = set([iindex for iindex, _, _ in user_history])
        
        # Get top-k recommendations (excluding padding, mask, session tokens and seen items)
        valid_scores = []
        for iindex in range(1, self.num_items + 1):
            if iindex not in seen_items:
                valid_scores.append((iindex, scores[iindex].item()))
        
        # Sort by score and return top-k
        valid_scores.sort(key=lambda x: x[1], reverse=True)
        return valid_scores[:top_k]
