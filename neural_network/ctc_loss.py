import torch
import torch.nn as nn

class CTCLoss(nn.Module):
    def __init__(self):
        super(CTCLoss, self).__init__()
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        """
        log_probs: [seq_len, batch, num_classes]
        targets: [batch, max_target_len] 
        input_lengths: [batch]
        target_lengths: [batch]
        """
        # Flatten targets pour CTC
        targets_flat = []
        for i, length in enumerate(target_lengths):
            targets_flat.extend(targets[i][:length].tolist())
        targets_flat = torch.LongTensor(targets_flat)
        
        loss = self.ctc_loss(log_probs, targets_flat, input_lengths, target_lengths)
        return loss

# Création du vocabulaire
def create_vocabulary():
    chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?-'
    char_to_int = {char: i+1 for i, char in enumerate(chars)}
    char_to_int['<BLANK>'] = 0  # Token blank pour CTC
    int_to_char = {v: k for k, v in char_to_int.items()}
    int_to_char[0] = ''  # Blank token
    
    return char_to_int, int_to_char, len(char_to_int)

char_to_int, int_to_char, vocab_size = create_vocabulary()