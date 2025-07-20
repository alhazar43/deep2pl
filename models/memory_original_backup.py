import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryHeadGroup(nn.Module):
    """
    Memory head group for DKVMN, handles read/write operations on memory matrix.
    """
    
    def __init__(self, memory_size, memory_state_dim, is_write=False):
        super(MemoryHeadGroup, self).__init__()
        self.memory_size = memory_size
        self.memory_state_dim = memory_state_dim
        self.is_write = is_write
        
        if self.is_write:
            self.erase_linear = nn.Linear(memory_state_dim, memory_state_dim, bias=True)
            self.add_linear = nn.Linear(memory_state_dim, memory_state_dim, bias=True)
            
            # Initialize weights
            nn.init.kaiming_normal_(self.erase_linear.weight)
            nn.init.kaiming_normal_(self.add_linear.weight)
            nn.init.constant_(self.erase_linear.bias, 0)
            nn.init.constant_(self.add_linear.bias, 0)
    
    def correlation_weight(self, embedded_query_vector, key_memory_matrix):
        """
        Calculate correlation weight between query and key memory.
        
        Args:
            embedded_query_vector: Shape (batch_size, key_dim)
            key_memory_matrix: Shape (memory_size, key_dim)
            
        Returns:
            correlation_weight: Shape (batch_size, memory_size)
        """
        # Compute similarity scores via matrix multiplication
        similarity_scores = torch.matmul(embedded_query_vector, key_memory_matrix.t())
        correlation_weight = F.softmax(similarity_scores, dim=1)
        return correlation_weight
    
    def read(self, value_memory_matrix, correlation_weight):
        """
        Read from value memory using correlation weights.
        
        Args:
            value_memory_matrix: Shape (batch_size, memory_size, value_dim)
            correlation_weight: Shape (batch_size, memory_size)
            
        Returns:
            read_content: Shape (batch_size, value_dim)
        """
        # Expand correlation weight for batch matrix multiplication
        correlation_weight_expanded = correlation_weight.unsqueeze(1)  # (batch_size, 1, memory_size)
        
        # Weighted sum over memory slots
        read_content = torch.bmm(correlation_weight_expanded, value_memory_matrix)
        read_content = read_content.squeeze(1)  # (batch_size, value_dim)
        
        return read_content
    
    def write(self, value_memory_matrix, correlation_weight, embedded_content_vector):
        """
        Write to value memory using erase-add mechanism.
        
        Args:
            value_memory_matrix: Shape (batch_size, memory_size, value_dim)
            correlation_weight: Shape (batch_size, memory_size)
            embedded_content_vector: Shape (batch_size, value_dim)
            
        Returns:
            new_value_memory_matrix: Shape (batch_size, memory_size, value_dim)
        """
        assert self.is_write, "This head group is not configured for writing"
        
        # Generate erase and add signals
        erase_signal = torch.sigmoid(self.erase_linear(embedded_content_vector))
        add_signal = torch.tanh(self.add_linear(embedded_content_vector))
        
        # Reshape for broadcasting
        erase_signal_expanded = erase_signal.unsqueeze(1)  # (batch_size, 1, value_dim)
        add_signal_expanded = add_signal.unsqueeze(1)      # (batch_size, 1, value_dim)
        correlation_weight_expanded = correlation_weight.unsqueeze(2)  # (batch_size, memory_size, 1)
        
        # Compute erase and add multiplications
        erase_mul = torch.bmm(correlation_weight_expanded, erase_signal_expanded)
        add_mul = torch.bmm(correlation_weight_expanded, add_signal_expanded)
        
        # Update memory: erase then add
        new_value_memory_matrix = value_memory_matrix * (1 - erase_mul) + add_mul
        
        return new_value_memory_matrix


class DKVMN(nn.Module):
    """
    Dynamic Key-Value Memory Network for knowledge tracing.
    """
    
    def __init__(self, memory_size, key_dim, value_dim, 
                 init_key_memory=None):
        super(DKVMN, self).__init__()
        self.memory_size = memory_size
        self.key_dim = key_dim
        self.value_dim = value_dim
        
        # Initialize key memory (static)
        if init_key_memory is not None:
            self.key_memory_matrix = nn.Parameter(init_key_memory)
        else:
            self.key_memory_matrix = nn.Parameter(
                torch.randn(memory_size, key_dim)
            )
            nn.init.kaiming_normal_(self.key_memory_matrix)
        
        # Memory head groups
        self.key_head = MemoryHeadGroup(memory_size, key_dim, is_write=False)
        self.value_head = MemoryHeadGroup(memory_size, value_dim, is_write=True)
        
        # Value memory will be initialized per batch
        self.value_memory_matrix = None
    
    def init_value_memory(self, batch_size, init_value_memory=None):
        """
        Initialize value memory for a batch.
        
        Args:
            batch_size: Batch size
            init_value_memory: Optional initial value memory
        """
        if init_value_memory is not None:
            if init_value_memory.dim() == 2:
                # Expand to batch dimension
                self.value_memory_matrix = init_value_memory.unsqueeze(0).expand(
                    batch_size, self.memory_size, self.value_dim
                ).contiguous()
            else:
                self.value_memory_matrix = init_value_memory
        else:
            self.value_memory_matrix = torch.zeros(
                batch_size, self.memory_size, self.value_dim
            )
    
    def attention(self, embedded_query_vector):
        """
        Compute attention weights for memory access.
        
        Args:
            embedded_query_vector: Shape (batch_size, key_dim)
            
        Returns:
            correlation_weight: Shape (batch_size, memory_size)
        """
        correlation_weight = self.key_head.correlation_weight(
            embedded_query_vector, self.key_memory_matrix
        )
        return correlation_weight
    
    def read(self, correlation_weight):
        """
        Read from value memory.
        
        Args:
            correlation_weight: Shape (batch_size, memory_size)
            
        Returns:
            read_content: Shape (batch_size, value_dim)
        """
        read_content = self.value_head.read(self.value_memory_matrix, correlation_weight)
        return read_content
    
    def write(self, correlation_weight, embedded_content_vector):
        """
        Write to value memory.
        
        Args:
            correlation_weight: Shape (batch_size, memory_size)
            embedded_content_vector: Shape (batch_size, value_dim)
            
        Returns:
            new_value_memory_matrix: Shape (batch_size, memory_size, value_dim)
        """
        self.value_memory_matrix = self.value_head.write(
            self.value_memory_matrix, correlation_weight, embedded_content_vector
        )
        return self.value_memory_matrix