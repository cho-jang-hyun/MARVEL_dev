import torch
import torch.nn as nn
import math
import numpy as np
from parameter import BUDGET_FEATURE_DIM

class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(SingleHeadAttention, self).__init__()
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = embedding_dim
        self.key_dim = self.value_dim
        self.tanh_clipping = 10
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, k, mask=None):

        n_batch, n_key, n_dim = k.size()
        n_query = q.size(1)

        k_flat = k.reshape(-1, n_dim)
        q_flat = q.reshape(-1, n_dim)

        shape_k = (n_batch, n_key, -1)
        shape_q = (n_batch, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(shape_q)
        K = torch.matmul(k_flat, self.w_key).view(shape_k)

        U = self.norm_factor * torch.matmul(Q, K.transpose(1, 2))
        U = self.tanh_clipping * torch.tanh(U)

        if mask is not None:
            U = U.masked_fill(mask == 1, -1e8)
        attention = torch.log_softmax(U, dim=-1)  

        return attention

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads=8, gated_attention=True):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.gated_attention = gated_attention
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = self.embedding_dim // self.n_heads
        self.key_dim = self.value_dim
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_value = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.value_dim))
        if gated_attention:
            # Paper-style headwise gating: query-dependent sigmoid gate after attention output.
            self.w_gate = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, 1))
        self.w_out = nn.Parameter(torch.Tensor(self.n_heads, self.value_dim, self.embedding_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, k=None, v=None, key_padding_mask=None, attn_mask=None):
        if k is None:
            k = q
        if v is None:
            v = q

        n_batch, n_key, n_dim = k.size()
        n_query = q.size(1)
        n_value = v.size(1)

        k_flat = k.contiguous().view(-1, n_dim)
        v_flat = v.contiguous().view(-1, n_dim)
        q_flat = q.contiguous().view(-1, n_dim)
        shape_v = (self.n_heads, n_batch, n_value, -1)
        shape_k = (self.n_heads, n_batch, n_key, -1)
        shape_q = (self.n_heads, n_batch, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(shape_q)
        K = torch.matmul(k_flat, self.w_key).view(shape_k) 
        V = torch.matmul(v_flat, self.w_value).view(shape_v)  
        if self.gated_attention:
            G = torch.matmul(q_flat, self.w_gate).view(self.n_heads, n_batch, n_query, 1)

        U = self.norm_factor * torch.matmul(Q, K.transpose(2, 3)) 

        if attn_mask is not None:
            attn_mask = attn_mask.view(1, n_batch, n_query, n_key).expand_as(U)

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.repeat(1, n_query, 1)
            key_padding_mask = key_padding_mask.view(1, n_batch, n_query, n_key).expand_as(U) 

        if attn_mask is not None and key_padding_mask is not None:
            mask = (attn_mask + key_padding_mask)
        elif attn_mask is not None:
            mask = attn_mask
        elif key_padding_mask is not None:
            mask = key_padding_mask
        else:
            mask = None

        if mask is not None:
            U = U.masked_fill(mask > 0, -1e8)

        attention = torch.softmax(U, dim=-1)  
        heads = torch.matmul(attention, V)  
        if self.gated_attention:
            heads = heads * torch.sigmoid(G)
        out = torch.mm(
            heads.permute(1, 2, 0, 3).reshape(-1, self.n_heads * self.value_dim),
            self.w_out.view(-1, self.embedding_dim)
        ).view(-1, n_query, self.embedding_dim)

        return out, attention  


class Normalization(nn.Module):
    def __init__(self, embedding_dim):
        super(Normalization, self).__init__()
        self.normalizer = nn.LayerNorm(embedding_dim)

    def forward(self, input):
        return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head, gated_attention=True):
        super(EncoderLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head, gated_attention=gated_attention)
        self.normalization1 = Normalization(embedding_dim)
        self.feedForward = nn.Sequential(nn.Linear(embedding_dim, 512), nn.ReLU(inplace=True),
                                         nn.Linear(512, embedding_dim))
        self.normalization2 = Normalization(embedding_dim)

    def forward(self, src, key_padding_mask=None, attn_mask=None):
        h0 = src
        h = self.normalization1(src)
        h, _ = self.multiHeadAttention(q=h, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        h = h + h0
        h1 = h
        h = self.normalization2(h)
        h = self.feedForward(h)
        h2 = h + h1
        return h2


class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head, gated_attention=True):
        super(DecoderLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head, gated_attention=gated_attention)
        self.normalization1 = Normalization(embedding_dim)
        self.feedForward = nn.Sequential(nn.Linear(embedding_dim, 512),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(512, embedding_dim))
        self.normalization2 = Normalization(embedding_dim)

    def forward(self, tgt, memory, key_padding_mask=None, attn_mask=None):
        h0 = tgt
        tgt = self.normalization1(tgt)
        memory = self.normalization1(memory)
        h, w = self.multiHeadAttention(q=tgt, k=memory, v=memory, key_padding_mask=key_padding_mask,
                                       attn_mask=attn_mask)
        h = h + h0
        h1 = h
        h = self.normalization2(h)
        h = self.feedForward(h)
        h2 = h + h1
        return h2, w


class Encoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=8, n_layer=1, gated_attention=True):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(EncoderLayer(embedding_dim, n_head, gated_attention) for i in range(n_layer))

    def forward(self, src, key_padding_mask=None, attn_mask=None):
        for layer in self.layers:
            src = layer(src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return src


class Decoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=8, n_layer=1, gated_attention=True):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(embedding_dim, n_head, gated_attention) for i in range(n_layer)])

    def forward(self, tgt, memory, key_padding_mask=None, attn_mask=None):
        for layer in self.layers:
            tgt, w = layer(tgt, memory, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return tgt, w


class PositionalEncoding(nn.Module):
    """Positional encoding for temporal sequences."""
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]


class TrajectoryEncoder(nn.Module):
    """
    Encodes detected robot trajectories using temporal transformer.

    Input:
        detected_trajectories: [batch, max_detected_agents, seq_len, feature_dim]
        trajectory_mask: [batch, max_detected_agents] - True for padded agents

    Output:
        trajectory_tokens: [batch, max_detected_agents * seq_len, trajectory_embedding_dim]
        trajectory_token_mask: [batch, 1, max_detected_agents * seq_len], True for invalid tokens
    """
    def __init__(self, feature_dim, trajectory_embedding_dim, seq_len, n_head=4, n_layer=2, gated_attention=True):
        super(TrajectoryEncoder, self).__init__()
        self.feature_dim = feature_dim
        self.trajectory_embedding_dim = trajectory_embedding_dim
        self.seq_len = seq_len
        self.gated_attention = gated_attention

        # Project trajectory features to embedding dimension
        self.feature_projection = nn.Linear(feature_dim, trajectory_embedding_dim)

        # Positional encoding for temporal information
        self.positional_encoding = PositionalEncoding(trajectory_embedding_dim, max_len=seq_len)

        # Temporal transformer encoder for each agent's trajectory
        self.temporal_encoder = Encoder(embedding_dim=trajectory_embedding_dim, n_head=n_head, n_layer=n_layer, gated_attention=gated_attention)

        # Output projection for each valid agent-time token.
        self.output_layer = nn.Sequential(
            nn.Linear(trajectory_embedding_dim, trajectory_embedding_dim),
            nn.ReLU(),
            nn.Linear(trajectory_embedding_dim, trajectory_embedding_dim)
        )
        # Kept for checkpoint compatibility with earlier pooled-embedding models.
        self.null_trajectory_embedding = nn.Parameter(torch.zeros(trajectory_embedding_dim))
        self.latest_debug = {}

    def forward(self, detected_trajectories, trajectory_mask, trajectory_node_indices=None):
        """
        Args:
            detected_trajectories: [batch, max_detected_agents, seq_len, feature_dim]
            trajectory_mask: [batch, max_detected_agents] - True for padded agents

        Returns:
            trajectory_tokens: [batch, max_detected_agents * seq_len, trajectory_embedding_dim]
            trajectory_token_mask: [batch, 1, max_detected_agents * seq_len]
        """
        batch_size, max_agents, seq_len, feature_dim = detected_trajectories.shape

        # Build timestep validity mask (True means invalid/padded timestep).
        # We should not use trajectory_node_indices here because an agent's valid coordinate 
        # might not have a corresponding node in the current agent's local graph yet.
        timestep_mask = detected_trajectories.abs().sum(dim=-1) == 0
        timestep_mask = timestep_mask | trajectory_mask.unsqueeze(-1)

        # Reshape to process all agents together: [batch * max_agents, seq_len, feature_dim]
        trajectories_flat = detected_trajectories.reshape(batch_size * max_agents, seq_len, feature_dim)

        # Project features to embedding dimension
        # [batch * max_agents, seq_len, trajectory_embedding_dim]
        embedded = self.feature_projection(trajectories_flat)

        # Add positional encoding
        embedded = self.positional_encoding(embedded)

        # Apply temporal transformer encoder with timestep mask.
        temporal_padding_mask = timestep_mask.reshape(batch_size * max_agents, 1, seq_len)
        temporal_features = self.temporal_encoder(embedded, key_padding_mask=temporal_padding_mask)
        temporal_features = temporal_features.reshape(batch_size, max_agents, seq_len, self.trajectory_embedding_dim)

        temporal_features = self.output_layer(temporal_features)

        # Keep all agent-time tokens. Invalid timesteps are zeroed and masked
        # later when action tokens cross-attend to trajectory tokens.
        valid_timestep = ~timestep_mask
        temporal_features = temporal_features * valid_timestep.unsqueeze(-1).float()
        trajectory_tokens = temporal_features.reshape(
            batch_size, max_agents * seq_len, self.trajectory_embedding_dim
        )
        trajectory_token_mask = timestep_mask.reshape(batch_size, 1, max_agents * seq_len)

        # Agent is usable only if not padded and it has at least one valid timestep.
        agent_valid = (~trajectory_mask) & valid_timestep.any(dim=2)

        # Debug metrics for TensorBoard.
        valid_token_flat = valid_timestep.reshape(batch_size, max_agents * seq_len)
        if valid_token_flat.any():
            token_norm = trajectory_tokens[valid_token_flat].norm(dim=-1).mean()
        else:
            token_norm = trajectory_tokens.new_tensor(0.0)
        self.latest_debug = {
            "detected_agents_mean": (~trajectory_mask).float().sum(dim=1).mean().detach(),
            "usable_agents_mean": agent_valid.float().sum(dim=1).mean().detach(),
            "valid_timestep_ratio": valid_timestep.float().mean().detach(),
            "embedding_norm": token_norm.detach(),
            "agent_attention_entropy": trajectory_tokens.new_tensor(0.0).detach(),
        }

        return trajectory_tokens, trajectory_token_mask


class PolicyNet(nn.Module):
    def __init__(self, node_dim, embedding_dim, num_angles_bin, use_trajectory=True, gated_attention=True,
                 budget_feature_dim=BUDGET_FEATURE_DIM):
        super(PolicyNet, self).__init__()

        self.use_trajectory = use_trajectory
        self.gated_attention = gated_attention

        # Graph Encoder
        self.initial_embedding = nn.Linear(node_dim, embedding_dim)
        self.encoder = Encoder(embedding_dim=embedding_dim, n_head=4, n_layer=6, gated_attention=gated_attention)

        # Local frontiers distribution encoder
        self.frontiers_embedding =  nn.Conv1d(num_angles_bin, embedding_dim, kernel_size=3, padding=1)
        self.node_frontiers_embedding = nn.Linear(embedding_dim * 2, embedding_dim)

        # Trajectory encoder
        if use_trajectory:
            from parameter import TRAJECTORY_FEATURE_DIM, TRAJECTORY_EMBEDDING_DIM, TRAJECTORY_HISTORY_LENGTH
            self.trajectory_encoder = TrajectoryEncoder(
                feature_dim=TRAJECTORY_FEATURE_DIM,
                trajectory_embedding_dim=TRAJECTORY_EMBEDDING_DIM,
                seq_len=TRAJECTORY_HISTORY_LENGTH,
                n_head=4,
                n_layer=2,
                gated_attention=gated_attention
            )
            self.trajectory_token_projection = nn.Linear(TRAJECTORY_EMBEDDING_DIM, embedding_dim)
            self.trajectory_node_ffn = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim)
            )
            self.trajectory_token_fusion = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim)
            )
            self.trajectory_action_attention = MultiHeadAttention(embedding_dim, n_heads=4, gated_attention=gated_attention)
            self.trajectory_action_norm = nn.LayerNorm(embedding_dim)

        # Decoder
        self.decoder = Decoder(embedding_dim=embedding_dim, n_head=4, n_layer=1, gated_attention=gated_attention)
        self.current_embedding = nn.Linear(embedding_dim * 3, embedding_dim)
        self.budget_state_embedding = nn.Linear(budget_feature_dim, embedding_dim)

        # Heading layer
        self.best_headings_embedding = nn.Linear(num_angles_bin, embedding_dim)
        self.visited_headings_embedding = nn.Linear(num_angles_bin, embedding_dim)
        self.neighboring_node_embedding = nn.Linear(embedding_dim * 3, embedding_dim)

        # pointer
        self.pointer = SingleHeadAttention(embedding_dim)

    def encode_graph(self, node_inputs, node_padding_mask, edge_mask, frontier_distribution):
        node_feature = self.initial_embedding(node_inputs)
        enhanced_node_feature = self.encoder(src=node_feature,
                                                         key_padding_mask=node_padding_mask,
                                                         attn_mask=edge_mask)

        frontier_distribution = frontier_distribution.permute(0, 2, 1)
        frontiers_feature = self.frontiers_embedding(frontier_distribution)
        frontiers_feature = frontiers_feature.permute(0, 2, 1)

        enhanced_node_feature = self.node_frontiers_embedding(torch.cat((enhanced_node_feature, frontiers_feature), dim=-1))

        return enhanced_node_feature

    def encode_trajectory_nodes(self, enhanced_node_feature, trajectory_node_indices, trajectory_mask):
        """
        Extract and encode node embeddings for detected trajectory positions.

        Args:
            enhanced_node_feature: [batch, num_nodes, embedding_dim]
            trajectory_node_indices: [batch, max_agents, seq_len], -1 for invalid
            trajectory_mask: [batch, max_agents], True for padded agents

        Returns:
            trajectory_node_features: [batch, max_agents, seq_len, embedding_dim]
        """
        batch_size, num_nodes, embedding_dim = enhanced_node_feature.shape
        _, max_agents, seq_len = trajectory_node_indices.shape

        # Initialize output tensor
        trajectory_node_features = torch.zeros(
            batch_size, max_agents, seq_len, embedding_dim,
            dtype=enhanced_node_feature.dtype,
            device=enhanced_node_feature.device
        )

        # Process each batch
        for b in range(batch_size):
            for a in range(max_agents):
                # Skip if this agent is padded
                if trajectory_mask[b, a]:
                    continue

                for t in range(seq_len):
                    node_idx = trajectory_node_indices[b, a, t].item()

                    # Skip if invalid node index
                    if node_idx < 0 or node_idx >= num_nodes:
                        continue

                    # Extract node embedding
                    trajectory_node_features[b, a, t] = enhanced_node_feature[b, node_idx]

        # Apply FFN projection
        # Reshape to [batch * max_agents * seq_len, embedding_dim]
        valid_node_mask = (trajectory_node_indices >= 0) & (~trajectory_mask.unsqueeze(-1))
        traj_flat = trajectory_node_features.reshape(-1, embedding_dim)
        traj_projected = self.trajectory_node_ffn(traj_flat)
        # Reshape back to [batch, max_agents, seq_len, embedding_dim]
        trajectory_node_features = traj_projected.reshape(batch_size, max_agents, seq_len, embedding_dim)
        trajectory_node_features = trajectory_node_features * valid_node_mask.unsqueeze(-1).float()

        return trajectory_node_features

    def encode_trajectory_context(self, enhanced_node_feature, detected_trajectories, trajectory_mask, trajectory_node_indices):
        if not self.use_trajectory or detected_trajectories is None or trajectory_mask is None:
            return None, None

        trajectory_tokens, trajectory_token_mask = self.trajectory_encoder(
            detected_trajectories,
            trajectory_mask,
            trajectory_node_indices,
        )
        trajectory_tokens = self.trajectory_token_projection(trajectory_tokens)

        if trajectory_node_indices is not None:
            trajectory_node_features = self.encode_trajectory_nodes(
                enhanced_node_feature,
                trajectory_node_indices,
                trajectory_mask,
            )
            batch_size, max_agents, seq_len, embedding_dim = trajectory_node_features.shape
            trajectory_node_features = trajectory_node_features.reshape(batch_size, max_agents * seq_len, embedding_dim)
            trajectory_tokens = self.trajectory_token_fusion(
                torch.cat((trajectory_tokens, trajectory_node_features), dim=-1)
            )

        return trajectory_tokens, trajectory_token_mask

    def apply_trajectory_action_attention(self, action_features, trajectory_tokens, trajectory_token_mask):
        if trajectory_tokens is None or trajectory_token_mask is None:
            return action_features

        has_valid_tokens = (~trajectory_token_mask.squeeze(1).bool()).any(dim=1)
        attended_features, attention = self.trajectory_action_attention(
            q=action_features,
            k=trajectory_tokens,
            v=trajectory_tokens,
            key_padding_mask=trajectory_token_mask,
        )
        if hasattr(self, 'trajectory_encoder'):
            if has_valid_tokens.any():
                attention_probs = attention.clamp_min(1e-8)
                attention_entropy = -(attention_probs * attention_probs.log()).sum(dim=-1)
                self.trajectory_encoder.latest_debug["agent_attention_entropy"] = (
                    attention_entropy[:, has_valid_tokens].mean().detach()
                )
            else:
                self.trajectory_encoder.latest_debug["agent_attention_entropy"] = (
                    action_features.new_tensor(0.0).detach()
                )
        attended_features = attended_features * has_valid_tokens.view(-1, 1, 1).float()
        return self.trajectory_action_norm(action_features + attended_features)

    def decode_state(self, enhanced_node_feature, current_index, node_padding_mask):
        embedding_dim = enhanced_node_feature.size()[2]
        current_node_feature = torch.gather(enhanced_node_feature, 1,
                                                  current_index.repeat(1, 1, embedding_dim))
        enhanced_current_node_feature, _ = self.decoder(current_node_feature,
                                                                    enhanced_node_feature,
                                                                    node_padding_mask)
        return current_node_feature, enhanced_current_node_feature

    def output_policy(self, current_node_feature, enhanced_current_node_feature,
                      enhanced_node_feature, current_edge, edge_padding_mask, headings_visited, neighbor_best_headings,
                      budget_state, trajectory_tokens=None, trajectory_token_mask=None):

        embedding_dim = enhanced_node_feature.size()[2]
        batch_size = enhanced_node_feature.size()[0]
        num_best_headings = neighbor_best_headings.size()[2]
        embedded_budget_state = self.budget_state_embedding(budget_state.unsqueeze(1))
        current_state_feature = self.current_embedding(torch.cat((enhanced_current_node_feature,
                                                                  current_node_feature,
                                                                  embedded_budget_state), dim=-1))

        neighboring_feature = torch.gather(enhanced_node_feature, 1,
                                           current_edge.repeat(1, 1, embedding_dim))

        enhanced_neighbor_best_headings = self.best_headings_embedding(neighbor_best_headings)
        all_headings_visited = self.visited_headings_embedding(headings_visited)
        all_neighbor_headings_visited = torch.gather(all_headings_visited, 1,
                                           current_edge.repeat(1, 1, embedding_dim))

        neighboring_nodes_feature = neighboring_feature.unsqueeze(2).repeat(1, 1, num_best_headings, 1)
        neighbor_headings_visited = all_neighbor_headings_visited.unsqueeze(2).repeat(1, 1, num_best_headings, 1)

        enhanced_neighbor_features = self.neighboring_node_embedding(torch.cat((neighboring_nodes_feature, neighbor_headings_visited,
                                                                                enhanced_neighbor_best_headings), dim=-1)).reshape(batch_size, -1, embedding_dim)
        enhanced_neighbor_features = self.apply_trajectory_action_attention(
            enhanced_neighbor_features,
            trajectory_tokens,
            trajectory_token_mask,
        )

        current_mask = edge_padding_mask.unsqueeze(-1).repeat(1, 1, 1, num_best_headings).reshape(batch_size, 1, -1)
        logp = self.pointer(current_state_feature, enhanced_neighbor_features, current_mask)
        logp = logp.squeeze(1)

        return logp

    def forward(self, node_inputs, node_padding_mask, edge_mask, current_index,
                current_edge, edge_padding_mask, frontier_distribution, headings_visited, neighbor_best_headings,
                budget_state,
                detected_trajectories=None, trajectory_mask=None, trajectory_node_indices=None):
        enhanced_node_feature = self.encode_graph(node_inputs, node_padding_mask, edge_mask, frontier_distribution)

        trajectory_tokens, trajectory_token_mask = self.encode_trajectory_context(
            enhanced_node_feature,
            detected_trajectories,
            trajectory_mask,
            trajectory_node_indices,
        )

        current_node_feature, enhanced_current_node_feature = self.decode_state(
            enhanced_node_feature, current_index, node_padding_mask)
        logp = self.output_policy(current_node_feature, enhanced_current_node_feature,
                                  enhanced_node_feature, current_edge, edge_padding_mask, headings_visited, neighbor_best_headings,
                                  budget_state, trajectory_tokens, trajectory_token_mask)

        return logp


class QNet(nn.Module):
    def __init__(self, node_dim, embedding_dim, num_angles_bin, train_algo, use_trajectory=True, gated_attention=True,
                 budget_feature_dim=BUDGET_FEATURE_DIM):
        super(QNet, self).__init__()

        self.use_trajectory = use_trajectory
        self.gated_attention = gated_attention

        # Graph encoder
        self.encoder = Encoder(embedding_dim=embedding_dim, n_head=4, n_layer=6, gated_attention=gated_attention)

        # Local frontiers distribution encoder
        self.frontiers_embedding = nn.Conv1d(num_angles_bin, embedding_dim, kernel_size=3, padding=1)
        self.node_frontiers_embedding = nn.Linear(embedding_dim * 2, embedding_dim)

        # Trajectory encoder
        if use_trajectory:
            from parameter import TRAJECTORY_FEATURE_DIM, TRAJECTORY_EMBEDDING_DIM, TRAJECTORY_HISTORY_LENGTH
            self.trajectory_encoder = TrajectoryEncoder(
                feature_dim=TRAJECTORY_FEATURE_DIM,
                trajectory_embedding_dim=TRAJECTORY_EMBEDDING_DIM,
                seq_len=TRAJECTORY_HISTORY_LENGTH,
                n_head=4,
                n_layer=2,
                gated_attention=gated_attention
            )
            self.trajectory_token_projection = nn.Linear(TRAJECTORY_EMBEDDING_DIM, embedding_dim)
            self.trajectory_node_ffn = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim)
            )
            self.trajectory_token_fusion = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim)
            )
            self.trajectory_action_attention = MultiHeadAttention(embedding_dim, n_heads=4, gated_attention=gated_attention)
            self.trajectory_action_norm = nn.LayerNorm(embedding_dim)

        # Decoder
        self.decoder = Decoder(embedding_dim=embedding_dim, n_head=4, n_layer=1, gated_attention=gated_attention)
        self.current_embedding = nn.Linear(embedding_dim * 3, embedding_dim)
        self.budget_state_embedding = nn.Linear(budget_feature_dim, embedding_dim)

        # Heading layer
        self.best_headings_embedding = nn.Linear(num_angles_bin, embedding_dim)
        self.visited_headings_embedding = nn.Linear(num_angles_bin, embedding_dim)
        self.neighboring_node_embedding = nn.Linear(embedding_dim * 3, embedding_dim)

        # Agent decoder
        if train_algo in (2, 3, 4, 5):
            self.initial_embedding = nn.Linear(node_dim + 1, embedding_dim)
        else:
            # Graph embedding
            self.initial_embedding = nn.Linear(node_dim, embedding_dim)

        if train_algo in (1, 3, 5):
            self.agent_decoder = Decoder(embedding_dim=embedding_dim, n_head=4, n_layer=1, gated_attention=gated_attention)
            self.all_agent_embedding = nn.Linear(embedding_dim * 2, embedding_dim)

            self.q_values_layer = nn.Linear(embedding_dim * 3, 1)
        else:
            self.q_values_layer = nn.Linear(embedding_dim * 2, 1)


    def encode_graph(self, node_inputs, node_padding_mask, edge_mask, frontier_distribution):
        node_feature = self.initial_embedding(node_inputs)
        enhanced_node_feature = self.encoder(src=node_feature,
                                                         key_padding_mask=node_padding_mask,
                                                         attn_mask=edge_mask)
        
        frontier_distribution = frontier_distribution.permute(0, 2, 1)
        frontiers_feature = self.frontiers_embedding(frontier_distribution)
        frontiers_feature = frontiers_feature.permute(0, 2, 1)

        enhanced_node_feature = self.node_frontiers_embedding(torch.cat((enhanced_node_feature, frontiers_feature), dim=-1))

        return enhanced_node_feature

    def encode_trajectory_nodes(self, enhanced_node_feature, trajectory_node_indices, trajectory_mask):
        """
        Extract and encode node embeddings for detected trajectory positions.

        Args:
            enhanced_node_feature: [batch, num_nodes, embedding_dim]
            trajectory_node_indices: [batch, max_agents, seq_len], -1 for invalid
            trajectory_mask: [batch, max_agents], True for padded agents

        Returns:
            trajectory_node_features: [batch, max_agents, seq_len, embedding_dim]
        """
        batch_size, num_nodes, embedding_dim = enhanced_node_feature.shape
        _, max_agents, seq_len = trajectory_node_indices.shape

        # Initialize output tensor
        trajectory_node_features = torch.zeros(
            batch_size, max_agents, seq_len, embedding_dim,
            dtype=enhanced_node_feature.dtype,
            device=enhanced_node_feature.device
        )

        # Process each batch
        for b in range(batch_size):
            for a in range(max_agents):
                # Skip if this agent is padded
                if trajectory_mask[b, a]:
                    continue

                for t in range(seq_len):
                    node_idx = trajectory_node_indices[b, a, t].item()

                    # Skip if invalid node index
                    if node_idx < 0 or node_idx >= num_nodes:
                        continue

                    # Extract node embedding
                    trajectory_node_features[b, a, t] = enhanced_node_feature[b, node_idx]

        # Apply FFN projection
        # Reshape to [batch * max_agents * seq_len, embedding_dim]
        valid_node_mask = (trajectory_node_indices >= 0) & (~trajectory_mask.unsqueeze(-1))
        traj_flat = trajectory_node_features.reshape(-1, embedding_dim)
        traj_projected = self.trajectory_node_ffn(traj_flat)
        # Reshape back to [batch, max_agents, seq_len, embedding_dim]
        trajectory_node_features = traj_projected.reshape(batch_size, max_agents, seq_len, embedding_dim)
        trajectory_node_features = trajectory_node_features * valid_node_mask.unsqueeze(-1).float()

        return trajectory_node_features

    def encode_trajectory_context(self, enhanced_node_feature, detected_trajectories, trajectory_mask, trajectory_node_indices):
        if not self.use_trajectory or detected_trajectories is None or trajectory_mask is None:
            return None, None

        trajectory_tokens, trajectory_token_mask = self.trajectory_encoder(
            detected_trajectories,
            trajectory_mask,
            trajectory_node_indices,
        )
        trajectory_tokens = self.trajectory_token_projection(trajectory_tokens)

        if trajectory_node_indices is not None:
            trajectory_node_features = self.encode_trajectory_nodes(
                enhanced_node_feature,
                trajectory_node_indices,
                trajectory_mask,
            )
            batch_size, max_agents, seq_len, embedding_dim = trajectory_node_features.shape
            trajectory_node_features = trajectory_node_features.reshape(batch_size, max_agents * seq_len, embedding_dim)
            trajectory_tokens = self.trajectory_token_fusion(
                torch.cat((trajectory_tokens, trajectory_node_features), dim=-1)
            )

        return trajectory_tokens, trajectory_token_mask

    def apply_trajectory_action_attention(self, action_features, trajectory_tokens, trajectory_token_mask):
        if trajectory_tokens is None or trajectory_token_mask is None:
            return action_features

        has_valid_tokens = (~trajectory_token_mask.squeeze(1).bool()).any(dim=1)
        attended_features, attention = self.trajectory_action_attention(
            q=action_features,
            k=trajectory_tokens,
            v=trajectory_tokens,
            key_padding_mask=trajectory_token_mask,
        )
        if hasattr(self, 'trajectory_encoder'):
            if has_valid_tokens.any():
                attention_probs = attention.clamp_min(1e-8)
                attention_entropy = -(attention_probs * attention_probs.log()).sum(dim=-1)
                self.trajectory_encoder.latest_debug["agent_attention_entropy"] = (
                    attention_entropy[:, has_valid_tokens].mean().detach()
                )
            else:
                self.trajectory_encoder.latest_debug["agent_attention_entropy"] = (
                    action_features.new_tensor(0.0).detach()
                )
        attended_features = attended_features * has_valid_tokens.view(-1, 1, 1).float()
        return self.trajectory_action_norm(action_features + attended_features)

    def decode_state(self, enhanced_node_feature, current_index, node_padding_mask):
        embedding_dim = enhanced_node_feature.size()[2]
        current_node_feature = torch.gather(enhanced_node_feature, 1,
                                                  current_index.repeat(1, 1, embedding_dim))
        enhanced_current_node_feature, _ = self.decoder(current_node_feature,
                                                                    enhanced_node_feature,
                                                                    node_padding_mask)
        return current_node_feature, enhanced_current_node_feature

    def output_q(self, current_node_feature, enhanced_current_node_feature, enhanced_node_feature,
                 current_edge, edge_padding_mask, headings_visited, neighbor_best_headings, budget_state,
                 current_index, all_agent_indices, all_agent_next_indices,
                 trajectory_tokens=None, trajectory_token_mask=None):
        embedding_dim = enhanced_node_feature.size()[2]
        num_best_headings = neighbor_best_headings.size()[2]
        batch_size = enhanced_node_feature.size()[0]
        embedded_budget_state = self.budget_state_embedding(budget_state.unsqueeze(1))
        current_state_feature = self.current_embedding(torch.cat((enhanced_current_node_feature,
                                                                  current_node_feature,
                                                                  embedded_budget_state), dim=-1))

        neighboring_feature = torch.gather(enhanced_node_feature, 1,
                                           current_edge.repeat(1, 1, embedding_dim))

        enhanced_neighbor_best_headings = self.best_headings_embedding(neighbor_best_headings)
        all_headings_visited = self.visited_headings_embedding(headings_visited)
        all_neighbor_headings_visited = torch.gather(all_headings_visited, 1,
                                           current_edge.repeat(1, 1, embedding_dim))

        neighboring_nodes_feature = neighboring_feature.unsqueeze(2).repeat(1, 1, num_best_headings, 1)
        neighbor_headings_visited = all_neighbor_headings_visited.unsqueeze(2).repeat(1, 1, num_best_headings, 1)

        enhanced_neighbor_features = self.neighboring_node_embedding(torch.cat((neighboring_nodes_feature, neighbor_headings_visited,
                                                                                enhanced_neighbor_best_headings), dim=-1)).reshape(batch_size, -1, embedding_dim)
        enhanced_neighbor_features = self.apply_trajectory_action_attention(
            enhanced_neighbor_features,
            trajectory_tokens,
            trajectory_token_mask,
        )
        
        if all_agent_indices != None:
            all_agent_node_feature = torch.gather(enhanced_node_feature, 1,
                                                all_agent_indices.repeat(1, 1, embedding_dim))
            all_agent_selected_neighboring_feature = torch.gather(enhanced_node_feature, 1,
                                                                all_agent_next_indices.repeat(1, 1, embedding_dim))

            all_agent_action_features = torch.cat((all_agent_node_feature, all_agent_selected_neighboring_feature), dim=-1)
            all_agent_action_features = self.all_agent_embedding(all_agent_action_features)

            agent_mask = all_agent_indices == current_index
            global_state_action_feature, _ = self.agent_decoder(current_state_feature, all_agent_action_features, agent_mask)
            action_features = torch.cat((current_state_feature.repeat(1, enhanced_neighbor_features.size()[1], 1), enhanced_neighbor_features, global_state_action_feature.repeat(1, enhanced_neighbor_features.size()[1], 1)), dim=-1)
            q_values = self.q_values_layer(action_features)
        else:
            action_features = torch.cat((current_state_feature.repeat(1, enhanced_neighbor_features.size()[1], 1), enhanced_neighbor_features), dim=-1)
            q_values = self.q_values_layer(action_features)
        return q_values

    def forward(self, node_inputs, node_padding_mask, edge_mask, current_index,
                current_edge, edge_padding_mask, frontier_distribution, headings_visited, neighbor_best_headings,
                budget_state,
                all_agent_indices=None, all_agent_next_indices=None, detected_trajectories=None, trajectory_mask=None, trajectory_node_indices=None):
        enhanced_node_feature = self.encode_graph(node_inputs, node_padding_mask, edge_mask, frontier_distribution)

        trajectory_tokens, trajectory_token_mask = self.encode_trajectory_context(
            enhanced_node_feature,
            detected_trajectories,
            trajectory_mask,
            trajectory_node_indices,
        )

        current_node_feature, enhanced_current_node_feature = self.decode_state(
            enhanced_node_feature, current_index, node_padding_mask)
        q_values = self.output_q(current_node_feature, enhanced_current_node_feature,
                                 enhanced_node_feature, current_edge, edge_padding_mask, headings_visited, neighbor_best_headings,
                                 budget_state, current_index, all_agent_indices, all_agent_next_indices,
                                 trajectory_tokens, trajectory_token_mask)
        return q_values
