"""S10 task 3: runtime corrections for two batch-dependent lines of the vendored refiner. Nothing under adapters/ is edited; install(model)
re-classes the existing attention modules to the subclasses below (same parameters, same state_dict keys) and the two switches in STATE
turn each correction on or off per forward call.

fix1  bevformer/encoder.py:237 hands the self-attention block its reference points as torch.cat([ref_pos, ref_pos]) (rows e0, e1, ..,
      e0, e1, ..) while temporal_self_attention.py:178 lays the values (and :220 the offsets) out per example (rows e0, e0, e1, e1, ..).
      With fix1 the reference points are re-laid per example (e0, e0, e1, e1, ..) before the block uses them.
fix2  bevformer/spatial_cross_attention.py:156 takes the camera-visibility mask of batch row 0 (mask_per_img[0]) for every row.
      With fix2 each row uses its own mask.
fix3  (found by this task, Q1) temporal_self_attention.py:195 pairs each row's query with value[:bs] before the offset and weight
      layers. With the per-example layout of :178 value[:bs] is (e0, e0, e1, e1, ..), so row j is paired with example j // 2.
      With fix3 row j is paired with its own example (pre-hooks on the two layers swap the first embed_dims channels).
At batch 1 all three corrections leave every tensor as it was.

STATE["rec"]: when a list, every self-attention call appends dict(ref_in, ref_used) and the pure-PyTorch sampler appends
dict(loc, w) (sampling locations and attention weights), and every cross-attention call appends dict(cam_mask)."""
import torch
from mmcv.runner import force_fp32
from navsim.agents.drive_jepa_perception_based.bevformer import spatial_cross_attention as _sca
from navsim.agents.drive_jepa_perception_based.bevformer import temporal_self_attention as _tsa

STATE = dict(fix1=False, fix2=False, fix3=False, rec=None)


def set_fix(fix1=False, fix2=False, fix3=False):
    STATE["fix1"] = bool(fix1); STATE["fix2"] = bool(fix2); STATE["fix3"] = bool(fix3)


_orig_msda = _tsa.multi_scale_deformable_attn_pytorch


def _msda_rec(value, spatial_shapes, sampling_locations, attention_weights):
    if STATE["rec"] is not None:
        STATE["rec"].append(dict(loc=sampling_locations.detach(), w=attention_weights.detach(), shapes=spatial_shapes.detach()))
    return _orig_msda(value, spatial_shapes, sampling_locations, attention_weights)


_tsa.multi_scale_deformable_attn_pytorch = _msda_rec


class TemporalSelfAttentionFix(_tsa.TemporalSelfAttention):
    def forward(self, query, key=None, value=None, identity=None, query_pos=None, key_padding_mask=None, reference_points=None, **kwargs):
        ref_in = reference_points
        if STATE["fix1"] and value is None and reference_points is not None:
            bs = query.shape[0] if self.batch_first else query.shape[1]
            assert reference_points.shape[0] == bs * self.num_bev_queue
            # encoder.py:237 built cat([ref, ref]); the values are stack([q, q], 1).reshape(2 bs, ..): give row 2j and 2j+1 example j's points
            reference_points = reference_points[:bs].repeat_interleave(self.num_bev_queue, 0)
        if STATE["rec"] is not None:
            STATE["rec"].append(dict(ref_in=ref_in.detach(), ref_used=reference_points.detach()))
        self._own_q = query if (STATE["fix3"] and value is None) else None      # (bs, num_query, c), before the positional encoding: what value[:bs] holds at batch 1
        try:
            return super().forward(query, key, value, identity, query_pos=query_pos, key_padding_mask=key_padding_mask, reference_points=reference_points, **kwargs)
        finally:
            self._own_q = None


def _own_pair_hook(tsa):
    def pre(lin, args):
        own = tsa.__dict__.get("_own_q")
        if own is None:
            return None
        x = args[0]; assert x.shape[:-1] == own.shape[:-1] and x.shape[-1] == 2 * own.shape[-1]
        return (torch.cat([own.to(x.dtype), x[..., own.shape[-1]:]], -1),)
    return pre


class SpatialCrossAttentionFix(_sca.SpatialCrossAttention):
    @force_fp32(apply_to=('query', 'key', 'value', 'query_pos', 'reference_points_cam'))
    def forward(self, query, key, value, residual=None, query_pos=None, key_padding_mask=None, reference_points=None, spatial_shapes=None,
                reference_points_cam=None, bev_mask=None, level_start_index=None, flag='encoder', **kwargs):
        if STATE["rec"] is not None:
            STATE["rec"].append(dict(cam_mask=(bev_mask.detach().sum(-1) > 0)))
        if not STATE["fix2"]:
            return super().forward(query, key, value, residual=residual, query_pos=query_pos, key_padding_mask=key_padding_mask, reference_points=reference_points,
                                   spatial_shapes=spatial_shapes, reference_points_cam=reference_points_cam, bev_mask=bev_mask, level_start_index=level_start_index, flag=flag, **kwargs)
        # spatial_cross_attention.py:137-229 with the row-0 mask (:156) replaced by each row's own mask
        assert residual is None and not self.full_attention
        query = query.to(torch.float32); key = key.to(torch.float32); value = value.to(torch.float32); reference_points_cam = reference_points_cam.to(torch.float32)
        with torch.autocast(device_type='cuda', enabled=False):
            inp_residual = query
            slots = torch.zeros_like(query)
            if query_pos is not None:
                query = query + query_pos
            bs, num_query, _ = query.size()
            D = reference_points_cam.size(3)
            indexes = [[mask_per_img[j].sum(-1).nonzero().squeeze(-1) for j in range(bs)] for mask_per_img in bev_mask]   # [camera][row]
            max_len = max(len(ix) for per_cam in indexes for ix in per_cam)
            queries_rebatch = query.new_zeros([bs, self.num_cams, max_len, self.embed_dims])
            reference_points_rebatch = reference_points_cam.new_zeros([bs, self.num_cams, max_len, D, 2])
            for j in range(bs):
                for i, reference_points_per_img in enumerate(reference_points_cam):
                    ix = indexes[i][j]
                    queries_rebatch[j, i, :len(ix)] = query[j, ix]
                    reference_points_rebatch[j, i, :len(ix)] = reference_points_per_img[j, ix]
            num_cams, l, bs, embed_dims = key.shape
            key = key.permute(2, 0, 1, 3).reshape(bs * self.num_cams, l, self.embed_dims)
            value = value.permute(2, 0, 1, 3).reshape(bs * self.num_cams, l, self.embed_dims)
            queries = self.deformable_attention(query=queries_rebatch.view(bs * self.num_cams, max_len, self.embed_dims), key=key, value=value,
                                                reference_points=reference_points_rebatch.view(bs * self.num_cams, max_len, D, 2),
                                                spatial_shapes=spatial_shapes, level_start_index=level_start_index).view(bs, self.num_cams, max_len, self.embed_dims)
            for j in range(bs):
                for i in range(self.num_cams):
                    ix = indexes[i][j]
                    slots[j, ix] += queries[j, i, :len(ix)]
            count = bev_mask.sum(-1) > 0
            count = count.permute(1, 2, 0).sum(-1)
            count = torch.clamp(count, min=1.0)
            slots = slots / count[..., None]
            slots = self.output_proj(slots)
            return self.dropout(slots) + inp_residual


def install(model):
    """Re-class every TemporalSelfAttention / SpatialCrossAttention inside `model`. Returns the names found."""
    found = dict(tsa=[], sca=[])
    for n, mod in model.named_modules():
        if type(mod) is _tsa.TemporalSelfAttention:
            assert mod.batch_first; mod.__class__ = TemporalSelfAttentionFix; mod.__dict__["_own_q"] = None; found["tsa"].append(n)
            mod.sampling_offsets.register_forward_pre_hook(_own_pair_hook(mod)); mod.attention_weights.register_forward_pre_hook(_own_pair_hook(mod))
        elif type(mod) is _sca.SpatialCrossAttention:
            mod.__class__ = SpatialCrossAttentionFix; found["sca"].append(n)
    return found
