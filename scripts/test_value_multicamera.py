"""Focused tests for real camera identity, fixed batching, and masked visual attention."""
import base64
from pathlib import Path

import cv2
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import flax.nnx as nnx

from openpi.models.model import Observation
from openpi.models.value_model import ValueModel, ValueHead
from openpi.models.value_model_config import ValueModelConfig
from openpi.policies.value_policy import ValueInputs
from openpi.shared.value_cameras import CAMERA_MODEL_KEYS, resolve_camera_names, write_camera_config
from openpi.training.agilex_hdf5_value_dataset import AgilexHdf5ValueDataset
from openpi.training.data_loader import _collate_fn


@pytest.mark.parametrize('names', [('head',), ('head','right'), ('head','left','right')])
def test_hdf5_selected_cameras_survive_transform_and_batch(tmp_path, names):
    folder=tmp_path/'episode_0';folder.mkdir()
    with h5py.File(folder/'episode.hdf5','w') as f:
        f.create_dataset('action',shape=(2,1))
        f.create_dataset('value_label',data=np.array([-1.,0.],np.float32))
        for camera,value in [('head',20),('left',80),('right',140)]:
            ok,jpeg=cv2.imencode('.jpg',np.full((8,12,3),value,np.uint8));assert ok
            image=base64.b64encode(jpeg.tobytes()).decode()
            f.create_dataset(f'observation/camera/{camera}',data=[image,image],dtype=h5py.string_dtype())
    raw=AgilexHdf5ValueDataset(str(tmp_path),camera_names=names)
    samples=[]
    for item in raw.__getitems__([1,0]):
        item['value']=item['value_label']
        samples.append(ValueInputs(camera_names=names)(item))
    batch=_collate_fn(samples)
    assert set(batch['image'])=={CAMERA_MODEL_KEYS[n] for n in names}
    for name in names:
        expected={'head':20,'left':80,'right':140}[name]
        np.testing.assert_allclose(batch['image'][CAMERA_MODEL_KEYS[name]],expected,atol=1)
        assert batch['image_mask'][CAMERA_MODEL_KEYS[name]].all()
    spec,_=ValueModelConfig(camera_names=names).inputs_spec(batch_size=2)
    assert set(spec.images)==set(batch['image'])


def test_right_only_and_missing_camera_have_stable_independent_slots(tmp_path):
    right=np.full((8,12,3),177,np.uint8)
    data={'images':{'cam_right_wrist':right},'value':-1}
    output=ValueInputs(camera_names=('right',))(data)
    assert list(output['image'])==['right_wrist_0_rgb']
    np.testing.assert_array_equal(output['image']['right_wrist_0_rgb'],right)
    partial=ValueInputs(camera_names=('head','left','right'))({'image':np.zeros_like(right),'right_wrist_image':right,'value':0})
    assert partial['image_mask']=={'base_0_rgb':True,'wrist_0_rgb':False,'right_wrist_0_rgb':True}
    assert not partial['image']['wrist_0_rgb'].any()
    with pytest.raises(ValueError):ValueModelConfig(camera_names=('head','head'))
    # A value-only HDF5 can contain a wrist camera without head/action/state arrays.
    folder=tmp_path/'episode_1';folder.mkdir()
    ok,jpeg=cv2.imencode('.jpg',right);assert ok
    with h5py.File(folder/'episode.hdf5','w') as f:
        f.create_dataset('value_label',data=np.array([-1.,0.],np.float32))
        image=base64.b64encode(jpeg.tobytes()).decode()
        f.create_dataset('observation/camera/right',data=[image,image],dtype=h5py.string_dtype())
    raw=AgilexHdf5ValueDataset(str(tmp_path),camera_names=('right',))
    assert len(raw)==2 and 'observation.images.cam_right_wrist' in raw[0]


def test_camera_metadata_roundtrip_and_conflicting_resume(tmp_path):
    step=tmp_path/'step_00000001';step.mkdir()
    write_camera_config(tmp_path,('head','left','right'))
    assert resolve_camera_names(checkpoint=step)==('head','left','right')
    assert resolve_camera_names(('right',),step)==('right',)
    with pytest.raises(ValueError,match='differs'):write_camera_config(tmp_path,('head',))
    assert resolve_camera_names(checkpoint=tmp_path/'legacy'/'step_1')==('head','left')


def test_masked_camera_does_not_change_cross_attention_or_value_gradient(monkeypatch):
    # Exercise the actual fusion path using tiny backbones, without loading model weights.
    import openpi.models.value_model as module
    class FakeEmbedder:
        def __init__(self,**kwargs):pass
        def encode(self):pass
        def apply(self,variables,tokens,method):return jnp.ones((*tokens.shape,8))
    monkeypatch.setattr(module,'Embedder',FakeEmbedder)
    class LLM(nnx.Module):
        def __init__(self):self.embedder=nnx.Param(jnp.zeros(1))
        def __call__(self,embeddings,mask):return embeddings
    class Images(nnx.Module):
        def __call__(self,x,train=False):return jnp.mean(x,axis=(1,2))[:,None,:],None
    class Tiny(ValueModel):
        def __init__(self):
            import types
            self.config=types.SimpleNamespace(vocab_size=16,embed_dim=8)
            self.img=Images();self.img_projection=nnx.Linear(3,8,rngs=nnx.Rngs(1));self.llm=LLM()
            self.cross_attn_norm=nnx.LayerNorm(8,rngs=nnx.Rngs(2))
            self.cross_attention=nnx.MultiHeadAttention(num_heads=2,in_features=8,decode=False,rngs=nnx.Rngs(3))
            self.value_head=ValueHead(8,4,201,0,nnx.Rngs(4))
    model=Tiny()
    def obs(right,mask):
        return Observation(images={'base_0_rgb':jnp.ones((1,2,2,3)),'right_wrist_0_rgb':jnp.full((1,2,2,3),right)},
            image_masks={'base_0_rgb':jnp.array([True]),'right_wrist_0_rgb':jnp.array([mask])},
            state=jnp.zeros((1,1)),tokenized_prompt=jnp.ones((1,2),jnp.int32),tokenized_prompt_mask=jnp.ones((1,2),bool))
    np.testing.assert_allclose(model(obs(0.,False)),model(obs(90.,False)),atol=1e-6)
    assert not np.allclose(model(obs(0.,True)),model(obs(90.,True)))
    @nnx.value_and_grad
    def loss(m):return jnp.mean(m(obs(3.,True),train=True)**2)
    value,grad=loss(model)
    assert np.isfinite(value)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(grad))
