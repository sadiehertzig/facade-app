"""Tests for facade_photoguard_encoder.py, the engine the server runs.

Analytic CPU fixtures only; no real weights, MPS/CUDA run, listener or provider.
TinyVAE stands in for Stable Diffusion's VAE encoder (8x average pooling), and
`reference_pgd` is an independent transcription of upstream PhotoGuard's
`pgd()` loop, so the port can be compared with upstream tensor for tensor.
"""
from dataclasses import replace
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn.functional as F

import facade_photoguard_encoder as pg


@pytest.fixture(autouse=True, scope="module")
def few_cpu_threads():
    """Run these small tensor tests on one CPU thread, so results are repeatable, then restore the setting."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class TinyVAE:
    def __init__(self):
        self.inputs = []
    def encode(self, image):
        self.inputs.append(image.detach().clone())
        pooled = F.avg_pool2d(image, 8)
        mean = torch.cat([pooled, pooled.mean(dim=1, keepdim=True)], dim=1)
        return SimpleNamespace(latent_dist=SimpleNamespace(mean=mean))


def components(dtype=torch.float32):
    vae = TinyVAE()
    target = vae.encode(pg._target_tensor(side=8).to(dtype)).latent_dist.mean.detach()
    vae.inputs.clear()
    return pg.LoadedComponents(vae, target, torch.device("cpu"), dtype)


def tensors(dtype=torch.float32):
    image = torch.linspace(-.98, .98, 3*8*8, dtype=dtype).reshape(1,3,8,8)
    mask = torch.zeros(1,1,8,8,dtype=dtype)
    mask[:,:,:,:3] = 1
    return image, mask


def recipe(iterations=3, **kwargs):
    return replace(pg.DEFAULT_RECIPE, side=8, iterations=iterations, **kwargs)


def reference_pgd(image, mask, comp, configured):
    """Literal reference ordering with isolated upstream global CPU RNG.

    Unlike the port, this retains intermediate graphs and uses the original
    .data update shape. It is deliberately independent of port projection,
    schedule and RNG helpers. MIT attribution is in the port/license file.
    """
    x = image.to(comp.dtype) * (1-mask.to(comp.dtype))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(configured.seed)
        x_adv = x.clone().detach() + (torch.rand(*x.shape)*2*configured.epsilon-configured.epsilon)
        for i in range(configured.iterations):
            actual_step_size = configured.step_size - (configured.step_size-configured.step_size/100)/configured.iterations*i
            x_adv.requires_grad_(True)
            loss = (comp.vae.encode(x_adv).latent_dist.mean-comp.target_mean).norm()
            grad, = torch.autograd.grad(loss, [x_adv])
            x_adv = x_adv-grad.detach().sign()*actual_step_size
            x_adv = torch.minimum(torch.maximum(x_adv,x-configured.epsilon),x+configured.epsilon)
            x_adv.data = torch.clamp(x_adv,min=-1,max=1)
            x_adv.grad = None
            x_adv.data *= 1-mask
    return x_adv.detach()


def test_exact_gray128_target_is_not_normalized_zero():
    """The gray target is exactly RGB 128 (hash checked), which maps to 0.0039, not to 0."""
    target = pg._target_tensor()
    expected = np.full((512,512,3),128,np.uint8)
    assert hashlib.sha256(expected.tobytes()).hexdigest() == pg.TARGET_RGB512_SHA256
    assert target.shape == (1,3,512,512)
    assert target.unique().item() == 0.003921627998352051
    assert target.half().unique().item() == 0.0039215087890625
    assert not torch.equal(target, torch.zeros_like(target))


def test_latent_objective_gradient_matches_independent_finite_difference():
    """The latent-distance loss's gradient matches a finite-difference estimate."""
    comp = components(torch.float64)
    image, _ = tensors(torch.float64)
    current = image.clone().requires_grad_(True)
    def objective(x):
        return (comp.vae.encode(x).latent_dist.mean-comp.target_mean).norm()
    derivative = torch.autograd.grad(objective(current), current)[0]
    direction = torch.linspace(-1,1,current.numel(),dtype=current.dtype).reshape_as(current)
    direction /= direction.norm()
    analytic = (derivative*direction).sum().item()
    h=1e-5
    observed=((objective(image+h*direction)-objective(image-h*direction))/(2*h)).item()
    assert analytic != 0
    assert observed == pytest.approx(analytic, rel=1e-7, abs=1e-9)


@pytest.mark.parametrize("iterations", [3,200])
def test_port_matches_reference_trajectory_including_initial_unmasked_noise(iterations):
    """The port reproduces upstream's PGD exactly at 3 and 200 steps, including the unmasked first iterate, without touching the global RNG."""
    image, mask = tensors()
    a,b = components(),components()
    configured=recipe(iterations)
    rng_before=torch.get_rng_state().clone()
    expected=reference_pgd(image,mask,a,configured)
    actual=pg._attack_tensors(image,mask,b,recipe=configured)
    assert torch.equal(torch.get_rng_state(),rng_before)
    assert torch.equal(actual,expected)
    assert len(a.vae.inputs)==len(b.vae.inputs)==iterations
    assert all(torch.equal(x,y) for x,y in zip(a.vae.inputs,b.vae.inputs))
    assert (b.vae.inputs[0]*mask).abs().max()>0, "upstream does not mask the initial noise"
    assert torch.equal(b.vae.inputs[1]*mask,torch.zeros_like(image))
    assert torch.equal(actual*mask,torch.zeros_like(image))
    assert ((actual-image)*(1-mask)).abs().max() <= configured.epsilon+1e-6


def test_full200_progress_export_bound_and_edit_region_preservation():
    """A 200-step run reports progress 0..200, stays within 16 levels and leaves the white region byte-identical."""
    image,mask=tensors()
    updates=[]
    result=pg._attack_tensors(image,mask,components(),recipe=recipe(200),
        progress=lambda current,total:updates.append((current,total)))
    assert updates==[(i,200) for i in range(201)]
    original=Image.fromarray(((image[0].permute(1,2,0).numpy()+1)*127.5).round().astype(np.uint8))
    mask_pil=Image.fromarray((mask[0,0].numpy()*255).astype(np.uint8))
    out=pg._output_image(original,result,mask_pil)
    delta=np.asarray(out).astype(np.int16)-np.asarray(original).astype(np.int16)
    assert np.abs(delta).max()<=16 and np.abs(delta).max()>0
    assert np.array_equal(np.asarray(out)[np.asarray(mask_pil)==255],np.asarray(original)[np.asarray(mask_pil)==255])


def test_same_job_seed_repeats_without_changing_global_rng():
    """The same seed gives the same output; a different seed gives a different random start."""
    image,mask=tensors()
    a,b,c=components(),components(),components()
    first=pg._attack_tensors(image,mask,a,recipe=recipe(seed=1234))
    second=pg._attack_tensors(image,mask,b,recipe=recipe(seed=1234))
    pg._attack_tensors(image,mask,c,recipe=recipe(seed=1235))
    assert torch.equal(first,second)
    assert torch.equal(a.vae.inputs[0],b.vae.inputs[0])
    assert not torch.equal(a.vae.inputs[0],c.vae.inputs[0])


def test_cancel_before_work_and_between_iterations_returns_no_output():
    """A cancel before work or between iterations raises AttackCancelled and returns nothing."""
    image,mask=tensors(); comp=components(); updates=[]
    with pytest.raises(pg.AttackCancelled):
        pg._attack_tensors(image,mask,comp,recipe=recipe(),cancel=lambda:True,
                           progress=lambda *args:updates.append(args))
    assert not comp.vae.inputs and not updates
    with pytest.raises(pg.AttackCancelled):
        pg._attack_tensors(image,mask,comp,recipe=recipe(200),cancel=lambda:len(comp.vae.inputs)>=5,
                           progress=lambda *args:updates.append(args))
    assert len(comp.vae.inputs)==5 and updates[-1]==(4,200)


def test_cancel_from_final_progress_never_returns_candidate():
    """A cancel set by the very last progress report still prevents any result."""
    image,mask=tensors(); progress=[]
    with pytest.raises(pg.AttackCancelled):
        pg._attack_tensors(image,mask,components(),recipe=recipe(),cancel=lambda:(3,3) in progress,
                           progress=lambda *args:progress.append(args))


@pytest.mark.parametrize("kind", ["nan_target","nan_objective","nan_gradient"])
def test_nonfinite_target_objective_or_gradient_fails_closed(kind):
    """A NaN in the target, the loss or the gradient raises FloatingPointError."""
    comp=components();image,mask=tensors()
    if kind=="nan_target":
        comp.target_mean.fill_(float('nan'))
    else:
        base=comp.vae.encode
        class BadGradient(torch.autograd.Function):
            @staticmethod
            def forward(ctx,x): return x.clone()
            @staticmethod
            def backward(ctx,grad): return torch.full_like(grad,float('nan'))
        def broken(x):
            mean=base(x).latent_dist.mean
            mean=mean*float('nan') if kind=="nan_objective" else BadGradient.apply(mean)
            return SimpleNamespace(latent_dist=SimpleNamespace(mean=mean))
        comp.vae.encode=broken
    with pytest.raises(FloatingPointError):
        pg._attack_tensors(image,mask,comp,recipe=recipe())


@pytest.mark.parametrize("kind", ["empty","full","soft","nan","shape"])
def test_invalid_tensors_do_not_reach_vae(kind):
    """Empty, full, soft or wrongly shaped masks and NaN images are refused before the VAE runs."""
    comp=components();image,mask=tensors()
    if kind=="empty":mask.zero_()
    if kind=="full":mask.fill_(1)
    if kind=="soft":mask[0,0,0,0]=.5
    if kind=="nan":image[0,0,0,0]=float('nan')
    if kind=="shape":image=image[:,:,:,:7]
    with pytest.raises(ValueError):
        pg._attack_tensors(image,mask,comp,recipe=recipe())
    assert not comp.vae.inputs


@pytest.mark.parametrize("color", [0,42,128,255])
def test_public_invalid_mask_rejected_before_model_load(color,monkeypatch):
    """protect() refuses a one-colour mask before it tries to load any model."""
    monkeypatch.setattr(pg,'load_local_components',lambda *a,**k:pytest.fail('must not load'))
    with pytest.raises(ValueError,match='binary 0/255'):
        pg.protect(Image.new('RGB',(512,512),128),Image.new('L',(512,512),color),snapshot=Path('missing'))


def test_export_uses_reference_truncation_and_refuses_out_of_bound():
    """Output pixels are truncated like ToPILImage, the white region is restored, and a delta over 16 is refused."""
    original=Image.new('RGB',(8,8),(128,128,128));mask=Image.new('L',(8,8),0);mask.paste(255,(0,0,3,8))
    candidate=torch.full((1,3,8,8),(130.9/127.5)-1)
    out=pg._output_image(original,candidate,mask)
    assert out.getpixel((7,7))==(130,130,130)
    assert out.getpixel((0,0))==(128,128,128)
    with pytest.raises(ValueError,match='byte bound'):
        pg._output_image(original,torch.ones_like(candidate),mask)


def test_snapshot_checks_only_exact_vae_and_rejects_unbound_optional_files(tmp_path,monkeypatch):
    """Snapshot checks look only at vae/, and fail on a changed byte or an extra .safetensors file there."""
    path=tmp_path/'vae/config.json';path.parent.mkdir();path.write_bytes(b'analytic config')
    monkeypatch.setattr(pg,'SOURCE_FILES',{'vae/config.json':hashlib.sha256(path.read_bytes()).hexdigest()})
    (tmp_path/'unet').mkdir();(tmp_path/'unet/not-consumed.bin').write_bytes(b'ignored')
    assert pg.verify_local_snapshot(tmp_path)==tmp_path
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):pg.verify_local_snapshot(tmp_path)
    path.write_bytes(b'analytic config');(path.parent/'optional.safetensors').write_bytes(b'not allowed')
    with pytest.raises(ValueError,match='inventory'):pg.verify_local_snapshot(tmp_path)


def test_real_cpu_loader_refused_before_snapshot_access(monkeypatch):
    """Loading onto the CPU is refused before the snapshot is read."""
    monkeypatch.setattr(pg,'version',lambda name:pg.REQUIRED_VERSIONS[name])
    monkeypatch.setattr(pg,'verify_local_snapshot',lambda *a,**k:pytest.fail('must not read'))
    with pytest.raises(ValueError,match='explicit CUDA or MPS'):
        pg.load_local_components(Path('missing'),device='cpu')


def test_public_requires_verified_components_and_autograd():
    """protect() refuses CPU test components and refuses to run under inference_mode."""
    image=Image.new('RGB',(512,512),128);mask=Image.new('L',(512,512));mask.paste(255,(0,0,256,512))
    with pytest.raises(ValueError,match='verified CUDA-FP16 or MPS-FP32'):
        pg.protect(image,mask,components=components())
    with torch.inference_mode(),pytest.raises(RuntimeError,match='inference_mode'):
        pg.protect(image,mask,components=components())


def test_public_closes_prepared_image_when_mask_preparation_fails(monkeypatch):
    """If the mask cannot be prepared, the already prepared photo is closed, not leaked."""
    image = Image.new("RGB",(512,512),(128,128,128))
    clean = image.copy()
    def prepare(value, *, mask=False):
        if mask:
            raise ValueError("invalid mask")
        return clean
    monkeypatch.setattr(pg,"_clean_pil",prepare)
    with pytest.raises(ValueError,match="invalid mask"):
        pg.protect(image,object())
    with pytest.raises(ValueError):
        clean.getpixel((0,0))


def test_public_closes_output_if_cancelled_after_encoding(monkeypatch):
    """A cancel after the output image exists closes that image instead of returning it."""
    image=Image.new("RGB",(512,512),(128,128,128))
    mask=Image.new("L",(512,512));mask.paste(255,(0,0,256,512))
    output=image.copy()
    comp=pg.LoadedComponents(None,None,torch.device("mps"),torch.float32,pg.source_digest())
    monkeypatch.setattr(pg,"_attack_tensors",lambda *a,**k:None)
    monkeypatch.setattr(pg,"_output_image",lambda *a,**k:output)
    checks=[]
    def cancel():
        checks.append(True)
        return len(checks)>=3
    with pytest.raises(pg.AttackCancelled):
        pg.protect(image,mask,components=comp,cancel=cancel)
    with pytest.raises(ValueError):
        output.getpixel((0,0))


def test_cli_protects_one_png_and_never_overwrites(tmp_path, monkeypatch, capsys):
    """The command-line tool writes one clean PNG, prints progress and a JSON summary, and refuses an existing output file."""
    import json
    image = Image.new("RGB", (512, 512), (128, 128, 128))
    mask = Image.new("L", (512, 512))
    mask.paste(255, (0, 0, 256, 512))
    image.save(tmp_path / "in.png")
    mask.save(tmp_path / "mask.png")
    comp = pg.LoadedComponents(None, None, torch.device("mps"), torch.float32, pg.source_digest())
    monkeypatch.setattr(pg, "load_local_components", lambda *a, **k: comp)

    def fake_protect(img, msk, *, components, cancel, progress):
        assert components is comp and not cancel()
        for iteration in range(201):
            progress(iteration, 200)
        return img.copy()

    monkeypatch.setattr(pg, "protect", fake_protect)
    out = tmp_path / "out.png"
    argv = ["--snapshot", str(tmp_path), "--input", str(tmp_path / "in.png"), "--mask", str(tmp_path / "mask.png"),
            "--output", str(out), "--device", "mps"]
    pg.main(argv)
    with Image.open(out) as result:
        assert result.size == (512, 512) and result.mode == "RGB" and not result.info
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith("iteration 0/200") and lines[-2].startswith("iteration 200/200")
    summary = json.loads(lines[-1])
    assert summary["max_absolute_byte_delta"] == 0 and summary["recipe"] == pg.RECIPE_ID
    with pytest.raises(SystemExit):
        pg.main(argv)  # the output already exists


def test_module_imports_only_the_standard_library_ml_stack_and_its_sibling():
    """The engine imports nothing beyond the standard library, numpy, torch, Pillow, diffusers and its sibling module."""
    import ast
    import sys
    allowed = set(sys.stdlib_module_names) | {"__future__", "numpy", "torch", "PIL", "diffusers",
                                                "facade_photoguard_diffusion"}
    tree = ast.parse(Path(pg.__file__).read_text())
    imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                for alias in node.names}
    imported |= {node.module.split(".")[0] for node in ast.walk(tree)
                 if isinstance(node, ast.ImportFrom) and node.module}
    assert imported <= allowed, imported - allowed
