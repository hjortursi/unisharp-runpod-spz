import base64
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import requests
import runpod

APP_DIR = Path("/workspace/UniSHARP")
MODEL_DIR = Path("/workspace/models")
CHECKPOINT = Path(os.environ.get("UNISHARP_CHECKPOINT", "/workspace/models/pretained_model.pt"))
READY_FILE = Path("/workspace/.unisharp_serverless_ready_v11")
SETUP_LOG = Path("/workspace/unisharp_setup.log")
DEFAULT_CAMERA = os.environ.get("UNISHARP_DEFAULT_CAMERA", "auto")
MAX_RESULT_INLINE_MB = float(os.environ.get("MAX_RESULT_INLINE_MB", "18"))
COMPACT_OUTPUT_DEFAULT = os.environ.get("COMPACT_OUTPUT_DEFAULT", "1") not in {"0", "false", "False"}
SPZ_DIR = Path("/workspace/spz")
SPZ_TOOL = SPZ_DIR / "build" / "ply_to_spz"
_setup_lock = threading.Lock()


def _run(cmd, cwd=None, env=None, timeout=None):
    SETUP_LOG.parent.mkdir(parents=True, exist_ok=True)
    with SETUP_LOG.open("a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n\n$ {cmd}\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            shell=True,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        log.write(f"\n[exit {proc.returncode}] {cmd}\n")
        log.flush()
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {cmd}")


def _tail(path=SETUP_LOG, n=16000):
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
        return data[-n:]
    except FileNotFoundError:
        return ""


def _ensure_setup():
    if READY_FILE.exists() and CHECKPOINT.exists() and (APP_DIR / "scripts" / "infer_unisharp.py").exists() and SPZ_TOOL.exists():
        return {"already_ready": True, "log_tail": _tail(n=4000)}
    with _setup_lock:
        if READY_FILE.exists() and CHECKPOINT.exists() and (APP_DIR / "scripts" / "infer_unisharp.py").exists() and SPZ_TOOL.exists():
            return {"already_ready": True, "log_tail": _tail(n=4000)}
        t0 = time.time()
        SETUP_LOG.write_text(f"UniSHARP lazy setup started at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n", encoding="utf-8")
        env = os.environ.copy()
        env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        env["DEBIAN_FRONTEND"] = "noninteractive"
        try:
            _run("apt-get update -qq && apt-get install -y -qq --no-install-recommends git git-lfs curl wget ca-certificates ffmpeg build-essential ninja-build cmake python3-dev libgl1 libglib2.0-0 libgomp1 libxext6 libsm6 libxrender1 && rm -rf /var/lib/apt/lists/*", env=env, timeout=900)
            _run("git lfs install --skip-repo || true", env=env, timeout=120)
            _run("python -m pip install --no-cache-dir --upgrade pip setuptools wheel", env=env, timeout=600)
            _run("python -m pip install --no-cache-dir runpod requests huggingface_hub hf_transfer", env=env, timeout=600)
            if not (APP_DIR / ".git").exists():
                _run(f"rm -rf {APP_DIR} && git clone --depth 1 https://github.com/Insta360-Research-Team/UniSHARP.git {APP_DIR}", env=env, timeout=600)
            if not (APP_DIR / "UniK3D" / ".git").exists():
                _run(f"git clone --depth 1 https://github.com/lpiccinelli-eth/UniK3D.git {APP_DIR / 'UniK3D'}", cwd=APP_DIR, env=env, timeout=600)
            _run("python -m pip install --no-cache-dir --force-reinstall --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0", cwd=APP_DIR, env=env, timeout=1800)
            _run("python -m pip install --no-cache-dir -r requirements.txt", cwd=APP_DIR, env=env, timeout=1800)
            # Pin xformers to the build matched to torch 2.8.0. Newer xformers pulls torch >=2.10/2.12,
            # which breaks torchvision 0.23.0 and recreates the torchvision::nms import failure.
            _run("python -m pip install --no-cache-dir wandb xformers==0.0.32.post2", cwd=APP_DIR, env=env, timeout=1200)
            _run("python - <<'PYCHK'\nimport torch, torchvision\nprint('torch', torch.__version__, 'cuda', torch.version.cuda)\nprint('torchvision', torchvision.__version__)\nPYCHK", cwd=APP_DIR, env=env, timeout=300)
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            if not CHECKPOINT.exists():
                _run("python - <<'PYHF'\nfrom huggingface_hub import hf_hub_download\np=hf_hub_download(repo_id='Insta360-Research/Unisharp', filename='pretained_model.pt', local_dir='/workspace/models', local_dir_use_symlinks=False)\nprint(p)\nPYHF", cwd=APP_DIR, env=env, timeout=1800)
            if not (SPZ_DIR / ".git").exists():
                _run(f"rm -rf {SPZ_DIR} && git clone --depth 1 https://github.com/nianticlabs/spz.git {SPZ_DIR}", env=env, timeout=600)
            if not SPZ_TOOL.exists():
                _run("cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)", cwd=SPZ_DIR, env=env, timeout=900)
            READY_FILE.write_text(str(time.time()), encoding="utf-8")
            return {"already_ready": False, "setup_seconds": round(time.time()-t0, 2), "log_tail": _tail(n=8000)}
        except Exception as e:
            return {"setup_failed": True, "error": str(e), "traceback": traceback.format_exc()[-8000:], "log_tail": _tail(n=16000)}


def _download_url(url: str, dst: Path) -> None:
    req = Request(url, headers={"User-Agent": "unisharp-runpod/1.0"})
    with urlopen(req, timeout=120) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f)


def _write_input_file(inp: dict, work: Path) -> Path:
    if inp.get("image_url"):
        suffix = Path(str(inp.get("filename") or inp["image_url"]).split("?")[0]).suffix or ".jpg"
        image_path = work / ("input" + suffix)
        _download_url(str(inp["image_url"]), image_path)
        return image_path
    if inp.get("image_base64"):
        suffix = str(inp.get("extension") or ".jpg")
        if not suffix.startswith("."):
            suffix = "." + suffix
        image_path = work / ("input" + suffix)
        image_path.write_bytes(base64.b64decode(inp["image_base64"]))
        return image_path
    raise ValueError("Provide input.image_url or input.image_base64")


def _zip_dir(src: Path, zip_path: Path, compact: bool = False) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in src.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(src).as_posix()
            if compact and not (rel.endswith(".gif") or rel.endswith("metadata.json") or rel.endswith(".ply") or rel.endswith(".jpg") or rel.endswith(".jpeg")):
                continue
            z.write(p, rel)


def _list_outputs(out_dir: Path):
    return sorted(str(p.relative_to(out_dir)) for p in out_dir.rglob("*") if p.is_file())



def _patch_infer_for_gaussians_only() -> None:
    """Add --gaussians-only to upstream infer script without changing Gaussian quality."""
    infer_path = APP_DIR / "scripts" / "infer_unisharp.py"
    text = infer_path.read_text(encoding="utf-8")
    if "--gaussians-only" in text:
        return
    old_arg = "    p.add_argument(\"--low-pass-filter-eps\", type=float, default=0.0)\n    return p\n"
    new_arg = "    p.add_argument(\"--low-pass-filter-eps\", type=float, default=0.0)\n    p.add_argument(\"--gaussians-only\", action=\"store_true\", help=\"Save Gaussian PLY + metadata only; skip preview rendering/GIFs.\")\n    return p\n"
    if old_arg not in text:
        raise RuntimeError("Could not patch argparser for --gaussians-only")
    text = text.replace(old_arg, new_arg, 1)
    old_block = """    model_output = out if isinstance(out, dict) else {"gaussians": out}
    (
        forward_distance_m,
"""
    new_block = """    model_output = out if isinstance(out, dict) else {"gaussians": out}
    if bool(getattr(args, "gaussians_only", False)):
        sample_dir = out_root / _slug_from_path(image_path)
        sample_dir.mkdir(parents=True, exist_ok=True)
        if camera_kind == "panorama":
            f_px = float(w) / (2.0 * math.pi)
        elif render_intrinsics is not None:
            k3 = render_intrinsics.detach().to(device=device, dtype=torch.float32)[0]
            f_px = float(0.5 * (float(k3[0, 0].detach().cpu()) + float(k3[1, 1].detach().cpu())))
        elif render_camera_params is not None:
            params = render_camera_params.detach().to(device=device, dtype=torch.float32)
            f_px = float(0.5 * (float(params[0, 0].detach().cpu()) + float(params[0, 1].detach().cpu())))
        else:
            f_px = float(w)
        _save_ply_if_requested(gaussians_world, sample_dir / "gaussians.ply", f_px=f_px, image_h=h, image_w=w, enabled=bool(args.save_ply))
        metadata = {
            "checkpoint": str(args.checkpoint),
            "checkpoint_step": int(step),
            "image": str(image_path),
            "camera_kind": camera_kind,
            "ray_stats": stats,
            "camera_json": str(args.camera_json) if args.camera_json is not None else None,
            "camera_json_entry": camera_json_entry,
            "aspect_camera_name": aspect_camera_name,
            "explicit_camera_intrinsics": args.camera_intrinsics,
            "explicit_camera_params": args.camera_params,
            "low_pass_filter_eps": float(args.low_pass_filter_eps),
            "gaussians_only": True,
            "preview_rendering_skipped": True,
            "height": int(h),
            "width": int(w),
        }
        (sample_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
        LOGGER.info("Saved Gaussian-only outputs -> %s", sample_dir)
        return
    (
        forward_distance_m,
"""
    if old_block not in text:
        raise RuntimeError("Could not patch _process_one gaussians-only block")
    text = text.replace(old_block, new_block, 1)
    infer_path.write_text(text, encoding="utf-8")


def _ply_vertex_count(ply_path: Path) -> int | None:
    try:
        with ply_path.open("rb") as f:
            for raw in f:
                line = raw.decode("ascii", errors="replace").strip()
                if line.startswith("element vertex "):
                    return int(line.split()[-1])
                if line == "end_header":
                    return None
    except Exception:
        return None
    return None


def _convert_first_ply_to_spz(out_dir: Path, spz_path: Path) -> tuple[Path, int | None]:
    ply_files = sorted(out_dir.rglob("*.ply"))
    if not ply_files:
        raise RuntimeError("No gaussians.ply produced; cannot create SPZ")
    ply = ply_files[0]
    vertex_count = _ply_vertex_count(ply)
    spz_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run([str(SPZ_TOOL), str(ply), str(spz_path)], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
    if proc.returncode != 0 or not spz_path.exists() or spz_path.stat().st_size <= 0:
        raise RuntimeError(f"SPZ conversion failed: {proc.stdout[-4000:]}")
    return spz_path, vertex_count


def _put_with_retries(url: str, data: bytes, timeout: int = 300) -> int:
    for attempt in range(1, 4):
        try:
            r = requests.put(url, data=data, timeout=timeout)
            r.raise_for_status()
            return attempt
        except Exception:
            if attempt >= 3:
                raise
            time.sleep(2 * attempt)
    return 3

def _make_splat_preview(ply_path: Path, out_path: Path) -> None:
    script = f"""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plyfile import PlyData
ply_path = Path({str(ply_path)!r})
out_path = Path({str(out_path)!r})
ply = PlyData.read(str(ply_path))
v = ply['vertex'].data
names = v.dtype.names
xyz = np.column_stack([v['x'], v['y'], v['z']]).astype(np.float32)
finite = np.isfinite(xyz).all(axis=1)
xyz = xyz[finite]
v = v[finite]
if all(c in names for c in ['red','green','blue']):
    rgb = np.column_stack([v['red'], v['green'], v['blue']]).astype(np.float32) / 255.0
elif all(c in names for c in ['f_dc_0','f_dc_1','f_dc_2']):
    C0 = 0.28209479177387814
    rgb = np.column_stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']]).astype(np.float32) * C0 + 0.5
    rgb = np.clip(rgb, 0, 1)
else:
    z = xyz[:,2]
    zn = (z - np.nanmin(z)) / (np.nanmax(z) - np.nanmin(z) + 1e-6)
    rgb = plt.get_cmap('viridis')(zn)[:,:3]
maxn = 180000
if len(xyz) > maxn:
    rng = np.random.default_rng(42)
    idx = rng.choice(len(xyz), maxn, replace=False)
    xyz = xyz[idx]; rgb = rgb[idx]
center = np.median(xyz, axis=0)
xyz = xyz - center
scale = np.percentile(np.linalg.norm(xyz, axis=1), 98) + 1e-6
xyz = xyz / scale
views = [(15,0,'front-ish'), (15,90,'right'), (15,180,'back'), (15,270,'left'), (70,45,'top'), (-45,45,'bottom')]
fig = plt.figure(figsize=(14,9), dpi=150)
for i,(elev,azim,title) in enumerate(views,1):
    ax = fig.add_subplot(2,3,i, projection='3d')
    ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c=rgb, s=0.08, linewidths=0, alpha=0.75)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f'{{title}} — {{len(v):,}} gaussians')
    ax.set_axis_off()
    ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_zlim(-1,1)
fig.tight_layout(pad=0.2)
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, bbox_inches='tight', pad_inches=0.02)
"""
    subprocess.run(["python", "-c", script], cwd=str(APP_DIR), check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)


def handler(job):
    t0 = time.time()
    inp = job.get("input") or {}
    setup = _ensure_setup()
    if setup.get("setup_failed"):
        return {"ok": False, **setup}
    try:
        _patch_infer_for_gaussians_only()
    except Exception as e:
        return {"ok": False, "error": f"Failed to patch UniSHARP infer script: {e}", "traceback": traceback.format_exc()[-8000:], **setup}
    if inp.get("setup_only"):
        return {"ok": True, "message": "UniSHARP serverless worker is ready", "checkpoint_exists": CHECKPOINT.exists(), "app_dir_exists": APP_DIR.exists(), "spz_tool_exists": SPZ_TOOL.exists(), "gaussians_only_patch": True, **setup}

    work = Path(tempfile.mkdtemp(prefix="unisharp_"))
    try:
        image_path = _write_input_file(inp, work)
        out_dir = work / "out"
        camera_json_path = None
        if inp.get("camera_json"):
            camera_json_path = work / "camera.json"
            camera_json_path.write_text(json.dumps(inp["camera_json"]), encoding="utf-8")
        cmd = ["python", str(APP_DIR / "scripts" / "infer_unisharp.py"), "--checkpoint", str(CHECKPOINT), "--image", str(image_path), "--out-dir", str(out_dir), "--camera", str(inp.get("camera") or DEFAULT_CAMERA)]
        if inp.get("save_ply", True):
            cmd.append("--save-ply")
        if inp.get("gaussians_only") or inp.get("skip_previews"):
            cmd.append("--gaussians-only")
        if camera_json_path:
            cmd += ["--camera-json", str(camera_json_path)]
        render_max_long_edge = inp.get("render_max_long_edge") or inp.get("panorama_max_long_edge")
        if render_max_long_edge is not None:
            render_max_long_edge = int(render_max_long_edge)
            if render_max_long_edge < 512 or render_max_long_edge > 8192:
                raise ValueError("render_max_long_edge must be between 512 and 8192")
            infer_path = APP_DIR / "scripts" / "infer_unisharp.py"
            text = infer_path.read_text(encoding="utf-8")
            text2 = __import__('re').sub(r"PANORAMA_MAX_LONG_EDGE = \d+", f"PANORAMA_MAX_LONG_EDGE = {render_max_long_edge}", text, count=1)
            if text2 == text:
                raise RuntimeError("Could not patch PANORAMA_MAX_LONG_EDGE in infer_unisharp.py")
            infer_path.write_text(text2, encoding="utf-8")
        if inp.get("low_pass_filter_eps") is not None:
            cmd += ["--low-pass-filter-eps", str(inp["low_pass_filter_eps"])]
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{APP_DIR}:{APP_DIR / 'UniK3D'}:" + env.get("PYTHONPATH", "")
        proc = subprocess.run(cmd, cwd=str(APP_DIR), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=int(inp.get("timeout_seconds", 900)))
        if proc.returncode != 0:
            return {"ok": False, "error": "UniSHARP inference failed", "returncode": proc.returncode, "log_tail": proc.stdout[-12000:]}
        if inp.get("spz_only") or str(inp.get("output_format") or "").lower() == "spz":
            spz_path, vertex_count = _convert_first_ply_to_spz(out_dir, work / "gaussians.spz")
            spz_bytes = spz_path.read_bytes()
            result = {"ok": True, "seconds": round(time.time() - t0, 2), "camera": str(inp.get("camera") or DEFAULT_CAMERA), "render_max_long_edge": inp.get("render_max_long_edge") or inp.get("panorama_max_long_edge"), "output_format": "spz", "spz_only": True, "spz_bytes": len(spz_bytes), "gaussians": vertex_count, "log_tail": proc.stdout[-4000:]}
            if inp.get("output_url"):
                result["upload_attempts"] = _put_with_retries(str(inp["output_url"]), spz_bytes, timeout=300)
                result["output_url"] = inp["output_url"]
            elif len(spz_bytes) <= MAX_RESULT_INLINE_MB * 1024 * 1024:
                result["spz_base64"] = base64.b64encode(spz_bytes).decode("ascii")
            else:
                result["warning"] = "SPZ too large for inline response; pass input.output_url as a presigned PUT URL."
            return result
        if inp.get("splat_preview") or inp.get("splat_preview_only"):
            ply_files = list(out_dir.rglob("*.ply"))
            if ply_files:
                _make_splat_preview(ply_files[0], ply_files[0].with_name("splat_preview.jpg"))
        zip_path = work / "unisharp_outputs.zip"
        compact = bool(inp.get("compact_output", COMPACT_OUTPUT_DEFAULT))
        if inp.get("splat_preview_only"):
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
                for p in out_dir.rglob("*"):
                    if p.is_file():
                        rel = p.relative_to(out_dir).as_posix()
                        if rel.endswith("splat_preview.jpg") or rel.endswith("metadata.json"):
                            z.write(p, rel)
        else:
            _zip_dir(out_dir, zip_path, compact=compact)
        size = zip_path.stat().st_size
        result = {"ok": True, "seconds": round(time.time() - t0, 2), "camera": str(inp.get("camera") or DEFAULT_CAMERA), "render_max_long_edge": inp.get("render_max_long_edge") or inp.get("panorama_max_long_edge"), "compact_output": compact, "gaussians_only": bool(inp.get("gaussians_only") or inp.get("skip_previews")), "files": _list_outputs(out_dir), "zip_bytes": size, "log_tail": proc.stdout[-4000:]}
        if inp.get("output_url"):
            zip_data = zip_path.read_bytes()
            result["upload_attempts"] = _put_with_retries(str(inp["output_url"]), zip_data, timeout=300)
            result["output_url"] = inp["output_url"]
        elif size <= MAX_RESULT_INLINE_MB * 1024 * 1024:
            result["zip_base64"] = base64.b64encode(zip_path.read_bytes()).decode("ascii")
        else:
            result["warning"] = "Result zip too large for inline response; pass input.output_url as a presigned PUT URL."
        return result
    except Exception as e:
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()[-12000:]}
    finally:
        shutil.rmtree(work, ignore_errors=True)


runpod.serverless.start({"handler": handler})