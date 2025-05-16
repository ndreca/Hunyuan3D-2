import os
import shutil
import subprocess
import time
import traceback

from PIL import Image
from torch import cuda, Generator
from cog import BasePredictor, BaseModel, Input, Path

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import FaceReducer, FloaterRemover, DegenerateFaceRemover, MeshlibCleaner, Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.shapegen.models.autoencoders import SurfaceExtractors
from hy3dgen.shapegen.utils import logger
from hy3dgen.texgen import Hunyuan3DPaintPipeline

CHECKPOINTS_PATH = "/src/checkpoints"
HUNYUAN3D_REPO = "andreca/hunyuan3d-2xet"
HUNYUAN3D_DIT_MODEL = "hunyuan3d-dit-v2-0-turbo"
HUNYUAN3D_VAE_MODEL = "hunyuan3d-vae-v2-0-turbo"
HUNYUAN3D_DELIGHT_MODEL = "hunyuan3d-delight-v2-0"
HUNYUAN3D_PAINT_MODEL = "hunyuan3d-paint-v2-0"
U2NET_PATH = os.path.join(CHECKPOINTS_PATH, ".u2net/")
U2NET_URL = "https://weights.replicate.delivery/default/comfy-ui/rembg/u2net.onnx.tar"

def download_if_not_exists(url, dest):
    if not os.path.exists(dest):
        start = time.time()
        os.makedirs(dest, exist_ok=True)
        logger.info(f"downloading url: {url}") 
        logger.info(f"downloading to: {dest}")
        subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
        duration = time.time() - start
        logger.info(f"downloading took: {duration:.2f}s")

class Output(BaseModel):
    mesh: Path

class Predictor(BasePredictor):
    def setup(self) -> None:
        try:
            start = time.time()
            logger.info("Setup started")
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ['U2NET_HOME'] = U2NET_PATH

            mc_algo = 'dmc'
            use_delight = False
            use_super = False
            
            download_if_not_exists(U2NET_URL, U2NET_PATH)
            self.i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                HUNYUAN3D_REPO,
                subfolder=HUNYUAN3D_DIT_MODEL
            )
            self.i23d_worker.enable_flashvdm(mc_algo=mc_algo)
            self.i23d_worker.vae.surface_extractor = SurfaceExtractors[mc_algo]()
            self.texgen_worker = Hunyuan3DPaintPipeline.from_pretrained(
                HUNYUAN3D_REPO, 
                subfolder=HUNYUAN3D_PAINT_MODEL, 
                use_delight=use_delight, 
                use_super=use_super
            )
            self.floater_remove_worker = FloaterRemover()
            self.degenerate_face_remove_worker = DegenerateFaceRemover()
            self.face_reduce_worker = FaceReducer()
            self.rmbg_worker = BackgroundRemover()
            self.cleaner_worker = MeshlibCleaner()
            duration = time.time() - start
            logger.info(f"Setup took: {duration:.2f}s")
        except Exception as e:
            logger.error(f"Setup failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _cleanup_gpu_memory(self):
        if cuda.is_available():
            cuda.empty_cache()
            cuda.ipc_collect()

    def _log_analytics_event(self, event_name, params=None):
        pass

    def predict(
        self,
        image: Path = Input(
            description="Input image for generating 3D shape",
            default=None
        ),
        steps: int = Input(
            description="Number of inference steps",
            default=50,
            ge=20,
            le=50,
        ),
        guidance_scale: float = Input(
            description="Guidance scale for generation",
            default=5.5,
            ge=1.0,
            le=20.0,
        ),
        max_facenum: int = Input(
            description="Maximum number of faces for mesh generation",
            default=40000,
            ge=10000,
            le=200000
        ),
        num_chunks: int = Input(
            description="Number of chunks for mesh generation",
            default=200000,
            ge=10000,
            le=200000
        ),
        seed: int = Input(
            description="Random seed for generation",
            default=1234
        ),
        octree_resolution: int = Input(
            description="Octree resolution for mesh generation",
            choices=[256, 384, 512],
            default=512
        ),
        remove_background: bool = Input(
            description="Whether to remove background from input image",
            default=True
        )
    ) -> Output:
        start_time = time.time()
        
        self._log_analytics_event("predict_started", {
            "steps": steps,
            "guidance_scale": guidance_scale,
            "max_facenum": max_facenum,
            "num_chunks": num_chunks,
            "seed": seed,
            "octree_resolution": octree_resolution,
            "remove_background": remove_background
        })

        if os.path.exists("output"):
            shutil.rmtree("output")
        
        os.makedirs("output", exist_ok=True)

        self._cleanup_gpu_memory()

        generator = Generator()
        generator = generator.manual_seed(seed)

        if image is not None:
            input_image = Image.open(str(image))
            if remove_background or input_image.mode == "RGB":
                input_image = self.rmbg_worker(input_image.convert('RGB'))
                self._cleanup_gpu_memory()
        else:
            self._log_analytics_event("predict_error", {"error": "no_image_provided"})
            raise ValueError("Image must be provided")

        input_image.save("output/input.png")

        try:
            mesh = self.i23d_worker(
                image=input_image,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                octree_resolution=octree_resolution,
                num_chunks=num_chunks
            )[0]
            self._cleanup_gpu_memory()

            mesh = self.floater_remove_worker(mesh)
            mesh = self.degenerate_face_remove_worker(mesh)
            mesh = self.cleaner_worker(mesh)
            mesh = self.face_reduce_worker(mesh, max_facenum=max_facenum)
            self._cleanup_gpu_memory()
            
            mesh = self.texgen_worker(mesh, input_image)
            self._cleanup_gpu_memory()
            
            output_path = Path("output/mesh.glb")
            mesh.export(str(output_path), include_normals=True)

            if not Path(output_path).exists():
                self._log_analytics_event("predict_error", {"error": "mesh_export_failed"})
                raise RuntimeError(f"Failed to generate mesh file at {output_path}")

            duration = time.time() - start_time
            self._log_analytics_event("predict_completed", {
                "duration": duration,
                "final_face_count": len(mesh.faces),
                "success": True
            })

            return Output(mesh=output_path)
        except Exception as e:
            logger.error(f"Predict failed: {str(e)}")
            logger.error(traceback.format_exc())
            self._log_analytics_event("predict_error", {
                "error": str(e),
                "error_type": type(e).__name__
            })
            raise