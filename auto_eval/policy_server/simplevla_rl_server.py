"""
FastAPI server for SimpleVLA-RL policy with action chunking and temporal ensembling.

Usage:
    python auto_eval/policy_server/simplevla_rl_server.py \
        --rl_checkpoint /path/to/rl/checkpoint \
        --sft_checkpoint /path/to/sft/checkpoint \
        --port 8000

Then expose via bore.pub:
    bore local 8000 --to bore.pub
"""

import os
import json_numpy
json_numpy.patch()
import logging
import traceback
from dataclasses import dataclass
from typing import Any, Dict
import json

import draccus
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import base class for action chunking
from auto_eval.policy_server.template_advanced import ActionChunkingObsHistoryPolicyServer

# Import SimpleVLARLPolicy from auto_eval
from auto_eval.robot.policy import SimpleVLARLPolicy

logging.basicConfig(level=logging.INFO)


class SimpleVLARLServer(ActionChunkingObsHistoryPolicyServer):
    def __init__(self, rl_checkpoint: str, sft_checkpoint: str):
        """Initialize server with SimpleVLARLPolicy and action chunking"""

        # Initialize base class with action chunking parameters
        # BRIDGE mode predicts 5 action chunks
        super().__init__(
            obs_horizon=1,              # No observation history
            action_pred_horizon=5,      # Predict 5 action chunks
            action_temporal_ensemble=True,  # Enable temporal ensembling
            action_exp_weight=0.0,      # Uniform weighting for temporal ensemble
        )

        # Set environment variable for BRIDGE (5 action chunks)
        os.environ["ROBOT_PLATFORM"] = "BRIDGE"

        # Create policy config with return_all_chunks=True
        policy_config = {
            "rl_checkpoint_path": rl_checkpoint,
            "sft_checkpoint_path": sft_checkpoint,
            "temperature": 1.6,
            "do_sample": True,
            "unnorm_key": "bridge_dataset",
            "return_all_chunks": True  # Return all 5 chunks for temporal ensembling
        }

        # Set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Initialize policy
        logging.info(f"Loading SimpleVLA-RL policy on {self.device}...")
        self.policy = SimpleVLARLPolicy(config=policy_config, device=self.device)

        logging.info(f"✅ SimpleVLA-RL Server initialized on {self.device}")
        logging.info(f"RL checkpoint: {rl_checkpoint}")
        logging.info(f"SFT checkpoint: {sft_checkpoint}")
        logging.info(f"Action chunking: 5 chunks with temporal ensembling")

    def predict_action_chunk(self, obs_dict: dict, instruction: str):
        """
        Predict action chunk for given observation with temporal ensembling.

        Args:
            obs_dict: Dict with "image" key containing (256, 256, 3) uint8 array
            instruction: Language instruction string

        Returns:
            action: numpy array of shape (7,) after temporal ensembling
        """
        # Create obs_dict matching policy expectations
        policy_obs_dict = {
            "image_primary": obs_dict["image"]
        }

        # Run policy inference - returns (5, 7) action chunks
        action_chunks = np.asarray(self.policy(policy_obs_dict, instruction))
        if action_chunks.ndim > 1 and action_chunks.shape[0] == 1:
            action_chunks = action_chunks[0]

        # Check if actions are chunked
        if len(action_chunks.shape) > 1:
            assert action_chunks.shape[0] == self.action_pred_horizon, \
                f"Expected {self.action_pred_horizon} chunks, got {action_chunks.shape[0]}"

            # Apply temporal ensembling if enabled
            if self.action_temporal_ensemble:
                action = self._apply_temporal_ensembling(action_chunks)
            else:
                # Just return the first chunk
                action = action_chunks[0]
        else:
            # Single action (shouldn't happen with return_all_chunks=True)
            action = action_chunks

        # Validate final action shape
        assert action.shape == (7,), f"Invalid action shape {action.shape}"

        return action

    def reset(self):
        """Reset server state (observation and action history)"""
        # Reset observation and action history from base class
        super().reset()
        # SimpleVLA-RL policy itself is stateless
        logging.info("Server reset: cleared observation and action history")


###############################################################################
# FastAPI Setup
###############################################################################
app = FastAPI()
server: SimpleVLARLServer = None


@app.on_event("startup")
def load_model_on_startup():
    """Load the SimpleVLARLServer once when the server starts"""
    global server

    # Get checkpoint paths from environment or use defaults
    rl_ckpt = os.environ.get(
        "SIMPLEVLA_RL_CHECKPOINT",
        "/projects/work/yang-lab2/as20482/checkpoints/SimpleVLA-RL/SimpleVLA-RL/openvla_worldgym_bridge_oft_ckpt20k_5chunk/actor/global_step_64"
    )
    sft_ckpt = os.environ.get(
        "SIMPLEVLA_SFT_CHECKPOINT",
        "/projects/work/yang-lab2/as20482/checkpoints/SimpleVLA-RL/openvla-oft-bridge-sft-20000"
    )

    logging.info("[Server Startup] Loading SimpleVLARLServer with config:")
    logging.info(f"  RL checkpoint:  {rl_ckpt}")
    logging.info(f"  SFT checkpoint: {sft_ckpt}")

    server = SimpleVLARLServer(
        rl_checkpoint=rl_ckpt,
        sft_checkpoint=sft_ckpt
    )

    logging.info("[Server Startup] Server loaded successfully")


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/reset")
def reset_server():
    """Reset server state"""
    global server
    if server is None:
        return {"error": "Server not ready"}
    server.reset()
    return {"status": "reset successful"}


@app.post("/act")
def act(payload: dict = Body(...)):
    """Predict action from observation and instruction"""
    global server

    if double_encode := "encoded" in payload:
        assert len(payload.keys()) == 1, "Only uses encoded payload!"
        payload = json.loads(payload["encoded"])

    logging.info("[Server] Received POST request at /act")

    if server is None:
        raise HTTPException(status_code=503, detail="Server not ready; model failed to load")

    # Validate required fields
    if "image" not in payload or "instruction" not in payload:
        raise HTTPException(
            status_code=400,
            detail="Missing required fields: image and instruction"
        )

    # Extract and validate inputs
    image = np.array(payload["image"])
    instruction = payload["instruction"]

    if image.shape != (256, 256, 3):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image shape {image.shape}, expected (256, 256, 3)"
        )

    # Create obs_dict
    obs_dict = {"image": image}

    try:
        # Predict action with temporal ensembling
        action = server.predict_action_chunk(obs_dict, instruction)

        # Return response
        if double_encode:
            return JSONResponse(json_numpy.dumps(action))
        else:
            return JSONResponse(action.tolist())

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        logging.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


###############################################################################
# Configuration dataclass
@dataclass
class DeployConfig:
    # Checkpoint paths
    rl_checkpoint: str = "/projects/work/yang-lab2/as20482/checkpoints/SimpleVLA-RL/SimpleVLA-RL/openvla_worldgym_bridge_oft_ckpt20k_5chunk/actor/global_step_64"
    sft_checkpoint: str = "/projects/work/yang-lab2/as20482/checkpoints/SimpleVLA-RL/openvla-oft-bridge-sft-20000"

    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000


# Main entry point
@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    # Set environment variables for startup function
    os.environ["SIMPLEVLA_RL_CHECKPOINT"] = cfg.rl_checkpoint
    os.environ["SIMPLEVLA_SFT_CHECKPOINT"] = cfg.sft_checkpoint

    # Run uvicorn server
    logging.info(f"🚀 Starting server on {cfg.host}:{cfg.port}")
    uvicorn.run(app, host=cfg.host, port=cfg.port)


if __name__ == "__main__":
    deploy()
