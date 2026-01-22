Quick Run Down

1.0: Create your conda environment:
```bash
conda create -n autoeval python=3.10 -y
conda activate autoeval
pip install -r requirements.txt
pip install imageio[ffmpeg] datetime
```

2.0: install ngrok:
Linux: https://ngrok.com/docs/guides/device-gateway/linux#1-install-the-ngrok-agent

MacOS: https://ngrok.com/docs/getting-started

2.1: get your ngrok token and authenticate following:
https://dashboard.ngrok.com/get-started/your-authtoken

3.0: Set policy configuration in `scripts/simplevla_rl_server/run_simplevla_rl_server.sbatch`:
Set `RL_CHECKPOINT` to the checkpoint directory of the RL model;
Set `SFT_CHECKPOINT` to the checkpoint directory of the base model;
Set `AUTO_EVAL_DIR` to the directory of this codebase.

4.0: run the server:
With SLURM:
```bash
sbatch scripts/simplevla_rl_server/run_simplevla_rl_server.sbatch
```
Then ssh to the allocated GPU.

Without SLURM:
Run what is inside `scripts/simplevla_rl_server/run_simplevla_rl_server.sbatch` directly in terminal.

4.1: run ngrok in the background:
```bash
nohup ngrok http 8000 > ngrok.log 2>&1 &
```
To check the IP address, run
```bash
ngrok http 8000
```
At this point you either see an interface popping up or error messages saying it is already online.
Both are fine. Find the link with the work `unadorned` in it, and that is the IP address.

4.2: Open the link to test connection. If it shows a page with `{"detail":"Not Found"}`, it is connected. If it shows a fancy page telling you there is an error, then the server is not running properly.

4.3 In case the server is not running properly, check the logs in `logs/simplevla_server_%j.err` first. If there is no error, use
```bash
ps aux | grep ngrok
```
to check running ngrok processes, and kill the first one with 
```bash
kill <PID>
``` 
Then return to step 4.1.