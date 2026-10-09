# The UT HPC cluster: a practical guide

A practical guide to the University of Twente EEMCS-HPC cluster as it applies to this thesis. The facts here were
checked on the cluster on 7 October 2026. Where they come from the wiki rather than a test, the text says so.
Official documentation: <https://hpc.wiki.utwente.nl/eemcs-hpc>.

## 1. What it is, in one picture

The cluster is about fifty computers in a server room that many people share. You never sit at any of them. You log
in to a **login node** (the "head node"). From there you hand in **jobs**: a script plus a list of what it needs,
such as "1 GPU, 4 CPU cores, 32 GB of memory, 6 hours". A program called **Slurm** keeps the queue. When a machine
has what you asked for free, Slurm runs your script there, writes everything it prints to a log file, and frees the
machine again when the script ends.

```
  JupyterHub (where you are now)            THE CLUSTER
  ┌──────────────────────┐   ssh    ┌────────────────────────┐
  │ /data/private/...    │ ───────► │ login node             │   you work here:
  │ your data, your A4000│          │ hpc-head1 / hpc-head2  │   edit, copy, submit, check
  └──────────────────────┘          └───────────┬────────────┘
                                                │ sbatch job.sh
                                                ▼
                                    ┌────────────────────────┐
                                    │ Slurm: queue and       │   decides who runs where
                                    │ scheduler              │   and when
                                    └───────────┬────────────┘
                      ┌─────────────────────────┼─────────────────────────┐
                      ▼                         ▼                         ▼
               ┌─────────────┐          ┌─────────────┐          ┌─────────────┐
               │ hpc-node29  │          │ hpc-node30  │   ...    │ ctit084 ... │   compute nodes:
               │ 2 x 96GB GPU│          │ 2 x 96GB GPU│          │ 4 x GPU     │   your job runs here,
               └─────────────┘          └─────────────┘          └─────────────┘   you never log in
                      └──────────── all of them see the same /home and /projects ─────────┘
```

**So is it like a computer you connect to?** Partly. The login node behaves like an ordinary Linux machine
reached over SSH: you get a shell, you can edit files, run `git`, copy data and install packages. The difference is
that the login node is only the front desk. **Nothing heavy runs on it**: no training, no encoding, no large data
processing. The heavy work goes into a job, and the job runs somewhere else, later, without you watching.

## 2. How it differs from the JupyterHub

| | JupyterHub (this machine) | UT HPC cluster |
|---|---|---|
| What you get | One container that is yours, with one GPU | A shared pool of about 50 machines |
| How you run code | Directly, in a terminal or notebook | Write a job script, submit it, wait for it to start |
| When it starts | Immediately | When resources free up: seconds, minutes, or hours on a busy day |
| GPU | 1 x RTX A4000, 16 GB, 57 bf16 TFLOP/s measured | 2 x RTX PRO 6000 Blackwell per ITC node, 96 GB each, 276 bf16 TFLOP/s measured |
| Interactive work | Normal | Only short sessions (about 1 hour) through `sinteractive` |
| Your data | `/data/private`, 115 TB drive, chips and caches already there | **Not visible from the cluster.** Data must be copied over or rebuilt there |
| Python environment | Poetry env with Python 3.11 | Must be built separately; no Python 3.11 module exists (see section 8) |
| Internet | Direct | Login node: direct. Compute nodes: only through the UT proxy |
| Runs in parallel | One thing at a time on one GPU | Up to 8 jobs at once (your limit) on different GPUs |

The two systems are **completely separate computers with separate files**. Nothing you do on one appears on the
other unless you copy it.

## 3. Connecting

The cluster only accepts connections from UT addresses (`130.89.*`). The JupyterHub has one, so from here it works
directly. From your laptop you need to be on campus or on [eduVPN](https://utwente.nl/vpn).

**From the JupyterHub** (key-based login, no password, already set up):

```bash
ssh utwente-hpc                 # login node hpc-head2
ssh hpc-head1                   # the other login node, same files
ssh utwente-hpc 'squeue -u $USER'   # run a single command and come back
```

This works because of an entry in `~/.ssh/config` on the JupyterHub that uses the key
`~/.ssh/id_ed25519_utwente_hpc`, which you installed on the cluster with `ssh-copy-id`.

**From the laptop:** `ssh s3598535@hpc-head2.ewi.utwente.nl` with your UT password. To skip the password, install
a key from the laptop with `ssh-copy-id` in the same way.

**Copying files** (run these on the JupyterHub):

```bash
rsync -avP local_dir/ utwente-hpc:~/target_dir/          # JupyterHub -> cluster, resumable
rsync -avP utwente-hpc:~/results/ local_results/          # cluster -> JupyterHub
```

Measured speed from the JupyterHub is about **110 MB/s**, roughly 400 GB per hour. For this project that means:

| Data | Size | Copy time |
|---|---|---|
| EE 2021 chip set | 284 GB | about 45 min |
| TerraMind v1 large feature cache | 744 GB | about 2 h |
| Prithvi-EO-2.0 600M feature cache | 1.2 TB | about 3 h |

## 4. Where files live

Every machine in the cluster sees the same shared folders, so a file you write on the login node is there when your
job starts on a compute node.

| Location | What it is for | Size and access | Lifetime |
|---|---|---|---|
| `/home/s3598535` | Your personal space: code, environments, job scripts, logs, small results | **1 TB**, only you | As long as your account exists |
| `/projects/itc/<group>` | Shared working space for a research group: large data, shared results | 100 TB pool. `tech` belongs to group `itc-tech-members`, which **you are not in yet**, so you cannot write there | Managed by the group |
| `/datasets/itc/<group>` | Large, static, read-mostly datasets | 159 TB pool, read-only for you | Managed by the group |
| `/local/<jobid>` | Fast scratch disk **inside the compute node**, for the duration of one job | About 6.4 TB free NVMe on each ITC GPU node | Deleted by your job script when the job ends. Not visible from other nodes |

Important points:

- **Nothing is backed up.** The wiki is explicit that the cluster is not for archiving. Your home has hourly ZFS
  snapshots in `~/.zfs/snapshot/` (three were present when checked), so a file deleted in the last few hours can be
  recovered from there. That is a safety net, not a backup. Keep the master copy of anything important on the
  JupyterHub drive or in git.
- **When your account ends, your data is deleted.**
- `/local` is the place for data a job reads intensively, such as a feature cache: copy it to `/local/$SLURM_JOB_ID`
  at the start of the job, read it from there at NVMe speed, and copy results back to `/home` before the job ends.
- **1 TB is not enough for the full thesis data.** The chips plus the two large caches are about 2.2 TB. A project
  directory, or membership of `itc-tech-members`, has to be requested from the ITC contact person.

## 5. Partitions: which machines you can use

A **partition** is a named group of machines with its own access rules. You pick one with `-p` when you submit. You
may also list several (`-p itc-gpu,main-gpu`) and Slurm uses whichever frees up first.

Your Slurm account is **`itc-tech`**. That gives you the ITC partitions and the general ones:

| Partition | Machines | Hardware per machine | Use it for |
|---|---|---|---|
| **`itc-gpu`** | hpc-node29, hpc-node30 | 2 x RTX PRO 6000 Blackwell 96 GB, 32 cores, 256 GB RAM, 7 TB local NVMe | **Main choice** for encoding and training. Only ITC users, so the queue is short |
| `itc-cpu` | hpc-node27, 28, 32 | No GPU, 2 x AMD EPYC 9654 (192 cores), 1.9 TB RAM | Heavy CPU work: chip building, rasterising, statistics |
| `main-gpu` | 26 machines (ctit084 to 094 and hpc-node01 to 18, with a few gaps) | 1 to 4 GPUs each: Quadro RTX 6000 24 GB, A40 48 GB, L40 or L40S 48 GB | Fallback when `itc-gpu` is full. Open to everyone, so busier |
| `main` | the above plus CPU machines | mixed | The default when you name no partition |
| `main-cpu` | 6 CPU machines | no GPU | Light CPU jobs |

Other partitions you see in `sinfo` (`dmb`, `mia`, `tfe-gpu`, `students` and so on) belong to other groups and will
reject your jobs. `students` is only for EEMCS course work.

Choosing a GPU type inside a partition: `--gres=gpu:1` takes any GPU, `--gres=gpu:blackwell:1` asks for a Blackwell
card, and `--constraint=a40` (or `l40s`, `rtx6000pro`) asks for a specific model.

## 6. Limits

| Limit | Value | Where it comes from |
|---|---|---|
| Jobs running at the same time | **8** | Your QoS `research` |
| Jobs in the queue (running plus waiting) | **100** | Your QoS `research` |
| GPUs per job | No QoS cap; physically 2 per `itc-gpu` node, 4 on most `main-gpu` nodes | Hardware |
| Run time per job | **24 h by default** if you do not set `--time`. The partitions set no maximum, so longer can be asked for | Partition settings |
| Interactive sessions | About 1 hour, by the cluster's rules | Wiki |
| Home space | 1 TB | Storage quota |
| Work on the login node | **Not allowed** beyond editing, copying, installing and submitting | Wiki rule |
| Internet from jobs | Only via `proxy.utwente.nl:3128` | Network setup |
| Being kicked off | Never: preemption is off, so a running job keeps its resources until it ends or hits its time limit | Slurm config |

Two things matter in practice:

- **Memory is shared, not reserved.** Slurm here hands out cores and GPUs only, so jobs on the same node share its
  RAM. Still set `--mem` to a realistic value: it makes Slurm place the job on a node that has that much memory.
- **Ask for a realistic `--time`.** Slurm uses backfill: a short job can jump into a gap before a long one starts. A
  job that asks for 2 hours starts much sooner than one that asks for 24.

Your priority in the queue drops as you use more resources (fair share), and recovers over time.

## 7. Day to day

### Submitting a batch job (the normal way)

Write a script whose `#SBATCH` lines state what it needs, then hand it in with `sbatch`. This template is based on
the test job that ran successfully:

```bash
#!/bin/bash -l
#SBATCH -J my-run                      # short name shown in the queue
#SBATCH -p itc-gpu                     # partition (itc-gpu,main-gpu to accept either)
#SBATCH --gres=gpu:1                   # one GPU
#SBATCH -c 8                           # CPU cores (data loading workers)
#SBATCH --mem=64G                      # memory
#SBATCH --time=06:00:00                # time limit; the job is killed when it is reached
#SBATCH --output=logs/%x_%j.log        # %x = job name, %j = job id; the logs/ folder must exist
#SBATCH --mail-type=END,FAIL           # optional e-mail when it finishes or fails
#SBATCH --mail-user=<your UT e-mail>

# Compute nodes reach the internet only through the UT proxy
export HTTP_PROXY=http://proxy.utwente.nl:3128 HTTPS_PROXY=http://proxy.utwente.nl:3128
export http_proxy=$HTTP_PROXY https_proxy=$HTTPS_PROXY

# Fast local scratch, removed whenever the job ends (also on failure or cancel)
Scratch="/local/${SLURM_JOB_ID}"
mkdir -p "$Scratch"; trap 'rm -rf "$Scratch"' EXIT

nvidia-smi
python my_script.py
```

Two details in the first line and the proxy block are not optional. `#!/bin/bash -l` makes the `module` command
available inside the job. Without the proxy, any download from a job times out.

```bash
sbatch job.sh                 # submit; prints the job id
squeue -u $USER               # my jobs: PD = waiting, R = running; the last column says why it waits
scancel <jobid>               # cancel one job
tail -f logs/my-run_<jobid>.log   # follow the output while it runs
sacct -j <jobid> -o JobID,State,Elapsed,MaxRSS,NodeList   # what happened, after it ends
seff <jobid>                  # how much of the requested CPU and memory the job actually used
```

### An interactive session (short tests only)

```bash
module load slurm/utils
sinteractive -p itc-gpu --gres=gpu:1 -c 4 --mem 32G --time 60
```

This gives you a shell on a compute node with a GPU for up to 60 minutes, which is useful for checking that an
environment works before submitting a long job. Leave it with `exit` so the GPU goes back to the pool.

### Looking at the cluster

```bash
sinfo -p itc-gpu,main-gpu                 # partitions: idle, mix (partly used), alloc (full)
squeue -p itc-gpu                         # who is using and waiting for the ITC GPUs
sshare -U                                 # your fair-share standing
```

Web dashboard (from a UT network): <http://hpc-status.ewi.utwente.nl/slurm>.

### Long-running work on the login node

Copies and installs that take long should survive a dropped SSH connection. Run them inside `tmux` (available on
the login node): `tmux new -s copy`, start the command, detach with `Ctrl-b d`, reattach later with
`tmux attach -t copy`.

## 8. Software

The machines start almost empty. Software comes from **modules**, which you load per session or per job:

```bash
module avail                     # list everything
module load python/3.13.14       # example
module load nvidia/cuda-12.4     # CUDA toolkit, only needed to compile CUDA code
module list                      # what is loaded now
```

Relevant for this project:

- **Python:** modules `python/3.10.7` and `python/3.13.14`, plus the system Python 3.10.12, and
  `anaconda3` / `miniconda3` modules. **There is no Python 3.11**, which the repository uses, so the environment
  will need conda (`miniconda3/25.7`) or a standalone Python 3.11.
- **PyTorch:** pip wheels bring their own CUDA runtime, so no CUDA module is needed. `torch 2.11.0+cu128` was
  installed in 51 s inside a test job and ran on the Blackwell GPU. The driver on the ITC nodes is 595.58 and supports
  CUDA up to 13.2.
- **Containers:** `singularity/3.x` modules exist. Apptainer and Docker are not available.
- **Installing:** into your home (a virtual environment, or `pip install --user`). Software that needs root goes
  through the cluster admins.

## 9. Running this project's workflow

The JupyterHub develops the code, extracts the chips and analyses the results. The cluster runs the K-shot workflow
(`scripts/run_kshot.py`): feature encoding and decoder fitting. **Code is never edited on the cluster.** Every change
goes JupyterHub, git, cluster, so the two machines cannot drift apart. The repository is public, so the cluster
clones and pulls over HTTPS without a key; a read-only deploy key, `~/.ssh/id_ed25519_github`, is ready on the
cluster for the day the repository is made private.

The clone mirrors the JupyterHub's layout, so every path in a configuration or a result file reads the same on both:

```
~/ExplainedGMF4Agri/                    clone of github.com/davidrers/ExplainedGMF4Agri
  data/eurocrops_chips/EE_2021/         copied from the JupyterHub, 284 GB
  data/eurocrops_chips/EE_2021_mini/    the pilot chip set, links into EE_2021
  data/eurocrops/{parquet,vector}/      copied from the JupyterHub, about 3.6 GB
  results/                              written by the jobs
```

**One-time set-up**, on the login node, from the clone: `bash -l scripts/cluster/setup_env.sh`. It creates a
Python 3.11 environment with the `miniconda3/25.7` module (from conda-forge), installs Poetry 2.2.1, runs
`poetry install` from the same lock as the JupyterHub, installs THOR with `scripts/env/install_thor.sh`, downloads
the encoder weights on the login node, where the internet is direct, and ends with `pytest`.

**The loop:**

| Step | Where | Command |
|---|---|---|
| 1. Change code, run `pytest`, commit, push | JupyterHub | git |
| 2. A new chip set: build it, then copy it | JupyterHub | `bash scripts/cluster/push_data.sh eurocrops_chips/<set> ...` (rsync, resumable, checks the free space first, leaves logs and pid files behind) |
| 3. Update | Cluster | `git pull`, and `poetry install` if the lock changed |
| 4. Run | Cluster | `bash scripts/cluster/submit.sh kshot <set> [arms]` for the main workflow, or `submit.sh experiments/<folder>/experiment.yaml <set>` for an experiment, one job per arm; `DRY_RUN=1` prints the `sbatch` commands; `TIME`, `CPUS` and `MEM` override the defaults of 2 days, 32 CPUs and 120 GB |
| 5. Bring the results back | JupyterHub | `bash scripts/cluster/pull_results.sh kshot [<set>]`; `DEST=` puts them elsewhere than `results/` |

Steps 3 to 5 can be driven from the JupyterHub with `ssh utwente-hpc '<command>'`. `squeue -u $USER` shows the jobs;
their logs are in `results/<experiment>/<set>/logs/<arm>_<job id>.log`.

**One job** (`scripts/cluster/kshot.sbatch`) runs one arm on `itc-gpu` with one GPU. It sets the UT proxy, creates
`/local/$SLURM_JOB_ID` on the node's NVMe, and runs the workflow with that directory as scratch. A feature cache
found under `data/embeddings/` in the clone is used as it is; a missing one is computed into `/local`, serves the
arm's whole sweep and is deleted when the job ends, fails or is cancelled. Each cell writes its `results.json` and its
test predictions to the home results directory as it finishes, so a resubmitted job skips the finished cells.

## 10. Open points

1. **Storage:** request a project directory under `/projects/itc` (or membership of `itc-tech-members`). The chips
   and one token-grid cache do not fit together in the 1 TB home, so every job encodes its arm again; a project
   directory linked to `data/embeddings/` in the clone would keep the caches between jobs.
