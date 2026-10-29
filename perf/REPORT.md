# Optimizing CorridorKey 2.0 inference

At the same Hann window, `--optimized` writes the same bytes as the original
CorridorKey 2.0 runtime, alpha and foreground, with much less GPU memory and
in less time. On an RTX 5080, a 120-frame 4K green-screen shot (EXR frames,
8-frame windows) keys in 3.9 minutes with `--fast` instead of 12.1, and the
GPU peaks at 4.0 GiB instead of 11.4. On a 6 GB RTX 3060 Laptop, 1080p runs 40-frame windows
instead of 8 and still finishes sooner.

The model and its weights are unchanged. Every option is opt-in and keeps the
output byte-identical on its own, and without the flags the runtime takes the
original code path. The raw JSON, logs and comparison tables behind every
number are in `perf/results/`.

## Results

### Test clips

| clip | what it is | source |
|---|---|---|
| Tears of Steel 04_5h | a live-action green-screen shot at 4096x2160, [linear EXR frames](https://media.xiph.org/mango/tearsofsteel-footage-exr/04_5h/linear/) | [Tears of Steel](https://mango.blender.org/), Blender Foundation's open movie, (CC) |
| hair_detail | a close-up with fine, flyaway hair, 1280x1080 | [Hollywood Camera Work](https://www.hollywoodcamerawork.com/green-screen-plates.html) green-screen plate |
| motion_blur | a fast arm swing with heavy motion blur, 1280x1080 | [Hollywood Camera Work](https://www.hollywoodcamerawork.com/green-screen-plates.html) green-screen plate |
| converge | a person walking behind a foreground pole, at 1080p and 1440p | [Hollywood Camera Work](https://www.hollywoodcamerawork.com/green-screen-plates.html) green-screen plate |

The quality checks use five more Hollywood Camera Work plates: smoke, tinted
glasses, translucent fabric, hands with a green cup, and a wide cyc with set
pieces.

### RTX 5080

Whole runs, model load included, with a 12 GB memory budget and alpha and
foreground written as video (`perf/results/phase6/box2/`).

| clip | original | `--optimized` | `--fast` |
|---|---|---|---|
| 4K green-screen shot (Tears of Steel), 120 EXR frames, 8/4 low-VRAM | 724.3 s | 259.3 / 262.8 / 264.8 s | 235.4 / 236.1 s |
| hair close-up (hair_detail), 1280x1080, 80 frames, automatic window | 60.9 s at 48/24 low-VRAM | 26.5 / 26.8 s at 80/40 | 24.8 / 24.6 s at 80/40 |
| arm swing with motion blur (motion_blur), 1280x1080, 50 frames, automatic window | 38.5 s at 80/40 low-VRAM | 18.1 / 18.0 s at 80/40 | 17.1 / 17.3 s at 80/40 |

8/4 means 8-frame Hann windows with a stride of 4. Peak GPU memory on the 4K
clip, from nvidia-smi: 11.4 GiB for the original and 4.0 GiB for `--fast`.
At 4K all three run the same window and compute the same key (on this card
the original's own output varies slightly from run to run with the carry hint
on, see "When the original does not repeat itself"). At HD each version picks
its own window for the budget, and the optimized runs fit 80/40 in normal
mode, where the original needs low-VRAM mode for 48/24.

`--fast` is `--optimized` with more x264 threads. The frames it hands the
encoder are identical, but x264's output depends on its thread count, so the
video files differ slightly from a two-thread encode.

### RTX 3060 Laptop, 6 GB, Windows 11

The walking shot (converge), 48 frames, automatic settings, model load
included (`perf/laptop_run.py`, `perf/results/phase4/laptop_e2e_table.txt`):

| clip | original | `--optimized` | `--fast` |
|---|---|---|---|
| 1080p | 114.4 s at 8/4 | 89.1 s at 40/20 | 88.2 s |
| 1440p | 205.1 s at 4/2 | 161.6 s at 28/14 | 156.2 s |

Peak reserved memory barely moves (4.75 -> 4.63 GiB at 1080p, 4.24 -> 4.38
GiB at 1440p) while the windows get five and seven times longer.

### Memory per window

Peak allocated memory of a low-VRAM stage-1 window, measured with
`perf/vram_table.py` on an RTX 6000 Ada:

| window | original | `--optimized` |
|---|---|---|
| 1080p, 24 frames | 7.69 GiB | 2.76 GiB |
| 1080p, 48 frames | 13.20 GiB | 4.65 GiB |
| 1440p, 32 frames | 12.59 GiB | 4.22 GiB |
| 4K, 24 frames | 20.78 GiB | 5.67 GiB |
| 4K, 48 frames | 39.66 GiB | 10.75 GiB |

Each extra frame of window costs 81 MiB instead of 235 at 1080p, and 217 MiB
instead of 806 at 4K. The fixed part, model included, went from about 2.8 to
1.2 GiB at 1080p.

### What auto mode picks

Windows picks for a given amount of free GPU memory, with Linux in brackets
where it differs. L is low-VRAM mode, N normal mode. On a 6 GB card the
original picks 8L at 720p and 1080p, 4L at 1440p and 2L at 3K and 4K.

| free memory | 720p | 1080p | 1440p | 3K | 4K |
|---|---|---|---|---|---|
| 4.95 GiB (6 GB card) | 48L | 40L | 28L (32L) | 16L (24L) | 10L (16L) |
| 6.95 GiB (8 GB) | 64L (80L) | 56L (64L) | 40L (48L) | 24L (40L) | 16L (24L) |
| 10.95 GiB (12 GB) | 80N | 80L | 64L (80L) | 40L (64L) | 28L (40L) |
| 14.95 GiB (16 GB) | 80N | 80N | 80L | 64L (80L) | 40L (56L) |
| 22.95 GiB (24 GB) | 80N | 80N | 80N | 80L | 64L (80L) |

Every one of these picks ran end to end with its peak under the budget
("Auto mode's table and margins" below).

## Using it

```
python infer.py --input clip.mp4 --output_dir out --optimized
python infer.py --input clip.mp4 --output_dir out --fast
```

- With `--hann_chunk` and `--hann_stride`, `--optimized` runs exactly that
  window and writes the original's files.
- Without them it picks the window from its own measured table, so the
  memory it saves goes into longer windows, and longer windows change the
  key ("Longer windows" below).
- `--fast` gives each video encoder half the CPU cores, up to 8. On fast GPUs
  at HD the encoder often sets the pace and `--fast` is 5-10% faster. At 4K
  on a 16-core CPU the extra encoder threads competed with frame preparation
  and made it slower (322 vs 299 s), so measure on your machine.
- In the launcher, "Auto, optimized" is the default memory profile. "Auto,
  original" keeps the shipped settings, and "Manual" asks whether to use the
  optimized engine.

`--optimized` turns on the options below, and each also works on its own:

| option | what it does |
|---|---|
| `--bf16_weights` | Linear and Conv weights stored as bf16, all weights frozen |
| `--lean_activations` | stage 1 without padded or duplicate activation copies |
| `--lean_foundations` | frees the parts of MoGe that inference never runs |
| `--activation_arena` | one allocator segment for each window's working memory (Windows) |
| `--prefetch_frames` | reads the next window while the GPU works |
| `--cache_region_indices` | keeps shifted-window index tensors on the GPU |
| `--lean_reads` | skips display views that only the outputs need |
| `--read_ahead` | reads stage 1's first window during the prepass, later frames on worker threads |
| `--pinned_io` | quantizes outputs on the GPU and copies them without stalling it |
| `--fast_load` | builds the model without random initialization |
| `--decode_once` | decodes and converts each frame once |
| `--gpu_prep` | runs frame-folder color conversions on the GPU |
| `--async_baton` | keeps low-VRAM stage-1 windows back to back on the GPU |

It also turns on expandable allocator segments on Linux and reads the auto
window from `_VRAM_PEAK_GB_OPTIMIZED`.

## How I checked the output

The bar for every option is byte-identical output at the same window. That
bar can be checked because the original is deterministic in most setups: on
the first test machine (an RTX 3090), rerunning the test clips at 1080p and
720p, and rendering one clip alone while others shared the GPU, matched bit
for bit (`perf/results/phase1/`). So any difference after a change comes from
the change.

- `perf/same_files.py` compares every output file of two runs byte for byte,
  EXR and video, all four outputs.
- `perf/goldens.py` and `perf/compare.py` render reference EXRs of the test
  clips and compare alpha and foreground, overall, in the edge band and as
  flicker.
- Where work moved from the CPU to the GPU, I checked every input the
  conversion can see:
  - `perf/quantize_check.py`: the GPU's 16-bit output quantization equals
    the CPU's for all 1,065,353,217 float32 values in [0, 1].
  - `perf/gpu_prep_probe.py`: GPU and CPU frame conversion give the same bf16
    bits for every 8-bit value and every finite half float, and the CPU's
    vectorized and scalar loops agree, so the result does not depend on the
    thread count. 32-bit float EXRs are different: 778 + 693 of the float32
    patterns in [-65504, 65504] round one bf16 step apart, so frames whose
    EXR header stores R, G or B as 32-bit float stay on the CPU.
  - `perf/results/phase6/scripts/bf16_cast_check.py`: all 2,130,706,434
    float32 values in [-1, 1] convert to bf16 identically on both.
  - `perf/load_probe.py --verify`: with `--fast_load`, all 1,815 parameters
    and buffers are bitwise equal to a normal load.
- Each stage-1 memory change was also switched off on its own against the
  rest with `perf/lean_probe.py` (stage-1 forward on random inputs): bitwise
  equal at every size.

### Across GPU architectures

None of the options adds a CUDA kernel. They take copies off the critical
path and move standard PyTorch operations between the CPU and the GPU. What
can change from one GPU to the next is which kernels PyTorch and cuDNN pick,
and whether those give the same bits twice. So each card ran the same check
(`perf/results/phase6/scripts/arch_check.sh`): 24 frames from the middle of
the hair close-up at 8/4 low-VRAM, float EXR alpha and foreground, 48 files.

1. `perf/gpu_prep_probe.py`, which exits 1 on any mismatch.
2. The original twice and `--optimized` once with deterministic cuDNN
   (`perf/results/phase6/scripts/det_run.py` sets
   `torch.backends.cudnn.deterministic = True` and `benchmark = False`, then
   runs `infer.py` unchanged).
3. The original and `--optimized` once each with PyTorch's defaults.

PyTorch 2.9.1+cu128 throughout. Its kernels are built for sm_70, sm_75,
sm_80, sm_86, sm_90, sm_100 and sm_120; Ada (sm_89) uses the sm_86 build.

| GPU | architecture | probe | deterministic cuDNN: original vs itself, vs `--optimized` | defaults: original vs `--optimized` |
|---|---|---|---|---|
| Tesla V100 SXM2 32 GB | Volta, sm_70 | 0 mismatches | 48/48, 48/48 | 48/48 |
| Quadro RTX 6000 | Turing, sm_75 | 0 mismatches | 48/48, 48/48 | 48/48 |
| A100 SXM4 40 GB | Ampere, sm_80 | 0 mismatches | 48/48, 48/48 | 48/48 |
| RTX 3060 Laptop, Windows | Ampere, sm_86 | 0 mismatches | not run | 96/96 at 1080p 8/4 (`perf/results/phase6/laptop/laptop_final.log`) |
| RTX 4090 | Ada, sm_89 | exit 0, output not saved | 48/48, 48/48 | 0/48, and the original differs from itself as much: alpha MAE 8.30e-5 between two runs, 8.16e-5 against `--optimized` |
| H100 PCIe | Hopper, sm_90 | 0 mismatches | 48/48, 48/48 | 48/48 |
| RTX 5080 | Blackwell, sm_120 | 0 mismatches | not run | with `--no_carry_hint`: 96/96 at 1080p, 32/32 at 4K |

Logs are in `perf/results/phase6/` (`arch_v100/`, `arch_turing/`,
`arch_a100/`, `arch_h100/`, `rtx4090/`, `laptop/`, `box2/tier3p.log`).
Datacenter Blackwell (sm_100) was not run.

Speed on the same check, model load included:

| GPU | clip, window | original | `--optimized` |
|---|---|---|---|
| H100 PCIe | hair close-up, 24 frames, auto 80/40 | 26.4 / 26.4 s | 17.7 / 17.7 s |
| A100 SXM4 40 GB | hair close-up, 24 frames, auto 80/40 | 26.5 / 26.8 s | 16.5 / 16.6 s |
| Quadro RTX 6000 | hair close-up, 24 frames, 8/4 low-VRAM | 81.3 / 76.1 s | 54.3 / 53.5 s |
| Tesla V100 SXM2 32 GB | hair close-up, 24 frames, 8/4 low-VRAM | 52.6 / 54.2 s | 37.7 / 37.5 s |

Volta and Turing are below what the README asks for (RTX 30-series or
newer), and this check shows why. PyTorch has no fused bf16 attention for
sm_70 or sm_75: flash attention needs sm_80, and the memory-efficient kernel
takes only fp16 and fp32 there (`sdpa.txt` in `arch_v100/` and
`arch_turing/`). So every attention call takes the math path and builds its
whole score matrix. At the automatic settings (80/40, normal mode) the
original and `--optimized` both stop on a 23.73 GiB allocation in full
spatial attention, on the 32 GB V100 as on the 24 GB Quadro (`oom.txt` in the
same folders). 8/4 low-VRAM fits, which is where the Volta and Turing timings
come from.

### When the original does not repeat itself

Stage 1's aux head is a ConvTranspose2d, which cuDNN runs as a backward-data
convolution. For larger inputs the algorithm it picks accumulates with
atomics, so the same input gives slightly different bits from run to run,
and the carry hint feeds that aux into every later window. Where it starts
depends on the GPU: on an RTX 3060 from 3K at 48 frames, on an RTX 6000 Ada
from 1440p at 24, and on an RTX 5080 already at 1080p with 8-frame windows
(`perf/deconv_probe.py`). On the RTX 4090 the original also varied under
cuDNN's default algorithms with the carry hint off. On those cards I compare
with the carry hint off, or with deterministic cuDNN, and both versions then
repeat and agree.

## Memory

### Weights

The original loads every weight as fp32 and runs under
`torch.autocast(bfloat16)`. `load_model` leaves `requires_grad` on, and
autocast caches the bf16 cast of every fp32 leaf that requires grad for the
whole autocast region, which in stage 1 is a whole window. At a 1-frame
window that makes the 2.80 GiB floor:

| part | GiB |
|---|---|
| stage-1 weights, fp32 | 1.73 |
| autocast's cached bf16 copies of them | about 0.8 |
| one frame of activations | about 0.2 |

- `--frozen_weights` turns `requires_grad` off, so autocast caches nothing.
- `--bf16_weights` also stores Linear and Conv weights and biases as bf16.
  Norms, scale parameters and buffers stay fp32. Storing bf16 weights with
  `requires_grad` still on is not exact: one CSwin QKV projection then
  returns different bits for identical inputs and weights. So the flag
  always freezes too.
- `--lean_foundations` frees MoGe's neck and four heads, which inference
  never runs (only its encoder is called): 65 MiB, or 32 MiB with bf16
  weights.

Stage-1 forward at 1080p on random inputs (`perf/floor_probe.py`), peak
allocated GiB, all with identical output:

| window | original | frozen | bf16 weights |
|---|---|---|---|
| 1 frame | 2.80 | 2.03 | 1.15 |
| 5 frames | 3.54 | 3.02 | 2.17 |
| 8 frames | 4.12 | 3.79 | 2.94 |

### Stage-1 activations

Memory snapshots at windows of 4 and 8 frames (`perf/mem_peak.py --diff`) put
everything that grows with the window in stage 1's /16 Swin path. At 1080p
the original's 196 MiB per frame there was:

| MiB/frame | what |
|---|---|
| 80.0 | four padded fp32 copies in `SwinAttention3D`'s low-VRAM window path (72 rows padded to 80): the padded input, unused on that path, the padded gate source, the attention output buffer and the unpartition copy |
| 67.5 | four fp32 /16 tensors, fp32 only because fp32 scale parameters promote the bf16 stream |
| 25.3 | the full-resolution input held for the whole forward, the stem output, and a second permuted copy of it for the late skip |
| 23.5 | the rest: QKV permute copies, the hint, window stacks, features |

`--lean_activations` removes most of it, and every change is exact:

- Batched window attention cuts windows straight from the unpadded input,
  zero-fills only past the bottom and right edges, and writes each batch's
  result straight into the unpadded output.
- The normalized input of every attention, MLP and local-mix branch only
  reaches Linear and Conv layers, which autocast casts to bf16 anyway, and
  copies of itself. Casting it to bf16 once gives those layers the same
  values and halves every copy. The one fp32-sensitive operation, the CSwin
  gate product, still rounds to the same bf16 value: a bf16 x bf16 product is
  exact in fp32, and both paths round it to bf16 once.
- The QKV projection, its depthwise convolution and the 7x7 local mix run
  one frame at a time.
- Residual adds, the MLP gate and the skip connections run in place where the
  runner owns the stream.
- Stage 1 takes its window inputs from a holder it empties, so no input stays
  alive next to its permuted copy.
- The QKV grid is built per window-batch band instead of for the whole frame.
- The late skip waits in registered host memory, in an exact-size buffer
  (`corridorkey/pinned.py`), and its copies there and back do not block.
- The stem patchifies one frame at a time. The whole-window call needed cuDNN
  layout buffers 2.5 times the size of its input (1.0 GiB at 1080p with 24
  frames).
- The CSwin strip intermediates are freed as they are consumed, which cut
  the stage-1 peak by 14 to 17% from 1440p up.

The "Memory per window" table above is the result.

### The prepass

The low-VRAM prepasses (RVM, then C-RADIO and MoGe) decoded the clip twice
and kept full-size frames. Now one read feeds both, RVM's input is downscaled
one frame at a time, and the prepass keeps only the downscaled frames the
foundation models read. At 4K and 8/4 the prepass peak went from 3.19 GiB to
about 1.5 GiB (`perf/results/phase6/box2/tier3i/`, `tier3q/`).

### The allocator

PyTorch's caching allocator sizes each cudaMalloc for the request that
caused it, and freed memory merges only inside its own segment. With the
window's large tensors that left reserved memory far above allocated: at
1080p with all exact options the gap was about 1.8 GiB, and 20- and 24-frame
windows failed on single requests of 600 and 812 MiB.

- On Linux `--optimized` turns on expandable segments, unless
  `PYTORCH_CUDA_ALLOC_CONF` is already set, and the gap drops to about 0.3
  GiB.
- Windows builds of PyTorch have no expandable segments, and the Windows
  driver spills to system RAM before an allocation fails, so nothing ever
  makes the cache flush. `--activation_arena` allocates and frees one block
  of the window's working size before stage 1. The window's large tensors
  then come out of that one segment, where blocks freed together merge back.
  In normal mode, reserved memory went from 1.12-1.29 times the table's
  estimate to 1.04-1.05.
- `perf/alloc_sim.py` replays recorded allocation traces through a
  re-implementation of the caching allocator's rules. It reproduces the peak
  reserved memory of 13 recorded snapshots to the MiB, and I tried every
  allocator idea there before running it on a GPU.

### Auto mode's table and margins

The automatic window reads peak memory from a measured table, so memory
savings only become longer windows once the table is measured again.
`perf/vram_table.py` measures it the way the runtime uses it, per resolution
and window. On the original code path it reproduces upstream's own table
within 0.04 GiB, so the method matches. `--optimized` reads
`_VRAM_PEAK_GB_OPTIMIZED` in `corridorkey/runtime.py`, measured the same way.

The margin on top depends on the allocator in use: x1.05 + 0.3 GiB with
expandable segments, and on Windows x1.15 + 0.3 GiB for low-VRAM windows up
to 1440p, x1.5 + 0.3 GiB from 3K up and x1.08 + 0.3 GiB in normal mode.
`perf/tier_check.py` ran every pick in the table above end to end: on
Windows' terms (default allocator, no cap) each peak reserved stayed under
its budget, from 720p to 4K at 4.95 to 22.95 GiB, and every Linux pick ran
under a hard cap.

## Time

### Where it went

The starting point was an RTX 3060 desktop standing in for the laptop (the
same GA106 chip, the memory-efficient attention Windows uses, capped at the
laptop's 5.0 GiB): 1080p, 8/4 low-VRAM, 48 frames, 74.5 s.

| part | s | share |
|---|---|---|
| stage 2 | 33.6 | 45% |
| stage 1 | 29.1 | 39% |
| foundation prepass | 4.4 | 6% |
| RVM prepass | 4.1 | 6% |
| encoder after the last frame | 2.4 | 3% |
| model swaps and setup | 1.0 | 1% |

In low-VRAM mode every frame was decoded and prepared four times, and the
stage-1 and stage-2 reads ran inside the window loop, so the GPU waited on
numpy. Nsight Systems on a short run showed the GPU idle 35% of the time,
with 4,162 `cudaStreamSynchronize` calls taking 2.06 of 7.69 s: per-tensor
model swaps through pageable memory, and index uploads in the shifted-window
attention.

### What changed

| option | change | measured |
|---|---|---|
| `--prefetch_frames` | the next window decoded and prepared on a background thread, and one read of the clip for both prepasses | 1080p 8/4: 72.2 -> 61.4 s; prepasses on a 4-core host 21.4 -> 4.2 s |
| `--cache_region_indices` | shifted-window index tensors stay on the GPU; each upload was a full GPU sync | stage 2 on a 4-core host 545 -> 495 ms per frame; no change on a fast host |
| `--lean_reads`, `--read_ahead` | frames the model needs read on worker threads ahead of time, without display views nobody uses | not measured on their own |
| `--pinned_io` | stage-2 output quantized to 16 bits on the GPU and copied through pinned memory one batch late | stage-2 loop x1.115 on the RTX 5080, x1.029 on the laptop |
| `--fast_load` | the model built without random initialization; every value then comes from the checkpoints | model load with bf16 weights 4.36 -> 1.47 s |
| `--decode_once` | each frame decoded and converted once, the model's input read back from a temporary file (13 MB per 1080p frame, 53 MB per 4K frame) | 4K 8/4 with `--fast`: 294.4 -> 250.2 s |
| `--gpu_prep` | frame-folder color conversions on the GPU, with `--decode_once` | 4K 4/2, 32 frames: prepass 11.75 -> 6.49 s |
| `--async_baton` | low-VRAM stage-1 windows back to back on the GPU | next section |

Windows itself spends 13-20 s importing torch before the model loads; that
part is outside the runtime.

### The 4K stalls

At 4K, stage 1's kernels for the 120-frame clip add up to 70.6 s, but stage
1 kept the GPU busy only 63% (4/2) to 72% (8/4) of its time (torch.profiler).
py-spy on the main thread showed where it waited, per stage-1 window at 4K
4/2:

- a blocking copy of the carry hint to the host: 2.77 s over 15 windows,
  most of it waiting for the GPU
- a blocking copy of the weighted window output into pageable memory: 3.29 s
- the blending and the writes to the memory-mapped output: 2.89 s
- inside the forward, the late skip's blocking copy to host memory: 5.71 s

In stage 2 the output conversion swapped channels with a Python list index,
which uploads an index tensor with a blocking copy and so waits for the
whole frame (12.8 s over 32 frames).

`--async_baton` keeps the carry hint on the GPU (its bf16 conversion moves
there, and all 2,130,706,434 float32 values in [-1, 1] convert identically),
uploads the Hann weights once, and copies each window's output into an
exact-size registered buffer without blocking, then blends it while the GPU
runs the next window. The late skip's copies stopped blocking, and the
channel swap became a `flip`, which needs no index upload. 4K Tears of
Steel, 120 frames, `--fast`
(`perf/results/phase6/box2/tier3l.log`):

| window | before | + final `--gpu_prep`, `flip`, late skip | + `--async_baton` |
|---|---|---|---|
| 4/2 | 280.1 / 276.1 s | 251.4 / 251.7 s | 223.7 / 226.2 s |
| 8/4 | 262.6 / 262.2 s | 263.4 / 251.5 s | 235.4 / 236.1 s |

The first version took its buffers from PyTorch's pinned-memory cache. Stage
1 got 19% faster, but stage 2 slowed from 54-59 s to 85-120 s: the cache
rounds each buffer up to a power of two (the 570 MB window buffer to 1 GB)
and keeps it, and stage 2's own pinned copies then crawled. The exact-size
registered buffer avoids that without an allocator setting.

### The video encoder

The original writes video with x264 at two threads per output, which at 4K
encodes about one frame per second, slower than stage 2 produces them. So
the default run waits on x264. Tears of Steel's 120 frames encoded on their
own, CRF 12, 4:4:4 10-bit (`perf/results/phase6/box1/tier2b/encode.jsonl`):

| encoder | time | frames/s | PSNR | size |
|---|---|---|---|---|
| x264 medium, 2 threads (the default) | 121.4 s | 0.99 | 39.83 dB | 389 MB |
| x264 medium, 4 threads | 70.7 s | 1.70 | 39.83 dB | 389 MB |
| x264 medium, 8 threads (`--fast` on 16+ cores) | 43.9 s | 2.74 | 39.83 dB | 389 MB |
| NVENC HEVC 4:4:4, QP 12 | 16.8 s | 7.13 | 38.27 dB | 360 MB |

Eight threads give the same quality and size at 2.7 frames per second.
NVENC frees the CPU, but it needs a different codec and more bits for the
same quality, so I left it out. A shorter x264 lookahead (`rc-lookahead=10`)
saves 1-2 s at 1080p for at most 0.13 dB; I left that out too.

## Longer windows: what changes in the key

At the same window the output is identical, so any visible change in auto
mode comes from the window length alone. Against the original's 8-frame
render of the 8 test clips at 1080p
(`perf/results/phase3/cmp_opt16_vs_shipped.txt`, `cmp_opt24_vs_shipped.txt`):

| | mean abs alpha difference | edge flicker, 8-clip mean |
|---|---|---|
| 8 frames (original) | | 0.196 |
| 16 frames | 0.0019 (0.0002 to 0.0069 per clip) | 0.188 (-4.4%) |
| 24 frames | 0.0026 (0.0002 to 0.0100 per clip) | 0.192 (-2.5%) |

Edge flicker is the mean frame-to-frame alpha change inside the soft edge
(0.02 < alpha < 0.98), and the subject's own motion sets a floor under it.
It drops 4 to 13% on hair, glasses, cup and cyc, and is flat on the others
except smoke at 24 frames (+8%). The largest changes are local: on cyc the
24-frame key removes green tracking markers and tape that the 8-frame key
keeps; on glasses it makes both tinted lenses equally transparent where the
8-frame key makes one nearly opaque; on smoke the change is almost all in
the off-screen wall and rigging. Whether that is better is a compositor's
call, so "Auto, original" stays in the launcher.

## What I tried and dropped

| idea | exact | result |
|---|---|---|
| channels-last input for the 7x7 depthwise convs (cuDNN instead of PyTorch's generic kernel) | no | 33% slower: 81.3 vs 61.3 s at 1080p 8/4 |
| two or four frames per stage-2 call | no, max alpha difference 0.0117-0.0156 | 7-9% slower |
| smaller Swin window batches | no | no change in peak memory |
| cuDNN or flash attention in `--fast` | no, alpha MAE 0.7-2.4e-4 | stage 2 20% slower on the laptop; whole runs slower on the RTX 5080 (34.1 vs 32.6 s) |
| torch.compile | no, closer to an fp32 reference | 4K 8/4: 461.8 s with a warm cache vs 360.7 s without; out of memory in normal mode at HD |
| CUDA graphs for low-VRAM stage 2 | yes | x1.006, within noise |
| stage 2 alongside low-VRAM stage 1 | yes | 4% faster at HD, 13% slower at 4K (336.8 vs 298.6 s): the encoders then compete with stage 1 for the CPU |
| normal mode at 4K with overlapping windows | yes | 339.6 vs 303.7 s: the foundation models run twice per frame at 8/4 |
| splitting the frame conversions across threads | no | PyTorch's vectorized and scalar pow(1/2.4) disagree on 91% of float32 inputs |
| capping the process on Windows (`set_per_process_memory_fraction` with `max_split_size_mb`) | yes | worked in emulation; `CUDA error: unknown error` on the real laptop once the cache filled the card |
| `max_split_size_mb` alone | yes | no more failures, but with no cap it reserved 35.32 GiB for 10.13 GiB allocated |
| `roundup_power2_divisions` | yes | 3.37 vs 3.65 GiB at best (simulated), far short of the arena |
| a per-frame half-precision check for `--gpu_prep` | yes | prepass 10% slower; the EXR header says the same for free |
| bigger foundation-prepass batches; parallel decode for every stage-1 window | yes | no gain |

torch.compile is the interesting one. Against a full fp32 reference
(`perf/infer_with.py --fp32`, 1080p 8/4), it is closer than the bf16 path on
every mean error, because its fused kernels keep more in fp32: hair alpha
MAE 3.37e-4 against 4.08e-4 for the original. But it is slower here and runs
out of memory at HD, so it is not in `infer.py`.

## Upstream behavior I left alone

- The nondeterministic aux ConvTranspose2d ("When the original does not
  repeat itself").
- The sRGB conversion depends on the CPU thread count: PyTorch's vectorized
  and scalar pow(1/2.4) disagree on 91% of float32 inputs, so the way a frame
  is split across threads changes bits (`perf/elementwise_check.py`).
- The README's 6 GB row lists 720p at 24/12, which does not fit the 5.0 GiB
  a 6 GB Windows laptop has free. The runtime's own selector picks 8/4 there.
- On Volta and Turing the automatic settings run out of memory, because
  PyTorch has no fused bf16 attention below sm_80.
- `load_model` leaves `requires_grad` on, so autocast caches bf16 copies of
  every weight. `--bf16_weights` and `--frozen_weights` fix that, opt-in.

## Reproducing

PyTorch 2.9.1+cu128, cuDNN 9.10.2 and Python 3.12 everywhere: Windows 11 on
the laptop, Linux on rented GPUs (Vast.ai). The test clips come from Hollywood
Camera Work's green-screen plates
(https://www.hollywoodcamerawork.com/green-screen-plates.html), which are
free to use but not to re-host, so they are not in the repo;
`perf/prepare_clips.py` builds the clips from the downloaded plates. The 4K
clip is Tears of Steel, (CC) Blender Foundation, mango.blender.org.

| tool | what it does |
|---|---|
| `perf/bench.py` | timing and memory per stage in one process; `--sdpa` and `--mem_cap_gib` emulate the laptop, `--mem_snapshot` and `--torch_profile` record traces |
| `perf/infer_with.py` | runs `infer.py` with a memory budget, a forced attention backend, fp32 or torch.compile |
| `perf/same_files.py` | byte-for-byte comparison of two output folders |
| `perf/goldens.py`, `perf/compare.py` | reference renders and quality metrics |
| `perf/vram_table.py`, `perf/build_vram_table.py` | the auto-VRAM table |
| `perf/tier_check.py` | runs every auto pick end to end against its budget |
| `perf/alloc_sim.py`, `perf/frag_analyze.py`, `perf/mem_peak.py` | allocator replay and snapshot analysis |
| `perf/gpu_prep_probe.py`, `perf/quantize_check.py`, `perf/load_probe.py` | exhaustive exactness checks |
| `perf/laptop_run.py` | end-to-end wall time on Windows, model load included |
| `perf/results/phase6/scripts/arch_check.sh` | the per-GPU check above |

Not tested: datacenter Blackwell (B200), and anything outside NVIDIA, since
the runtime needs CUDA. The only real Windows machine was the RTX 3060
Laptop; the other Windows numbers come from Linux runs with the default
allocator and no cap, which is how Windows behaves. And `--fast` does not
yet pick its encoder thread count by resolution and core count.
